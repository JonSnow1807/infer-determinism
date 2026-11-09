# Second H100 run

This is an executable coverage plan, not a second GPU result. The first run used four Q/KV heads and D64. This increment adds 67 attention cases (15 included in quick); optional libraries and unsupported kernels can reduce the runnable count. Run `python3 -m pytest -q tests` before renting the GPU. CPU tests exercise adapter contracts with mocks; they cannot establish CUDA kernel support, graph safety, numerical accuracy, or H100 timing.

## Exact additions

All new cases use BF16 outputs and the existing `Case`, `BatchSpec`, and `cases()` APIs. Full-only selection is the optional `shapes["full_only"]` flag, checked before running a baseline. Names encode heads, dimension, window and FP8 mode; JSONL divergence/skip shapes include the effective configuration and scales.

| Family | Target / path | Full matrix | Quick subset | Added cases |
|---|---|---|---|---:|
| GQA/MQA | `torch_sdpa`, forced FLASH_ATTENTION, EFFICIENT_ATTENTION, CUDNN_ATTENTION, MATH | Q/KV heads 32/8, 64/8, 8/1; D128; BF16 | 32/8, all four backends | 12 |
| GQA/MQA | FlashInfer FA2 tensor-core paged decode and prefill | All three head pairs × D128/D256; BF16 | 32/8 D128, both wrappers | 12 |
| GQA/MQA | vLLM raw FA2 and FA3 paged varlen decode | All three head pairs × D128/D256; BF16 | 32/8 D128, both versions | 12 |
| Sliding window | FlashInfer decode/prefill; vLLM FA2/FA3 | 32/8 × D128/D256 × W1024/W4096; BF16 | D128 W1024, all four paths | 16 |
| FP8 KV | FlashInfer decode/prefill; vLLM FA3 | All three head pairs at D128, plus 32/8 D256 | 32/8 D128, all three paths | 12 |
| FP8 + sliding | FlashInfer decode/prefill; vLLM FA3 | 32/8 D128 W4096 | Full-only | 3 |

The FlashInfer and FA names start with `run2/`; SDPA names append `q32_kv8_d128`, `q64_kv8_d128`, or `q8_kv1_d128`. FA2 FP8 is deliberately not generated: FA3 is the requested FP8 interface. FA versions that the installation rejects remain visible as skips instead of disappearing from the manifest.

- SDPA uses native `enable_gqa=True`; it does not expand/repeat KV heads as a fallback. Q/K length is 32/32 for Flash/cuDNN and 8/32 for math/efficient, matching the original target's causal layout. Perturbations are repeat, batch size and probe position.
- New FlashInfer and FA probes use K257 without a window, K1152 for W1024, and K4224 for W4096. Companion K starts at 32 to bound memory, and the existing companion-length sweep still applies. FlashInfer Q is 1 for decode and 8 for prefill; FA Q is 1.
- New FlashInfer cases force `backend="fa2"`, tensor cores for decode, and `disable_split_kv=True`. New FA2/FA3 cases fix `num_splits=1`. Repeat, batch, position, companion KV length, and page-size perturbations apply. FA pages remain 16/32/64; FlashInfer retains 8/16/32/64. Existing fixed-split controls and chunked-prefill cases remain available.
- Every new FA case and every new FlashInfer window case includes graph replay with a three-request batch and probe at positions 0 and 2. Capture sees at most half the final probe KV length, crossing the window boundary on replay. Each request's capture length is capped by its own allocated KV length.

W means **total visible tokens including the current token**: FlashInfer receives `window_left=W-1`; FA receives `window_size=(W-1, 0)` and `causal=True`. Window values, precisions and head shapes are separate cases, not perturbations compared with one another.

## Graph contract

The runner compares graph replay with an eager call using the **same final batch, probe position and lengths**. Batch-versus-alone comparisons remain separate. This reference configuration is recorded in checks, divergences and standalone reproducers.

FlashInfer zero-initializes each new workspace, as required by its decode constructor. It keeps physical batch size and Q count fixed at three requests (three decode queries or 24 prefill queries). It allocates persistent CUDA graph metadata buffers at final capacity, plans and warms the final length, plans the shorter capture length, captures only `run`, then replans the final length outside capture and replays. Page indices retain each request's original physical pages while lengths change. Split KV is disabled because a fixed split alone can still change the number of CTAs when KV grows; FlashInfer's [decode documentation/source](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2/flashinfer/decode.py) describes this limitation.

FA3 and windowed FA2 capture one active query within capacity for three and replay all three. Both query prefix sums and visible KV lengths are restored in place; position 2 was inactive during capture. Non-window FA2 GQA/MQA keeps all three queries active and tests KV-length replay only: its [single-query GQA transpose optimization](https://github.com/vllm-project/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp) discards query prefix sums. New FA case shapes record this distinction as `graph_query_activation`. The graph call supplies no capture-time scheduler metadata. Eager FA3 scheduling receives the same heads, dtype, causal flag and window as its kernel. Both adapters poison the output buffer before replay and reject nonfinite/unwritten active outputs.

These raw FA probes are **split-controlled kernel probes**, not an implementation of vLLM's entire invariant policy. The native builder also fixes maximum sequence length to the model limit and disables AOT scheduling in invariant mode; this harness retains raw eager scheduling and batch-derived maximum length. See [NOTES](../NOTES.md) for why no `vllm_attention_layer` target is registered yet.

## FP8 contract

Storage is `torch.float8_e4m3fn`, with fixed, distinct per-tensor dequantization scales K=0.25 and V=0.125. Quantization is `clamp(x / scale, -448, 448).to(float8_e4m3fn)`; scales never depend on companion data. Zero padding is initialized. The probe's stored bytes remain identical across batch positions and page layouts.

FlashInfer keeps Q in BF16 and supplies `k_scale` and `v_scale` to `run`. FA3 requires FP8 Q/K/V together on the inspected interface, so its case also quantizes Q with scale 0.5. All three FA descales are float32 tensors shaped `[batch, num_kv_heads]`, containing a single repeated per-tensor value, including Q's descale. FA output is explicitly allocated in BF16. Thus FlashInfer FP8-KV and FA3 FP8-QKV are different numerical experiments, not cross-library comparisons. Shapes record Q dtype and all applicable scales; `Case.dtype` records the output dtype.

## API assumptions to verify on the box

The following calls are inferred from source/documentation, not executed against installed GPU libraries here. The first run had torch 2.14.0, vLLM 0.33.0 and FlashInfer 0.7.2.post1; upstream `main` can differ from those wheels. Capture exact versions and inspect signatures on the H100 before treating skips as coverage.

| API | Assumed signature / behavior | Failure policy |
|---|---|---|
| [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | `scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)` under a forced `sdpa_kernel` backend | Missing keyword or no available kernel is a skip; no KV repetition fallback. Backend GQA availability is version-dependent. |
| FlashInfer decode constructor | `BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2", use_tensor_cores=True)`; graph kwargs `use_cuda_graph`, `paged_kv_indptr_buffer`, `paged_kv_indices_buffer`, `paged_kv_last_page_len_buffer` | Missing constructor/keyword is a precise skip. |
| [FlashInfer prefill constructor](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2/flashinfer/prefill.py) | `BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")`; graph buffers use `_buf`, plus `qo_indptr_buf` | Missing keyword is a skip. Query capacity is established by the first plan. |
| FlashInfer plan/run | Decode `plan(ptr, indices, tail, Hq, Hkv, D, page, q_data_type=..., kv_data_type=..., window_left=..., disable_split_kv=True)`; prefill adds `qo_indptr` first and `causal=True`; `run(q, cache, k_scale=..., v_scale=..., out=...)` | Window/scales/out checked when requested. Mixed BF16-Q/FP8-KV FA2 support and D256 must be verified for each installed wrapper. |
| [vLLM raw FA interface](https://github.com/vllm-project/vllm/blob/main/vllm/vllm_flash_attn/flash_attn_interface.py) | `flash_attn_varlen_func(..., seqused_k, block_table, fa_version=2/3, num_splits=1, window_size=(W-1,0), causal=True, out=...)`; FA3 adds `q_descale`, `k_descale`, `v_descale` | Version-support helper, feature keyword checks and known build/dimension rejections produce skips. Verify paged FP8 D128/D256 kernel availability. |
| FA3 scheduler | `get_scheduler_metadata(B, 1, max_K, Hq, Hkv, D, seqused_k, qkv_dtype=..., cu_seqlens_q=..., page_size=..., num_splits=1, causal=True, window_size=...)` | Optional scheduler is used only if present; a present scheduler lacking required feature keywords is a skip. Graph launch omits precomputed metadata. |

Signature inspection cannot prove kernel support, and some C bindings expose no signature. Runtime handling recognizes explicit missing-keyword/no-kernel and known FA build/dimension restrictions. Other failures, including bad shape/stride/scale assertions, OOM, JIT failures and illegal memory accesses, remain **errors**, with their original messages. Unknown support rejections should be reviewed on the box and classified narrowly; do not turn all `RuntimeError`s into skips. A skip is never evidence of invariance.

## Expected differences and bug candidates

| Comparison | Interpretation |
|---|---|
| Different split counts, fixed split sizes, TF32 modes, FP8 versus BF16, or different windows | Different reduction trees, arithmetic or attention semantics can change bits by design. Separate FP8/window cases are not compared across configurations. |
| Existing FlashInfer default planner or FA3 `num_splits=0` context cases under batch changes | Heuristic split selection can change reductions; divergence is expected context, not an invariant-mode regression by itself. |
| New split-controlled FlashInfer/FA cases under batch, position, companion length or page changes | Candidate failure of the requested invariance controls. Reproduce and check the installed backend's contract before attributing it to vLLM. No raw result alone establishes a native vLLM defect. |
| Replay versus eager for the same final batch, including window/FP8 cases | Unexpected; investigate graph metadata, scheduling and kernel execution. Unwritten/nonfinite output is an error, not a successful equality check. |
| Repeats with identical inputs and execution settings | Unexpected divergence; investigate nondeterminism. |
| PyTorch SDPA batch changes | Useful invariance observations; SDPA does not promise bitwise equality across all backends/shapes. A mismatch alone is not a correctness bug. |

The sliding probes target the failure area reported in [vLLM #65587](https://github.com/vllm-project/vllm/issues/65587), but do not reproduce a whole Gemma-2 model, its alternating layers, softcapping, scheduler, or native graph runner. An H100 preflight should also check a small masked reference: changing KV strictly outside the causal window must not affect output, while changing an included V token must. CPU mocks verify window forwarding and boundaries but cannot verify that the GPU kernel obeys the mask.

## Execution and budget

Quick retains the original small kernels and adds only 32/8 D128 GQA, W1024, and representative FP8 cases. All D256, W4096, 64/8 and 8/1 additions are full-only. To make room, compiled torch GEMMs, legacy vLLM GEMMs other than `mm` K4096/N4096, and legacy reductions wider than D4096 are now full-only. `big_m` perturbations are also full-only. Full mode retains the original matrices. Quick's design budget is about five minutes on an H100 with dependencies and JIT caches ready; this is **not timed/verified here**, and first-use compilation can exceed it.

Run targets in fresh processes so `init_batch_invariance()` cannot alter subsequent raw-kernel targets. For the three changed attention targets:

```bash
export VLLM_BATCH_INVARIANT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p results/run2
python3 -m pip freeze > results/run2/versions.txt
nvidia-smi > results/run2/nvidia-smi.txt
for target in torch_sdpa flashinfer vllm_flash_attn; do
  time python3 -m infer_determinism run --quick --targets "$target" --out "results/run2/${target}-quick.jsonl"
  python3 -m infer_determinism report "results/run2/${target}-quick.jsonl" > "results/run2/${target}-quick.md"
done
# Same loop without --quick runs the entire added matrix.
# Preserve the reviewed diff too: no commit is required for this work.
git diff > results/run2/reviewed.patch
```

Also retain new untracked helper/test/docs files with the run snapshot; `git diff` alone omits them. Pin the captured package versions when repeating a run. Review every skip/error before claiming coverage. Keep native invariant-mode runs separate and preserve their deliberate cuBLAS environment setting from the first-run notes.

CPU validation in this checkout is recorded in the task response; run commands are `python3 -m pytest -q tests` and `python3 -m infer_determinism run --quick --out /tmp/infer-determinism-run2-cpu.jsonl`. With no CUDA, all seven GPU targets must skip without importing vLLM/FlashInfer or producing errors.
