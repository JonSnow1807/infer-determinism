# infer-determinism on H100 NVL (2026-11-06)

The per-target JSONL files in this directory are the retained full run. This is the run operator's field report; supplemental repeat observations mentioned below are not included in those JSONL files.

## Environment
- 1x NVIDIA H100 NVL 94 GB, driver 595.99.02, NGC pytorch:26.08 image, fresh Python 3.12 virtual environment
- torch 2.14.0+cu130 (CUDA runtime 13.0, cuBLAS 13.1.1.3, cuDNN 9.24.0.43), triton 3.8.0, vllm 0.33.0, flashinfer-python 0.7.2.post1
  (vllm picked torch 2.14.0; torchaudio 2.11.0 mismatch is irrelevant here). `vllm-flash-attn` is not a separate dist; FA comes bundled in vllm.
- `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` for all runs, `VLLM_BATCH_INVARIANT=1`, `PYTHONUNBUFFERED=1`, image torch library removed from `LD_LIBRARY_PATH`.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` for all targets EXCEPT `vllm_invariant_mode`, which ran with it unset so that vLLM's own
  `init_batch_invariance()` sets it (observed value afterwards: `:16:8`, fp32 matmul precision `ieee`).

## Adapter / harness fixes (in this repository)
1. flashinfer: `plan(..., data_type=bf16)` rejected by 0.7.x for decode -> `q_data_type/kv_data_type` (error `dtype of q bfloat16 does not match q_data_type float16`).
2. torch_sdpa: FLASH_ATTENTION/CUDNN_ATTENTION were skipped for bf16/fp16 because flash rejects `is_causal` with Q != K (confirmed via the warning text). Q is now 32 (=K) for those two backends. Remaining skips (fp32 flash, fp32 cudnn) are real: backends do not support fp32.
3. flashinfer: `fixed_split_size` baseline was 0 and the perturbation compared fixed=1 vs auto, which is by design different; also units are PAGES in FlashInfer. Now: `BatchSpec.fixed_split_size` is in tokens (converted to pages), baseline fixed=64 tokens with 1024-token KV so splitting really occurs, added `disable_split_kv` field + cases (decode/prefill, tensor-core and CUDA-core decode), plus context-only default-planner long-KV cases. fixed_split_size sweep values changed to 16..512 tokens.
4. flashinfer: workspace 32 MB -> 1 GiB (long-KV x batch 257 overflowed `batch_prefill_tmp_v`; 12 errors in first attempt).
5. vllm_flash_attn: added long-KV (2048) cases: `num_splits=1` baseline (what vLLM forces under VLLM_BATCH_INVARIANT=1, see `v1/attention/backends/flash_attn.py`) and context-only default-splits case. Original 32-token cases kept.
6. core: new `big_m` perturbation (M up to 4096), `other_kv_length` list extended with 1025/3000, `disable_split_kv` field.
7. New target `vllm_invariant_mode`: calls `init_batch_invariance()` then plain `torch.mm/F.linear/softmax/log_softmax/mean` + `rms_norm_batch_invariant` at realistic dims (K/N 4096..14336, vocab 32000/128256, bf16/fp16/fp32). Reason: on SM90 vLLM does NOT install aten::mm overrides (it relies on cuBLAS settings); the original `vllm_batch_invariant` target only calls the Triton kernels directly, which is not what runs for raw torch ops on H100.
8. runner: `torch.backends.cudnn.allow_tf32` getter raises after `init_batch_invariance()` (mixed legacy/new TF32 API) and crashed divergence-record creation; wrapped in a safe getter, also records `matmul_fp32_precision`.
- Original targets ran with no further fixes: torch_matmul, vllm_batch_invariant, moe, vllm_flash_attn (adapter paths/signatures were right for vllm 0.33.0: `vllm.model_executor.determinism.batch_invariant`, `vllm.vllm_flash_attn`).
- Remaining skips: torch_sdpa float32/FLASH_ATTENTION and float32/CUDNN_ATTENTION ("No available kernel": fp32 unsupported by those backends).

## Full report table (grouped by case; see the per-target Markdown reports for first divergent configs)
Per-target record counts: torch_matmul 164 divergences/392 checks/8 pass; torch_sdpa 0 div/340 checks/10 pass/2 skip;
vllm_batch_invariant 0 div/408 checks/12 pass; vllm_invariant_mode 94 div/2586 checks/50 pass; moe 0 div/34 checks/1 pass;
flashinfer 34 div/415 checks/7 pass; vllm_flash_attn 24 div/256 checks/5 pass. 0 errors in final runs.

| Target | Case | Perturbation | First-divergence max abs / #elems / max ULP | Class |
|---|---|---|---|---|
| torch_matmul | float32/{mm,linear}/{eager,compile} | batch, tile_m, position | mm: 3.8e-6 / 95 / 320; linear: 5.7e-6 / 92 / 120 | by design (plain torch fp32 GEMM, no claim); bf16/fp16 mm/linear did NOT diverge |
| vllm_invariant_mode | mm/float32/K4096xN4096 | batch, tile_m, big_m, position | 1.07e-6 (M=2) .. 9.5e-6 (M>=31) / 3644..3990 of 4096 / 77824 (M=2), 41254 (M>=31) | BUG candidate (B1) |
| vllm_invariant_mode | mm/float32/K4096xN14336 | batch, tile_m, big_m, position | 1.19e-6 (M=2) .. 1.10e-5 / 11806..13976 of 14336 / 28672..42790 | BUG candidate (B1) |
| flashinfer | fa2_disable_split_kv_cudacore/batch_decode_long | batch (B>=31), position | 4.9e-4..9.8e-4 / 92..140 of 256 / 28..37 | BUG candidate (B2), already filed |
| flashinfer | ctx_default_planner/batch_decode_long | batch, position | 4.9e-4..9.8e-4 / 92..140 / 28..37 | by design (default planner picks split by batch; no claim) |
| flashinfer | fa2_fixed_split64tok/batch_{decode,prefill}_long | fixed_split_size (16,32,128,..) | 6.1e-4..9.8e-4 / 115..1182 / 39..29177 | by design (changing the split size changes reduction order) |
| vllm_flash_attn | FA3/ctx_paged_varlen_long_default_splits | batch, position | 2.4e-4 / 97 / 33 | by design (num_splits=0 heuristic; vLLM forces 1 in invariant mode) |

No divergence (all checks equal) for: vllm_batch_invariant (matmul/log_softmax/mean/rms_norm, bf16/fp16/fp32, hidden 128); vllm_invariant_mode bf16
mm/linear/linear_bias/matmul3d, fp16 mm, softmax, log_softmax, mean, rms_norm at the tested dims (bf16, fp32 for the elementwise ops) incl. M up to 4096;
vllm_flash_attn FA2 and FA3 with `num_splits=1`, long KV 2048, batch/position/companion-KV/block_size/CUDA-graph replay (graph replay==eager inputs) ;
flashinfer fa2 `fixed_split_size=64 tokens` (decode+prefill, tensor cores) and `disable_split_kv` (tensor-core decode, prefill) across batch/position/companion KV/block_size/chunk;
moe (fused_experts); torch_sdpa all runnable backends (repeat/batch/position); repeat-run determinism everywhere (no run-to-run nondeterminism found).

## BUG candidates

### B1. fp32 `torch.mm`/`F.linear` is not batch-invariant on H100 after vLLM `init_batch_invariance()`
- Claim: `VLLM_BATCH_INVARIANT=1` / `enable_batch_invariant_mode()` (vllm 0.33.0 `model_executor/determinism/batch_invariant.py`): "Hopper (SM90) ...: the only source of batch variance is split-k, which we disable via the cuBLAS workspace config" (no aten::mm override installed on SM90). Also sets fp32 precision to ieee.
- Minimal repro: `repro/min_fp32_mm.py` (also generated `repro/vim_fp32_mm_k4096n4096_b2.py`, record `a10fb5178f1cebd5`). K=N=4096, `torch.mm(x[:1], w.t())` vs `torch.mm(x[:M], w.t())[:1]`.
- Observed (identical on 3/3 runs, clean shell each): fp32 M=1 vs M=2/3/8: max abs 1.19e-6, 3606/4096 elements differ; M=31..64: max abs 9.78e-6, 3972 elements differ. Generated repro (M=2, different data): max abs 1.07e-6, 3644 elems, 77824 ULP, 3/3 "bitwise equal: False". bf16 same shapes: equal for every M (3/3). vLLM's own Triton `matmul_batch_invariant` fp32 M=1 vs 64: equal (3/3).
- Existing issues: `gh search` for vllm "batch invariant fp32/float32/Hopper cublas": none specific. Related: vllm #55131 (open; Triton matmul TF32 / not invariant), PR #55466 (open; fp32 ieee in Triton dots), #52960 (merged; fp32 N=1 OOR SM89), and issue #65569 (open; proposes routing Linear to cuBLAS on SM90/SM100 because "cuBLAS is already batch-invariant under the mode's settings" - its evidence is bf16/fp16 only; my result says fp32 is not).
- Impact scope (verified in source): in vllm 0.33.0 vLLM `Linear` layers call the Triton `linear_batch_invariant` (`layers/linear.py:241`), so model Linear layers are not affected; only raw torch fp32 GEMMs outside that path (e.g. direct `torch.mm`/`F.linear` in model code) and the #65569 proposal would be. Low severity.
- Root-cause guess: cuBLAS SGEMM (non-tensor-op fp32, ieee) heuristic selects different kernel/tile (different K accumulation order) for M=1 vs M>=2 vs M>=31 even with split-K disabled. Not verified by profiling kernel names -> treat as unconfirmed.

### B2. FlashInfer CUDA-core decode ignores `disable_split_kv` -> batch-variant for B>=31
- Claim: `BatchDecodeWithPagedKVCacheWrapper.plan(disable_split_kv=True)` (documented batch-invariance control).
- Repro: `repro/flashinfer_cudacore_disable_split_kv_b31.py` (record `7dadfae07cec90a5`): `use_tensor_cores=False`, backend fa2, 4 heads, D=64, bf16, page 16, KV 1024; probe alone vs probe in batch of 31 identical companions-length requests.
- Observed 3/3 runs: bitwise equal False, max abs 4.88e-4, 140/256 elements, 28 ULP. Batch 64: 9.8e-4/127/37; 128 and 257: 4.9e-4/92/33. Same flag with `use_tensor_cores=True` (and prefill) was invariant in all checks.
- Existing issue: flashinfer-ai/flashinfer #7981 (open) describes exactly this ("CUDA-core decode silently ignores disable_split_kv", changes bits past batch 16). Related: #7982 (cuDNN SDPA differs across processes on H100), #7937, #7924. No new filing needed.
- Root-cause: per #7981 the CUDA-core planner `DecodePlan` (scheduler.cuh) decides split_kv from workload only. I did not read the C++ source myself ("consistent with the issue text, not independently verified"). Note my observation of "past 16" is not reproduced precisely: first divergence in my sweep is at B=31 (sizes tested: 2,3,7,8,15,16,31,...; 15/16 equal).

## Not bugs / context
- Plain torch fp32 GEMM varying with M (torch_matmul), FlashInfer default planner, FA3 default num_splits, changing fixed_split_size: all by design, reported only as context.
- No run-to-run nondeterminism, no CUDA-graph replay != eager mismatch (FA2/FA3, num_splits 0 and 1), no vLLM bf16/fp16 invariant-op failure found. Coverage limits: head dim 64, 4 heads, no sliding window/GQA/FP8 KV, single GPU, M tested up to 4096.

## Files
- Per-target `.jsonl` and `.md` final results; `repro/` scripts; `versions.txt`. The earlier smoke runs and superseded full run are not retained here.
