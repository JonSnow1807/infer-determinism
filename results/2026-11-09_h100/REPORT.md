# infer-determinism run 2 + batch-invariant decode profile (2026-11-09, H100 NVL)

This is the run operator's field report. The summary is in [README.md](README.md).

Environment: 1x H100 NVL 94 GB (shared with a concurrent PyTorch source build and test job; the harness and profiling jobs did not overlap each other, but timings carry contention noise), driver 580.178.04 (run 1 used 595.99.02), NGC pytorch:26.08 image, fresh venv.
torch 2.14.0+cu130, triton 3.8.0, vllm 0.33.0, flashinfer-python 0.7.2.post1 (flashinfer-cubin 0.6.6 mismatched -> cubins downloaded on demand), the same library versions as run 1.
Env: VLLM_BATCH_INVARIANT=1, CUBLAS_WORKSPACE_CONFIG=:4096:8, TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0, image torch libs stripped from LD_LIBRARY_PATH. Note: the image sets PIP_CONSTRAINT, which makes pip silently backtrack to vllm 0.19.1/torch 2.10; `unset PIP_CONSTRAINT` is required.
Files: `torch_sdpa|flashinfer|vllm_flash_attn.{jsonl,md}`, `repro/`, `part2_profile/`, `versions.txt`, `pipfreeze.txt`.

## Part 1: run 2 (GQA/MQA, D128/D256, sliding window, FP8 KV, graph replay)
Only run-2 cases were run (`--include "run2/|q[0-9]+_kv[0-9]+_d[0-9]+"`, full matrix, no `--quick`); no run-1 config was repeated.
No adapter fixes were needed (0 errors, 0 skips in all three targets). Only harness change: new `--include REGEX` option (runner.py, __main__.py). 24 CPU tests pass.

| Target | cases | checks | divergences | errors/skips |
|---|---:|---:|---:|---|
| torch_sdpa (FLASH, EFFICIENT, CUDNN, MATH; 32/8, 64/8, 8/1; D128) | 12 (8 pass) | 358 | 50 | 0/0 |
| flashinfer FA2 decode+prefill (GQA/MQA D128/D256, window 1024/4096, FP8 KV, FP8+window) | 27 | 1394 | 6 | 0/0 |
| vllm_flash_attn FA2/FA3 num_splits=1 (same matrix, FA3 FP8) | 25 | 1175 | 0 | 0/0 |

### Triage
- vLLM FA2 and FA3 with num_splits=1: all 25 cases invariant to batch, position, companion KV length, page size and graph replay == eager, including D256, windows, FP8 (FA3). No finding.
- FlashInfer FA2 with disable_split_kv=True, decode (tensor cores): invariant everywhere incl. GQA/MQA, D256, window, FP8, graph replay. Prefill without window: invariant across batch/position/page/chunk.
- torch_sdpa: 50 divergences = MATH backend (by design, no claim; <=4 ULP) and cuDNN q64_kv8 at batch 257 only (84 ULP; SDPA makes no batch-invariance claim; same family as flashinfer #7982). Class: by design / no claim. FLASH and EFFICIENT invariant.

### BUG candidate C1: FlashInfer FA2 paged prefill, CUDA-graph wrapper != eager wrapper at head dim 128 (same batch)
- Cases: `run2/fa2_nosplit/batch_prefill/q32_kv8_d128/window1024`, `.../window4096`, `.../window4096/fp8_e4m3` (graph_replay, probe 0 and 2). Records 2c181fbf4a9186f4, 98a826d7c64c0515, 83a5a00afc5be2f7 (+ probe-2 twins). D256 prefill and all decode: equal.
- Claim: run-2 contract "graph replay must equal eager for the same final batch" (disable_split_kv=True, fixed everything else); this is the property vLLM relies on if FlashInfer prefill is used under graphs with VLLM_BATCH_INVARIANT=1. FlashInfer itself does not document bitwise graph==eager, so severity is low and "claim" is the harness/user contract, not a FlashInfer docstring.
- Observed 3/3 (fresh processes, `repro/fi_<id>.py`, run with `python -m infer_determinism repro <id>`): bitwise equal False; max abs 9.8e-4 (w1024, 5704 elements differ), 4.9e-4 (w4096, 4462; FP8 4527).
- Not a capture-length or window effect: `repro/diag_graph_capture_length_sweep.py` shows every capture length (1 .. final length, even capture length == final length) differs from eager for D128, and every one is equal for D256.
- Root cause (read from installed source, not tested by patching): `scheduler.cuh` PrefillPlan chooses `cta_tile_q` from the average packed qo length in the eager path (here 8 q x 4 gqa = 32) but from `max_qo_len = (total_rows - batch + 1) * gqa_group` (here 22*4 = 88) when `enable_cuda_graph`; a different Q tile changes the online-softmax tiling and thus the bits. Same mechanism as open flashinfer-ai/flashinfer#7924 (cta_tile_q depends on batch-mates; PR #2830 "fixed cta_tile_q" open). D256 probably maps both to the same tile.
- Upstream search (flashinfer, vllm, flash-attention, pytorch; issues+PRs; terms: cuda graph, bitwise, batch invariant, sliding window, window_left, cta_tile_q): no issue states the graph-vs-eager tile difference. Related: flashinfer #7924 + PR #2830 (tile choice), #9009 (batch-invariant tracker), #7937, #7981 (known, run 1), #7982 (cuDNN). Treat C1 as a new angle on #7924, not a new bug class. Nothing has been posted upstream.

## Part 2: vLLM batch-invariant decode, Qwen3-8B bf16 TP=1 (vllm#65569 / tracker #27433)
Method: offline `LLM`, in=512, out=256, random token ids, prefix caching off, CUDA graphs + torch.compile on (default), max_model_len 1024. Decode tok/s = B*255/(T256 - T1) (medians of 3, T1 = prefill + 1 token). Per-kernel time: torch profiler over 8 steady-state decode steps (engine.step()), kernel-sum is device time per step. Two repetitions (rep1 in `part2_profile/`, rep2 in `part2_profile/rep2/`); GPU was shared, so expect ~10% noise.
GEMM shapes from a third, eager run with a recording wrapper around `mm_batch_invariant` (`inv1_b*_eager.json`). Note: in vllm 0.33.0 under the mode, `UnquantizedLinearMethod.apply` calls Triton `linear_batch_invariant` -> `matmul_persistent` for ALL Linear layers on SM90 (including lm_head), not cuBLAS.

| B | mode | decode tok/s (rep1 / rep2) | ms per step | kernel-sum ms/step | GEMM ms/step (share) | attention ms/step (share) |
|---|---|---|---|---|---|---|
| 32 | default | 4279 / 3661 | 7.5 / 8.7 | 6.9 / 8.3 | cuBLAS 4.95 / 6.20 (72-75%) | 1.4 / 1.5 (18-20%) |
| 32 | invariant | 3492 / 3403 | 9.2 / 9.4 | 8.5 / 8.5 | Triton 6.00 / 6.08 (71%) | 1.3 / 1.3 (15%) |
| 128 | default | 8202 / 9166 | 15.6 / 14.0 | 14.2 / 11.8 | cuBLAS 7.37 / 6.33 (52-54%) | 5.7 / 4.5 (39-40%) |
| 128 | invariant | 8612 / 8549 | 14.9 / 15.0 | 13.6 / 13.4 | Triton 7.88 / 7.71 (58%) | 4.2 / 4.2 (31%) |

- Share of decode device time in the Triton batch-invariant GEMM (`matmul_kernel_persistent`): 71% at B=32, 58% at B=128 (1160 launches per 8 steps = 4 per layer x 36 + lm_head).
- End-to-end slowdown vs default: B=32 +22% and +8% decode ms/step in rep1 and rep2, each against its own default run (decode tok/s -18% and -7%; noisy); B=128 -5% to +7% (within noise; invariant mode's attention is cheaper: num_splits=1, no combine kernel, about -0.3 to -1.5 ms/step, which offsets a GEMM that is ~+0.3 to +1.5 ms/step slower).
- Dominant GEMM shapes (M = batch; N x K; share of Triton GEMM time at B=32 / B=128, time per call B=32 / B=128):
  gate_up 24576x4096: 45% / 42% (76 / 85 us); down 4096x12288: 24% / 25% (41 / 50 us); qkv 6144x4096: 14% / 16% (24 / 32 us); o_proj 4096x4096: 10% / 10% (17 / 19 us); lm_head 151936x4096: 6% / 7% (once per step, 355 / 509 us). Every Linear is M=B, K in {4096, 12288}.
- Standalone microbench (`gemm_micro.py`, unloaded GPU, no graphs): Triton-BI vs cuBLAS time ratio at M=32: qkv 1.62x, o_proj 2.70x (Triton ~28 us launch-bound floor), gate_up 1.31x, down 1.28x, lm_head 1.02x; at M=128: 1.48x, 1.74x, 1.19x, 1.25x, 1.23x; at M=512 about 1.15-1.23x everywhere. So the real GEMM-level gap is 1.2-1.3x on the two biggest shapes, and small-N shapes are inflated by launch overhead that CUDA graphs hide in the e2e run.
- Takeaway for the CUTLASS fixed-order GEMM question: the ceiling is the GEMM gap of roughly 1.2-1.3x on gate_up/down (about 70% of GEMM time), i.e. at most ~5-15% end-to-end decode at B=32 and ~0-10% at B=128 (where invariant mode already breaks even because of cheaper attention). A hand-built kernel would need to be within ~1.0-1.1x of cuBLAS to matter; the evidence here does not show a large win, and bit-exactness to the Triton order is not required (only batch invariance).
- Caveats: one model, two batch sizes, two reps, shared GPU, decode includes sampler/scheduler CPU time (hence wall ms/step > kernel-sum).
