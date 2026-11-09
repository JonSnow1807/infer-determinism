# H100 NVL run 2: 2026-11-09

This run executed the second-run cases from the [run-2 plan](../../docs/run2-plan.md): GQA/MQA head layouts (32/8, 64/8, 8/1), head dimensions 128 and 256, sliding windows of 1024 and 4096 tokens, scaled e4m3 FP8 KV cache, and CUDA-graph replay. Only run-2 cases were selected (`--include`), in full mode. It also profiled vLLM decode with and without `VLLM_BATCH_INVARIANT=1`. The run operator's field report is [REPORT.md](REPORT.md).

## Environment

One NVIDIA H100 NVL 94 GB with driver 580.178.04, a different machine from run 1 (driver 595.99.02). Library versions match run 1: PyTorch 2.14.0, vLLM 0.33.0, FlashInfer 0.7.2.post1, Triton 3.8.0 ([`versions.txt`](versions.txt), [`pipfreeze.txt`](pipfreeze.txt)). The JSONL metadata records `VLLM_BATCH_INVARIANT=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The GPU was shared with an unrelated PyTorch build and test job, so the profiling timings in part 2 carry contention noise. The bitwise comparisons in part 1 are not affected by timing.

## Part 1: recorded outcomes

| Target | Cases | Equal checks | Divergences | Errors / skips | Report |
|---|---:|---:|---:|---|---|
| `torch_sdpa` | 12 | 358 | 50 | 0 / 0 | [Markdown](torch_sdpa.md) · [JSONL](torch_sdpa.jsonl) |
| `flashinfer` | 27 | 1,394 | 6 | 0 / 0 | [Markdown](flashinfer.md) · [JSONL](flashinfer.jsonl) |
| `vllm_flash_attn` | 25 | 1,175 | 0 | 0 / 0 | [Markdown](vllm_flash_attn.md) · [JSONL](vllm_flash_attn.jsonl) |

- **vLLM FlashAttention FA2 and FA3 with `num_splits=1`:** no divergence in any of the 25 cases, across batch, position, companion KV length, page size and graph replay, including D256, sliding windows and FA3 with FP8 KV.
- **FlashInfer decode with `disable_split_kv=True` (tensor cores):** no divergence, including GQA/MQA, D256, windows, FP8 KV and graph replay. Prefill without a window had no divergence across batch, position, page size and chunking.
- **`torch_sdpa`:** the 50 divergences come from the MATH backend (at most 4 ULP) and from the cuDNN backend in the 64/8 layout at batch size 257 only. SDPA makes no batch-invariance claim, so these are expected. FLASH and EFFICIENT had no divergence.

## Candidate

- **FlashInfer FA2 paged prefill at head dimension 128: CUDA-graph replay differs from eager for the same batch.** All 6 FlashInfer divergences are this case, in the window-1024, window-4096 and window-4096 FP8 cases (two probe positions each). The maximum absolute difference is 9.8e-4 (window 1024) and 4.9e-4 (window 4096), and it reproduced in 3 of 3 fresh processes. [`repro/diag_graph_capture_length_sweep.py`](repro/diag_graph_capture_length_sweep.py) shows that every capture length differs from eager at D128, including a capture length equal to the final length, and that D256 is equal. A likely cause, read from the installed source and not tested by patching, is that the prefill planner picks its query tile size from the maximum query length when CUDA graphs are enabled and from the average otherwise. This is the same mechanism as the open [flashinfer-ai/flashinfer#7924](https://github.com/flashinfer-ai/flashinfer/issues/7924). FlashInfer does not document bitwise equality between graph and eager execution, so severity is low. Reproduce with the `repro/fi_*.py` scripts.

## Part 2: cost of vLLM batch-invariant decode

Qwen3-8B, bf16, one GPU, 512 input and 256 output tokens, CUDA graphs on, two repetitions, torch profiler over 8 decode steps ([`part2_profile/`](part2_profile/)).

| Batch | Mode | Decode ms/step (rep 1 / rep 2) | GEMM share of decode device time |
|---|---|---|---|
| 32 | default | 7.5 / 8.7 | cuBLAS 72-75% |
| 32 | invariant | 9.2 / 9.4 | Triton 71% |
| 128 | default | 15.6 / 14.0 | cuBLAS 52-54% |
| 128 | invariant | 14.9 / 15.0 | Triton 58% |

- In invariant mode every `Linear` layer, including `lm_head`, runs through vLLM's Triton persistent matmul.
- At batch 32, invariant decode was 22% and 8% slower per step in the two repetitions. At batch 128 it was within noise of the default, because its attention is cheaper and offsets the slower GEMM.
- Standalone, the Triton GEMM is 1.2-1.3x slower than cuBLAS on the two largest shapes (gate_up and down projections). So a faster fixed-order GEMM could gain at most about 5-15% end-to-end at batch 32 and 0-10% at batch 128 in this setup.

## Coverage

One GPU, one model for profiling, two batch sizes and two profiling repetitions. An equal comparison establishes invariance only for the recorded case and perturbation. It does not establish numerical accuracy or invariance for other shapes and software stacks.
