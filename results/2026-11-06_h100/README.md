# H100 NVL run: 2026-11-06

This run compared raw output bits for a fixed probe request against variations in batch size and position, GEMM M dimension, companion KV length, attention split settings, and supported graph replay paths. The JSONL files contain every comparison; each per-target Markdown report lists the first divergent configuration for each case and perturbation.

## Environment

[`versions.txt`](versions.txt) records an NVIDIA H100 NVL with driver 595.99.02; PyTorch 2.14.0, vLLM 0.33.0, FlashInfer 0.7.2.post1, Triton 3.8.0, CUDA runtime package 13.0.96, cuBLAS 13.1.1.3, and cuDNN 9.24.0.43. The JSONL metadata records `VLLM_BATCH_INVARIANT=1`. `CUBLAS_WORKSPACE_CONFIG` was `:4096:8` for six targets and unset for `vllm_invariant_mode`, whose adapter calls `init_batch_invariance()`.

## Recorded outcomes

The counts below are record counts from the seven JSONL files. A `pass` is a case summary; a `check` is an equal comparison. Both can coexist with divergences in the same target.

| Target | Equal checks | Divergences | Passes | Skips | Report |
|---|---:|---:|---:|---:|---|
| `torch_matmul` | 392 | 164 | 8 | 0 | [Markdown](torch_matmul.md) · [JSONL](torch_matmul.jsonl) |
| `torch_sdpa` | 340 | 0 | 10 | 2 | [Markdown](torch_sdpa.md) · [JSONL](torch_sdpa.jsonl) |
| `vllm_batch_invariant` | 408 | 0 | 12 | 0 | [Markdown](vllm_batch_invariant.md) · [JSONL](vllm_batch_invariant.jsonl) |
| `vllm_invariant_mode` | 2,586 | 94 | 50 | 0 | [Markdown](vllm_invariant_mode.md) · [JSONL](vllm_invariant_mode.jsonl) |
| `moe` | 34 | 0 | 1 | 0 | [Markdown](moe.md) · [JSONL](moe.jsonl) |
| `flashinfer` | 415 | 34 | 7 | 0 | [Markdown](flashinfer.md) · [JSONL](flashinfer.jsonl) |
| `vllm_flash_attn` | 256 | 24 | 5 | 0 | [Markdown](vllm_flash_attn.md) · [JSONL](vllm_flash_attn.jsonl) |

The `torch_matmul` divergences were confined to float32 `mm` and `linear` in eager and compiled modes. Plain PyTorch GEMM has no batch-invariance guarantee here. Of FlashInfer's 34 divergences, 10 came from deliberately changing `fixed_split_size`, which changes the reduction partition; 12 came from the default planner's batch-dependent split choice. The 24 vLLM FlashAttention divergences were in the FA3 context case using default `num_splits=0`; that setting leaves split selection to the heuristic. These are expected consequences of the tested configurations.

No divergence was recorded for the runnable `torch_sdpa` backends, vLLM's direct batch-invariant matmul/log-softmax/mean/RMS-norm kernels, fused MoE, FA2 or FA3 with `num_splits=1`, or the tested FlashInfer tensor-core decode and prefill paths with fixed split settings or `disable_split_kv`. The vLLM invariant-mode target's tested bfloat16/float16 GEMMs and softmax, log-softmax, mean, and RMS norm cases also had no divergence. Repeat comparisons were equal in all recorded cases. The two `torch_sdpa` skips were float32 FlashAttention and cuDNN attention, each reporting no available kernel.

## Candidates

- **Float32 `torch.mm` after vLLM initialization on SM90.** The `vllm_invariant_mode` target recorded 94 divergences across `K4096xN4096` and `K4096xN14336` float32 `mm` cases after calling `init_batch_invariance()`. The first `K4096xN4096` batch comparison changed 3,644 of 4,096 output elements (maximum absolute difference 1.0728836059570312e-06). This is a low-severity candidate for vLLM because its model `Linear` layer uses the Triton batch-invariant path; the observation concerns direct float32 `torch.mm`. Reproduce with [the minimal script](repro/min_fp32_mm.py) or [the generated case script](repro/vim_fp32_mm_k4096n4096_b2.py).
- **FlashInfer CUDA-core decode with `disable_split_kv=True`.** The decode case recorded 12 divergences across batch and position variations; its first batch divergence was at batch size 31, changing 140 of 256 elements (maximum absolute difference 0.00048828125). The corresponding tensor-core decode and prefill cases had no divergence. This behavior was already reported upstream in [flashinfer-ai/flashinfer#7981](https://github.com/flashinfer-ai/flashinfer/issues/7981). Reproduce with [the CUDA-core script](repro/flashinfer_cudacore_disable_split_kv_b31.py).

## Coverage

The attention probes used head dimension 64. This run did not cover GQA, sliding-window attention, FP8 KV cache, or more than one GPU. An equal comparison establishes invariance only for the recorded case and perturbation; it does not establish numerical accuracy or invariance for other shapes and software stacks.
