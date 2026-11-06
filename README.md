# infer-determinism

Reproducible inference matters for reinforcement learning and evaluations: a request can receive different floating-point bits when unrelated requests share its batch. `infer-determinism` probes this batch invariance at the kernel level. A divergence identifies a change to reproduce and explain; it does not, by itself, establish a numerical correctness defect.

## Method

Each case runs a probe request alone as a reference, then repeats it while varying supported batch size, probe position, companion KV length, tile boundary, split setting, or graph replay. The probe inputs and weights stay fixed. Outputs are compared by their raw floating-point bits; JSONL records also report changed element counts, maximum absolute difference, and maximum ordered-bit ULP distance. The harness does not compare against a high-precision oracle.

## Install and run

Use Python 3.10+ and install PyTorch with CUDA support for GPU targets. vLLM and FlashInfer are optional dependencies for their respective targets. On an Ubuntu H100 host, `bash scripts/run_h100.sh` creates `.venv`, installs the GPU dependencies, runs the CPU tests, and writes quick and full suite outputs. `FULL_BUDGET=90m` changes its full-suite time limit.

In an existing environment:

```bash
python -m pip install -e '.[test]'
export VLLM_BATCH_INVARIANT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m infer_determinism run --quick --out results.jsonl
python -m infer_determinism report results.jsonl > report.md
python -m infer_determinism repro RECORD_ID --file results.jsonl > repro.py
```

Use `--targets` to select from `torch_matmul`, `torch_sdpa`, `vllm_batch_invariant`, `vllm_invariant_mode`, `flashinfer`, `vllm_flash_attn`, and `moe`. The `vllm_invariant_mode` target calls vLLM's `init_batch_invariance()`; to reproduce the first H100 run's configuration for that target, leave `CUBLAS_WORKSPACE_CONFIG` unset before launching it. The other H100 targets used `:4096:8`.

## Read the results

The JSONL starts with environment metadata. `check` records are equal comparisons; `divergence` records include the compared configurations and a standalone reproducer; `pass` summarizes a case; `skip` and `error` give reasons. A skip is not a pass. The generated reproducer embeds harness code but still requires the target library and a compatible GPU stack. The Markdown report shows the first divergent configuration per case and perturbation; count all divergence records in JSONL for the full total.

## First results

The [2026-11-06 H100 run](results/2026-11-06_h100/README.md) reports per-target counts, invariant observations, expected divergences, two bug candidates, and reproduction scripts. The [runner's report](results/2026-11-06_h100/REPORT.md) and per-target JSONL and Markdown files are retained alongside it.

## Status and limits

The published GPU evidence is one H100 NVL run. The attention cases cover head dimension 64, without GQA, sliding-window attention, or FP8 KV cache, and use a single GPU. Results characterize the tested configurations and software versions, not all inference paths or cross-machine reproducibility. Run `python3 -m pytest -q tests` for the CPU harness tests.
