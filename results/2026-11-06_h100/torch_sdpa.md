# Inference determinism report

Source: [`torch_sdpa.jsonl`](torch_sdpa.jsonl)

| Target | Case | Perturbation | First divergent config | Max abs diff | Differing elements | Max ULP | Record ID |
|---|---|---|---|---:|---:|---:|---|
| — | No divergences found | — | — | — | — | — | — |

## Run summary

- check: 340
- pass: 10
- divergence: 0
- skip: 2
- error: 0

## Skips and errors

- **torch_sdpa/float32/FLASH_ATTENTION**: FLASH_ATTENTION: No available kernel. Aborting execution.
- **torch_sdpa/float32/CUDNN_ATTENTION**: CUDNN_ATTENTION: No available kernel. Aborting execution.
