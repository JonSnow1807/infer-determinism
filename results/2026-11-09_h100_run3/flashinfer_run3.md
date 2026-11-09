# Inference determinism report

Source: `/root/reports/run3/flashinfer_run3.jsonl`

| Target | Case | Perturbation | First divergent config | Max abs diff | Differing elements | Max ULP | Record ID |
|---|---|---|---|---:|---:|---:|---|
| — | No divergences found | — | — | — | — | — | — |

## Run summary

- check: 1065
- pass: 21
- divergence: 0
- skip: 1
- error: 0

## Skips and errors

- **flashinfer/run3/trtllm_gen/batch_decode/q32_kv8_d128**: trtllm-gen attention kernels require SM100/SM103 (this GPU is sm9x); vLLM also disables TRTLLM attention when VLLM_BATCH_INVARIANT=1
