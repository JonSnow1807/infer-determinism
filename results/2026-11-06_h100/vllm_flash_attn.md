# Inference determinism report

Source: [`vllm_flash_attn.jsonl`](vllm_flash_attn.jsonl)

| Target | Case | Perturbation | First divergent config | Max abs diff | Differing elements | Max ULP | Record ID |
|---|---|---|---|---:|---:|---:|---|
| vllm_flash_attn | FA3/ctx_paged_varlen_long_default_splits | batch | `{"batch_size": 7, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 2048, "probe_index": 0, "probe_kv_length": 2048, "seed": 1729, "tf32": false}` | 0.000244140625 | 97 | 33 | `f56d029b136374bb` |
| vllm_flash_attn | FA3/ctx_paged_varlen_long_default_splits | position | `{"batch_size": 7, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 2048, "probe_index": 3, "probe_kv_length": 2048, "seed": 1729, "tf32": false}` | 0.000244140625 | 97 | 33 | `238a057e7d86866a` |

## Run summary

- check: 256
- pass: 5
- divergence: 24
- skip: 0
- error: 0

## Skips and errors

