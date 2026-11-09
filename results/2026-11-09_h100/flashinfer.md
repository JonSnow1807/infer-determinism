# Inference determinism report

Source: `/root/logs/idet2/flashinfer.jsonl`

| Target | Case | Perturbation | First divergent config | Max abs diff | Differing elements | Max ULP | Record ID |
|---|---|---|---|---:|---:|---:|---|
| flashinfer | run2/fa2_nosplit/batch_prefill/q32_kv8_d128/window1024 | graph_replay | `{"batch_size": 3, "block_size": 16, "chunk_size": 0, "disable_split_kv": true, "fixed_split_size": 0, "graph_capture_kv_length": 576, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 1152, "seed": 1729, "tf32": false}` | 0.0009765625 | 5704 | 28424 | `2c181fbf4a9186f4` |
| flashinfer | run2/fa2_nosplit/batch_prefill/q32_kv8_d128/window4096 | graph_replay | `{"batch_size": 3, "block_size": 16, "chunk_size": 0, "disable_split_kv": true, "fixed_split_size": 0, "graph_capture_kv_length": 2112, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 4224, "seed": 1729, "tf32": false}` | 0.00048828125 | 4462 | 27715 | `98a826d7c64c0515` |
| flashinfer | run2/fa2_nosplit/batch_prefill/q32_kv8_d128/window4096/fp8_e4m3 | graph_replay | `{"batch_size": 3, "block_size": 16, "chunk_size": 0, "disable_split_kv": true, "fixed_split_size": 0, "graph_capture_kv_length": 2112, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 4224, "seed": 1729, "tf32": false}` | 0.00048828125 | 4527 | 28154 | `83a5a00afc5be2f7` |

## Run summary

- check: 1394
- pass: 27
- divergence: 6
- skip: 0
- error: 0

## Skips and errors

