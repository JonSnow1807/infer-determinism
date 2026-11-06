# Inference determinism report

Source: [`flashinfer.jsonl`](flashinfer.jsonl)

| Target | Case | Perturbation | First divergent config | Max abs diff | Differing elements | Max ULP | Record ID |
|---|---|---|---|---:|---:|---:|---|
| flashinfer | ctx_default_planner/batch_decode_long | batch | `{"batch_size": 31, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 1024, "probe_index": 0, "probe_kv_length": 1024, "seed": 1729, "tf32": false}` | 0.00048828125 | 140 | 28 | `db09a3ea4a83716b` |
| flashinfer | ctx_default_planner/batch_decode_long | position | `{"batch_size": 31, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 1024, "probe_index": 15, "probe_kv_length": 1024, "seed": 1729, "tf32": false}` | 0.00048828125 | 140 | 28 | `b9e7a68d54976e39` |
| flashinfer | fa2_fixed_split64tok/batch_decode_long | fixed_split_size | `{"batch_size": 1, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 16, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 1024, "probe_index": 0, "probe_kv_length": 1024, "seed": 1729, "tf32": false}` | 0.00067138671875 | 141 | 86 | `f6e35c27f66adf4c` |
| flashinfer | fa2_fixed_split64tok/batch_prefill_long | fixed_split_size | `{"batch_size": 1, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 16, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 1024, "probe_index": 0, "probe_kv_length": 1024, "seed": 1729, "tf32": false}` | 0.0009765625 | 1182 | 29065 | `4cd5ab951aded653` |
| flashinfer | fa2_disable_split_kv_cudacore/batch_decode_long | batch | `{"batch_size": 31, "block_size": 16, "chunk_size": 0, "disable_split_kv": true, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 1024, "probe_index": 0, "probe_kv_length": 1024, "seed": 1729, "tf32": false}` | 0.00048828125 | 140 | 28 | `7dadfae07cec90a5` |
| flashinfer | fa2_disable_split_kv_cudacore/batch_decode_long | position | `{"batch_size": 31, "block_size": 16, "chunk_size": 0, "disable_split_kv": true, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 1024, "probe_index": 15, "probe_kv_length": 1024, "seed": 1729, "tf32": false}` | 0.00048828125 | 140 | 28 | `25749b582674ec7b` |

## Run summary

- check: 415
- pass: 7
- divergence: 34
- skip: 0
- error: 0

## Skips and errors

