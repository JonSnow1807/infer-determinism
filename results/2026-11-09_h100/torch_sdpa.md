# Inference determinism report

Source: `/root/logs/idet2/torch_sdpa.jsonl`

| Target | Case | Perturbation | First divergent config | Max abs diff | Differing elements | Max ULP | Record ID |
|---|---|---|---|---:|---:|---:|---|
| torch_sdpa | bfloat16/MATH/q32_kv8_d128 | batch | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 0.0001220703125 | 3 | 1 | `031f3638d6a23f76` |
| torch_sdpa | bfloat16/MATH/q32_kv8_d128 | position | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 1, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 0.0001220703125 | 3 | 1 | `d9def24264dc77a4` |
| torch_sdpa | bfloat16/CUDNN_ATTENTION/q64_kv8_d128 | batch | `{"batch_size": 257, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 0.00390625 | 76 | 84 | `9cd7ac73480e06f6` |
| torch_sdpa | bfloat16/CUDNN_ATTENTION/q64_kv8_d128 | position | `{"batch_size": 257, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 128, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 0.00390625 | 76 | 84 | `9aa641207d5179a4` |
| torch_sdpa | bfloat16/MATH/q64_kv8_d128 | batch | `{"batch_size": 7, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 0.00390625 | 15 | 4 | `40224fb0e98ade0e` |
| torch_sdpa | bfloat16/MATH/q64_kv8_d128 | position | `{"batch_size": 7, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 3, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 0.00390625 | 15 | 4 | `a6ccc2b8cf03dc67` |
| torch_sdpa | bfloat16/MATH/q8_kv1_d128 | batch | `{"batch_size": 8, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 0.0001220703125 | 2 | 1 | `0cd8e844cfc0e86e` |
| torch_sdpa | bfloat16/MATH/q8_kv1_d128 | position | `{"batch_size": 8, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 4, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 0.0001220703125 | 2 | 1 | `5f216e1603b43ab4` |

## Run summary

- check: 358
- pass: 8
- divergence: 50
- skip: 0
- error: 0

## Skips and errors

