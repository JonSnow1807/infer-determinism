# Inference determinism report

Source: [`torch_matmul.jsonl`](torch_matmul.jsonl)

| Target | Case | Perturbation | First divergent config | Max abs diff | Differing elements | Max ULP | Record ID |
|---|---|---|---|---:|---:|---:|---|
| torch_matmul | float32/mm/eager | batch | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 3.814697265625e-06 | 95 | 320 | `c82e22da9b5ea94e` |
| torch_matmul | float32/mm/eager | tile_m | `{"batch_size": 31, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 3.814697265625e-06 | 95 | 320 | `fd46c602a191bab4` |
| torch_matmul | float32/mm/eager | position | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 1, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 3.814697265625e-06 | 95 | 320 | `b141041e076bb1ec` |
| torch_matmul | float32/mm/compile | batch | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 3.814697265625e-06 | 95 | 320 | `fa7844bc2f5ddd14` |
| torch_matmul | float32/mm/compile | tile_m | `{"batch_size": 31, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 3.814697265625e-06 | 95 | 320 | `b9ace22ce9ef8810` |
| torch_matmul | float32/mm/compile | position | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 1, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 3.814697265625e-06 | 95 | 320 | `568b3927ecf3d284` |
| torch_matmul | float32/linear/eager | batch | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 5.7220458984375e-06 | 92 | 120 | `0bc921ab95e1f0b7` |
| torch_matmul | float32/linear/eager | tile_m | `{"batch_size": 31, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 5.7220458984375e-06 | 92 | 120 | `4d5aba495afe5199` |
| torch_matmul | float32/linear/eager | position | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 1, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 5.7220458984375e-06 | 92 | 120 | `c17bc3853fbd333a` |
| torch_matmul | float32/linear/compile | batch | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 5.7220458984375e-06 | 92 | 120 | `8d905475d2c01937` |
| torch_matmul | float32/linear/compile | tile_m | `{"batch_size": 31, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 5.7220458984375e-06 | 92 | 120 | `9dbb3a44d8372584` |
| torch_matmul | float32/linear/compile | position | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 1, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 5.7220458984375e-06 | 92 | 120 | `9da024c8bffb4368` |

## Run summary

- check: 392
- pass: 8
- divergence: 164
- skip: 0
- error: 0

## Skips and errors

