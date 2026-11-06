# Inference determinism report

Source: [`vllm_invariant_mode.jsonl`](vllm_invariant_mode.jsonl)

| Target | Case | Perturbation | First divergent config | Max abs diff | Differing elements | Max ULP | Record ID |
|---|---|---|---|---:|---:|---:|---|
| vllm_invariant_mode | mm/float32/K4096xN4096 | batch | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 1.0728836059570312e-06 | 3644 | 77824 | `a10fb5178f1cebd5` |
| vllm_invariant_mode | mm/float32/K4096xN4096 | tile_m | `{"batch_size": 31, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 9.5367431640625e-06 | 3990 | 41254 | `b250f8edffb18a27` |
| vllm_invariant_mode | mm/float32/K4096xN4096 | big_m | `{"batch_size": 500, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 9.5367431640625e-06 | 3990 | 41254 | `de2bc4e71654f185` |
| vllm_invariant_mode | mm/float32/K4096xN4096 | position | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 1, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 1.0728836059570312e-06 | 3644 | 77824 | `e7201b302974456d` |
| vllm_invariant_mode | mm/float32/K4096xN14336 | batch | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 1.1920928955078125e-06 | 11806 | 28672 | `3fce74eb94e87d27` |
| vllm_invariant_mode | mm/float32/K4096xN14336 | tile_m | `{"batch_size": 31, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 1.0967254638671875e-05 | 13976 | 42790 | `76d37f22e24302ea` |
| vllm_invariant_mode | mm/float32/K4096xN14336 | big_m | `{"batch_size": 500, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 0, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 1.0967254638671875e-05 | 13976 | 42790 | `67446fff4e68790a` |
| vllm_invariant_mode | mm/float32/K4096xN14336 | position | `{"batch_size": 2, "block_size": 16, "chunk_size": 0, "disable_split_kv": false, "fixed_split_size": 0, "graph_capture_kv_length": 0, "num_splits": 0, "other_kv_length": 32, "probe_index": 1, "probe_kv_length": 32, "seed": 1729, "tf32": false}` | 1.1920928955078125e-06 | 11806 | 28672 | `bfb3b4648533ad5c` |

## Run summary

- check: 2586
- pass: 50
- divergence: 94
- skip: 0
- error: 0

## Skips and errors

