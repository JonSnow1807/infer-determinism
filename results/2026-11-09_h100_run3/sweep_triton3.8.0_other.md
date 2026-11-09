# Autotune-config sweep report

Source: `/root/reports/run3/sweep_triton3.8.0_other.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.8.0 @ /root/venv3/lib/python3.12/site-packages/triton/__init__.py, vllm 0.33.0, fla 0.6.0

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|
| mamba2_ssd_varlen | autotune | out=2.32e-03, final_state=1.65e-03 | True (3 runs) | - | _chunk_cumsum_fwd_kernel: BLOCK_SIZE_H=4 num_warps=4 num_stages=3; _chunk_state_fwd_kernel: BLOCK_SIZE_K=32 BLOCK_SIZE_M=64 BLOCK_SIZE_N=128 num_warps=4 num_stages=4; _state_passing_fwd_kernel: BLOCK_SIZE=1024 num_warps=4 num_stages=3; _bmm_chunk_fwd_kernel: BLOCK_SIZE_K=32 BLOCK_SIZE_M=128 BLOCK_SIZE_N=64 num_warps=4 num_stages=4; _chunk_scan_fwd_kernel: BLOCK_SIZE_K=32 BLOCK_SIZE_M=64 BLOCK_SIZE_N=64 num_warps=4 num_stages=1 |
| mamba2_ssd_varlen | first | out=2.32e-03, final_state=1.65e-03 | True (1 runs) | - | _chunk_cumsum_fwd_kernel: BLOCK_SIZE_H=2 num_warps=4 num_stages=3; _chunk_state_fwd_kernel: BLOCK_SIZE_K=32 BLOCK_SIZE_M=32 BLOCK_SIZE_N=32 num_warps=4 num_stages=3; _state_passing_fwd_kernel: BLOCK_SIZE=64 num_warps=4 num_stages=3; _bmm_chunk_fwd_kernel: BLOCK_SIZE_K=64 BLOCK_SIZE_M=128 BLOCK_SIZE_N=256 num_warps=8 num_stages=3; _chunk_scan_fwd_kernel: BLOCK_SIZE_K=32 BLOCK_SIZE_M=32 BLOCK_SIZE_N=32 num_warps=8 num_stages=2 |
| gdn_fla_varlen | autotune | o=4.32e-03, final_state=3.33e-03 | True (3 runs) | - | l2norm_fwd_kernel: BT=8 num_warps=1 num_stages=3; chunk_local_cumsum_scalar_kernel: num_warps=2 num_stages=3; chunk_gated_delta_rule_fwd_kkt_solve_kernel: BK=64 num_warps=1 num_stages=3; recompute_w_u_fwd_kernel: num_warps=4 num_stages=3; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=64 num_warps=4 num_stages=2; chunk_fwd_kernel_o: BK=64 BV=64 num_warps=4 num_stages=3 |
| gdn_fla_varlen | first | o=4.32e-03, final_state=3.33e-03 | True (1 runs) | - | l2norm_fwd_kernel: BT=8 num_warps=1 num_stages=3; chunk_local_cumsum_scalar_kernel: num_warps=1 num_stages=3; chunk_gated_delta_rule_fwd_kkt_solve_kernel: BK=32 num_warps=1 num_stages=3; recompute_w_u_fwd_kernel: num_warps=2 num_stages=2; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=2 num_stages=2; chunk_fwd_kernel_o: BK=128 BV=128 num_warps=8 num_stages=3 |

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| mamba2_ssd_varlen | _chunk_cumsum_fwd_kernel | 0 | 6 | 0 | 0 | 0 | 0 | - | - |
| mamba2_ssd_varlen | _chunk_state_fwd_kernel | 1 | 14 | 0 | 0 | 0 | 0 | - | - |
| mamba2_ssd_varlen | _state_passing_fwd_kernel | 2 | 6 | 0 | 0 | 0 | 0 | - | - |
| mamba2_ssd_varlen | _bmm_chunk_fwd_kernel | 3 | 9 | 0 | 0 | 0 | 0 | - | - |
| mamba2_ssd_varlen | _chunk_scan_fwd_kernel | 4 | 12 | 0 | 0 | 0 | 0 | - | - |
| gdn_fla_varlen | l2norm_fwd_kernel | 0 | 25 | 0 | 0 | 0 | 0 | - | - |
| gdn_fla_varlen | l2norm_fwd_kernel | 1 | 25 | 0 | 0 | 0 | 0 | - | - |
| gdn_fla_varlen | chunk_local_cumsum_scalar_kernel | 2 | 4 | 0 | 0 | 0 | 0 | - | - |
| gdn_fla_varlen | chunk_gated_delta_rule_fwd_kkt_solve_kernel | 3 | 6 | 0 | 0 | 0 | 0 | - | - |
| gdn_fla_varlen | recompute_w_u_fwd_kernel | 4 | 9 | 0 | 0 | 0 | 0 | - | - |
| gdn_fla_varlen | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 5 | 12 | 0 | 0 | 0 | 0 | - | - |
| gdn_fla_varlen | chunk_fwd_kernel_o | 6 | 3 | 0 | 0 | 0 | 0 | - | - |

## Flagged configs

None.

## Skips and errors

None.
