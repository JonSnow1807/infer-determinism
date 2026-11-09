# Autotune-config sweep report

Source: `/root/reports/run3/sweep_triton3.8.0_kda.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.8.0 @ /root/venv3/lib/python3.12/site-packages/triton/__init__.py, vllm 0.33.0, fla 0.6.0

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|
| kda_vllm_varlen | autotune | o=4.38e-03, final_state=3.52e-03 | True (3 runs) | - | chunk_local_cumsum_vector_kernel: BS=64 num_warps=8 num_stages=3; chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter: BK=32 num_warps=1 num_stages=3; chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra: num_warps=2 num_stages=3; merge_16x16_to_64x64_inverse_kernel: num_warps=2 num_stages=5; recompute_w_u_fwd_kernel: num_warps=4 num_stages=3; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=64 num_warps=4 num_stages=2; chunk_gla_fwd_kernel_o: BK=64 BV=128 num_warps=4 num_stages=3 |
| kda_vllm_varlen | first | o=4.38e-03, final_state=3.52e-03 | True (1 runs) | - | chunk_local_cumsum_vector_kernel: BS=32 num_warps=2 num_stages=3; chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter: BK=32 num_warps=1 num_stages=2; chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra: num_warps=1 num_stages=3; merge_16x16_to_64x64_inverse_kernel: num_warps=2 num_stages=2; recompute_w_u_fwd_kernel: num_warps=2 num_stages=2; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=2 num_stages=2; chunk_gla_fwd_kernel_o: BK=32 BV=64 num_warps=2 num_stages=2 |
| kda_kimik3_varlen | autotune | o=4.38e-03, final_state=3.53e-03 | True (3 runs) | - | chunk_local_cumsum_vector_kernel: BS=64 num_warps=8 num_stages=3; chunk_kda_fwd_kernel_intra_token_parallel: BH=8 num_warps=2 num_stages=3; chunk_kda_fwd_kernel_inter_solve_fused: BK=32 num_warps=1 num_stages=3; recompute_w_u_fwd_kernel: num_warps=4 num_stages=3; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=64 num_warps=4 num_stages=2; chunk_gla_fwd_kernel_o: BK=64 BV=128 num_warps=4 num_stages=3 |
| kda_kimik3_varlen | first | o=4.38e-03, final_state=3.53e-03 | True (1 runs) | - | chunk_local_cumsum_vector_kernel: BS=32 num_warps=2 num_stages=3; chunk_kda_fwd_kernel_intra_token_parallel: BH=1 num_warps=1 num_stages=3; chunk_kda_fwd_kernel_inter_solve_fused: BK=32 num_warps=1 num_stages=3; recompute_w_u_fwd_kernel: num_warps=2 num_stages=2; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=2 num_stages=2; chunk_gla_fwd_kernel_o: BK=32 BV=64 num_warps=2 num_stages=2 |

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kda_vllm_varlen | chunk_local_cumsum_vector_kernel | 0 | 6 | 0 | 0 | 0 | 0 | - | - |
| kda_vllm_varlen | chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter | 1 | 24 | 0 | 0 | 0 | 0 | - | - |
| kda_vllm_varlen | chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra | 2 | 4 | 0 | 0 | 0 | 0 | - | - |
| kda_vllm_varlen | merge_16x16_to_64x64_inverse_kernel | 3 | 12 | 0 | 0 | 0 | 0 | - | - |
| kda_vllm_varlen | recompute_w_u_fwd_kernel | 4 | 9 | 0 | 0 | 0 | 0 | - | - |
| kda_vllm_varlen | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 5 | 12 | 0 | 0 | 0 | 0 | - | - |
| kda_vllm_varlen | chunk_gla_fwd_kernel_o | 6 | 36 | 0 | 0 | 0 | 0 | - | - |
| kda_kimik3_varlen | chunk_local_cumsum_vector_kernel | 0 | 6 | 0 | 0 | 0 | 0 | - | - |
| kda_kimik3_varlen | chunk_kda_fwd_kernel_intra_token_parallel | 1 | 16 | 0 | 0 | 0 | 0 | - | - |
| kda_kimik3_varlen | chunk_kda_fwd_kernel_inter_solve_fused | 2 | 6 | 0 | 0 | 0 | 0 | - | - |
| kda_kimik3_varlen | recompute_w_u_fwd_kernel | 3 | 9 | 0 | 0 | 0 | 0 | - | - |
| kda_kimik3_varlen | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 4 | 12 | 0 | 0 | 0 | 0 | - | - |
| kda_kimik3_varlen | chunk_gla_fwd_kernel_o | 5 | 36 | 0 | 0 | 0 | 0 | - | - |

## Flagged configs

None.

## Skips and errors

None.
