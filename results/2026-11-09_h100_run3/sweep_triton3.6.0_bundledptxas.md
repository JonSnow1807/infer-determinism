# Autotune-config sweep report

Source: `/root/reports/run3/sweep_triton3.6.0_bundledptxas.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.6.0 @ /root/tri/3.6.0/triton/__init__.py, vllm 0.33.0, fla 0.6.0

ptxas used for sm_90: /root/tri/3.6.0/triton/backends/nvidia/bin/ptxas (Build cuda_12.8.r12.8/compiler.35583870_0)

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|
| gdn_vllm_fixed | autotune | o=4.31e-03, final_state=3.31e-03 | True (3 runs) | - | chunk_local_cumsum_scalar_kernel: num_warps=1 num_stages=3; chunk_scaled_dot_kkt_fwd_kernel: BK=64 num_warps=2 num_stages=4; merge_16x16_to_64x64_inverse_kernel: num_warps=2 num_stages=3; recompute_w_u_fwd_kernel: num_warps=4 num_stages=4; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=4 num_stages=3; chunk_fwd_kernel_o: BK=64 BV=128 num_warps=4 num_stages=3 |
| gdn_vllm_fixed | first | o=4.31e-03, final_state=3.31e-03 | True (1 runs) | - | chunk_local_cumsum_scalar_kernel: num_warps=1 num_stages=3; chunk_scaled_dot_kkt_fwd_kernel: BK=32 num_warps=2 num_stages=2; merge_16x16_to_64x64_inverse_kernel: num_warps=2 num_stages=2; recompute_w_u_fwd_kernel: num_warps=2 num_stages=2; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=2 num_stages=2; chunk_fwd_kernel_o: BK=64 BV=64 num_warps=2 num_stages=2 |
| gdn_vllm_varlen | autotune | o=4.32e-03, final_state=3.33e-03 | True (3 runs) | - | chunk_local_cumsum_scalar_kernel: num_warps=4 num_stages=3; chunk_scaled_dot_kkt_fwd_kernel: BK=64 num_warps=2 num_stages=3; merge_16x16_to_64x64_inverse_kernel: num_warps=2 num_stages=2; recompute_w_u_fwd_kernel: num_warps=4 num_stages=3; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=4 num_stages=2; chunk_fwd_kernel_o: BK=64 BV=128 num_warps=4 num_stages=3 |
| gdn_vllm_varlen | first | o=4.32e-03, final_state=3.33e-03 | True (1 runs) | - | chunk_local_cumsum_scalar_kernel: num_warps=1 num_stages=3; chunk_scaled_dot_kkt_fwd_kernel: BK=32 num_warps=2 num_stages=2; merge_16x16_to_64x64_inverse_kernel: num_warps=2 num_stages=2; recompute_w_u_fwd_kernel: num_warps=2 num_stages=2; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=2 num_stages=2; chunk_fwd_kernel_o: BK=64 BV=64 num_warps=2 num_stages=2 |
| kda_vllm_varlen | autotune | o=4.38e-03, final_state=3.52e-03 | True (3 runs) | - | chunk_local_cumsum_vector_kernel: BS=64 num_warps=8 num_stages=3; chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter: BK=32 num_warps=1 num_stages=3; chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra: num_warps=1 num_stages=3; merge_16x16_to_64x64_inverse_kernel: num_warps=2 num_stages=2; recompute_w_u_fwd_kernel: num_warps=4 num_stages=3; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=4 num_stages=2; chunk_gla_fwd_kernel_o: BK=64 BV=128 num_warps=4 num_stages=3 |
| kda_vllm_varlen | first | o=4.38e-03, final_state=3.52e-03 | True (1 runs) | - | chunk_local_cumsum_vector_kernel: BS=32 num_warps=2 num_stages=3; chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter: BK=32 num_warps=1 num_stages=2; chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra: num_warps=1 num_stages=3; merge_16x16_to_64x64_inverse_kernel: num_warps=2 num_stages=2; recompute_w_u_fwd_kernel: num_warps=2 num_stages=2; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=2 num_stages=2; chunk_gla_fwd_kernel_o: BK=32 BV=64 num_warps=2 num_stages=2 |
| kda_kimik3_varlen | autotune | o=4.38e-03, final_state=3.53e-03 | True (3 runs) | - | chunk_local_cumsum_vector_kernel: BS=64 num_warps=8 num_stages=3; chunk_kda_fwd_kernel_intra_token_parallel: BH=4 num_warps=1 num_stages=3; chunk_kda_fwd_kernel_inter_solve_fused: BK=32 num_warps=1 num_stages=3; recompute_w_u_fwd_kernel: num_warps=4 num_stages=3; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=4 num_stages=2; chunk_gla_fwd_kernel_o: BK=64 BV=128 num_warps=4 num_stages=2 |
| kda_kimik3_varlen | first | o=4.38e-03, final_state=3.53e-03 | True (1 runs) | - | chunk_local_cumsum_vector_kernel: BS=32 num_warps=2 num_stages=3; chunk_kda_fwd_kernel_intra_token_parallel: BH=1 num_warps=1 num_stages=3; chunk_kda_fwd_kernel_inter_solve_fused: BK=32 num_warps=1 num_stages=3; recompute_w_u_fwd_kernel: num_warps=2 num_stages=2; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=2 num_stages=2; chunk_gla_fwd_kernel_o: BK=32 BV=64 num_warps=2 num_stages=2 |
| mamba2_ssd_varlen | autotune | out=2.32e-03, final_state=1.65e-03 | True (3 runs) | - | _chunk_cumsum_fwd_kernel: BLOCK_SIZE_H=4 num_warps=4 num_stages=3; _chunk_state_fwd_kernel: BLOCK_SIZE_K=64 BLOCK_SIZE_M=64 BLOCK_SIZE_N=128 num_warps=4 num_stages=2; _state_passing_fwd_kernel: BLOCK_SIZE=1024 num_warps=4 num_stages=3; _bmm_chunk_fwd_kernel: BLOCK_SIZE_K=32 BLOCK_SIZE_M=64 BLOCK_SIZE_N=128 num_warps=4 num_stages=4; _chunk_scan_fwd_kernel: BLOCK_SIZE_K=32 BLOCK_SIZE_M=64 BLOCK_SIZE_N=64 num_warps=4 num_stages=1 |
| mamba2_ssd_varlen | first | out=2.32e-03, final_state=1.65e-03 | True (1 runs) | - | _chunk_cumsum_fwd_kernel: BLOCK_SIZE_H=2 num_warps=4 num_stages=3; _chunk_state_fwd_kernel: BLOCK_SIZE_K=32 BLOCK_SIZE_M=32 BLOCK_SIZE_N=32 num_warps=4 num_stages=3; _state_passing_fwd_kernel: BLOCK_SIZE=64 num_warps=4 num_stages=3; _bmm_chunk_fwd_kernel: BLOCK_SIZE_K=64 BLOCK_SIZE_M=128 BLOCK_SIZE_N=256 num_warps=8 num_stages=3; _chunk_scan_fwd_kernel: BLOCK_SIZE_K=32 BLOCK_SIZE_M=32 BLOCK_SIZE_N=32 num_warps=8 num_stages=2 |
| gdn_fla_varlen | autotune | o=4.32e-03, final_state=3.33e-03 | True (3 runs) | - | l2norm_fwd_kernel: BT=64 num_warps=2 num_stages=3; chunk_local_cumsum_scalar_kernel: num_warps=1 num_stages=3; chunk_gated_delta_rule_fwd_kkt_solve_kernel: BK=64 num_warps=1 num_stages=3; recompute_w_u_fwd_kernel: num_warps=4 num_stages=3; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=64 num_warps=4 num_stages=2; chunk_fwd_kernel_o: BK=64 BV=64 num_warps=4 num_stages=3 |
| gdn_fla_varlen | first | o=4.32e-03, final_state=3.33e-03 | True (1 runs) | - | l2norm_fwd_kernel: BT=8 num_warps=1 num_stages=3; chunk_local_cumsum_scalar_kernel: num_warps=1 num_stages=3; chunk_gated_delta_rule_fwd_kkt_solve_kernel: BK=32 num_warps=1 num_stages=3; recompute_w_u_fwd_kernel: num_warps=2 num_stages=2; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=2 num_stages=2; chunk_fwd_kernel_o: BK=128 BV=128 num_warps=8 num_stages=3 |

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| gdn_vllm_fixed | chunk_local_cumsum_scalar_kernel | 0 | 4 | 0 | 0 | 0 | 0 | o=1.12e-07 | - |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | 1 | 27 | 0 | 0 | 6 | 6 | A=1.66e-03 | BK=32 num_warps=4 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=4 |
| gdn_vllm_fixed | merge_16x16_to_64x64_inverse_kernel | 2 | 12 | 0 | 0 | 0 | 0 | Ai=1.61e-04 | - |
| gdn_vllm_fixed | recompute_w_u_fwd_kernel | 3 | 9 | 0 | 0 | 0 | 0 | - | - |
| gdn_vllm_fixed | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 4 | 12 | 0 | 0 | 0 | 0 | - | - |
| gdn_vllm_fixed | chunk_fwd_kernel_o | 5 | 24 | 0 | 0 | 0 | 0 | o=2.32e-03 | - |
| gdn_vllm_varlen | chunk_local_cumsum_scalar_kernel | 0 | 4 | 0 | 0 | 0 | 0 | o=1.12e-07 | - |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | 1 | 27 | 0 | 0 | 5 | 5 | A=1.66e-03 | BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=4 |
| gdn_vllm_varlen | merge_16x16_to_64x64_inverse_kernel | 2 | 12 | 0 | 0 | 0 | 0 | Ai=1.61e-04 | - |
| gdn_vllm_varlen | recompute_w_u_fwd_kernel | 3 | 9 | 0 | 0 | 0 | 0 | - | - |
| gdn_vllm_varlen | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 4 | 12 | 0 | 0 | 0 | 0 | - | - |
| gdn_vllm_varlen | chunk_fwd_kernel_o | 5 | 24 | 0 | 0 | 0 | 0 | o=2.32e-03 | - |
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

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 284 | 2018331920 | A=1.66e-03 | A=0.00e+00 |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 24663 | 2053811242 | A=1.66e-02 | A=1.65e-02 |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic | 7926 | 2037230070 | A=7.61e-03 | A=7.43e-03 |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 13896 | 2040733410 | A=1.13e-02 | A=1.12e-02 |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic | 454 | 2030083068 | A=4.09e-03 | A=3.73e-03 |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic | 170 | 2017816478 | A=1.66e-03 | A=0.00e+00 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref | 13525 | 2045414691 | A=9.47e-03 | A=9.32e-03 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 18826 | 2043003890 | A=1.39e-02 | A=1.38e-02 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 18210 | 2041300646 | A=1.31e-02 | A=1.30e-02 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic | 280 | 2018651337 | A=2.83e-03 | A=2.29e-03 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic | 760 | 2023528008 | A=1.66e-03 | A=0.00e+00 |

## Skips and errors

None.
