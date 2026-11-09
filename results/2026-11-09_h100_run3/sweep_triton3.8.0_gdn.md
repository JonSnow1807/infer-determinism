# Autotune-config sweep report

Source: `/root/reports/run3/sweep_triton3.8.0_gdn.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.8.0 @ /root/venv3/lib/python3.12/site-packages/triton/__init__.py, vllm 0.33.0, fla 0.6.0

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|
| gdn_vllm_fixed | autotune | o=4.31e-03, final_state=3.31e-03 | True (3 runs) | - | chunk_local_cumsum_scalar_kernel: num_warps=4 num_stages=3; chunk_scaled_dot_kkt_fwd_kernel: BK=64 num_warps=2 num_stages=3; merge_16x16_to_64x64_inverse_kernel: num_warps=2 num_stages=4; recompute_w_u_fwd_kernel: num_warps=4 num_stages=4; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=4 num_stages=3; chunk_fwd_kernel_o: BK=64 BV=128 num_warps=4 num_stages=3 |
| gdn_vllm_fixed | first | o=4.31e-03, final_state=3.31e-03 | True (1 runs) | - | chunk_local_cumsum_scalar_kernel: num_warps=1 num_stages=3; chunk_scaled_dot_kkt_fwd_kernel: BK=32 num_warps=2 num_stages=2; merge_16x16_to_64x64_inverse_kernel: num_warps=2 num_stages=2; recompute_w_u_fwd_kernel: num_warps=2 num_stages=2; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=2 num_stages=2; chunk_fwd_kernel_o: BK=64 BV=64 num_warps=2 num_stages=2 |
| gdn_vllm_varlen | autotune | o=4.32e-03, final_state=3.33e-03 | True (3 runs) | - | chunk_local_cumsum_scalar_kernel: num_warps=2 num_stages=3; chunk_scaled_dot_kkt_fwd_kernel: BK=64 num_warps=2 num_stages=3; merge_16x16_to_64x64_inverse_kernel: ?; recompute_w_u_fwd_kernel: num_warps=4 num_stages=3; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: ?; chunk_fwd_kernel_o: BK=128 BV=128 num_warps=4 num_stages=4 |
| gdn_vllm_varlen | first | o=4.32e-03, final_state=3.33e-03 | True (1 runs) | - | chunk_local_cumsum_scalar_kernel: num_warps=1 num_stages=3; chunk_scaled_dot_kkt_fwd_kernel: BK=32 num_warps=2 num_stages=2; merge_16x16_to_64x64_inverse_kernel: num_warps=2 num_stages=2; recompute_w_u_fwd_kernel: num_warps=2 num_stages=2; chunk_gated_delta_rule_fwd_kernel_h_blockdim64: BV=32 num_warps=2 num_stages=2; chunk_fwd_kernel_o: BK=64 BV=64 num_warps=2 num_stages=2 |

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| gdn_vllm_fixed | chunk_local_cumsum_scalar_kernel | 0 | 4 | 0 | 0 | 0 | 0 | o=1.12e-07 | - |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | 1 | 27 | 0 | 0 | 7 | 7 | A=1.66e-03 | BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=3; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=4 |
| gdn_vllm_fixed | merge_16x16_to_64x64_inverse_kernel | 2 | 12 | 0 | 0 | 0 | 0 | Ai=1.61e-04 | - |
| gdn_vllm_fixed | recompute_w_u_fwd_kernel | 3 | 9 | 0 | 0 | 0 | 0 | - | - |
| gdn_vllm_fixed | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 4 | 12 | 0 | 0 | 0 | 0 | - | - |
| gdn_vllm_fixed | chunk_fwd_kernel_o | 5 | 24 | 0 | 0 | 0 | 0 | o=2.32e-03 | - |
| gdn_vllm_varlen | chunk_local_cumsum_scalar_kernel | 0 | 4 | 0 | 0 | 0 | 0 | o=1.12e-07 | - |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | 1 | 27 | 0 | 0 | 8 | 8 | A=1.66e-03 | BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=3; BK=32 num_warps=8 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2 |
| gdn_vllm_varlen | merge_16x16_to_64x64_inverse_kernel | 2 | 12 | 0 | 0 | 0 | 0 | Ai=1.61e-04 | - |
| gdn_vllm_varlen | recompute_w_u_fwd_kernel | 3 | 9 | 0 | 0 | 0 | 0 | - | - |
| gdn_vllm_varlen | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 4 | 12 | 0 | 0 | 0 | 0 | - | - |
| gdn_vllm_varlen | chunk_fwd_kernel_o | 5 | 24 | 0 | 0 | 0 | 0 | o=2.32e-03 | - |

## Flagged configs

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 1908 | 2026533100 | A=1.66e-03 | A=0.00e+00 |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=3 | nondeterministic | 320 | 2020479031 | A=1.66e-03 | A=0.00e+00 |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 81689 | 2042356117 | A=2.88e-02 | A=2.88e-02 |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic | 2839 | 2031444080 | A=6.58e-03 | A=6.37e-03 |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic | 683 | 2029777106 | A=1.66e-03 | A=0.00e+00 |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic | 3616 | 2038569382 | A=7.02e-03 | A=6.83e-03 |
| gdn_vllm_fixed | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic | 39 | 2024619051 | A=1.66e-03 | A=0.00e+00 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 2513 | 2026218761 | A=4.12e-03 | A=3.77e-03 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 1000 | 2021453386 | A=3.72e-03 | A=3.34e-03 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=3 | nondeterministic | 7 | 1978326479 | A=1.96e-03 | A=1.05e-03 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=4 | nondeterministic | 519 | 2046491843 | A=2.44e-03 | A=1.79e-03 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 96890 | 2054976715 | A=3.13e-02 | A=3.12e-02 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 26372 | 2041793250 | A=1.78e-02 | A=1.78e-02 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 28000 | 2050682192 | A=1.66e-02 | A=1.65e-02 |
| gdn_vllm_varlen | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic | 2358 | 2037040400 | A=4.50e-03 | A=4.19e-03 |

## Skips and errors

None.
