# Autotune-config sweep report

Source: `/root/reports/run3/control_triton3.7.1.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.7.1 @ /root/tri/3.7.1/triton/__init__.py, vllm 0.33.0, fla 0.6.0

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 8 | 8 | A=1.66e-03 | BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=3; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 27 | A=2.58e-01 | BK=32 num_warps=2 num_stages=2; BK=32 num_warps=2 num_stages=3; BK=32 num_warps=2 num_stages=4; BK=32 num_warps=4 num_stages=2; BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=2; BK=32 num_warps=8 num_stages=3; BK=32 num_warps=8 num_stages=4; BK=64 num_warps=2 num_stages=2; BK=64 num_warps=2 num_stages=3; BK=64 num_warps=2 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4; BK=128 num_warps=2 num_stages=2; BK=128 num_warps=2 num_stages=3; BK=128 num_warps=2 num_stages=4; BK=128 num_warps=4 num_stages=2; BK=128 num_warps=4 num_stages=3; BK=128 num_warps=4 num_stages=4; BK=128 num_warps=8 num_stages=2; BK=128 num_warps=8 num_stages=3; BK=128 num_warps=8 num_stages=4 |

## Flagged configs

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 77 | 2005835398 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 1451 | 2031362208 | A=2.69e-03 | A=2.12e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=3 | nondeterministic | 478 | 2027783658 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 86865 | 2052479146 | A=3.00e-02 | A=2.99e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic | 2129 | 2042304410 | A=4.11e-03 | A=3.76e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic | 2690 | 2038602316 | A=3.02e-03 | A=2.52e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic | 1142 | 2032613980 | A=2.52e-03 | A=1.89e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic | 114 | 1996326084 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=2 num_stages=2 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=2 num_stages=3 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=2 num_stages=4 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=2 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=2 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=3 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=4 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=2 num_stages=2 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=2 num_stages=3 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=2 num_stages=4 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=2 num_stages=2 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=2 num_stages=3 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=2 num_stages=4 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=2 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=3 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=4 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=2 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=3 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=4 | wrong_vs_ref | 0 | 0 | A=2.58e-01 | A=0.00e+00 |

## Skips and errors

None.
