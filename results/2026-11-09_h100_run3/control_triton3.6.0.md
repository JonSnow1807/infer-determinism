# Autotune-config sweep report

Source: `/root/reports/run3/control_triton3.6.0.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.6.0 @ /root/tri/3.6.0/triton/__init__.py, vllm 0.33.0, fla 0.6.0

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 2 | 2 | A=1.66e-03 | BK=32 num_warps=4 num_stages=4; BK=64 num_warps=4 num_stages=2 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 27 | A=2.58e-01 | BK=32 num_warps=2 num_stages=2; BK=32 num_warps=2 num_stages=3; BK=32 num_warps=2 num_stages=4; BK=32 num_warps=4 num_stages=2; BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=2; BK=32 num_warps=8 num_stages=3; BK=32 num_warps=8 num_stages=4; BK=64 num_warps=2 num_stages=2; BK=64 num_warps=2 num_stages=3; BK=64 num_warps=2 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4; BK=128 num_warps=2 num_stages=2; BK=128 num_warps=2 num_stages=3; BK=128 num_warps=2 num_stages=4; BK=128 num_warps=4 num_stages=2; BK=128 num_warps=4 num_stages=3; BK=128 num_warps=4 num_stages=4; BK=128 num_warps=8 num_stages=2; BK=128 num_warps=8 num_stages=3; BK=128 num_warps=8 num_stages=4 |

## Flagged configs

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 232 | 2011520623 | A=2.50e-03 | A=1.88e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic | 366 | 2022411273 | A=2.47e-03 | A=1.83e-03 |
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
