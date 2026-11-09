# Autotune-config sweep report

Source: `/root/reports/run3/control_triton3.8.0.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.8.0 @ /root/venv3/lib/python3.12/site-packages/triton/__init__.py, vllm 0.33.0, fla 0.6.0

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 11 | 11 | A=1.66e-03 | BK=32 num_warps=4 num_stages=2; BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=3; BK=32 num_warps=8 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 27 | A=2.58e-01 | BK=32 num_warps=2 num_stages=2; BK=32 num_warps=2 num_stages=3; BK=32 num_warps=2 num_stages=4; BK=32 num_warps=4 num_stages=2; BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=2; BK=32 num_warps=8 num_stages=3; BK=32 num_warps=8 num_stages=4; BK=64 num_warps=2 num_stages=2; BK=64 num_warps=2 num_stages=3; BK=64 num_warps=2 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4; BK=128 num_warps=2 num_stages=2; BK=128 num_warps=2 num_stages=3; BK=128 num_warps=2 num_stages=4; BK=128 num_warps=4 num_stages=2; BK=128 num_warps=4 num_stages=3; BK=128 num_warps=4 num_stages=4; BK=128 num_warps=8 num_stages=2; BK=128 num_warps=8 num_stages=3; BK=128 num_warps=8 num_stages=4 |

## Flagged configs

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=2 | nondeterministic | 150 | 2031939673 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 661 | 2030585399 | A=4.48e-03 | A=4.16e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 2162 | 2044141047 | A=4.94e-03 | A=4.65e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=3 | nondeterministic | 170 | 2028759755 | A=2.62e-03 | A=2.02e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=4 | nondeterministic | 27 | 2025914863 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 93145 | 2051444760 | A=3.12e-02 | A=3.12e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic | 2396 | 2038602316 | A=4.15e-03 | A=3.81e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic | 815 | 2026637025 | A=2.83e-03 | A=2.29e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic | 1956 | 2035148259 | A=5.66e-03 | A=5.41e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic | 39 | 1986601697 | A=2.08e-03 | A=1.26e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic | 77 | 2012743617 | A=1.66e-03 | A=0.00e+00 |
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
