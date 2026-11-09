# Autotune-config sweep report

Source: `/root/reports/run3/control_triton3.9.0.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.9.0 @ /root/tri/3.9.0/triton/__init__.py, vllm 0.33.0, fla 0.6.0

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 9 | 9 | A=1.66e-03 | BK=32 num_warps=4 num_stages=2; BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 27 | A=2.58e-01 | BK=32 num_warps=2 num_stages=2; BK=32 num_warps=2 num_stages=3; BK=32 num_warps=2 num_stages=4; BK=32 num_warps=4 num_stages=2; BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=2; BK=32 num_warps=8 num_stages=3; BK=32 num_warps=8 num_stages=4; BK=64 num_warps=2 num_stages=2; BK=64 num_warps=2 num_stages=3; BK=64 num_warps=2 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4; BK=128 num_warps=2 num_stages=2; BK=128 num_warps=2 num_stages=3; BK=128 num_warps=2 num_stages=4; BK=128 num_warps=4 num_stages=2; BK=128 num_warps=4 num_stages=3; BK=128 num_warps=4 num_stages=4; BK=128 num_warps=8 num_stages=2; BK=128 num_warps=8 num_stages=3; BK=128 num_warps=8 num_stages=4 |

## Flagged configs

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=2 | nondeterministic | 460 | 2007066853 | A=2.81e-03 | A=2.26e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 4405 | 2037435430 | A=8.92e-03 | A=8.76e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref | 6080 | 2044053555 | A=6.26e-03 | A=6.04e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 217021 | 2051968050 | A=5.32e-02 | A=5.32e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 75372 | 2047527120 | A=2.73e-02 | A=2.73e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 71039 | 2059540410 | A=2.51e-02 | A=2.51e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 18559 | 2055759675 | A=1.66e-02 | A=1.65e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 17363 | 2053220299 | A=1.64e-02 | A=1.64e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 14611 | 2051151956 | A=1.42e-02 | A=1.41e-02 |
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
