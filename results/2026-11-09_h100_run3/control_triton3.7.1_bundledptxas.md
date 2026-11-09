# Autotune-config sweep report

Source: `/root/reports/run3/control_triton3.7.1_bundledptxas.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.7.1 @ /root/tri/3.7.1/triton/__init__.py, vllm 0.33.0, fla 0.6.0

ptxas used for sm_90: /root/tri/3.7.1/triton/backends/nvidia/bin/ptxas (Build cuda_12.8.r12.8/compiler.35583870_0)

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 9 | 9 | A=1.66e-03 | BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=3; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 9 | 9 | A=1.66e-03 | BK=32 num_warps=4 num_stages=2; BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4 |
| kkt_issue65385_betaafter | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 0 | A=2.48e-07 | - |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 0 | A=2.29e-07 | - |

## Flagged configs

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 1590 | 2033820072 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 928 | 2032810400 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=3 | nondeterministic | 77 | 1988144973 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 96687 | 2046383726 | A=3.23e-02 | A=3.23e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 39554 | 2047682368 | A=2.08e-02 | A=2.07e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 36231 | 2046122295 | A=1.92e-02 | A=1.91e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic | 6105 | 2034184558 | A=6.63e-03 | A=6.42e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic | 39 | 1999766218 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic | 43 | 1993846115 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=2 | nondeterministic | 77 | 2009543504 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 125 | 2022739367 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 4047 | 2042263520 | A=3.79e-03 | A=3.40e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 40438 | 2046383726 | A=1.82e-02 | A=1.81e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 17594 | 2050231792 | A=1.17e-02 | A=1.15e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 17458 | 2042659373 | A=1.19e-02 | A=1.18e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref | 6394 | 2050973044 | A=8.95e-03 | A=8.80e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic, wrong_vs_ref | 6750 | 2052292215 | A=7.68e-03 | A=7.50e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic, wrong_vs_ref | 6629 | 2040954345 | A=8.84e-03 | A=8.68e-03 |

## Skips and errors

None.
