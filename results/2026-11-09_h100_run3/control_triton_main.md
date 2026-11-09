# Autotune-config sweep report

Source: `/root/reports/run3/control_triton_main.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.9.0 @ /root/tri/main/triton/__init__.py, vllm 0.33.0, fla 0.6.0

ptxas used for sm_90: /usr/local/cuda/bin/ptxas (Build cuda_13.4.r13.4/compiler.38657139_0)

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 10 | 10 | A=1.66e-03 | BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=3; BK=32 num_warps=8 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4 |
| kkt_issue65385_betaafter | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 0 | A=2.48e-07 | - |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 16 | 16 | A=1.66e-03 | BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=3; BK=32 num_warps=8 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4; BK=128 num_warps=4 num_stages=2; BK=128 num_warps=4 num_stages=3; BK=128 num_warps=4 num_stages=4; BK=128 num_warps=8 num_stages=2; BK=128 num_warps=8 num_stages=3; BK=128 num_warps=8 num_stages=4 |
| kkt_k256_betaafter | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 0 | A=3.80e-07 | - |

## Flagged configs

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 1469 | 2035791450 | A=4.09e-03 | A=3.74e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 3555 | 2043018154 | A=7.08e-03 | A=6.88e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=3 | nondeterministic | 281 | 2028132914 | A=2.33e-03 | A=1.63e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=4 | nondeterministic | 425 | 2019903980 | A=1.79e-03 | A=6.66e-04 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 328977 | 2059263183 | A=5.78e-02 | A=5.78e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 16059 | 2043181972 | A=1.25e-02 | A=1.24e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 15469 | 2037248439 | A=1.09e-02 | A=1.07e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 44926 | 2049997832 | A=2.39e-02 | A=2.38e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 20138 | 2048217684 | A=1.84e-02 | A=1.84e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 16360 | 2045692609 | A=1.53e-02 | A=1.52e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 3919 | 2010545489 | A=5.19e-03 | A=4.92e-03 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 7420 | 2017560480 | A=7.63e-03 | A=7.44e-03 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=3 | nondeterministic | 29 | 1952684616 | A=1.66e-03 | A=0.00e+00 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=4 | nondeterministic | 309 | 1980244027 | A=1.93e-03 | A=9.83e-04 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic | 9141 | 2012106124 | A=3.32e-03 | A=2.88e-03 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 374466 | 2035849739 | A=4.50e-02 | A=4.49e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 547293 | 2042292067 | A=5.89e-02 | A=5.89e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 87744 | 2036747150 | A=2.44e-02 | A=2.44e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 230790 | 2042299306 | A=4.25e-02 | A=4.24e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 253024 | 2040905890 | A=4.24e-02 | A=4.24e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 914052 | 2043351706 | A=9.43e-02 | A=9.42e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 310949 | 2043291749 | A=4.51e-02 | A=4.51e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 270749 | 2040379309 | A=4.21e-02 | A=4.21e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 540928 | 2044887265 | A=6.30e-02 | A=6.30e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 548403 | 2046923491 | A=7.24e-02 | A=7.24e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 520626 | 2044832257 | A=6.91e-02 | A=6.91e-02 |

## Skips and errors

None.
