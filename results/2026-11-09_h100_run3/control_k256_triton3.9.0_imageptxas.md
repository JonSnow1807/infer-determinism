# Autotune-config sweep report

Source: `/root/reports/run3/control_k256_triton3.9.0_imageptxas.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.9.0 @ /root/tri/3.9.0/triton/__init__.py, vllm 0.33.0, fla 0.6.0

ptxas used for sm_90: /root/tri/3.9.0/triton/backends/nvidia/bin/ptxas-blackwell (Build cuda_13.4.r13.4/compiler.38657139_0)

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 15 | 15 | A=1.66e-03 | BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4; BK=128 num_warps=4 num_stages=2; BK=128 num_warps=4 num_stages=3; BK=128 num_warps=4 num_stages=4; BK=128 num_warps=8 num_stages=2; BK=128 num_warps=8 num_stages=3; BK=128 num_warps=8 num_stages=4 |
| kkt_k256_betaafter | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 0 | A=3.80e-07 | - |

## Flagged configs

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 7994 | 2028259146 | A=6.17e-03 | A=5.94e-03 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref | 11039 | 2022941709 | A=5.73e-03 | A=5.48e-03 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=4 | nondeterministic | 103 | 2011454899 | A=1.66e-03 | A=0.00e+00 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 235442 | 2039502140 | A=3.25e-02 | A=3.25e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 611259 | 2044031590 | A=6.29e-02 | A=6.29e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 741617 | 2039502140 | A=6.81e-02 | A=6.81e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 27382 | 2036230164 | A=1.42e-02 | A=1.41e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 119679 | 2040905890 | A=2.90e-02 | A=2.89e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 118824 | 2046029407 | A=2.87e-02 | A=2.86e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 880274 | 2048675013 | A=9.29e-02 | A=9.29e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 1157039 | 2047747110 | A=1.12e-01 | A=1.12e-01 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 1009743 | 2050499724 | A=9.66e-02 | A=9.66e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 150762 | 2043911668 | A=3.20e-02 | A=3.19e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 524046 | 2042685411 | A=6.98e-02 | A=6.98e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 507121 | 2045385397 | A=6.82e-02 | A=6.81e-02 |

## Skips and errors

None.
