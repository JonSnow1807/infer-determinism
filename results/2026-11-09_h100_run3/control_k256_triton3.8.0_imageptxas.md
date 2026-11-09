# Autotune-config sweep report

Source: `/root/reports/run3/control_k256_triton3.8.0_imageptxas.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.8.0 @ /root/venv3/lib/python3.12/site-packages/triton/__init__.py, vllm 0.33.0, fla 0.6.0

ptxas used for sm_90: /usr/local/cuda/bin/ptxas (Build cuda_13.4.r13.4/compiler.38657139_0)

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
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 3419 | 2016656973 | A=5.50e-03 | A=5.24e-03 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 6362 | 2024056536 | A=3.66e-03 | A=3.26e-03 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=4 | nondeterministic | 292 | 1999150828 | A=2.22e-03 | A=1.48e-03 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 84986 | 2029677606 | A=2.11e-02 | A=2.10e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 250986 | 2043058121 | A=3.65e-02 | A=3.64e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 433452 | 2036813201 | A=5.25e-02 | A=5.25e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 31630 | 2031611740 | A=1.52e-02 | A=1.51e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 33339 | 2032988923 | A=1.29e-02 | A=1.28e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 49958 | 2028301432 | A=1.77e-02 | A=1.76e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 702055 | 2042992552 | A=7.50e-02 | A=7.49e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 1210232 | 2048823641 | A=1.21e-01 | A=1.21e-01 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 1129925 | 2047280877 | A=1.14e-01 | A=1.14e-01 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 379301 | 2043911668 | A=5.50e-02 | A=5.50e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 181925 | 2037749394 | A=3.28e-02 | A=3.28e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 147606 | 2039346332 | A=2.97e-02 | A=2.97e-02 |

## Skips and errors

None.
