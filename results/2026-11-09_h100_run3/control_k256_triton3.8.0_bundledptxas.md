# Autotune-config sweep report

Source: `/root/reports/run3/control_k256_triton3.8.0_bundledptxas.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.8.0 @ /root/venv3/lib/python3.12/site-packages/triton/__init__.py, vllm 0.33.0, fla 0.6.0

ptxas used for sm_90: /root/venv3/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas (Build cuda_12.9.r12.9/compiler.36037853_0)

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
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 1939 | 2011039992 | A=1.77e-03 | A=6.35e-04 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 6742 | 2020233472 | A=6.59e-03 | A=6.38e-03 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=4 | nondeterministic | 514 | 2017389496 | A=1.66e-03 | A=0.00e+00 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 76254 | 2033656438 | A=1.98e-02 | A=1.97e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 256504 | 2036747150 | A=3.49e-02 | A=3.49e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 404071 | 2039502140 | A=4.67e-02 | A=4.66e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 28225 | 2032040101 | A=1.31e-02 | A=1.30e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 45812 | 2037870463 | A=1.71e-02 | A=1.70e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 53245 | 2033940985 | A=1.79e-02 | A=1.78e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 712125 | 2042992552 | A=7.66e-02 | A=7.66e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 528899 | 2043291749 | A=6.69e-02 | A=6.69e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 593633 | 2044580115 | A=7.19e-02 | A=7.18e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 557733 | 2044887265 | A=6.90e-02 | A=6.90e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 26608 | 2029505595 | A=1.14e-02 | A=1.13e-02 |
| kkt_k256 | chunk_scaled_dot_kkt_fwd_kernel | BK=128 num_warps=8 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 25213 | 2032675595 | A=1.11e-02 | A=1.10e-02 |

## Skips and errors

None.
