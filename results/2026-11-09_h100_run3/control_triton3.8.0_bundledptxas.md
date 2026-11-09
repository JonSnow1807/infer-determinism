# Autotune-config sweep report

Source: `/root/reports/run3/control_triton3.8.0_bundledptxas.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.8.0 @ /root/venv3/lib/python3.12/site-packages/triton/__init__.py, vllm 0.33.0, fla 0.6.0

ptxas used for sm_90: /root/venv3/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas (Build cuda_12.9.r12.9/compiler.36037853_0)

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 6 | 6 | A=1.66e-03 | BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=3; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 10 | 10 | A=1.66e-03 | BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=3; BK=32 num_warps=8 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4 |
| kkt_issue65385_betaafter | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 0 | A=2.48e-07 | - |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 0 | A=2.29e-07 | - |

## Flagged configs

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 1827 | 2039463383 | A=1.99e-03 | A=1.10e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=3 | nondeterministic | 13 | 2000941452 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 95099 | 2049322258 | A=2.92e-02 | A=2.91e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 24820 | 2043743252 | A=1.57e-02 | A=1.56e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 33984 | 2048014649 | A=1.48e-02 | A=1.47e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref | 6601 | 2041983727 | A=9.24e-03 | A=9.09e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 2957 | 2035494291 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 4271 | 2045869327 | A=9.86e-03 | A=9.72e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=3 | nondeterministic | 183 | 2020021999 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=4 | nondeterministic | 52 | 2010608166 | A=2.10e-03 | A=1.29e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 54303 | 2049781419 | A=2.53e-02 | A=2.52e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 22201 | 2052755670 | A=1.46e-02 | A=1.45e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 22484 | 2053381061 | A=1.20e-02 | A=1.18e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref | 6462 | 2043904280 | A=8.13e-03 | A=7.96e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 9763 | 2043809326 | A=1.12e-02 | A=1.11e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic | 6054 | 2041043931 | A=9.49e-03 | A=9.34e-03 |

## Skips and errors

None.
