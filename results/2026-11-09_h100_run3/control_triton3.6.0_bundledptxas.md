# Autotune-config sweep report

Source: `/root/reports/run3/control_triton3.6.0_bundledptxas.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.6.0 @ /root/tri/3.6.0/triton/__init__.py, vllm 0.33.0, fla 0.6.0

ptxas used for sm_90: /root/tri/3.6.0/triton/backends/nvidia/bin/ptxas (Build cuda_12.8.r12.8/compiler.35583870_0)

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 7 | 7 | A=1.66e-03 | BK=32 num_warps=4 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 5 | 5 | A=1.66e-03 | BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4 |
| kkt_issue65385_betaafter | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 0 | A=2.48e-07 | - |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 0 | A=2.29e-07 | - |

## Flagged configs

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 27 | 2029699376 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 32672 | 2042512109 | A=1.84e-02 | A=1.84e-02 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref | 13663 | 2052086113 | A=8.33e-03 | A=8.16e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref | 10769 | 2045864163 | A=9.48e-03 | A=9.33e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic | 162 | 2006639810 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic | 35 | 2016107599 | A=2.03e-03 | A=1.16e-03 |
| kkt_issue65385 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic | 39 | 2013978810 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 23566 | 2045152562 | A=1.32e-02 | A=1.31e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 16970 | 2047512847 | A=1.26e-02 | A=1.25e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref | 8329 | 2042750626 | A=9.47e-03 | A=9.32e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic | 1687 | 2035344532 | A=5.01e-03 | A=4.73e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic | 1702 | 2030966606 | A=6.75e-03 | A=6.54e-03 |

## Skips and errors

None.
