# Autotune-config sweep report

Source: `/root/reports/run3/control_v0191_triton3.7.1.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.7.1 @ /root/tri/3.7.1/triton/__init__.py, vllm 0.33.0, fla 0.6.0

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 10 | 10 | A=1.66e-03 | BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=3; BK=32 num_warps=8 num_stages=4; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 0 | A=2.29e-07 | - |

## Flagged configs

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 473 | 2026059604 | A=3.61e-03 | A=3.21e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic | 2583 | 2042211093 | A=2.38e-03 | A=1.71e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=3 | nondeterministic | 35 | 37310214 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=4 | nondeterministic | 182 | 2002847356 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 132884 | 2050973044 | A=3.54e-02 | A=3.54e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic | 2222 | 2031354584 | A=3.99e-03 | A=3.62e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic | 1423 | 2031969052 | A=4.60e-03 | A=4.29e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 13237 | 2043357662 | A=1.12e-02 | A=1.11e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic | 47 | 2011801988 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic | 35 | 1996873200 | A=1.66e-03 | A=0.00e+00 |

## Skips and errors

None.
