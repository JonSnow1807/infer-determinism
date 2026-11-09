# Autotune-config sweep report

Source: `/root/reports/run3/control_v0191_triton3.8.0.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.8.0 @ /root/venv3/lib/python3.12/site-packages/triton/__init__.py, vllm 0.33.0, fla 0.6.0

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 10 | 10 | A=1.66e-03 | BK=32 num_warps=4 num_stages=3; BK=32 num_warps=4 num_stages=4; BK=32 num_warps=8 num_stages=2; BK=32 num_warps=8 num_stages=3; BK=64 num_warps=4 num_stages=2; BK=64 num_warps=4 num_stages=3; BK=64 num_warps=4 num_stages=4; BK=64 num_warps=8 num_stages=2; BK=64 num_warps=8 num_stages=3; BK=64 num_warps=8 num_stages=4 |
| kkt_issue65385_fla | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 0 | A=2.29e-07 | - |

## Flagged configs

| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |
|---|---|---|---|---:|---:|---|---|
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=3 | nondeterministic | 2116 | 2044044199 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=4 num_stages=4 | nondeterministic, wrong_vs_ref | 5822 | 2041581631 | A=4.77e-03 | A=4.47e-03 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=2 | nondeterministic | 54 | 2007653764 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=32 num_warps=8 num_stages=3 | nondeterministic | 29 | 1965452109 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 108765 | 2056270437 | A=3.43e-02 | A=3.43e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=3 | nondeterministic | 125 | 2006524814 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=4 num_stages=4 | nondeterministic | 242 | 2013280580 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=2 | nondeterministic, wrong_vs_ref, outlier_vs_consensus | 16524 | 2048710285 | A=1.42e-02 | A=1.41e-02 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=3 | nondeterministic | 78 | 2010450082 | A=1.66e-03 | A=0.00e+00 |
| kkt_issue65385_v0191 | chunk_scaled_dot_kkt_fwd_kernel | BK=64 num_warps=8 num_stages=4 | nondeterministic | 189 | 2020260213 | A=1.66e-03 | A=0.00e+00 |

## Skips and errors

None.
