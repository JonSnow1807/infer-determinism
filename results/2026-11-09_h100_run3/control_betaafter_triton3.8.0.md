# Autotune-config sweep report

Source: `/root/reports/run3/control_betaafter_triton3.8.0.jsonl`

GPU: NVIDIA H100 NVL (sm90), driver 580.159.03, torch 2.14.0, triton 3.8.0 @ /root/venv3/lib/python3.12/site-packages/triton/__init__.py, vllm 0.33.0, fla 0.6.0

ptxas used for sm_90: /usr/local/cuda/bin/ptxas (Build cuda_13.4.r13.4/compiler.38657139_0)

## End-to-end outputs vs float32 reference

| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |
|---|---|---|---|---|---|

## Per-kernel config sweep

| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| kkt_issue65385_betaafter | chunk_scaled_dot_kkt_fwd_kernel | 0 | 27 | 0 | 0 | 0 | 0 | A=2.48e-07 | - |

## Flagged configs

None.

## Skips and errors

None.
