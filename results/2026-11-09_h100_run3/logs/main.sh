#!/bin/bash
unset PIP_CONSTRAINT
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTHONUNBUFFERED=1 PYTHONPATH=/root/tri/main TRITON_CACHE_DIR=/root/tc/main
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -E 'torch|python3' | paste -sd: -)
PY=/root/venv3/bin/python; OUT=/root/reports/run3
cd /root/triton-main && echo "triton main commit: $(git rev-parse HEAD)" | tee $OUT/triton_main_commit.txt
cd /root/infer-determinism
$PY scripts/min_wgmma_rega_race.py 2>&1 | grep -v -E "opaque|deprecated|warn" | tee $OUT/min_rega_race_triton_main.txt
$PY -m infer_determinism sweep --scenarios kkt_issue65385 kkt_issue65385_betaafter kkt_k256 kkt_k256_betaafter --repeats 5 --out $OUT/control_triton_main.jsonl
$PY -m infer_determinism sweep-report $OUT/control_triton_main.jsonl > $OUT/control_triton_main.md
