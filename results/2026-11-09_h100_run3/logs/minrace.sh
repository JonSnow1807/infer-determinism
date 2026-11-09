#!/bin/bash
unset PIP_CONSTRAINT
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -E 'torch|python3' | paste -sd: -)
PY=/root/venv3/bin/python; X=/root/infer-determinism/scripts/min_wgmma_rega_race.py; OUT=/root/reports/run3/min_rega_race.txt
: > $OUT
for v in 3.6.0 3.7.1 3.8.0 3.9.0; do
  pp=""; [ "$v" != 3.8.0 ] && pp=/root/tri/$v
  PYTHONPATH=$pp TRITON_CACHE_DIR=/root/tc/min_img_$v $PY $X 2>&1 | grep -v -E "opaque|deprecated|UserWarning|warnings.warn" | tee -a $OUT
  [ "$v" != 3.9.0 ] && env -u TRITON_PTXAS_PATH PYTHONPATH=$pp TRITON_CACHE_DIR=/root/tc/min_bun_$v $PY $X 2>&1 | grep -v -E "opaque|deprecated|UserWarning|warnings.warn" | tee -a $OUT
done
