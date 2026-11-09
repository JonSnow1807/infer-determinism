#!/bin/bash
# K=256 kkt sweeps: image ptxas (CUDA 13.4.1) for 3.8.0 and 3.9.0, bundled ptxas for 3.8.0.
unset PIP_CONSTRAINT
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -E 'torch|python3' | paste -sd: -)
PY=/root/venv3/bin/python; OUT=/root/reports/run3; cd /root/infer-determinism
run() {  # $1 triton version, $2 ptxas mode
  local v=$1 mode=$2 pp=""
  [ "$v" != 3.8.0 ] && pp=/root/tri/$v
  local f=$OUT/control_k256_triton${v}_${mode}ptxas
  if [ "$mode" = bundled ]; then
    env -u TRITON_PTXAS_PATH PYTHONPATH=$pp TRITON_CACHE_DIR=/root/tc/k256_$mode $PY -m infer_determinism sweep --scenarios kkt_k256 kkt_k256_betaafter --repeats 5 --out $f.jsonl
  else
    PYTHONPATH=$pp TRITON_CACHE_DIR=/root/tc/k256_$mode $PY -m infer_determinism sweep --scenarios kkt_k256 kkt_k256_betaafter --repeats 5 --out $f.jsonl
  fi
  $PY -m infer_determinism sweep-report $f.jsonl > $f.md
}
run 3.8.0 image; run 3.9.0 image; run 3.8.0 bundled
