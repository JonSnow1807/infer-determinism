#!/bin/bash
# Triton 3.9.0 (bundled ptxas 13.4): (1) sepB variant; (2) dump rega TTGIR, set the in-loop
# warp_group_dot_wait pendings 1 -> 0 and rerun through TRITON_KERNEL_OVERRIDE.
unset PIP_CONSTRAINT
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTHONPATH=/root/tri/3.9.0
PY=/root/venv3/bin/python; X=/root/infer-determinism/scripts/min_race_experiments.py
D=/root/reports/run3/irdump/override; rm -rf $D /root/tc/ovr_*; mkdir -p $D
TRITON_CACHE_DIR=/root/tc/ovr_a $PY $X sepB 2>&1 | grep triton
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=$D/dump TRITON_CACHE_DIR=/root/tc/ovr_b $PY $X rega 2>&1 | grep triton
h=$(dirname $(ls $D/dump/*/scaled_gram_rega.ttgir)); hash=$(basename $h)
mkdir -p $D/override/$hash; cp $h/scaled_gram_rega.ttgir $D/override/$hash/
python3 - "$D/override/$hash/scaled_gram_rega.ttgir" <<'PYEOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
n = s.count("{pendings = 1 : i32}")
s = s.replace("{pendings = 1 : i32}", "{pendings = 0 : i32}")
open(p, "w").write(s); print(f"edited {n} warp_group_dot_wait pendings 1 -> 0 in {p}")
PYEOF
TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=$D/override TRITON_CACHE_DIR=/root/tc/ovr_c TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=$D/after $PY $X rega 2>&1 | grep -E "triton|verride"
grep -c "pendings = 0" $(ls $D/after/*/scaled_gram_rega.ttgir) 2>/dev/null; grep -o "wgmma.wait_group.sync.aligned [0-9]" $(ls $D/after/*/scaled_gram_rega.ptx) | sort | uniq -c
