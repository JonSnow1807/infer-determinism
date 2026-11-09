#!/bin/bash
# Dump PTX/SASS of the minimal register-A kernel (BK=64, num_warps=4, num_stages=2) under triton 3.9.0 and 3.8.0+ptxas13.4.
unset PIP_CONSTRAINT
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
PY=/root/venv3/bin/python; D=/root/reports/run3/irdump/minrace
mkdir -p $D
cat > /root/one_config.py <<'PYEOF'
import sys, torch
sys.argv = [sys.argv[0], "--runs", "2", "--rows", "8192"]
import importlib.util
spec = importlib.util.spec_from_file_location("m", "/root/infer-determinism/scripts/min_wgmma_rega_race.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
x = torch.randn(8192, 128, device="cuda").bfloat16(); s = torch.rand(8192, device="cuda").bfloat16(); out = torch.empty(128, 64, 64, device="cuda")
for kern in (m.scaled_gram_rega, m.scaled_gram_smem):
    kern[(128,)](x, s, out, K=128, BT=64, BK=64, num_warps=4, num_stages=2)
torch.cuda.synchronize(); print("ok")
PYEOF
for tag in t390 t380_ptx134; do
  pp=/root/tri/3.9.0; [ $tag = t380_ptx134 ] && pp=""
  TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=$D/$tag TRITON_CACHE_DIR=/root/tc/sass_$tag PYTHONPATH=$pp $PY /root/one_config.py 2>&1 | grep -v -E "opaque|deprecated|warn"
  for c in $(find $D/$tag -name "*.cubin"); do /usr/local/cuda/bin/cuobjdump -sass $c > ${c%.cubin}.sass 2>&1; done
done
for f in $(find $D -name "*.ptx"); do n=$(grep -c "wgmma.mma_async" $f); echo "$f wgmma=$n waits=$(grep -o 'wgmma.wait_group.sync.aligned [0-9]*' $f | sort | uniq -c | tr '\n' ' ')"; done
for f in $(find $D -name "*.sass"); do echo "$f HGMMA=$(grep -c HGMMA $f) DEPBAR=$(grep -c 'WARPGROUP.DEPBAR' $f) ARRIVE=$(grep -c 'WARPGROUP.ARRIVE' $f)"; done
