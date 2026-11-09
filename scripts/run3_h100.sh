#!/usr/bin/env bash
# Run 3 stages on a rented H100 (NGC pytorch image, repo copied to /root/infer-determinism).
# Results go to /root/reports/run3 (pulled continuously by vast.sh sync).
#   run3_h100.sh setup            venv with vLLM 0.33.0 (torch 2.14.0, triton 3.8.0, FlashInfer 0.7.2.post1),
#                                 fla-core 0.6.0, and triton 3.6.0 / 3.7.1 / 3.9.0 overlays
#   run3_h100.sh control VER      positive control: vllm#65385 kkt sweep + the issue's script under triton VER
#   run3_h100.sh sweep VER [SCENARIOS...]   autotune-config sweep under triton VER
#   run3_h100.sh fa3              FlashInfer FA3 / trtllm-gen run-3 cases (full matrix)
#   run3_h100.sh spec METHOD      speculative decoding under VLLM_BATCH_INVARIANT=1 (none|ngram|eagle3)
set -uo pipefail
REPO=/root/infer-determinism
OUT=/root/reports/run3
VENV=/root/venv3
mkdir -p "$OUT" /root/tri /root/tc
unset PIP_CONSTRAINT PYTORCH_BUILD_VERSION PYTORCH_BUILD_NUMBER
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTHONUNBUFFERED=1
# The image's torch libraries must not shadow the venv's wheels.
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -E 'torch|python3' | paste -sd: -)
PY=$VENV/bin/python
# The NGC image exports TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas (CUDA 13.4.1), which overrides the ptxas
# a pip-installed Triton bundles for sm_90 on releases < 3.9. PTXAS=bundled reproduces a plain pip install.
SUFFIX=""
if [ "${PTXAS:-image}" = bundled ]; then unset TRITON_PTXAS_PATH TRITON_PTXAS_BLACKWELL_PATH; SUFFIX=_bundledptxas; fi

overlay() {  # print the PYTHONPATH that selects triton $1 (empty = venv's own pin)
  local v=$1 pinned
  pinned=$($PY -c 'import importlib.metadata as m; print(m.version("triton"))' 2>/dev/null)
  if [ "$v" = "$pinned" ]; then echo ""; else echo "/root/tri/$v"; fi
}

case "${1:-}" in
setup)
  python3 -m venv "$VENV"
  $PY -m pip install -q -U pip uv
  $VENV/bin/uv pip install --python $PY 'vllm==0.33.0' 'pytest>=8' 2>&1 | tail -5
  $VENV/bin/uv pip install --python $PY --no-deps 'fla-core==0.6.0' 2>&1 | tail -2
  $VENV/bin/uv pip install --python $PY einops 2>&1 | tail -1
  $VENV/bin/uv pip install --python $PY --no-deps -e "$REPO" 2>&1 | tail -1
  # Positive-control kernel source (vLLM v0.19.1) beside the 0.33 vendored ops; adds a module, changes nothing else.
  cp "$REPO/scripts/vendored/chunk_scaled_dot_kkt_v0191.py" "$($PY -c 'import os, vllm.third_party.flash_linear_attention.ops as m; print(os.path.dirname(m.__file__))')/"
  for v in 3.6.0 3.7.1 3.9.0; do
    $VENV/bin/uv pip install --no-deps --target "/root/tri/$v" "triton==$v" 2>&1 | tail -1
  done
  $PY -m pip freeze > "$OUT/pipfreeze.txt"
  {
    nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv
    nvidia-smi | head -4
    $PY -c 'import torch, triton; print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)'
    $PY -c 'import vllm; print("vllm", vllm.__version__)'
    $PY -c 'import flashinfer; print("flashinfer", flashinfer.__version__)'
    $PY -c 'import importlib.metadata as m; print("fla-core", m.version("fla-core"))'
    for v in 3.6.0 3.7.1 3.9.0; do PYTHONPATH=/root/tri/$v $PY -c 'import triton; print("overlay triton", triton.__version__, triton.__file__)'; done
    (cd "$REPO" && git rev-parse HEAD 2>/dev/null; git -C "$REPO" status --short 2>/dev/null | head)
  } > "$OUT/versions.txt" 2>&1
  cat "$OUT/versions.txt"
  (cd "$REPO" && $PY -m pytest -q tests 2>&1 | tail -3) | tee "$OUT/cpu_tests_on_box.txt"
  ;;
control)
  v=${2:?triton version}
  export PYTHONPATH=$(overlay "$v") TRITON_CACHE_DIR=/root/tc/$v$SUFFIX
  $PY "$REPO/scripts/repro_kkt_issue65385.py" 2>&1 | tee "$OUT/control_issue_script_triton$v$SUFFIX.txt"
  $PY "$REPO/scripts/repro_kkt_issue65385.py" vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt_v0191 2>&1 | tee "$OUT/control_issue_script_v0191_triton$v$SUFFIX.txt"
  $PY -m infer_determinism sweep --scenarios ${CONTROL_SCENARIOS:-kkt_issue65385_v0191 kkt_issue65385 kkt_issue65385_fla} --repeats 5 --out "$OUT/control${OUTTAG:+_$OUTTAG}_triton$v$SUFFIX.jsonl"
  $PY -m infer_determinism sweep-report "$OUT/control${OUTTAG:+_$OUTTAG}_triton$v$SUFFIX.jsonl" > "$OUT/control${OUTTAG:+_$OUTTAG}_triton$v$SUFFIX.md"
  ;;
sweep)
  v=${2:?triton version}; shift 2
  export PYTHONPATH=$(overlay "$v") TRITON_CACHE_DIR=/root/tc/$v$SUFFIX
  scen=${*:-gdn_vllm_fixed gdn_vllm_varlen kda_vllm_varlen kda_kimik3_varlen mamba2_ssd_varlen gdn_fla_varlen}
  tag=${OUTTAG:+_$OUTTAG}$SUFFIX
  $PY -m infer_determinism sweep --scenarios $scen --repeats 3 --out "$OUT/sweep_triton$v$tag.jsonl"
  $PY -m infer_determinism sweep-report "$OUT/sweep_triton$v$tag.jsonl" > "$OUT/sweep_triton$v$tag.md"
  ;;
fa3)
  export VLLM_BATCH_INVARIANT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  $PY -m infer_determinism run --targets flashinfer --include '^run3/' --out "$OUT/flashinfer_run3.jsonl"
  $PY -m infer_determinism report "$OUT/flashinfer_run3.jsonl" > "$OUT/flashinfer_run3.md"
  ;;
spec)
  m=${2:?method}
  export VLLM_BATCH_INVARIANT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_ENABLE_HF_TRANSFER=0
  $PY "$REPO/scripts/spec_decode_bi.py" --method "$m" --gpu-mem "${GPU_MEM:-0.55}" --out "$OUT/spec_$m.json" 2>&1 | tail -40
  ;;
*)
  sed -n '2,10p' "$0"; exit 2;;
esac
