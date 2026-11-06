#!/usr/bin/env bash
set -euo pipefail

# Run from a checkout on Ubuntu with an NVIDIA H100 and a working CUDA driver.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.9' vllm flashinfer-python 'pytest>=8' -e .
python -m pip freeze | grep -Ei '^(torch|vllm|flashinfer-python|triton|nvidia-[^=]+)==' | tee gpu-package-versions.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -m pytest -q

export VLLM_BATCH_INVARIANT=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
mkdir -p results
set +e
python -m infer_determinism run --quick --out results/quick.jsonl
quick_status=$?
set -e
python -m infer_determinism report results/quick.jsonl > results/quick.md

# GNU timeout is present on standard Ubuntu. Override with FULL_BUDGET for a
# longer rental window; exit 124 keeps the partial quick suite available.
full_budget="${FULL_BUDGET:-45m}"
set +e
timeout "$full_budget" python -m infer_determinism run --out results/full.jsonl
full_status=$?
set -e
if [[ -f results/full.jsonl ]]; then
  python -m infer_determinism report results/full.jsonl > results/full.md
fi
echo "Full suite exit status: $full_status"
echo "Quick suite exit status: $quick_status"
if [[ "$full_status" -ne 0 ]]; then exit "$full_status"; fi
exit "$quick_status"
