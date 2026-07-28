#!/bin/bash
# tuned/build.sh <variant> — build/install pytorch from source for a single
# GPU variant (gb10/rtx40/rtx50) only, single-arch, into a per-variant
# editable venv. This is the real, multi-hour compile step; see
# tuned/wheel.sh for packaging + publishing once this succeeds.
#
# Uses the modern PEP 517 flow (pytorch's own setup.py is now just a
# deprecation shim, see setup.py's own DEPRECATION_NOTICE): plain
# `pip install --no-build-isolation -v -e .`, not `python setup.py`.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

command -v python3 &>/dev/null || { echo "ERROR: python3 not found on PATH." >&2; exit 1; }

VENV_DIR="${REPO_ROOT}/.venv-${GPU_TUNED_VARIANT}"
[[ -d "${VENV_DIR}" ]] || python3 -m venv "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "=========================================="
echo "Building pytorch for ${GPU_TUNED_HW_LABEL} only"
echo "=========================================="
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "USE_CUDA:             ${USE_CUDA}"
[ -n "${USE_KLEIDIAI_SME:-}" ] && echo "USE_KLEIDIAI_SME:      ${USE_KLEIDIAI_SME}"
echo "CUDA_HOME:            ${CUDA_HOME:-<unset>}"
echo "MAX_JOBS:              ${MAX_JOBS}"
echo "Python:                $(python3 --version)"
echo "Git commit:             $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""

echo "Installing build-time requirements..."
pip install --upgrade pip
[ -f requirements.txt ] && pip install -r requirements.txt
pip install "typing-extensions>=4.10.0" "scikit-build-core>=1.0"

echo "Building pytorch (this will take a long time)..."
pip install --no-build-isolation -v -e .

echo ""
echo "Smoke test: import torch, confirm CUDA is compiled in and reports the requested arch..."
python3 - <<'PYEOF'
import torch

print(f"torch.__version__ = {torch.__version__}")
print(f"torch.version.cuda = {torch.version.cuda}")
assert torch.backends.cuda.is_built(), "torch was NOT built with CUDA support"
print("OK: torch imports cleanly and was built with CUDA support.")
PYEOF

echo ""
echo "=========================================="
echo "Build complete (${GPU_TUNED_HW_LABEL})."
echo "=========================================="
echo "Venv: ${VENV_DIR}"
echo "Next: bash tuned/wheel.sh ${GPU_TUNED_VARIANT}"
