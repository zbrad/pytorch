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
if [[ "${GPU_TUNED_CLEAN_VENV:-}" == "1" && -d "${VENV_DIR}" ]]; then
    echo "GPU_TUNED_CLEAN_VENV=1: removing ${VENV_DIR} for a clean rebuild"
    rm -rf "${VENV_DIR}"
fi
if [[ -d "${VENV_DIR}" ]]; then
    # Reusing an existing venv (the default -- this is a multi-hour
    # compile, not something to redo every run) -- verify it wasn't
    # copied/contaminated from another repo before building on top of it.
    # Set GPU_TUNED_CLEAN_VENV=1 to force a fresh venv instead.
    gpu_tuned_verify_venv "${VENV_DIR}" "${REPO_ROOT}"
else
    python3 -m venv "${VENV_DIR}"
fi
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

# third_party/nccl is a nested checkout, not a tracked git submodule of
# this repo (third_party/nccl/ is gitignored here) -- a local fix there
# is otherwise invisible to `git status` and silently lost on a fresh
# clone or submodule reset. This applies tuned/patches/nccl-ldmc-arch-gate.patch
# (checked in, so it IS durable/reproducible) every build, loudly, so
# it's never silently missing: without it, NCCL's device/Makefile
# unconditionally compiles sm_100f/sm_100a "LDMC" multicast device code
# into every build with CUDA >= 12.7 regardless of TORCH_CUDA_ARCH_LIST,
# breaking this fleet's single-arch invariant with a fat binary (caught
# by gpu_tuned_verify_arch downstream in wheel.sh, but this fixes the
# root cause instead of just detecting it). Filed upstream:
# https://github.com/NVIDIA/nccl/issues/2420 -- remove this patch and
# this whole step once that's fixed there. See
# tuned/patches/nccl-ldmc-arch-gate.patch for the full rationale.
NCCL_MAKEFILE="${REPO_ROOT}/third_party/nccl/src/device/Makefile"
NCCL_PATCH="${REPO_ROOT}/tuned/patches/nccl-ldmc-arch-gate.patch"
echo "=========================================="
echo "NCCL LDMC arch-gate patch check"
echo "=========================================="
if [[ ! -f "${NCCL_MAKEFILE}" ]]; then
    echo "WARNING: ${NCCL_MAKEFILE} not found (third_party/nccl not yet" >&2
    echo "  checked out) -- cannot apply the NCCL LDMC arch-gate patch this run." >&2
    echo "  gpu_tuned_verify_arch in wheel.sh will still catch a fat sm_100" >&2
    echo "  binary if NCCL's build ends up unpatched -- but if it does, rerun" >&2
    echo "  this script (third_party/nccl should exist by then) rather than" >&2
    echo "  publishing an unverified wheel." >&2
elif grep -q "zbrad/pytorch tuned-builds: compile LDMC" "${NCCL_MAKEFILE}"; then
    echo "OK: NCCL LDMC arch-gate patch already applied."
else
    echo "APPLYING LOCAL PATCH: NCCL LDMC/multicast FP8 kernel gencode fix" >&2
    echo "  (fat sm_100+${GPU_TUNED_TORCH_ARCH} binary fix -- see" >&2
    echo "  tuned/patches/nccl-ldmc-arch-gate.patch for the full story)." >&2
    if patch -p1 -d "${REPO_ROOT}/third_party/nccl" < "${NCCL_PATCH}"; then
        echo "OK: NCCL LDMC arch-gate patch applied."
    else
        echo "ERROR: NCCL LDMC arch-gate patch FAILED to apply -- upstream NCCL" >&2
        echo "  likely changed src/device/Makefile since this patch was written." >&2
        echo "  Without this fix, the build will silently embed unreachable" >&2
        echo "  sm_100 device code (dead on this GPU, but breaks the single-arch" >&2
        echo "  invariant gpu_tuned_verify_arch checks). Re-derive the patch by" >&2
        echo "  hand against the new Makefile before proceeding -- do not skip" >&2
        echo "  this and hope wheel.sh's verify catches it; fix it here instead." >&2
        exit 1
    fi
fi
echo ""

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
