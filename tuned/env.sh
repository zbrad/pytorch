#!/bin/bash
# tuned/env.sh <variant> — device config + build-env setup for a tuned
# single-arch pytorch build (gb10/rtx40/rtx50). Source this file with the
# variant as $1; do not execute it directly.
#
# Exported: GPU_TUNED_VARIANT/PLATFORM/TORCH_ARCH/HW_LABEL/USE_KLEIDIAI_SME
# (from tuned/devices/<variant>.conf), CUDA_HOME (autodetected highest
# installed toolkit), USE_CUDA, TORCH_CUDA_ARCH_LIST, MAX_JOBS, and (gb10
# only) USE_KLEIDIAI_SME.
#
# USE_CUDA/TORCH_CUDA_ARCH_LIST/USE_KLEIDIAI_SME are plain shell env vars,
# not CMAKE_ARGS -- confirmed against pytorch's own real CI recipe
# (.ci/pytorch/build.sh does `export USE_CUDA=0`/`export USE_CUDA=1`
# directly), not guessed. MAX_JOBS is natively aliased to
# CMAKE_BUILD_PARALLEL_LEVEL via pyproject.toml's own
# [tool.scikit-build.env] table.

GPU_TUNED_ARG_VARIANT="$1"
if [[ -z "${GPU_TUNED_ARG_VARIANT}" ]]; then
    echo "ERROR: env.sh requires a variant argument (gb10/rtx40/rtx50)" >&2
    return 1 2>/dev/null || exit 1
fi

GPU_TUNED_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=devices/rtx50.conf
source "${GPU_TUNED_SELF_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" || return 1 2>/dev/null || exit 1
export GPU_TUNED_VARIANT GPU_TUNED_PLATFORM GPU_TUNED_TORCH_ARCH GPU_TUNED_HW_LABEL GPU_TUNED_USE_KLEIDIAI_SME

# Fail loudly if this script runs on the wrong host, rather than letting a
# mismatched build silently produce a wrong-architecture wheel that only
# surfaces as a confusing failure several steps later (same pattern as
# raft/cuvs's tuned/env.sh).
if [[ "$(uname -m)" != "${GPU_TUNED_PLATFORM}" ]]; then
    echo "ERROR: tuned/env.sh: expected platform '${GPU_TUNED_PLATFORM}' for" \
         "variant '${GPU_TUNED_VARIANT}', but uname -m reports '$(uname -m)'." >&2
    return 1 2>/dev/null || exit 1
fi

# --- Resolve CUDA_HOME to the highest installed toolkit when not explicitly set ---
pytorch_installed_cuda_toolkits() {
    local d
    for d in /usr/local/cuda-[0-9]*; do
        [ -d "$d" ] && basename "$d" | sed 's/^cuda-//'
    done | sort -V
}

if [ -z "${CUDA_HOME:-}" ]; then
    _pytorch_highest="$(pytorch_installed_cuda_toolkits | tail -1)"
    if [ -n "$_pytorch_highest" ]; then
        export CUDA_HOME="/usr/local/cuda-${_pytorch_highest}"
    else
        echo "[tuned/env] WARNING: no /usr/local/cuda-<ver> toolkit found; leaving CUDA_HOME unset." >&2
        echo "[tuned/env]          Set CUDA_HOME explicitly to an installed toolkit." >&2
    fi
    unset _pytorch_highest
fi
[ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"

# CUDA_VERSION_COMPACT (e.g. "133" for CUDA 13.3) -- used by tuned/wheel.sh
# for the version string, derived from CUDA_HOME's own directory name so it
# always matches whatever toolkit actually built this variant, not a
# separately-tracked guess.
if [ -n "${CUDA_HOME:-}" ]; then
    CUDA_VERSION_COMPACT="$(basename "$CUDA_HOME" | sed -E 's/^cuda-([0-9]+)\.([0-9]+).*/\1\2/')"
    export CUDA_VERSION_COMPACT
fi

export USE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${GPU_TUNED_TORCH_ARCH}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
[ -n "${GPU_TUNED_USE_KLEIDIAI_SME}" ] && export USE_KLEIDIAI_SME="${GPU_TUNED_USE_KLEIDIAI_SME}"

echo "[tuned/env] GPU_TUNED_VARIANT=${GPU_TUNED_VARIANT} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} CUDA_HOME=${CUDA_HOME:-<unset>} MAX_JOBS=${MAX_JOBS}"
