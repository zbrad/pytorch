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

# shellcheck source=common.sh
# Vendored from https://github.com/zbrad/tuned-common (pinned commit --
# see common.sh's own header/sync instructions to update). Provides
# gpu_tuned_verify_arch/assert_platform/installed_cuda_toolkits, shared
# verbatim across the fleet instead of hand-copied-and-edited per repo.
# NOT used for embed_build_info here: kept local on purpose (same
# reasoning as zbrad/raft's/zbrad/cuvs's/zbrad/faiss's tuned/env.sh).
source "${GPU_TUNED_SELF_DIR}/common.sh" || return 1 2>/dev/null || exit 1

# Fail loudly if this script runs on the wrong host, rather than letting a
# mismatched build silently produce a wrong-architecture wheel that only
# surfaces as a confusing failure several steps later (same pattern as
# raft/cuvs's tuned/env.sh).
gpu_tuned_assert_platform "${GPU_TUNED_PLATFORM}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- Resolve CUDA_HOME to the highest installed toolkit when not explicitly set ---
if [ -z "${CUDA_HOME:-}" ]; then
    _pytorch_highest="$(gpu_tuned_installed_cuda_toolkits | tail -1)"
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

# gpu_tuned_verify_arch now comes from common.sh (sourced above); call
# sites pass GPU_TUNED_TORCH_ARCH explicitly (the shared version takes it
# as an arg instead of reading a global, since different repos in the
# fleet name their arch var differently).

# embed_build_info <so_path> <variant> <package> <version> [hw_label] —
# thin wrapper over gpu_tuned_embed_build_info (common.sh) that pins the
# section name to .pytorch_build_info (this repo's pre-existing name,
# via the explicit [section-name] override -- otherwise the shared
# function would derive .torch_build_info from the "torch" package arg
# this repo's call sites pass, a rename), while keeping the real package
# name ("torch") in the message.
embed_build_info() {
    local so_path="$1" variant="$2" package="$3" version="$4" hw_label="$5"
    gpu_tuned_embed_build_info "${so_path}" "${variant}" "${package}" "${version}" \
        "${hw_label}" "https://github.com/zbrad/pytorch" "pytorch_build_info"
}
