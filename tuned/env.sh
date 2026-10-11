#!/bin/bash
# tuned/env.sh <variant> — device config + build-env setup for a tuned
# single-arch pytorch build (gb10/rtx40/rtx50). Source this file with the
# variant as $1; do not execute it directly.
#
# Exported: GPU_TUNED_VARIANT/PLATFORM/TORCH_ARCH/HW_LABEL/USE_KLEIDIAI_SME
# (from tuned/devices/<variant>.conf), CUDA_HOME (an explicit
# /usr/local/cuda-X.Y; autodetected highest installed when unset),
# GPU_TUNED_BUILD_ROOT (tuned/build/cu<ver>: venv/ cmake/ dist/ ccache/), USE_CUDA, TORCH_CUDA_ARCH_LIST, MAX_JOBS, and (gb10
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

# --- CUDA toolkit: only an explicit /usr/local/cuda-X.Y path, never the
# unversioned /usr/local/cuda (or cuda-13) alternatives symlinks, which
# silently follow whatever `update-alternatives` points at. CUDA_HOME
# defaults to the highest installed versioned toolkit.
if [ -z "${CUDA_HOME:-}" ]; then
    _pytorch_highest="$(gpu_tuned_installed_cuda_toolkits | tail -1)"
    if [ -z "$_pytorch_highest" ]; then
        echo "ERROR: no /usr/local/cuda-X.Y toolkit found; set CUDA_HOME to one." >&2
        return 1 2>/dev/null || exit 1
    fi
    export CUDA_HOME="/usr/local/cuda-${_pytorch_highest}"
    unset _pytorch_highest
fi
if [[ ! "${CUDA_HOME}" =~ ^/usr/local/cuda-[0-9]+\.[0-9]+$ || ! -d "${CUDA_HOME}" ]]; then
    echo "ERROR: CUDA_HOME='${CUDA_HOME}' must be an existing /usr/local/cuda-X.Y directory (not /usr/local/cuda or cuda-<major>)." >&2
    return 1 2>/dev/null || exit 1
fi

# Drop unversioned /usr/local/cuda[-<major>] entries so nothing resolves
# through them, then put the explicit toolkit first.
for _pytorch_var in PATH LD_LIBRARY_PATH LIBRARY_PATH CPATH; do
    _pytorch_val="$(printf '%s' "${!_pytorch_var:-}" | tr ':' '\n' | grep -Ev '^/usr/local/cuda(-[0-9]+)?(/|$)' | paste -sd: - || true)"
    export "${_pytorch_var}=${_pytorch_val}"
done
unset _pytorch_var _pytorch_val
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_PATH="${CUDA_HOME}" CUDACXX="${CUDA_HOME}/bin/nvcc"
# -DBUILD_TEST=OFF: skip the C++ test binaries (GCC 14.2 ICEs on the
# vec_test_all_types_SVE128 target). Passed as -D, not the BUILD_TEST env var,
# because EnvVarForwarding.cmake ignores the env var once the cache has a value.
export CMAKE_ARGS="-DCUDAToolkit_ROOT=${CUDA_HOME} -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc -DBUILD_TEST=OFF ${CMAKE_ARGS:-}"

# CUDA_VERSION_COMPACT (e.g. "133" for CUDA 13.3) -- used by tuned/wheel.sh
# for the version string and by the venv name below, derived from
# CUDA_HOME's own directory name so it always matches the toolkit in use.
CUDA_VERSION_COMPACT="$(basename "$CUDA_HOME" | sed -E 's/^cuda-([0-9]+)\.([0-9]+).*/\1\2/')"
export CUDA_VERSION_COMPACT

# Everything a tuned build produces lives under tuned/build/cu<ver>/ (ignored
# by git via build/): venv/, cmake/ (CMake build tree, via scikit-build-core's
# SKBUILD_BUILD_DIR), dist/ (wheels) and ccache/. One tree per toolkit, shared by
# build.sh/wheel.sh/verify_abi.sh.
GPU_TUNED_BUILD_ROOT="${GPU_TUNED_SELF_DIR}/build/cu${CUDA_VERSION_COMPACT}"
GPU_TUNED_VENV_DIR="${GPU_TUNED_BUILD_ROOT}/venv"
GPU_TUNED_CMAKE_DIR="${GPU_TUNED_BUILD_ROOT}/cmake"
GPU_TUNED_DIST_DIR="${GPU_TUNED_BUILD_ROOT}/dist"
export GPU_TUNED_BUILD_ROOT GPU_TUNED_VENV_DIR GPU_TUNED_CMAKE_DIR GPU_TUNED_DIST_DIR
export SKBUILD_BUILD_DIR="${GPU_TUNED_CMAKE_DIR}"
# ccache is isolated per toolkit too, so timings and hits never cross toolkits.
# CCACHE_DIR makes ccache ignore ~/.config/ccache/ccache.conf, so its 5 GiB
# default max_size would apply (too small for a PyTorch build): set it here.
export CCACHE_DIR="${GPU_TUNED_BUILD_ROOT}/ccache"
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-60G}"

# gpu_tuned_assert_cmake_cache_consistent: fail if an existing CMake cache
# was configured for a different toolkit than CUDA_HOME (it would silently
# keep using the old one) or for a different source tree (a build dir copied
# from another checkout; common.sh's gpu_tuned_verify_venv only looks at
# <repo>/build, not this directory).
gpu_tuned_assert_cmake_cache_consistent() {
    local cache="${GPU_TUNED_CMAKE_DIR}/CMakeCache.txt" cached repo_root
    [[ -f "${cache}" ]] || return 0
    repo_root="$(cd "${GPU_TUNED_SELF_DIR}/.." && pwd)"
    cached="$(sed -n 's/^CMAKE_CUDA_COMPILER:FILEPATH=//p' "${cache}" | head -1)"
    if [[ -n "${cached}" && "${cached}" != "${CUDA_HOME}/bin/nvcc" ]]; then
        echo "ERROR: ${GPU_TUNED_CMAKE_DIR} was configured with ${cached}, but CUDA_HOME=${CUDA_HOME}. Remove it first: rm -rf ${GPU_TUNED_CMAKE_DIR}" >&2
        return 1
    fi
    cached="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${cache}" | head -1)"
    if [[ -n "${cached}" && "${cached}" != "${repo_root}" ]]; then
        echo "ERROR: ${cache} was configured for source tree ${cached}, not ${repo_root} -- looks copied from another checkout. Remove it first: rm -rf ${GPU_TUNED_CMAKE_DIR}" >&2
        return 1
    fi
}

# gpu_tuned_assert_versioned_runpath <so>: fail if the library's
# RUNPATH/RPATH points through the unversioned /usr/local/cuda[-<major>].
gpu_tuned_assert_versioned_runpath() {
    local bad
    bad="$(readelf -d "$1" | sed -n 's/.*\(RUNPATH\|RPATH\).*\[\(.*\)\]/\2/p' | tr ':' '\n' | grep -E '^/usr/local/cuda(-[0-9]+)?(/|$)' || true)"
    if [[ -n "${bad}" ]]; then
        echo "ERROR: $1 has an unversioned CUDA path in its RUNPATH: ${bad}" >&2
        return 1
    fi
}

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
