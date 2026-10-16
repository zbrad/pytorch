#!/bin/bash
# tuned/env.sh <variant> — device config + build-env setup for a tuned
# single-arch pytorch build (gb10/rtx40/rtx50). Source this file with the
# variant as $1; do not execute it directly.
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10     (or CUDA_VER=13.4)
#
# Device facts come from tuned/common-data/devices/<variant>.conf and the CUDA toolkit from
# tuned/common-data/toolkits/<id>.env (both vendored from zbrad/tuned-common), never from
# /usr/local/cuda or from what happens to be installed. With neither CUDA_TAG nor CUDA_VER set
# the toolkit named in toolkits/DEFAULT is used, and the available list and the reason are
# printed. Repo-specific settings live in tuned/repo.env, loaded last; it may not change any
# shared key.
#
# Exported: GPU_TUNED_VARIANT/PLATFORM/CUDA_ARCH/HW_LABEL (shared), GPU_TUNED_TORCH_ARCH and
# GPU_TUNED_USE_KLEIDIAI_SME (repo.env), CUDA_HOME/CUDA_TAG/CUDA_VER (the toolkit),
# GPU_TUNED_BUILD_ROOT (tuned/build/cu<ver>: venv/ cmake/ dist/ releases/ ccache/), USE_CUDA,
# TORCH_CUDA_ARCH_LIST, MAX_JOBS, and (gb10 only) USE_KLEIDIAI_SME.
#
# USE_CUDA/TORCH_CUDA_ARCH_LIST/USE_KLEIDIAI_SME are plain shell env vars,
# not CMAKE_ARGS -- confirmed against pytorch's own real CI recipe
# (.ci/pytorch/build.sh does `export USE_CUDA=0`/`export USE_CUDA=1`
# directly), not guessed. MAX_JOBS is natively aliased to
# CMAKE_BUILD_PARALLEL_LEVEL via pyproject.toml's own
# [tool.scikit-build.env] table.

GPU_TUNED_ARG_VARIANT="${1:-}"
if [[ -z "${GPU_TUNED_ARG_VARIANT}" ]]; then
    echo "ERROR: env.sh requires a variant argument (gb10/rtx40/rtx50)" >&2
    return 1 2>/dev/null || exit 1
fi

GPU_TUNED_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_TUNED_DATA_DIR="${GPU_TUNED_SELF_DIR}/common-data"
if [[ ! -f "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" ]]; then
    echo "ERROR: no device '${GPU_TUNED_ARG_VARIANT}' in ${GPU_TUNED_DATA_DIR}/devices (have: $(cd "${GPU_TUNED_DATA_DIR}/devices" && ls | sed 's/\.conf$//' | tr '\n' ' '))" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck source=common-data/devices/gb10.conf
source "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" || return 1 2>/dev/null || exit 1
export GPU_TUNED_VARIANT GPU_TUNED_PLATFORM GPU_TUNED_CUDA_ARCH GPU_TUNED_HW_LABEL

# shellcheck source=common.sh
# Vendored from https://github.com/zbrad/tuned-common (see sync-common.sh).
source "${GPU_TUNED_SELF_DIR}/common.sh" || return 1 2>/dev/null || exit 1

# Fail loudly if this script runs on the wrong host, rather than letting a
# mismatched build silently produce a wrong-architecture wheel that only
# surfaces as a confusing failure several steps later.
gpu_tuned_assert_platform "${GPU_TUNED_PLATFORM}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- CUDA toolkit: CUDA_TAG (cu134) or CUDA_VER (13.4) select one from
#     tuned/common-data/toolkits; neither means toolkits/DEFAULT, stated loudly. ---
_pytorch_tk="${CUDA_TAG:-}"
_pytorch_req_ver="${CUDA_VER:-}"
if [ -z "${_pytorch_tk}" ] && [ -n "${_pytorch_req_ver}" ]; then
    _pytorch_tk="cu${_pytorch_req_ver//./}"                     # 13.4 -> cu134
fi
# An inherited CUDA_HOME must agree with the toolkit (gpu_tuned_load_toolkit refuses otherwise).
gpu_tuned_load_toolkit "${GPU_TUNED_DATA_DIR}" "${GPU_TUNED_VARIANT}" "${_pytorch_tk}" || return 1 2>/dev/null || exit 1
if [ -n "${_pytorch_req_ver}" ] && [ "${_pytorch_req_ver}" != "${TOOLKIT_CUDA_VER}" ]; then
    echo "[tuned/env] ERROR: CUDA_VER=${_pytorch_req_ver} conflicts with toolkit ${TOOLKIT_ID} (CUDA ${TOOLKIT_CUDA_VER})." >&2
    return 1 2>/dev/null || exit 1
fi
unset _pytorch_tk _pytorch_req_ver
gpu_tuned_verify_toolkit || return 1 2>/dev/null || exit 1

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

# CUDA_VERSION_COMPACT (e.g. "133" for CUDA 13.3) is the toolkit id's digits, used by
# tuned/wheel.sh for the version string.
CUDA_VERSION_COMPACT="${CUDA_TAG#cu}"
export CUDA_VERSION_COMPACT

# --- Repo-specific settings (torch arch spelling, KleidiAI, upstream ref), loaded after the
#     shared device and toolkit values; may not change any of them. ---
gpu_tuned_load_repo_env "${GPU_TUNED_SELF_DIR}/repo.env" || return 1 2>/dev/null || exit 1
export GPU_TUNED_TORCH_ARCH GPU_TUNED_USE_KLEIDIAI_SME

# Everything a tuned build produces lives under tuned/build/cu<ver>/ (ignored
# by git via build/): venv/, cmake/ (CMake build tree, via scikit-build-core's
# SKBUILD_BUILD_DIR), dist/ (wheels), releases/ (test and smoke logs) and ccache/. One tree per toolkit, shared by
# build.sh/wheel.sh/verify_abi.sh.
GPU_TUNED_REPO_ROOT="$(cd "${GPU_TUNED_SELF_DIR}/.." && pwd)"
# One variant per toolkit tree: refuses a tree last used by another variant.
gpu_tuned_claim_toolkit_tree "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1
GPU_TUNED_BUILD_ROOT="${GPU_TUNED_REPO_ROOT}/tuned/build/${CUDA_TAG}"
GPU_TUNED_VENV_DIR="${GPU_TUNED_BUILD_ROOT}/venv"
GPU_TUNED_CMAKE_DIR="${GPU_TUNED_BUILD_ROOT}/cmake"
GPU_TUNED_DIST_DIR="${GPU_TUNED_BUILD_ROOT}/dist"
GPU_TUNED_RELEASES_DIR="${GPU_TUNED_BUILD_ROOT}/releases"
export GPU_TUNED_BUILD_ROOT GPU_TUNED_VENV_DIR GPU_TUNED_CMAKE_DIR GPU_TUNED_DIST_DIR GPU_TUNED_RELEASES_DIR
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
    # FindNCCL's cached NCCL_INCLUDE_DIR wins over the environment, so a tree configured against
    # one NCCL silently keeps it after GPU_TUNED_NCCL_DIR (or the pin) changes.
    cached="$(sed -n 's/^NCCL_INCLUDE_DIR:PATH=//p' "${cache}" | head -1)"
    if [[ -n "${GPU_TUNED_NCCL_DIR:-}" && "${cached}" != "${GPU_TUNED_NCCL_DIR}/include" ]]; then
        echo "ERROR: ${cache} was configured against NCCL '${cached:-<in-tree>}', but this build uses ${GPU_TUNED_NCCL_DIR}. Remove the tree first: rm -rf ${GPU_TUNED_CMAKE_DIR}" >&2
        return 1
    fi
    if [[ -z "${GPU_TUNED_NCCL_DIR:-}" && -n "${cached}" ]]; then
        echo "ERROR: ${cache} was configured against the prebuilt NCCL '${cached}', but this build uses the in-tree NCCL. Remove the tree first: rm -rf ${GPU_TUNED_CMAKE_DIR}" >&2
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
export MAX_JOBS="${MAX_JOBS:-$(gpu_tuned_jobs)}"
[ -n "${GPU_TUNED_USE_KLEIDIAI_SME}" ] && export USE_KLEIDIAI_SME="${GPU_TUNED_USE_KLEIDIAI_SME}"

# Prebuilt NCCL, pinned. tuned/nccl.pins names a zbrad/nccl release per variant and toolkit; it is
# fetched, checksum-verified and extracted once into build/<toolkit>/nccl/<tag>/ and used unless
# GPU_TUNED_NCCL_BUILTIN=1 (build NCCL in-tree, the old path) or GPU_TUNED_NCCL_DIR is already set.
if [ -z "${GPU_TUNED_NCCL_DIR:-}" ] && [ -z "${GPU_TUNED_NCCL_BUILTIN:-}" ] && [ -f "${GPU_TUNED_SELF_DIR}/nccl.pins" ]; then
    _nccl_tag="$(awk -v v="${GPU_TUNED_VARIANT}" -v c="${CUDA_TAG}" '$1 !~ /^#/ && $1==v && $2==c {print $3}' "${GPU_TUNED_SELF_DIR}/nccl.pins")"
    if [ -n "${_nccl_tag}" ]; then
        bash "${GPU_TUNED_SELF_DIR}/fetch_nccl.sh" "${_nccl_tag}" "${GPU_TUNED_BUILD_ROOT}/nccl/${_nccl_tag}" >&2 || { unset _nccl_tag; return 1 2>/dev/null || exit 1; }
        GPU_TUNED_NCCL_DIR="${GPU_TUNED_BUILD_ROOT}/nccl/${_nccl_tag}"
    else
        echo "[tuned/env] ERROR: no NCCL pin for ${GPU_TUNED_VARIANT} ${CUDA_TAG} in tuned/nccl.pins." >&2
        echo "  Pin a zbrad/nccl release for it, choose a pinned toolkit (CUDA_TAG=...), or set GPU_TUNED_NCCL_BUILTIN=1 to build NCCL in-tree on purpose." >&2
        unset _nccl_tag
        return 1 2>/dev/null || exit 1
    fi
    unset _nccl_tag
fi
# By default PyTorch clones NCCL into third_party/nccl and builds it in-tree.
# GPU_TUNED_NCCL_DIR=<dir> instead links a prebuilt single-arch NCCL (a zbrad/nccl release
# tarball, extracted: <dir>/lib/libnccl_static.a and <dir>/include/nccl.h) through PyTorch's own
# USE_SYSTEM_NCCL + USE_STATIC_NCCL, with no PyTorch source change. Opt-in; unset keeps the
# in-tree build and its tuned/patches/nccl-ldmc-arch-gate.patch step.
if [ -n "${GPU_TUNED_NCCL_DIR:-}" ]; then
    if [ ! -f "${GPU_TUNED_NCCL_DIR}/lib/libnccl_static.a" ] || [ ! -f "${GPU_TUNED_NCCL_DIR}/include/nccl.h" ]; then
        echo "ERROR: GPU_TUNED_NCCL_DIR=${GPU_TUNED_NCCL_DIR} has no lib/libnccl_static.a and include/nccl.h." >&2
        return 1 2>/dev/null || exit 1
    fi
    export USE_SYSTEM_NCCL=1 USE_STATIC_NCCL=1
    export NCCL_ROOT="${GPU_TUNED_NCCL_DIR}" NCCL_INCLUDE_DIR="${GPU_TUNED_NCCL_DIR}/include" NCCL_LIB_DIR="${GPU_TUNED_NCCL_DIR}/lib"
    echo "[tuned/env] prebuilt NCCL: ${GPU_TUNED_NCCL_DIR} (USE_SYSTEM_NCCL=1 USE_STATIC_NCCL=1)"
    # What was linked, for the build-info stamp (the "deps" field) and the release notes: NCCL's
    # own version, the zbrad/nccl release it came from, and its source describe. Read from the
    # tarball's BUILD_INFO, so it cannot disagree with the files that were really linked.
    _nccl_bi="${GPU_TUNED_NCCL_DIR}/BUILD_INFO"
    GPU_TUNED_NCCL_VERSION="$(sed -n 's/^nccl_version=//p' "${_nccl_bi}" 2>/dev/null | head -1)"
    GPU_TUNED_NCCL_TAG="$(sed -n 's/^tag=//p' "${_nccl_bi}" 2>/dev/null | head -1)"
    GPU_TUNED_NCCL_SOURCE="$(sed -n 's/^source=//p' "${_nccl_bi}" 2>/dev/null | head -1)"
    unset _nccl_bi
    if [ -z "${GPU_TUNED_NCCL_VERSION}" ] || [ -z "${GPU_TUNED_NCCL_TAG}" ]; then
        echo "ERROR: ${GPU_TUNED_NCCL_DIR}/BUILD_INFO lacks nccl_version or tag; is this a zbrad/nccl release extraction?" >&2
        return 1 2>/dev/null || exit 1
    fi
    GPU_TUNED_NCCL_URL="https://github.com/zbrad/nccl/releases/tag/${GPU_TUNED_NCCL_TAG}"
    export GPU_TUNED_NCCL_VERSION GPU_TUNED_NCCL_TAG GPU_TUNED_NCCL_SOURCE GPU_TUNED_NCCL_URL
    export GPU_TUNED_BUILD_INFO_DEPS="nccl ${GPU_TUNED_NCCL_VERSION} (${GPU_TUNED_NCCL_SOURCE}, static) ${GPU_TUNED_NCCL_URL}"
else
    export GPU_TUNED_BUILD_INFO_DEPS="nccl in-tree (third_party/nccl $(git -C "${GPU_TUNED_SELF_DIR}/../third_party/nccl" describe --tags --long 2>/dev/null || echo unknown))"
fi

export GPU_TUNED_UPSTREAM_REF

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
