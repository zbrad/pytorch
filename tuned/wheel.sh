#!/bin/bash
# tuned/wheel.sh <variant> — package a built pytorch tree into a wheel and
# publish it as a real GitHub release, matching the tag scheme already
# established by the (manually) published GB10 wheel
# (v2.14.0.dev20260703-gb10-cu133) and zbrad/flash-attention's matching
# GB10 tag (v2.7.2.post1-gb10-cu133). Requires tuned/build.sh <variant> to
# have already succeeded (this reuses that venv, does not rebuild).
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

VENV_DIR="${GPU_TUNED_VENV_DIR}"
[[ -d "${VENV_DIR}" ]] || {
    echo "ERROR: ${VENV_DIR} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2
    exit 1
}
gpu_tuned_verify_venv "${VENV_DIR}" "${REPO_ROOT}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

[[ -n "${CUDA_VERSION_COMPACT:-}" ]] || {
    echo "ERROR: CUDA_VERSION_COMPACT not set (CUDA_HOME must resolve to a" \
         "/usr/local/cuda-X.Y directory) -- cannot derive the version string." >&2
    exit 1
}

# Base version from version.txt, with any PEP 440 pre-release suffix
# (a0/b0/rc1/...) stripped -- e.g. "2.14.0a0" -> "2.14.0". We never bump
# this ourselves -- it only moves when upstream's version.txt does, which
# can sit flat across many of our own rebuild cycles -- so it can't carry
# "how far is this build from the last one" on its own. That's what the
# local-version segment below (variant/cuda/date) is for.
BASE_VERSION="$(sed -E 's/[a-z]+[0-9]+$//' "${REPO_ROOT}/version.txt" | tr -d '[:space:]')"
GIT_SHA="$(git rev-parse --short HEAD)"
# Local-version label: <variant>.cu<cuda>.<YYYYMMDD>[.<seq>], e.g.
# gb10.cu133.20261011 (docs/VERSIONING.md in zbrad/tuned-common). The date is HEAD's
# UTC committer date, so rebuilding a commit reproduces its label; the sequence is the
# lowest unused one among this repo's release tags (GPU_TUNED_LABEL_SEQ=0 skips that
# lookup for a dev build that will never be published). This replaces the old
# tuning.<N> commit counter, which a release-branch merge could inflate silently.
# BASE_VERSION stays upstream's, so the wheel still satisfies torch==<upstream> pins.
PYTORCH_LOCAL_VERSION="$(gpu_tuned_build_local_version "${GPU_TUNED_VARIANT}" "${REPO_ROOT}")" || exit 1
[[ "${PYTORCH_LOCAL_VERSION}" == *".cu${CUDA_VERSION_COMPACT}."* ]] || {
    echo "ERROR: label ${PYTORCH_LOCAL_VERSION} disagrees with CUDA_VERSION_COMPACT=${CUDA_VERSION_COMPACT}" \
         "(from ${CUDA_HOME}); nvcc and the directory name must report the same release." >&2
    exit 1
}
export PYTORCH_BUILD_VERSION="${BASE_VERSION}+${PYTORCH_LOCAL_VERSION}"
export PYTORCH_BUILD_NUMBER=1

echo "=========================================="
echo "Packaging pytorch wheel (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "PYTORCH_BUILD_VERSION: ${PYTORCH_BUILD_VERSION}"
echo ""

pip install --upgrade build wheel
rm -rf "${GPU_TUNED_DIST_DIR}"
python3 -m build --wheel --no-isolation --outdir "${GPU_TUNED_DIST_DIR}"

WHEEL="$(ls "${GPU_TUNED_DIST_DIR}"/torch-*.whl 2>/dev/null | head -1)"
[[ -z "${WHEEL}" ]] && { echo "ERROR: no wheel found in ${GPU_TUNED_DIST_DIR}" >&2; exit 1; }
echo "Built wheel: $(basename "${WHEEL}") ($(du -sh "${WHEEL}" | awk '{print $1}'))"

# libtorch_cuda.so is where TORCH_CUDA_ARCH_LIST's actual device code
# lands -- verify + stamp it here, on the wheel's OWN contents, not a
# pre-build copy: python -m build's own fresh `cmake --install` (into a
# throwaway temp prefix) and its subsequent RPATH-rewrite pass both
# discard any section added to a pre-build .so, confirmed empirically --
# a test marker stamped before the build was completely absent afterward,
# tried against both the repo-root copy and the canonical build/lib/ CMake
# output. Unpack -> stamp -> repack (regenerates RECORD correctly, unlike
# a raw zip edit) is the only path that actually reaches what ships.
echo "Stamping build-info into the wheel's own libtorch_cuda.so"
UNPACK_DIR="$(mktemp -d)"
python3 -m wheel unpack "${WHEEL}" --dest "${UNPACK_DIR}"
WHEEL_SO="$(find "${UNPACK_DIR}" -name libtorch_cuda.so)"
[[ -z "${WHEEL_SO}" ]] && { echo "ERROR: libtorch_cuda.so not found inside ${WHEEL}." >&2; exit 1; }
gpu_tuned_assert_versioned_runpath "${WHEEL_SO}"
gpu_tuned_verify_arch "${WHEEL_SO}" "${GPU_TUNED_TORCH_ARCH}"
embed_build_info "${WHEEL_SO}" "${GPU_TUNED_VARIANT}" "torch" "${PYTORCH_BUILD_VERSION}" "${GPU_TUNED_HW_LABEL}"
gpu_tuned_verify_build_info "${WHEEL_SO}" "torch" "${PYTORCH_BUILD_VERSION}" "pytorch_build_info"
rm -f "${WHEEL}"
UNPACKED_CONTENT_DIR="$(find "${UNPACK_DIR}" -maxdepth 1 -mindepth 1 -type d)"
python3 -m wheel pack "${UNPACKED_CONTENT_DIR}" --dest-dir "${GPU_TUNED_DIST_DIR}"
rm -rf "${UNPACK_DIR}"
WHEEL="$(ls "${GPU_TUNED_DIST_DIR}"/torch-*.whl 2>/dev/null | head -1)"
echo "Re-packed with build-info stamp: $(basename "${WHEEL}")"

# Load the finished wheel once in a clean venv before anything is published: label, one torch
# at exactly this version, the stamp in libtorch_cuda.so, CUDA usable, device capability.
# Needs the target GPU on this host. Skip only for a wheel that will never be published.
if [[ "${PYTORCH_WHEEL_SKIP_SMOKE:-}" == 1 ]]; then
    echo "SKIP: wheel smoke test (PYTORCH_WHEEL_SKIP_SMOKE=1); do not publish this wheel"
else
    GPU_TUNED_SMOKE_EXPECT_CAP="${GPU_TUNED_TORCH_ARCH%a}" \
        gpu_tuned_smoke_wheel "${WHEEL}" torch torch pytorch_build_info libtorch_cuda.so || exit 1
fi

# No separate -variant-cu suffix here: PYTORCH_BUILD_VERSION's own local
# segment already carries variant/cuda/date, so appending them
# again would just duplicate them in the tag.
RELEASE_TAG="v${PYTORCH_BUILD_VERSION}"
RELEASE_TITLE="PyTorch ${BASE_VERSION} — ${GPU_TUNED_VARIANT} ${PYTORCH_LOCAL_VERSION#*.} (${GIT_SHA}) — ${GPU_TUNED_HW_LABEL} wheel"

# PYTORCH_WHEEL_NO_PUBLISH=1 stops here: everything above (label, packaging, stamp, smoke
# test) has run, and nothing is sent to GitHub. For testing the pipeline end to end.
if [[ -n "${PYTORCH_WHEEL_NO_PUBLISH:-}" ]]; then
    echo ""
    echo "PYTORCH_WHEEL_NO_PUBLISH set: built ${RELEASE_TAG}, not published."
    echo "Wheel: ${WHEEL}"
    exit 0
fi

echo ""
echo "Publishing wheel to GitHub release ${RELEASE_TAG}..."
gpu_tuned_publish_release "zbrad/pytorch" "${RELEASE_TAG}" "${RELEASE_TITLE}" \
    "torch ${PYTORCH_BUILD_VERSION} wheel for ${GPU_TUNED_HW_LABEL}, single-arch (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}), CUDA ${CUDA_HOME:-unknown}.${GPU_TUNED_NCCL_URL:+ NCCL ${GPU_TUNED_NCCL_VERSION} (${GPU_TUNED_NCCL_SOURCE}) statically linked, from ${GPU_TUNED_NCCL_URL}.}" \
    "${WHEEL}#$(basename "${WHEEL}")"

echo ""
echo "Release: https://github.com/zbrad/pytorch/releases/tag/${RELEASE_TAG}"
echo "Done."
