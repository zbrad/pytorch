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

VENV_DIR="${REPO_ROOT}/.venv-${GPU_TUNED_VARIANT}"
[[ -d "${VENV_DIR}" ]] || {
    echo "ERROR: ${VENV_DIR} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2
    exit 1
}
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

[[ -n "${CUDA_VERSION_COMPACT:-}" ]] || {
    echo "ERROR: CUDA_VERSION_COMPACT not set (CUDA_HOME must resolve to a" \
         "/usr/local/cuda-X.Y directory) -- cannot derive the version string." >&2
    exit 1
}

# Base version from version.txt, with any PEP 440 pre-release suffix
# (a0/b0/rc1/...) stripped -- e.g. "2.14.0a0" -> "2.14.0". The dev-date +
# git-sha + variant + cuda-tag suffix appended below is what actually makes
# each tuned build's version string unique, matching the shape of the
# already-published GB10 wheel (2.14.0.dev20260703+git9dcbe34504.gb10.cu133).
BASE_VERSION="$(sed -E 's/[a-z]+[0-9]+$//' "${REPO_ROOT}/version.txt" | tr -d '[:space:]')"
GIT_SHA="$(git rev-parse --short HEAD)"
BUILD_DATE="$(date -u +%Y%m%d)"

export PYTORCH_BUILD_VERSION="${BASE_VERSION}.dev${BUILD_DATE}+git${GIT_SHA}.${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}"
export PYTORCH_BUILD_NUMBER=1

echo "=========================================="
echo "Packaging pytorch wheel (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "PYTORCH_BUILD_VERSION: ${PYTORCH_BUILD_VERSION}"
echo ""

pip install --upgrade build
rm -rf "${REPO_ROOT}/dist"
python3 -m build --wheel --no-isolation

WHEEL="$(ls "${REPO_ROOT}"/dist/torch-*.whl 2>/dev/null | head -1)"
[[ -z "${WHEEL}" ]] && { echo "ERROR: no wheel found in dist/" >&2; exit 1; }
echo "Built wheel: $(basename "${WHEEL}") ($(du -sh "${WHEEL}" | awk '{print $1}'))"

RELEASE_TAG="v${PYTORCH_BUILD_VERSION}-${GPU_TUNED_VARIANT}-cu${CUDA_VERSION_COMPACT}"
RELEASE_TITLE="PyTorch ${PYTORCH_BUILD_VERSION} — ${GPU_TUNED_HW_LABEL} wheel"

echo ""
echo "Publishing wheel to GitHub release ${RELEASE_TAG}..."
gh release create "${RELEASE_TAG}" \
    --repo zbrad/pytorch \
    --title "${RELEASE_TITLE}" \
    --target "tuned-builds" \
    --notes "torch ${PYTORCH_BUILD_VERSION} wheel for ${GPU_TUNED_HW_LABEL}, single-arch (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}), CUDA ${CUDA_HOME:-unknown}." \
    "${WHEEL}#$(basename "${WHEEL}")"

echo ""
echo "Release: https://github.com/zbrad/pytorch/releases/tag/${RELEASE_TAG}"
echo "Done."
