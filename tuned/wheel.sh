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
# (a0/b0/rc1/...) stripped -- e.g. "2.14.0a0" -> "2.14.0". We never bump
# this ourselves -- it only moves when upstream's version.txt does, which
# can sit flat across many of our own rebuild cycles -- so it can't carry
# "how far is this build from the last one" on its own. That's what the
# local-version segment below (variant/cuda/tuning count) is for.
BASE_VERSION="$(sed -E 's/[a-z]+[0-9]+$//' "${REPO_ROOT}/version.txt" | tr -d '[:space:]')"
GIT_SHA="$(git rev-parse --short HEAD)"
# Commits on tuned-builds *since it diverged from main* (main..HEAD), not
# total HEAD history: total history is dominated by upstream's own commit
# count (hundreds of thousands, moving on every fetch-and-merge regardless
# of whether any of our own code changed) and would swamp the one signal
# this is actually for -- how much of *our* tuned-builds work (scripts,
# patches, conflict resolutions) has landed since an earlier wheel was
# built. `main` is kept fast-forwarded to upstream/main by the same
# fetch-and-rebuild flow that produced this checkout, so this is genuinely
# "commits ahead of upstream," not a stray branch reference.
#
# Surfaced as "tuning-vN" in the local version segment below, not a
# `.devN` release-segment -- same label llama.cpp's install.sh uses for
# its own pinned-tooling tag, but counting differently: llama.cpp bumps
# its tuning-vN by hand and only when tooling changes (its tuned-builds
# has no common ancestor with upstream/master at all -- verified via
# `git merge-base`, likely a history-recovery artifact -- so main..HEAD
# isn't even computable there). Here it's this exact commit count, always.
#
# No `.devN` marker and no `+git<sha>`: BASE_VERSION alone reads as if it
# were a final upstream release (e.g. "2.15.0"), which would be a real
# resolver hazard for a PyPI-distributed package -- but GB10/aarch64/sm121
# has no official upstream wheel at all (see this file's header), nothing
# else can claim that exact version for this platform, and installs here
# only ever happen via `@ URL` (requirements/gb10.txt in vllm et al.), not
# index resolution, so no other install path can collide with it either.
# The exact commit is still fully recoverable from TUNED_COMMIT_COUNT
# against tuned-builds' own history (barring another rewrite) and, belt
# and suspenders, from GIT_SHA in the release title below.
TUNED_COMMIT_COUNT="$(git rev-list --count main..HEAD)"

export PYTORCH_BUILD_VERSION="${BASE_VERSION}+${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.tuning-v${TUNED_COMMIT_COUNT}"
export PYTORCH_BUILD_NUMBER=1

echo "=========================================="
echo "Packaging pytorch wheel (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "PYTORCH_BUILD_VERSION: ${PYTORCH_BUILD_VERSION}"
echo ""

# libtorch_cuda.so is where TORCH_CUDA_ARCH_LIST's actual device code
# lands -- verify + stamp it before packaging, same discipline as
# raft/cuvs/faiss's tuned/wheel.sh|package.sh.
TORCH_CUDA_SO="${REPO_ROOT}/torch/lib/libtorch_cuda.so"
if [[ -f "${TORCH_CUDA_SO}" ]]; then
    gpu_tuned_verify_arch "${TORCH_CUDA_SO}" "${GPU_TUNED_TORCH_ARCH}"
    embed_build_info "${TORCH_CUDA_SO}" "${GPU_TUNED_VARIANT}" "torch" "${PYTORCH_BUILD_VERSION}" "${GPU_TUNED_HW_LABEL}"
else
    echo "ERROR: ${TORCH_CUDA_SO} not found -- run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2
    exit 1
fi

pip install --upgrade build
rm -rf "${REPO_ROOT}/dist"
python3 -m build --wheel --no-isolation

WHEEL="$(ls "${REPO_ROOT}"/dist/torch-*.whl 2>/dev/null | head -1)"
[[ -z "${WHEEL}" ]] && { echo "ERROR: no wheel found in dist/" >&2; exit 1; }
echo "Built wheel: $(basename "${WHEEL}") ($(du -sh "${WHEEL}" | awk '{print $1}'))"

# No separate -variant-cu suffix here: PYTORCH_BUILD_VERSION's own local
# segment already carries variant/cuda/tuning-count, so appending them
# again would just duplicate them in the tag.
RELEASE_TAG="v${PYTORCH_BUILD_VERSION}"
RELEASE_TITLE="PyTorch ${BASE_VERSION} — ${GPU_TUNED_VARIANT} tuning-v${TUNED_COMMIT_COUNT} (cu${CUDA_VERSION_COMPACT}, ${GIT_SHA}) — ${GPU_TUNED_HW_LABEL} wheel"

echo ""
echo "Publishing wheel to GitHub release ${RELEASE_TAG}..."
gpu_tuned_publish_release "zbrad/pytorch" "${RELEASE_TAG}" "${RELEASE_TITLE}" \
    "torch ${PYTORCH_BUILD_VERSION} wheel for ${GPU_TUNED_HW_LABEL}, single-arch (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}), CUDA ${CUDA_HOME:-unknown}." \
    "${WHEEL}#$(basename "${WHEEL}")"

echo ""
echo "Release: https://github.com/zbrad/pytorch/releases/tag/${RELEASE_TAG}"
echo "Done."
