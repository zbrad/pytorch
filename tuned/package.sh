#!/bin/bash
# tuned/package.sh <variant> — package the wheel tuned/build.sh made and full_test.sh tested:
# stamp a staged copy, write the checksum, package-info.txt and a DRAFT release-notes file. It
# never builds, tests, tags or publishes (docs/ARCHITECTURE.md B3). Stage order:
#   build.sh -> full_test.sh -> package.sh -> smoke_test.sh -> (review and commit the notes) -> release.sh
#
#   CUDA_TAG=cu134 bash tuned/package.sh gb10
#
# Input:  tuned/build/<cuda-tag>/wheel/torch-*.whl (unstamped; never modified here, H3), the
#         release-label and build-commit build.sh wrote, and the passing test log.
# Output: tuned/build/<cuda-tag>/dist/{torch-*.whl, torch-*.whl.sha256, package-info.txt} and
#         tuned/releases/RELEASE_NOTES_<version>.md (DRAFT) if no notes exist yet.
#
# libtorch_cuda-tuned.so is where TORCH_CUDA_ARCH_LIST's device code lands. The stamp is added to a
# staged unpack of the wheel (a section added to a pre-build .so does not survive the build) and the
# wheel is repacked, which regenerates RECORD correctly (a raw zip edit would not).
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/package.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"
# shellcheck source=no_cpu_blas.sh
source "${REPO_ROOT}/tuned/no_cpu_blas.sh"

VENV_DIR="${GPU_TUNED_VENV_DIR}"
[[ -d "${VENV_DIR}" ]] || { echo "ERROR: ${VENV_DIR} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
gpu_tuned_verify_venv "${VENV_DIR}" "${REPO_ROOT}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

BUILT="$(ls "${GPU_TUNED_WHEEL_DIR}"/torch-*.whl 2>/dev/null | head -1)"
[[ -n "${BUILT}" ]] || { echo "ERROR: no wheel in ${GPU_TUNED_WHEEL_DIR}. Run tuned/build.sh ${GPU_TUNED_VARIANT} for ${CUDA_TAG} first." >&2; exit 1; }
LABEL="$(cat "${GPU_TUNED_BUILD_ROOT}/release-label" 2>/dev/null || true)"
BUILD_COMMIT="$(cat "${GPU_TUNED_BUILD_ROOT}/build-commit" 2>/dev/null || true)"
[[ -n "${LABEL}" && -n "${BUILD_COMMIT}" ]] || { echo "ERROR: release-label or build-commit missing in ${GPU_TUNED_BUILD_ROOT}: the last build did not finish." >&2; exit 1; }
[[ "${BUILD_COMMIT}" != *-dirty ]] || { echo "ERROR: the build was made from a dirty tree (${BUILD_COMMIT}); commit and rebuild." >&2; exit 1; }

# The label is the one build.sh allocated and baked in; it is never recomputed here (E4).
BUILT_NAME="$(basename "${BUILT}")"
VERSION="$(gpu_tuned_wheel_version "${BUILT}" torch)"
[[ "${VERSION}" == *"+${GPU_TUNED_VARIANT}.${CUDA_TAG}.${LABEL}" ]] || {
    echo "ERROR: ${BUILT_NAME} is version ${VERSION}, not +${GPU_TUNED_VARIANT}.${CUDA_TAG}.${LABEL} from release-label." >&2
    exit 1
}
gpu_tuned_check_wheel_label "${BUILT}"
BASE_VERSION="${VERSION%%+*}"
RELEASE_TAG="v${VERSION}"
gpu_tuned_check_release_tag "${RELEASE_TAG}"

# H6: refuse a build from other source than HEAD; the commit the test log was run at must be the
# build's, and HEAD may differ from it only in tuned/.
git cat-file -e "${BUILD_COMMIT}^{commit}" 2>/dev/null || { echo "ERROR: build commit ${BUILD_COMMIT} is not in this checkout." >&2; exit 1; }
STALE_SRC="$(git diff --name-only "${BUILD_COMMIT}" HEAD -- . ':!tuned' | head -5)"
if [[ -n "${STALE_SRC}" ]]; then
    echo "ERROR: the wheel is from ${BUILD_COMMIT}, but source outside tuned/ changed since:" >&2
    echo "${STALE_SRC}" | sed 's/^/    /' >&2
    echo "  Rebuild and re-test before packaging." >&2
    exit 1
fi
TEST_LOG="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
[[ -f "${TEST_LOG}" ]] || { echo "ERROR: ${TEST_LOG} not found; run tuned/full_test.sh ${GPU_TUNED_VARIANT} for ${CUDA_TAG} before packaging." >&2; exit 1; }
grep -q "^=== Full test suite summary: .* passed ===$" "${TEST_LOG}" || { echo "ERROR: ${TEST_LOG} does not show a full, passing run." >&2; exit 1; }
[[ "${BUILT}" -nt "${TEST_LOG}" ]] && { echo "ERROR: ${BUILT_NAME} is newer than the test log; re-run tuned/full_test.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }
TESTED_COMMIT="$(sed -n 's/^commit: *//p' "${TEST_LOG}" | head -1)"
TESTED_SHA="$(sed -n 's/^wheel:.*sha256 //p' "${TEST_LOG}" | head -1)"
[[ "${TESTED_SHA}" == "$(sha256sum "${BUILT}" | awk '{print $1}')" ]] || { echo "ERROR: the test log tested a different wheel (sha256 ${TESTED_SHA:-none}) than ${BUILT_NAME}." >&2; exit 1; }

echo "=========================================="
echo "Packaging pytorch wheel (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "Version:      ${VERSION}"
echo "Release tag:  ${RELEASE_TAG}"
echo "Built at:     ${BUILD_COMMIT}; tested at ${TESTED_COMMIT}; packaging at $(git rev-parse HEAD)"
echo ""

mkdir -p "${GPU_TUNED_DIST_DIR}"
# package-info.txt is removed first and written last: its presence means every check below passed.
rm -f "${GPU_TUNED_DIST_DIR}/package-info.txt" "${GPU_TUNED_DIST_DIR}"/torch-*.whl "${GPU_TUNED_DIST_DIR}"/torch-*.whl.sha256

echo "Stamping build-info into a staged copy of the wheel's libtorch_cuda-tuned.so"
UNPACK_DIR="$(mktemp -d)"
trap 'rm -rf "${UNPACK_DIR}"' EXIT
python3 -m wheel unpack "${BUILT}" --dest "${UNPACK_DIR}"
WHEEL_SO="$(find "${UNPACK_DIR}" -name libtorch_cuda-tuned.so)"
[[ -n "${WHEEL_SO}" ]] || { echo "ERROR: libtorch_cuda-tuned.so not found inside ${BUILT_NAME}." >&2; exit 1; }
gpu_tuned_assert_versioned_runpath "${WHEEL_SO}"

# F4 gate: no shared object in the wheel keeps a stock torch name, as a file, a SONAME or a NEEDED entry.
# A check that cannot run is a failure, not a skip (H4): readelf and patchelf must exist.
command -v readelf >/dev/null && command -v patchelf >/dev/null || { echo "ERROR: readelf and patchelf are needed for the library-name gate." >&2; exit 1; }
STOCK_RE='^lib(c10|c10_cuda|torch|torch_cpu|torch_cuda|torch_cuda_linalg|torch_python|torch_global_deps|caffe2_nvrtc|shm)\.so(\.[0-9]+)*$'
NAME_VIOLATIONS=0
while IFS= read -r so; do
    base="$(basename "${so}")"
    [[ "${base}" =~ ${STOCK_RE} ]] && { echo "VIOLATION: ${so#"${UNPACK_DIR}"/} ships under the stock name ${base}." >&2; NAME_VIOLATIONS=1; }
    readelf -d "${so}" 2>/dev/null | sed -n 's/.*\(NEEDED\|SONAME\).*\[\(.*\)\]/\2/p' | grep -E "${STOCK_RE}" | sed "s#^#VIOLATION: ${base} names the stock library #" >&2 && NAME_VIOLATIONS=1
done < <(find "${UNPACK_DIR}" -type f -name '*.so*')
[[ "${NAME_VIOLATIONS}" -eq 0 ]] || { echo "ERROR: the wheel still carries stock torch library names; it was built without the -tuned rename (tuned/build.sh does it)." >&2; exit 1; }
# Every lib*.so in torch/lib is ours (-tuned) or an assumed system/CUDA library; every NEEDED resolves in the stage or is assumed.
ASSUME='^(libc|libm|libdl|librt|libpthread|libstdc\+\+|libgcc_s|libgomp|libatomic|libnuma|ld-linux-[a-z0-9_-]+|libcuda|libcudart|libcublas|libcublasLt|libcusparse|libcufft|libcusolver|libcurand|libnvrtc|libnvJitLink|libcufile|libnvToolsExt|libcupti|libcudnn[a-z_0-9]*|libnccl)\.so(\.[0-9.]+)*$'
gpu_tuned_verify_unique_sonames "$(dirname "${WHEEL_SO}")" - "" "${ASSUME}" || { echo "ERROR: torch/lib fails the unique-name gate." >&2; exit 1; }
echo "OK: every shared object in the wheel carries a -tuned name, SONAME and NEEDED entries"
# GPU-only wheel: no CPU BLAS, LAPACK or MPI, in the build flags it was compiled with or in any NEEDED entry.
gpu_tuned_verify_no_cpu_blas "${UNPACK_DIR}" || { echo "ERROR: the wheel was built with, or links, a CPU BLAS/LAPACK/MPI library (see tuned/no_cpu_blas.sh)." >&2; exit 1; }
echo "OK: no CPU BLAS, LAPACK or MPI in the wheel"
gpu_tuned_verify_arch "${WHEEL_SO}" "${GPU_TUNED_TORCH_ARCH}"
embed_build_info "${WHEEL_SO}" "${GPU_TUNED_VARIANT}" "torch" "${VERSION}" "${GPU_TUNED_HW_LABEL}"
gpu_tuned_verify_build_info "${WHEEL_SO}" "torch" "${VERSION}" "pytorch_build_info"
UNPACKED_CONTENT_DIR="$(find "${UNPACK_DIR}" -maxdepth 1 -mindepth 1 -type d)"
python3 -m wheel pack "${UNPACKED_CONTENT_DIR}" --dest-dir "${GPU_TUNED_DIST_DIR}"
WHEEL="$(ls "${GPU_TUNED_DIST_DIR}"/torch-*.whl | head -1)"
[[ "$(basename "${WHEEL}")" == "${BUILT_NAME}" ]] || { echo "ERROR: repacked wheel is $(basename "${WHEEL}"), expected ${BUILT_NAME}." >&2; exit 1; }
gpu_tuned_check_wheel_label "${WHEEL}"
(cd "${GPU_TUNED_DIST_DIR}" && sha256sum "${BUILT_NAME}" > "${BUILT_NAME}.sha256")
WHEEL_SHA="$(awk '{print $1}' "${WHEEL}.sha256")"
echo "Wheel:  ${BUILT_NAME} ($(du -sh "${WHEEL}" | awk '{print $1}'))"
echo "SHA256: ${WHEEL_SHA}"

# H5: a DRAFT notes file if none is tracked yet; release.sh refuses it until a person has reviewed
# it and removed the banner. The numbers come from the attached test log (G4).
NOTES="tuned/releases/RELEASE_NOTES_${RELEASE_TAG#v}.md"
if [[ ! -f "${NOTES}" ]]; then
    mkdir -p tuned/releases
    {
        echo "<!-- DRAFT — not yet published: review every line, then delete this banner and commit. -->"
        echo "# PyTorch ${BASE_VERSION} — ${GPU_TUNED_VARIANT} ${CUDA_TAG} (${RELEASE_TAG#v})"
        echo ""
        echo "Single-arch wheel for ${GPU_TUNED_HW_LABEL} (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}), CUDA ${CUDA_VER}."
        echo ""
        echo "- Release label: ${LABEL} (UTC date of the tuned commit)"
        echo "- Built from commit ${BUILD_COMMIT}; tested at ${TESTED_COMMIT}"
        echo "- Upstream base: ${GPU_TUNED_UPSTREAM_REF} merge-base $(git merge-base HEAD "${GPU_TUNED_UPSTREAM_REF}" 2>/dev/null || echo unknown)"
        if [[ -n "${GPU_TUNED_NCCL_VERSION:-}" ]]; then
            echo "- NCCL: ${GPU_TUNED_NCCL_VERSION} (${GPU_TUNED_NCCL_SOURCE}), statically linked, from ${GPU_TUNED_NCCL_URL}"
        else
            echo "- NCCL: built in-tree"
        fi
        echo "- Wheel sha256: ${WHEEL_SHA}"
        echo ""
        echo "## Tests"
        echo ""
        echo "Scope (from the attached test log): $(sed -n 's/^scope: *//p' "${TEST_LOG}" | head -1)"
        echo "Modules: $(sed -n 's/^modules: *//p' "${TEST_LOG}" | head -1)"
        echo "Result: $(grep '^=== Full test suite summary' "${TEST_LOG}" | tail -1)"
        echo ""
        echo "## Install"
        echo ""
        echo '```'
        echo "pip install \"torch @ https://github.com/zbrad/pytorch/releases/download/${RELEASE_TAG}/${BUILT_NAME}\""
        echo '```'
    } > "${NOTES}"
    echo "Wrote DRAFT release notes: ${NOTES}"
fi

# Written last.
cat > "${GPU_TUNED_DIST_DIR}/package-info.txt" <<INFO
release_tag=${RELEASE_TAG}
wheel=${BUILT_NAME}
version=${VERSION}
upstream_version=${BASE_VERSION}
label=${LABEL}
commit=$(git rev-parse HEAD)
built_commit=${BUILD_COMMIT}
tested_commit=${TESTED_COMMIT}
cuda_tag=${CUDA_TAG}
variant=${GPU_TUNED_VARIANT}
sha256=${WHEEL_SHA}
INFO

echo ""
echo "Packaged (not published): ${WHEEL} (${RELEASE_TAG})"
echo "Next: bash tuned/smoke_test.sh ${GPU_TUNED_VARIANT}, review and commit ${NOTES}, then bash tuned/release.sh ${GPU_TUNED_VARIANT}"
