#!/bin/bash
# tuned/release.sh <variant> — run the release gates for the wheel tuned/package.sh made. Stage order:
#   build.sh -> full_test.sh -> package.sh -> smoke_test.sh -> (review and commit
#   tuned/releases/RELEASE_NOTES_<version>.md) -> this -> publish.sh.
# It never builds, tests or packages, and never publishes. If every gate passes it writes a ready-marker
# (RELEASE_READY_<variant>); tuned/publish.sh <variant> publishes exactly what the gates saw.
# docs/ARCHITECTURE.md (zbrad/tuned-common) lists every gate.
#
#   CUDA_TAG=cu134 bash tuned/release.sh gb10
#
# The release is named by the wheel's date label (v<upstream>+<variant>.<cuda>.<YYYYMMDD>[.<seq>]),
# read from package-info.txt and never recomputed here: the notes commit moves HEAD and its date.
# The gates are all reported together; no marker is written unless every one passes.
set -uo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/release.sh <variant>}"

# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1
cd "${REPODIR}" || exit 1

INFO="${GPU_TUNED_DIST_DIR}/package-info.txt"
TEST_LOG="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
SMOKE_LOG="${GPU_TUNED_RELEASES_DIR}/SMOKE_RESULTS_${GPU_TUNED_VARIANT}.log"
MARKER="${GPU_TUNED_RELEASES_DIR}/RELEASE_READY_${GPU_TUNED_VARIANT}"
gpu_tuned_release_marker_clear "${MARKER}"

FAILURES=0
fail() { echo "  FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }
ok() { echo "  ok:   $*"; }

if [[ ! -f "${INFO}" ]]; then
    echo "ERROR: ${INFO} not found -- run 'bash tuned/package.sh ${GPU_TUNED_VARIANT}' at the commit being released first." >&2
    exit 1
fi
info() { sed -n "s/^$1=//p" "${INFO}"; }
RELEASE_TAG="$(info release_tag)"; WHEEL_NAME="$(info wheel)"; WHEEL="${GPU_TUNED_DIST_DIR}/${WHEEL_NAME}"
PACKAGED_COMMIT="$(info commit)"; BASE_VERSION="$(info upstream_version)"
# The compared build artifact is the unstamped wheel in wheel/, not the packaged copy in dist/:
# packaging writes dist/ on every run, so its mtime would move past the test log.
BUILT_WHEEL="${GPU_TUNED_WHEEL_DIR}/${WHEEL_NAME}"
RELEASE_NOTES_FILE="tuned/releases/RELEASE_NOTES_${RELEASE_TAG#v}.md"
RELEASE_TITLE="PyTorch ${BASE_VERSION} — ${GPU_TUNED_VARIANT} ${CUDA_TAG} (${RELEASE_TAG#v}) — ${GPU_TUNED_HW_LABEL} wheel"

echo "Release ${RELEASE_TAG} (PyTorch ${BASE_VERSION}, packaged at ${PACKAGED_COMMIT}):"

gpu_tuned_check_release_tag "${RELEASE_TAG}" >/dev/null && ok "tag carries a date label" || fail "tag ${RELEASE_TAG} carries no date label"
if ! { grep -qxF "cuda_tag=${CUDA_TAG}" "${INFO}" && grep -qxF "variant=${GPU_TUNED_VARIANT}" "${INFO}"; }; then
    fail "package-info.txt is for another toolkit or variant"
fi

# Clean tree, HEAD pushed, tag free.
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    fail "tracked files differ from HEAD; commit or stash them"
else
    ok "clean tracked tree"
fi
if UPSTREAM_BRANCH="$(git rev-parse --abbrev-ref '@{upstream}' 2>/dev/null)" && git merge-base --is-ancestor HEAD "${UPSTREAM_BRANCH}" 2>/dev/null; then
    ok "HEAD is pushed (${UPSTREAM_BRANCH})"
else
    fail "HEAD is not on the branch's upstream; push first"
fi
if git ls-remote --exit-code --tags origin "refs/tags/${RELEASE_TAG}" >/dev/null 2>&1; then
    fail "tag ${RELEASE_TAG} already exists on origin; rebuild for a new label"
else
    ok "tag ${RELEASE_TAG} is free on origin"
fi

# Test gate: a full, passing log newer than the built wheel, for THIS wheel.
if [[ ! -f "${BUILT_WHEEL}" ]]; then
    fail "${BUILT_WHEEL} not found -- cannot tell whether the test results are fresh; run tuned/build.sh ${GPU_TUNED_VARIANT}"
elif [[ ! -f "${TEST_LOG}" ]]; then
    fail "${TEST_LOG} not found -- run tuned/full_test.sh ${GPU_TUNED_VARIANT}"
elif [[ "${BUILT_WHEEL}" -nt "${TEST_LOG}" ]]; then
    fail "${BUILT_WHEEL} is newer than ${TEST_LOG}: the test results predate the build; re-run tuned/full_test.sh"
elif ! grep -q '^=== Full test suite summary: .* passed ===$' "${TEST_LOG}"; then
    fail "${TEST_LOG} does not show a full, passing run"
elif [[ "$(sed -n 's/^wheel:.*sha256 //p' "${TEST_LOG}" | head -1)" != "$(sha256sum "${BUILT_WHEEL}" | awk '{print $1}')" ]]; then
    fail "${TEST_LOG} tested a different wheel than ${BUILT_WHEEL}"
else
    ok "full test run passed on this wheel, log newer than the build"
fi

# Package gate: the packaged wheel matches package-info.txt, was made after the tests, and a passing
# smoke test of THIS wheel is newer than it.
if [[ ! -f "${WHEEL}" ]]; then
    fail "${WHEEL} not found"
else
    got="$(sha256sum "${WHEEL}" | awk '{print $1}')"
    [[ "${got}" == "$(info sha256)" ]] && ok "wheel sha256 matches package-info.txt" || fail "wheel sha256 does not match package-info.txt"
    if [[ -f "${TEST_LOG}" && "${TEST_LOG}" -nt "${WHEEL}" ]]; then
        fail "the test log is newer than the packaged wheel; package again after testing"
    else
        ok "wheel was packaged after the tests"
    fi
    if [[ ! -f "${SMOKE_LOG}" ]]; then
        fail "${SMOKE_LOG} not found -- run tuned/smoke_test.sh ${GPU_TUNED_VARIANT}"
    elif [[ "${WHEEL}" -nt "${SMOKE_LOG}" ]]; then
        fail "${WHEEL} is newer than the smoke test log; re-run tuned/smoke_test.sh ${GPU_TUNED_VARIANT}"
    elif ! grep -q '^=== Smoke test summary: passed ===$' "${SMOKE_LOG}"; then
        fail "${SMOKE_LOG} does not show a pass"
    else
        ok "smoke test of this wheel passed"
    fi
fi

# Only the release notes may change between the packaged commit and HEAD.
changed="$(git diff --name-only "${PACKAGED_COMMIT}" HEAD -- . ':!tuned/releases' 2>/dev/null || echo '?')"
if [[ -z "${changed}" ]]; then
    ok "only release notes changed since the packaged commit ${PACKAGED_COMMIT:0:11}"
else
    fail "files other than tuned/releases/ changed since the packaged commit ${PACKAGED_COMMIT:0:11}: $(echo "${changed}" | head -3 | tr '\n' ' ')"
fi

# Release notes: a tracked, reviewed file, not a draft.
if [[ ! -f "${RELEASE_NOTES_FILE}" ]]; then
    fail "${RELEASE_NOTES_FILE} not found"
elif grep -q 'DRAFT — not yet published' "${RELEASE_NOTES_FILE}"; then
    fail "${RELEASE_NOTES_FILE} is still a draft (DRAFT banner present)"
elif ! git ls-files --error-unmatch "${RELEASE_NOTES_FILE}" >/dev/null 2>&1; then
    fail "${RELEASE_NOTES_FILE} is not committed"
else
    ok "release notes ${RELEASE_NOTES_FILE##*/} committed"
fi

if [[ ${FAILURES} -gt 0 ]]; then
    echo "" >&2
    echo "ERROR: ${FAILURES} gate(s) failed; nothing was published." >&2
    exit 1
fi

echo ""
gpu_tuned_release_ready "${MARKER}" "zbrad/pytorch" "${RELEASE_TAG}" "${RELEASE_TITLE}" \
    "${RELEASE_NOTES_FILE}" \
    "${WHEEL}#${WHEEL_NAME}" \
    "${WHEEL}.sha256#${WHEEL_NAME}.sha256" \
    "${TEST_LOG}#Test results (${GPU_TUNED_VARIANT})" \
    "${SMOKE_LOG}#Package smoke test results (${GPU_TUNED_VARIANT})" || exit 1

echo "Done."
