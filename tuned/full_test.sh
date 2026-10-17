#!/bin/bash
# tuned/full_test.sh <variant> — test the wheel tuned/build.sh made, for ONE variant and ONE CUDA
# toolkit, and write a results log in that toolkit's tree.
#
#   CUDA_TAG=cu134 bash tuned/full_test.sh gb10
#
# The wheel (tuned/build/<cuda-tag>/wheel/torch-*.whl, unstamped) is installed into a throwaway venv
# with the test tooling from tuned/requirements-test.txt, and tested there, so what is tested is the
# artifact package.sh stamps and ships, not a source tree. Scope: tuned/verify_abi.sh,
# tuned/nccl_check.sh (when NCCL is a pinned prebuilt) and the modules in tuned/test_modules.txt
# (test/run_test.py --keep-going -i ...). That list is the recorded decision for
# docs/ARCHITECTURE.md G3, not upstream's whole suite.
#
# The log is tuned/build/<cuda-tag>/releases/TEST_RESULTS_<variant>.log and ends with a
# "Full test suite summary" line; tuned/release.sh refuses to publish unless it shows a pass and is
# newer than the wheel. GPU_TUNED_TEST_MODULES="test_cuda test_nn" runs a custom list instead: its
# log ends with "Partial test summary" and can never pass that gate.
export PATH="${CONDA_PREFIX:+$CONDA_PREFIX/bin:}$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/full_test.sh <variant>}"
REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPODIR}" || exit 1

# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/torch-*.whl 2>/dev/null | head -1)"
[[ -n "${WHEEL}" ]] || { echo "ERROR: no wheel in ${GPU_TUNED_WHEEL_DIR}. Run tuned/build.sh ${GPU_TUNED_VARIANT} for ${CUDA_TAG} first." >&2; exit 1; }
BUILD_COMMIT="$(cat "${GPU_TUNED_BUILD_ROOT}/build-commit" 2>/dev/null || true)"
LABEL="$(cat "${GPU_TUNED_BUILD_ROOT}/release-label" 2>/dev/null || true)"
[[ -n "${BUILD_COMMIT}" && -n "${LABEL}" ]] || { echo "ERROR: ${GPU_TUNED_BUILD_ROOT}/build-commit or release-label missing: the last build did not finish. Re-run tuned/build.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }
[[ "$(basename "${WHEEL}")" == torch-*"+${GPU_TUNED_VARIANT}.${CUDA_TAG}.${LABEL}-"* ]] \
    || { echo "ERROR: $(basename "${WHEEL}") does not carry label ${GPU_TUNED_VARIANT}.${CUDA_TAG}.${LABEL} from release-label." >&2; exit 1; }
[[ "${BUILD_COMMIT}" != *-dirty ]] || { echo "ERROR: the build was made from a dirty tree (${BUILD_COMMIT}); commit and rebuild." >&2; exit 1; }
git cat-file -e "${BUILD_COMMIT}^{commit}" 2>/dev/null || { echo "ERROR: build commit ${BUILD_COMMIT} is not in this checkout." >&2; exit 1; }
# H6: the wheel must be from the source at HEAD; only tuned/ may differ.
STALE_SRC="$(git diff --name-only "${BUILD_COMMIT}" HEAD -- . ':!tuned' | head -5)"
if [[ -n "${STALE_SRC}" ]]; then
    echo "ERROR: the wheel is from ${BUILD_COMMIT}, but source outside tuned/ changed since:" >&2
    echo "${STALE_SRC}" | sed 's/^/    /' >&2
    echo "  Rebuild (tuned/build.sh ${GPU_TUNED_VARIANT}) before testing." >&2
    exit 1
fi

# Tests not needed for tuned builds (tuned/test_skips.txt): whole modules and test-name substrings, each with
# its reason. They are listed in the log header, never dropped silently.
SKIP_MODULES=(); SKIP_TESTS=(); SKIP_LOG=()
while IFS= read -r line; do
    line="${line%%$'\r'}"
    [[ "${line}" =~ ^[[:space:]]*(module|test)[[:space:]]+([^[:space:]]+)[[:space:]]*(.*)$ ]] || continue
    kind="${BASH_REMATCH[1]}"; name="${BASH_REMATCH[2]}"; reason="${BASH_REMATCH[3]}"
    [[ -n "${reason}" ]] || { echo "ERROR: tuned/test_skips.txt: '${kind} ${name}' has no reason." >&2; exit 1; }
    [[ "${kind}" == module ]] && SKIP_MODULES+=("${name}") || SKIP_TESTS+=("${name}")
    SKIP_LOG+=("${kind} ${name}: ${reason}")
done < <(grep -v '^[[:space:]]*#' tuned/test_skips.txt)
SKIP_EXPR=""
if [[ ${#SKIP_TESTS[@]} -gt 0 ]]; then
    SKIP_EXPR="$(printf '%s or ' "${SKIP_TESTS[@]}")"; SKIP_EXPR="not (${SKIP_EXPR% or })"
fi

MINI=0
if [[ -n "${GPU_TUNED_TEST_MODULES:-}" ]]; then
    read -r -a MODULES <<< "${GPU_TUNED_TEST_MODULES}"
    SCOPE="custom (GPU_TUNED_TEST_MODULES): not a release scope"
    SUMMARY_WORD="Partial"
elif [[ "${GPU_TUNED_TEST_SCOPE:-release}" == mini ]]; then
    MINI=1
    MODULES=()
    SCOPE="mini: a few tests per area from tuned/test_mini.txt + verify_abi.sh${GPU_TUNED_NCCL_DIR:+ + nccl_check.sh}; not a release scope"
    SUMMARY_WORD="Mini"
elif [[ -n "${GPU_TUNED_TEST_SCOPE:-}" && "${GPU_TUNED_TEST_SCOPE}" != release ]]; then
    echo "ERROR: GPU_TUNED_TEST_SCOPE='${GPU_TUNED_TEST_SCOPE}' must be release or mini." >&2; exit 1
else
    mapfile -t MODULES < <(sed -e 's/#.*//' -e 's/[[:space:]]*$//' tuned/test_modules.txt | grep -v '^$')
    SCOPE="tuned/test_modules.txt (${#MODULES[@]} modules) minus tuned/test_skips.txt + verify_abi.sh${GPU_TUNED_NCCL_DIR:+ + nccl_check.sh}"
    SUMMARY_WORD="Full"
fi
if [[ ${#SKIP_MODULES[@]} -gt 0 && ${#MODULES[@]} -gt 0 ]]; then
    KEPT=()
    for m in "${MODULES[@]}"; do
        skip=0; for sm in "${SKIP_MODULES[@]}"; do [[ "${m}" == "${sm}" ]] && skip=1; done
        [[ ${skip} -eq 1 ]] || KEPT+=("${m}")
    done
    MODULES=("${KEPT[@]}")
fi
[[ ${MINI} -eq 1 || ${#MODULES[@]} -gt 0 ]] || { echo "ERROR: no test modules selected." >&2; exit 1; }

mkdir -p "${GPU_TUNED_RELEASES_DIR}"
RESULTS_FILE="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
rm -f "${RESULTS_FILE}"

# The wheel and the test tooling in a clean venv: nothing from the build venv, the repo or the user
# site is on the path (G2).
TESTENV="$(mktemp -d)"
trap 'rm -rf "${TESTENV}"' EXIT
PY="${TESTENV}/venv/bin/python"
python3 -m venv "${TESTENV}/venv" || exit 1
echo "installing $(basename "${WHEEL}") and tuned/requirements-test.txt into a clean venv"
env -u PYTHONPATH -u PYTHONHOME "${PY}" -m pip install --quiet "${WHEEL}" -r tuned/requirements-test.txt \
    || { echo "ERROR: pip could not install the wheel and test tooling." >&2; exit 1; }

# The toolkit's own lib dir first, so the host's ld.so.conf (which follows the /usr/local/cuda
# symlink) cannot satisfy cudart/cublas from another toolkit.
export LD_LIBRARY_PATH="${GPU_TUNED_TOOLKIT_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# The libtorch_cuda-tuned.so a process REALLY loads, from /proc/self/maps, run from an empty directory
# (from the repo root, `import torch` would find the source tree's torch/ first).
cd "${TESTENV}" || exit 1
LIB="$(env -u PYTHONPATH "${PY}" -I -c 'import torch
print(next(l.split()[-1] for l in open("/proc/self/maps") if "libtorch_cuda-tuned.so" in l))')" || { echo "ERROR: import torch failed in the test venv." >&2; exit 1; }
cd "${REPODIR}" || exit 1
[[ "${LIB}" == "${TESTENV}/venv/"* ]] || { echo "ERROR: libtorch_cuda-tuned.so is loaded from ${LIB}, not from the test venv." >&2; exit 1; }
gpu_tuned_verify_toolkit_libs "${LIB}" || { echo "ERROR: libtorch_cuda-tuned.so does not resolve the ${CUDA_TAG} runtime libraries." >&2; exit 1; }
gpu_tuned_verify_arch "${LIB}" "${GPU_TUNED_TORCH_ARCH}" || exit 1

{
    echo "pytorch tuned test run -- variant=${GPU_TUNED_VARIANT} cuda=${CUDA_VER} (${CUDA_TAG})"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit:  $(git rev-parse HEAD)"
    echo "build:   ${BUILD_COMMIT} (label ${LABEL})"
    echo "wheel:   $(basename "${WHEEL}") sha256 $(sha256sum "${WHEEL}" | awk '{print $1}')"
    echo "torch:   $(cd "${TESTENV}" && "${PY}" -I -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))' 2>&1 | tail -1)"
    echo "nccl:    ${GPU_TUNED_NCCL_URL:-in-tree}"
    echo "tools:   $("${PY}" -m pip freeze 2>/dev/null | grep -iE '^(pytest|expecttest|hypothesis|numpy|setuptools|ninja)[=@ ]' | tr '\n' ' ')"
    echo "scope:   ${SCOPE}"
    echo "modules: ${MODULES[*]:-<mini: see tuned/test_mini.txt>}"
    for l in "${SKIP_LOG[@]}"; do echo "not run (tuned/test_skips.txt): ${l}"; done
    echo "skips by design: multi-GPU NCCL/distributed tests skip on a single-GPU host (counted in the output below); everything outside the module list above is not run."
    echo "gates: wheel installed in a clean venv; libtorch_cuda-tuned.so loaded from it; toolkit libs resolve under ${CUDA_HOME}; single-arch ${GPU_TUNED_TORCH_ARCH}; build commit ${BUILD_COMMIT} == HEAD outside tuned/"
    echo ""
} > "${RESULTS_FILE}"

STATUS=0
echo "=== tuned/verify_abi.sh ===" | tee -a "${RESULTS_FILE}"
bash tuned/verify_abi.sh "${GPU_TUNED_VARIANT}" "${TESTENV}/venv" 2>&1 | tee -a "${RESULTS_FILE}"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { STATUS=1; echo "verify_abi.sh FAILED" | tee -a "${RESULTS_FILE}"; }

if [[ -n "${GPU_TUNED_NCCL_DIR:-}" ]]; then
    echo "" | tee -a "${RESULTS_FILE}"
    echo "=== tuned/nccl_check.sh ===" | tee -a "${RESULTS_FILE}"
    bash tuned/nccl_check.sh "${GPU_TUNED_VARIANT}" "${TESTENV}/venv" 2>&1 | tee -a "${RESULTS_FILE}"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { STATUS=1; echo "nccl_check.sh FAILED" | tee -a "${RESULTS_FILE}"; }
fi

# run_test.py puts the repo root first on sys.path while it imports upstream's test tooling, which
# would load the source tree's torch/ (no built libraries). Import the wheel's torch first, from the
# test/ directory, then run the script in this interpreter.
echo "" | tee -a "${RESULTS_FILE}"
if [[ ${MINI} -eq 1 ]]; then
    # Mini tier: for each line of tuned/test_mini.txt, run the named tests with pytest through the test file's
    # own entry point, from test/ (so sys.path[0] is test/ and torch is the wheel's). A line passes when it
    # ends with passes only and at least one test really ran.
    echo "=== mini tier (tuned/test_mini.txt) ===" | tee -a "${RESULTS_FILE}"
    while IFS= read -r line; do
        line="${line%%#*}"; line="${line%"${line##*[![:space:]]}"}"
        [[ -n "${line}" ]] || continue
        file="${line%% *}"; expr="$(sed -e 's/^[^[:space:]]*[[:space:]]*//' <<< "${line}")"
        [[ "${expr}" == "-" ]] && expr=""
        kexpr="${expr:+(${expr})}"
        if [[ -n "${SKIP_EXPR}" ]]; then kexpr="${kexpr:+${kexpr} and }${SKIP_EXPR}"; fi
        out="$(cd "${REPODIR}/test" && env -u PYTHONPATH "${PY}" -bb "${file}" --use-pytest ${kexpr:+-k "${kexpr}"} </dev/null 2>&1)"; rc=$?
        summary="$(grep -E '^=+ .*(passed|failed|skipped|error|no tests ran).* in [0-9.]+s' <<< "${out}" | tail -1)"
        passed="$(sed -n 's/.*[ =]\([0-9][0-9]*\) passed.*/\1/p' <<< "${summary}")"
        if [[ ${rc} -eq 0 && "${passed:-0}" -gt 0 ]]; then
            echo "  ok:   ${file}${expr:+ -k ${expr}}: ${summary//=/}" | tee -a "${RESULTS_FILE}"
        else
            echo "  FAIL: ${file}${expr:+ -k ${expr}} (exit ${rc}; ${summary:-no pytest summary})" | tee -a "${RESULTS_FILE}"
            tail -15 <<< "${out}" | sed 's/^/        /' | tee -a "${RESULTS_FILE}"
            STATUS=1
        fi
    done < tuned/test_mini.txt
else
    # run_test.py puts the repo root first on sys.path while it imports upstream's test tooling, which
    # would load the source tree's torch/ (no built libraries). Import the wheel's torch first, from the
    # test/ directory, then run the script in this interpreter.
    echo "=== test/run_test.py --keep-going${SKIP_EXPR:+ --pytest-k-expr '${SKIP_EXPR}'} -i ${MODULES[*]} ===" | tee -a "${RESULTS_FILE}"
    (cd "${REPODIR}/test" && env -u PYTHONPATH "${PY}" -I -c '
import runpy, sys
import torch
assert torch.__file__.startswith(sys.prefix), torch.__file__
k = sys.argv[1]
sys.argv = ["run_test.py", "--keep-going"] + (["--pytest-k-expr", k] if k else []) + ["-i", *sys.argv[2:]]
runpy.run_path("run_test.py", run_name="__main__")
' "${SKIP_EXPR}" "${MODULES[@]}") 2>&1 | tee -a "${RESULTS_FILE}"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || STATUS=1
fi

echo "" >> "${RESULTS_FILE}"
if [[ "${STATUS}" -eq 0 ]]; then
    echo "=== ${SUMMARY_WORD} test suite summary: tests, verify_abi and nccl_check passed ===" | tee -a "${RESULTS_FILE}"
else
    echo "=== ${SUMMARY_WORD} test suite summary: FAILED (see above) ===" | tee -a "${RESULTS_FILE}"
fi
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RESULTS_FILE}"
echo "Results written to ${RESULTS_FILE}"
exit "${STATUS}"
