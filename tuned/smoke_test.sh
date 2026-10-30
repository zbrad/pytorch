#!/bin/bash
# tuned/smoke_test.sh <variant> — prove the PACKAGED wheel works, before it is released. It installs the
# stamped wheel tuned/package.sh made (named in package-info.txt) into a clean venv and checks:
#   1. the wheel's sha256 matches package-info.txt and its .sha256 sidecar;
#   2. gpu_tuned_smoke_wheel: the label parses, exactly one torch of exactly this version is
#      installed and imported from inside the venv, the build-info stamp in libtorch_cuda-tuned.so names this
#      version, CUDA works and the device capability is the variant's;
#   3. the libtorch_cuda-tuned.so the process maps is the one inside the venv (proved from /proc/self/maps);
#   4. three real tests from this repo pass against it: test_cuda.py test_memory_stats and
#      test_cublas_multiple_threads_same_device, and test_linalg.py test_matmul_small_brute_force_2d_Nd
#      (the NCCL all-reduce and broadcast run in tuned/nccl_check.sh, part of full_test.sh).
# A smoke test, not the suite: tuned/full_test.sh already ran the scope against the unstamped wheel.
# What is new here is that the bits in the package are the ones running. Writes
# SMOKE_RESULTS_<variant>.log beside TEST_RESULTS; tuned/release.sh requires a passing log newer
# than the packaged wheel. A pass needs the end marker, so an abort cannot read as a pass.
export PATH="${CONDA_PREFIX:+$CONDA_PREFIX/bin:}$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/smoke_test.sh <variant>}"
REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPODIR}" || exit 1
# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

INFO="${GPU_TUNED_DIST_DIR}/package-info.txt"
[[ -f "${INFO}" ]] || { echo "ERROR: ${INFO} not found. Run tuned/package.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
info() { sed -n "s/^$1=//p" "${INFO}"; }
WHEEL="${GPU_TUNED_DIST_DIR}/$(info wheel)"; WANT_SHA="$(info sha256)"; VERSION="$(info version)"
[[ -f "${WHEEL}" ]] || { echo "ERROR: ${WHEEL} not found." >&2; exit 1; }
mkdir -p "${GPU_TUNED_RELEASES_DIR}"
LOG="${GPU_TUNED_RELEASES_DIR}/SMOKE_RESULTS_${GPU_TUNED_VARIANT}.log"
rm -f "${LOG}"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
FAILS=0
fail() { echo "  FAIL: $*"; FAILS=$((FAILS + 1)); }
ok() { echo "  ok:   $*"; }

{
    echo "pytorch tuned smoke test -- variant=${GPU_TUNED_VARIANT} cuda=${CUDA_VER} (${CUDA_TAG})"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wheel:   $(basename "${WHEEL}")"
    echo "packaged commit: $(info commit)"
    echo ""

    got="$(sha256sum "${WHEEL}" | awk '{print $1}')"
    [[ "${got}" == "${WANT_SHA}" ]] && ok "wheel sha256 matches package-info.txt" || fail "wheel sha256 ${got} does not match package-info.txt (${WANT_SHA})"
    (cd "${GPU_TUNED_DIST_DIR}" && sha256sum -c "$(basename "${WHEEL}").sha256" >/dev/null 2>&1) && ok "sidecar .sha256 verifies" || fail "sidecar .sha256 does not verify"

    # Install into a clean venv and load it once. The venv is kept so the repo's tests run in it.
    export LD_LIBRARY_PATH="${GPU_TUNED_TOOLKIT_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    SMOKE_OUT="$(GPU_TUNED_SMOKE_KEEP=1 GPU_TUNED_SMOKE_EXPECT_CAP="${GPU_TUNED_TORCH_ARCH%a}" \
        gpu_tuned_smoke_wheel "${WHEEL}" torch torch tuned_build_info libtorch_cuda-tuned.so 2>&1)"
    SMOKE_RC=$?
    printf '%s\n' "${SMOKE_OUT}" | sed 's/^/  | /'
    VENV="$(printf '%s\n' "${SMOKE_OUT}" | sed -n 's/^venv kept: //p' | tail -1)"
    if [[ ${SMOKE_RC} -ne 0 || -z "${VENV}" || ! -x "${VENV}/bin/python" ]]; then
        fail "gpu_tuned_smoke_wheel did not pass"
    else
        ok "wheel installs and loads in a clean venv (label, version, stamp, CUDA, device capability)"
        PY="${VENV}/bin/python"
        env -u PYTHONPATH "${PY}" -m pip install --quiet -r tuned/requirements-test.txt || fail "test tooling did not install"
        # The libtorch_cuda-tuned.so a process maps, from an empty directory, must be inside the venv.
        LIB="$(cd "${WORK}" && env -u PYTHONPATH "${PY}" -I -c 'import torch
print(next(l.split()[-1] for l in open("/proc/self/maps") if "libtorch_cuda-tuned.so" in l))' 2>&1 | tail -1)"
        [[ "${LIB}" == "${VENV}/"* ]] && ok "libtorch_cuda-tuned.so is mapped from the clean venv: ${LIB}" || fail "libtorch_cuda-tuned.so maps from '${LIB}', not from ${VENV}"
        # Our libraries keep torch's stock names, so prove they cannot be mistaken for stock ones in this
        # process: every libtorch*/libc10* mapped is the venv's, and no library name is mapped from two
        # different files (a second libc10.so, say, from LD_LIBRARY_PATH or another torch).
        NAMES="$(cd "${WORK}" && env -u PYTHONPATH "${PY}" -I -c '
import collections, os, sys
import torch
torch.zeros(1, device="cuda")
by = collections.defaultdict(set)
for line in open("/proc/self/maps"):
    p = line.split()[-1] if line.split() else ""
    if ".so" in p and p.startswith("/"):
        by[os.path.basename(p).split(".so")[0]].add(os.path.realpath(p))
bad = [f"{k}: {sorted(v)}" for k, v in by.items() if len(v) > 1]
bad += [f"{k} from {sorted(v)}" for k, v in by.items() if k.startswith(("libtorch", "libc10")) and not all(x.startswith(sys.prefix) for x in v)]
print("\n".join(bad) if bad else "clean")
' 2>&1 | tail -5)"
        [[ "${NAMES}" == clean ]] && ok "every libtorch*/libc10* is the venv's and no library name maps from two files" || fail "library identity: ${NAMES}"

        # Three real tests, run the way test/run_test.py runs them (python -bb <file> -k <name>) from
        # test/, so sys.path[0] is test/ and torch is the venv's, never the source tree's.
        run_one() {
            local file="$1" name="$2" out
            out="$(cd "${REPODIR}/test" && env -u PYTHONPATH "${PY}" -bb "${file}" -k "${name}" 2>&1)"
            printf '%s\n' "${out}" | tail -8 | sed 's/^/  | /'
            local ran skipped
            ran="$(printf '%s\n' "${out}" | sed -n 's/^Ran \([0-9]*\) tests\?.*/\1/p' | tail -1)"
            skipped="$(printf '%s\n' "${out}" | sed -n 's/^OK (skipped=\([0-9]*\))$/\1/p' | tail -1)"
            # A pass needs the run to end OK and at least one test to have really executed.
            if printf '%s\n' "${out}" | grep -qE '^OK( \(skipped=[0-9]+\))?$' && [[ "${ran:-0}" -gt "${skipped:-0}" ]]; then
                ok "${file} -k ${name}: ran ${ran}, skipped ${skipped:-0}"
            else
                fail "${file} -k ${name} (failed, or no test really ran: ran ${ran:-0}, skipped ${skipped:-0})"
            fi
        }
        run_one test_cuda.py test_memory_stats
        run_one test_cuda.py test_cublas_multiple_threads_same_device
        run_one test_linalg.py test_matmul_small_brute_force_2d_Nd
    fi
    [[ -n "${VENV:-}" && "${VENV}" == /tmp/*/venv ]] && rm -rf "$(dirname "${VENV}")"

    echo ""
    if [[ ${FAILS} -eq 0 ]]; then
        echo "=== Smoke test summary: passed ==="
    else
        echo "=== Smoke test summary: FAILED (${FAILS}) ==="
    fi
    echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${LOG}"
grep -q '^=== Smoke test summary: passed ===$' "${LOG}"
