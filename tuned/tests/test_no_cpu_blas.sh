#!/bin/bash
# Tests for pytorch_verify_no_cpu_blas (tuned/no_cpu_blas.sh) on throwaway wheel trees with real ELF files.
# Usage: bash tuned/tests/test_no_cpu_blas.sh   (exit 0 = all pass)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../no_cpu_blas.sh
source "${HERE}/../no_cpu_blas.sh"
command -v gcc >/dev/null && command -v readelf >/dev/null || { echo "gcc and readelf are needed for these tests" >&2; exit 1; }
PASS=0; FAIL=0
W="$(mktemp -d)"; trap 'rm -rf "${W}"' EXIT

check() { # check <name> <want-rc> <stderr-regex or -> <dir>
    local name="$1" want="$2" err_re="$3" dir="$4" err rc ok=1
    err="$(pytorch_verify_no_cpu_blas "${dir}" 2>&1 >/dev/null)"; rc=$?
    [[ "${rc}" -eq "${want}" ]] || ok=0
    if [[ "${err_re}" == "-" ]]; then [[ -z "${err}" ]] || ok=0; else grep -qE "${err_re}" <<<"${err}" || ok=0; fi
    if [[ "${ok}" -eq 1 ]]; then PASS=$((PASS+1)); echo "ok   - ${name}"
    else FAIL=$((FAIL+1)); echo "FAIL - ${name} (rc=${rc}, want ${want})"; sed 's/^/     stderr: /' <<<"${err}"; fi
}

# tree <name> <blas> <lapack> [needed-soname]: a fake unpacked wheel. A needed soname is made real: a stub
# library with that SONAME, and torch/lib/libuser.so linked against it so readelf lists it as NEEDED.
tree() {
    local d="${W}/$1" blas="$2" lapack="$3" need="${4:-}"
    mkdir -p "${d}/torch/include/ATen" "${d}/torch/lib"
    printf '#define AT_BUILD_WITH_BLAS() %s\n#define AT_BUILD_WITH_LAPACK() %s\n' "${blas}" "${lapack}" > "${d}/torch/include/ATen/Config.h"
    echo 'int plain(void){return 1;}' > "${W}/plain.c"
    gcc -shared -fPIC -o "${d}/torch/lib/libtorch_cpu-tuned.so" "${W}/plain.c"
    if [[ -n "${need}" ]]; then
        echo 'int stub(void){return 2;}' > "${W}/stub.c"
        mkdir -p "${W}/stubs-$1"
        gcc -shared -fPIC -Wl,-soname,"${need}" -o "${W}/stubs-$1/${need}" "${W}/stub.c"
        echo 'extern int stub(void); int user(void){return stub();}' > "${W}/user.c"
        gcc -shared -fPIC -o "${d}/torch/lib/libuser.so" "${W}/user.c" -Wl,--no-as-needed -L"${W}/stubs-$1" -l:"${need}"
    fi
}

tree clean 0 0
tree blas_on 1 0
tree lapack_on 0 1
tree openblas 0 0 libopenblas.so.0
tree mpi 0 0 libmpi.so.40
tree gfortran 0 0 libgfortran.so.5
tree harmless 0 0 libm.so.6                 # a normal system library is not flagged
mkdir -p "${W}/noconfig/torch/lib"; cp "${W}/clean/torch/lib/libtorch_cpu-tuned.so" "${W}/noconfig/torch/lib/"
tree undefined 0 0; printf '#define SOMETHING_ELSE 1\n' > "${W}/undefined/torch/include/ATen/Config.h"

check "clean tree passes, nothing printed"                     0 "-"                                         "${W}/clean"
check "AT_BUILD_WITH_BLAS() 1 is a violation"                  1 "VIOLATION: AT_BUILD_WITH_BLAS\\(\\) is not 0" "${W}/blas_on"
check "AT_BUILD_WITH_LAPACK() 1 is a violation"                1 "VIOLATION: AT_BUILD_WITH_LAPACK\\(\\) is not 0" "${W}/lapack_on"
check "a flag missing from Config.h is a violation"            1 "not defined"                               "${W}/undefined"
check "NEEDED libopenblas.so.0 is a violation, library named"  1 "libuser\\.so needs.*libopenblas\\.so\\.0"  "${W}/openblas"
check "NEEDED libmpi.so.40 is a violation"                     1 "needs.*libmpi\\.so\\.40"                    "${W}/mpi"
check "NEEDED libgfortran.so.5 is a violation"                 1 "needs.*libgfortran\\.so\\.5"                "${W}/gfortran"
check "an ordinary system NEEDED (libm) is not flagged"        0 "-"                                         "${W}/harmless"
check "a wheel without ATen/Config.h fails: a check that cannot run is a failure" 1 "ATen/Config\\.h is not in the wheel" "${W}/noconfig"

echo; echo "passed=${PASS} failed=${FAIL}"
[[ "${FAIL}" -eq 0 ]]
