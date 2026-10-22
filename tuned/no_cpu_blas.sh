#!/bin/bash
# Sourced by tuned/package.sh. The tuned wheel is GPU-only: no CPU BLAS, LAPACK or MPI.
#
# gpu_tuned_verify_no_cpu_blas <unpacked-wheel-dir>
#   1. ATen/Config.h, which the wheel ships, says AT_BUILD_WITH_BLAS() and AT_BUILD_WITH_LAPACK() are 0.
#      That is the build configuration as compiled, not what the environment of the build node suggested.
#   2. No shared object in the wheel lists a CPU math or MPI library as NEEDED. The unique-soname gate
#      already rejects any NEEDED outside its assumed list; this one names the reason, so an entry added
#      to that list later cannot let one of these back in silently.
# Returns 0 if both hold, 1 otherwise (every violation is printed first). A check that cannot run is a
# failure (H4): a missing Config.h or readelf fails it.
GPU_TUNED_CPU_MATH_RE='^lib(openblas|blas|cblas|lapack|lapacke|flexiblas|blis|atlas|gfortran|mpi|mpi_cxx|mkl_[a-z_0-9]+|armpl[a-z_0-9]*|nvpl_[a-z_0-9]+)\.so(\.[0-9.]+)*$'

gpu_tuned_verify_no_cpu_blas() {
    local dir="${1:?usage: gpu_tuned_verify_no_cpu_blas <unpacked-wheel-dir>}" cfg flag so needed bad=0
    command -v readelf >/dev/null || { echo "ERROR: readelf is needed for the no-CPU-BLAS gate." >&2; return 1; }
    cfg="$(find "${dir}" -path '*/include/ATen/Config.h' -type f | head -1)"
    [[ -n "${cfg}" ]] || { echo "ERROR: ATen/Config.h is not in the wheel, so the BLAS/LAPACK build flags cannot be checked." >&2; return 1; }
    for flag in AT_BUILD_WITH_BLAS AT_BUILD_WITH_LAPACK; do
        if ! grep -qE "^#define ${flag}\(\) 0[[:space:]]*\$" "${cfg}"; then
            echo "VIOLATION: ${flag}() is not 0 in ${cfg#"${dir}"/} ($(grep -E "^#define ${flag}\(\)" "${cfg}" || echo 'not defined')); the wheel was built with a CPU BLAS/LAPACK." >&2
            bad=1
        fi
    done
    while IFS= read -r so; do
        needed="$(readelf -d "${so}" 2>/dev/null | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | grep -E "${GPU_TUNED_CPU_MATH_RE}" || true)"
        if [[ -n "${needed}" ]]; then
            echo "${needed}" | sed "s#^#VIOLATION: $(basename "${so}") needs the CPU math/MPI library #" >&2
            bad=1
        fi
    done < <(find "${dir}" -type f -name '*.so*')
    [[ "${bad}" -eq 0 ]]
}
