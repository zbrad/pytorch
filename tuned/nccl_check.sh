#!/bin/bash
# tuned/nccl_check.sh <variant> <venv-dir> — after tuned/build.sh with GPU_TUNED_NCCL_DIR set, check in
# a venv with the built wheel installed (tuned/full_test.sh passes its test venv) that torch really uses the prebuilt NCCL: its version is the one in the NCCL
# tarball's BUILD_INFO, libtorch_cuda has no dynamic libnccl dependency (the archive is linked in),
# libtorch_cuda is still single-arch, and an NCCL all-reduce runs correctly on the GPU.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/nccl_check.sh <variant> <venv-dir>}"
VENV_DIR="${2:?usage: tuned/nccl_check.sh <variant> <venv-dir>}"
# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1
[[ -n "${GPU_TUNED_NCCL_DIR:-}" ]] || { echo "ERROR: GPU_TUNED_NCCL_DIR is not set; nothing to check." >&2; exit 1; }
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

BUILD_INFO="${GPU_TUNED_NCCL_DIR}/BUILD_INFO"
WANT="$(sed -n 's/^nccl_version=//p' "${BUILD_INFO}" 2>/dev/null | head -1)"
[[ -n "${WANT}" ]] || { echo "ERROR: ${BUILD_INFO} has no nccl_version (is GPU_TUNED_NCCL_DIR the extracted tarball root?)" >&2; exit 1; }
# Find the libtorch_cuda-tuned.so the process REALLY loads, from /proc/self/maps.
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
cd "${WORK}"
LIB="$(python -I -c 'import torch
print(next(l.split()[-1] for l in open("/proc/self/maps") if "libtorch_cuda-tuned.so" in l))')"
case "${LIB}" in "${VENV_DIR}"/*) ;; *) echo "FAIL: libtorch_cuda-tuned.so is loaded from ${LIB}, not from the venv ${VENV_DIR}" >&2; exit 1 ;; esac
echo "ok:   torch loads ${LIB}"

fail() { echo "FAIL: $*" >&2; exit 1; }
if readelf -d "${LIB}" | grep -qi 'NEEDED.*libnccl'; then fail "libtorch_cuda-tuned.so still depends on a shared libnccl"; fi
echo "ok:   no dynamic libnccl dependency"
gpu_tuned_verify_arch "${LIB}" "${GPU_TUNED_TORCH_ARCH}" >/dev/null || fail "libtorch_cuda-tuned.so is not single-arch sm_${GPU_TUNED_TORCH_ARCH}"
echo "ok:   libtorch_cuda-tuned.so single-arch sm_${GPU_TUNED_TORCH_ARCH}"

python -I - "${WANT}" <<'PY'
import sys
import torch
import torch.distributed as dist

want = tuple(int(x) for x in sys.argv[1].split("."))
got = tuple(torch.cuda.nccl.version())[:3]
assert got == want, f"torch reports NCCL {got}, tarball is {want}"
print(f"ok:   torch.cuda.nccl.version() == {'.'.join(map(str, want))}")
dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29571", rank=0, world_size=1)
x = torch.arange(1024, dtype=torch.float32, device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
assert torch.equal(x.cpu(), torch.arange(1024, dtype=torch.float32)), "all_reduce changed the data"
y = torch.ones(8, device="cuda")
dist.broadcast(y, 0)
dist.destroy_process_group()
print("ok:   NCCL all_reduce and broadcast ran on the GPU")
PY
echo "NCCL check passed."
