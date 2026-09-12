#!/bin/bash
# tuned/verify_abi.sh <variant> — confirms an out-of-tree CUDA extension
# built against THIS torch (via torch.utils.cpp_extension, the exact
# mechanism flash-attn/vllm's own custom ops use) actually loads and runs.
#
# A torch rebuild can succeed and pass its own smoke test (import torch,
# check CUDA is compiled in) while still being ABI-incompatible with
# every out-of-tree extension that links against libtorch -- hit for real
# this session: flash-attention-vllm's prebuilt .so failed with
# `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb`
# (c10::cuda::c10_cuda_check_implementation) after a torch rebuild, only
# discovered by bypassing flash_attn_interface.py's swallowing try/except.
# That symbol backs C10_CUDA_KERNEL_LAUNCH_CHECK(), which virtually every
# real CUDA extension (flash-attn included) calls after every kernel
# launch -- so a minimal extension exercising the same macro, compiled
# fresh against this torch build, catches the same failure class without
# needing flash-attn/vllm's source at all. Independent of and much
# cheaper than the full downstream rebuild-and-retest cycle; it doesn't
# replace that cycle (it can't catch a mismatch specific to flash-attn's
# own code), just catches the core-ABI class of it early, standalone, in
# this repo.
#
# Requires tuned/build.sh <variant> to have already succeeded (reuses
# that venv, does not build torch).
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
gpu_tuned_verify_venv "${VENV_DIR}" "${REPO_ROOT}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "=========================================="
echo "Verifying out-of-tree CUDA extension ABI (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo ""

python3 - <<'PYEOF'
import torch
from torch.utils.cpp_extension import load_inline

cpp_source = "torch::Tensor gpu_tuned_abi_check_add_one(torch::Tensor input);"

cuda_source = r"""
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>

__global__ void gpu_tuned_abi_check_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

torch::Tensor gpu_tuned_abi_check_add_one(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    auto output = input.clone();
    int n = static_cast<int>(output.numel());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gpu_tuned_abi_check_kernel<<<blocks, threads>>>(output.data_ptr<float>(), n);
    // The exact symbol (c10::cuda::c10_cuda_check_implementation) that
    // came back undefined in the flash-attention-vllm incident this
    // check exists to catch -- see this file's header.
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

print("Compiling a minimal out-of-tree CUDA extension against this torch build...")
module = load_inline(
    name="gpu_tuned_abi_check",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["gpu_tuned_abi_check_add_one"],
    verbose=True,
)

x = torch.zeros(1024, device="cuda")
y = module.gpu_tuned_abi_check_add_one(x)
torch.cuda.synchronize()
assert torch.equal(y, torch.ones(1024, device="cuda")), (
    "extension compiled and loaded, but produced wrong output"
)
print(
    "OK: out-of-tree CUDA extension compiled, loaded, and ran correctly "
    "against this torch build (C10 ABI check passed)."
)
PYEOF
