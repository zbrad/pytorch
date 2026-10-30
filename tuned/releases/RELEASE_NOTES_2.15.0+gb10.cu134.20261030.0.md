# PyTorch 2.15.0 — gb10 cu134 (2.15.0+gb10.cu134.20261030.0)

Single-arch wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261030.0 (UTC date of the tuned commit)
- Built from commit 83c6a7cd8fa1fbcf711559df8c66b427fb6c820b; tested at 83c6a7cd8fa1fbcf711559df8c66b427fb6c820b
- Upstream base: upstream/main merge-base 5c6918c45adefb8eaf40e7f665ccd00f9879fbef
- NCCL: 2.30.7 (v2.30.7-1-11-gbef8ea32), statically linked, from https://github.com/zbrad/nccl/releases/tag/v0.20261012.0-gb10-cu134
- Wheel sha256: b1e534d2506304ee1b65588b95ad809a79fe06c18129dd14f79faf9ac6d88fdb

## Changes from 20261027

The same torch source: nothing outside `tuned/` changed since 20261027.0's build commit (b6053b6f191). It stays a
GPU-only build: no CPU BLAS, no CPU LAPACK, no MPI. What changed is the build-info stamp:

- `libtorch_cuda-tuned.so` carries one line of JSON build info in an ELF section `.tuned_build_info`
  (`gpu_tuned_read_build_info <so> | jq`), replacing the free-text `.pytorch_build_info` section. Its fields include the
  release, variant, arch, CUDA tag and nvcc version, `fork.commit` (now the commit the wheel was built from, not HEAD at
  packaging) and `deps.nccl` (version, zbrad/nccl release, source, link type).
- Anything that reads the old `.pytorch_build_info` section (flashinfer's and vllm's `tuned/build.sh`) must switch to
  `gpu_tuned_verify_build_info_json` before pinning this release. Consumers that pin 20261022.0 or 20261027.0 are unaffected.

## Tests

Scope (from the attached test log): tuned/test_modules.txt (13 modules) minus tuned/test_skips.txt + verify_abi.sh + nccl_check.sh
Modules: test_cuda test_cuda_primary_ctx test_cuda_trace test_nn test_linalg test_matmul_cuda test_tensor_creation_ops test_torch test_autograd test_reductions test_transformers test_cpp_extensions_jit distributed/test_nccl
Result: === Full test suite summary: tests, verify_abi and nccl_check passed ===

## Install

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0+gb10.cu134.20261030.0/torch-2.15.0+gb10.cu134.20261030.0-cp314-cp314-linux_aarch64.whl"
```
