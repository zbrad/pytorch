# PyTorch 2.15.0 — gb10 cu134 (2.15.0+gb10.cu134.20261112.0)

Single-arch wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261112.0 (UTC date of the tuned commit)
- Built from commit 691bca8d5834938dbc1061058dc3e09c5ce42e8a; tested at 691bca8d5834938dbc1061058dc3e09c5ce42e8a
- Upstream base: upstream/main merge-base 2cce586c7744b4f82386d55742097d4395f23f58 (upstream's v2.15.0, unchanged from 20261030.1)
- NCCL: 2.30.7 (v2.30.7-1-11-gbef8ea32), statically linked, from https://github.com/zbrad/nccl/releases/tag/v0.20261012.0-gb10-cu134
- Wheel sha256: b5e7495f45c02aae3205609b202a8d4c87bf20f1a5e4f6c39622f0df4e3becb5

## Changes from 20261030.1

The PyTorch source is the same v2.15.0 base. What changed is fork build code added for the rtx40, rtx50 and Windows
builds since 20261030.1. As recorded in each commit message, most of it has no effect on a gb10 wheel:

- `cmake/Codegen.cmake` and `tuned/env.sh` (`4574038836d`, `450ca8c038d`, `a9cbf519036`): `TORCH_CUDA_EXACT_ARCH_LIST=1`
  builds only the arches in `TORCH_CUDA_ARCH_LIST`. Its commit records "No effect on gb10 or rtx40 (checked)"; the
  single-arch sm_121a gate passes as before.
- `TorchConfig.cmake.in` (`a8067cb240d`): CMake consumers now find `c10` and `caffe2_nvrtc` by their `-tuned` names.
  On Linux this removes a "library c10 not found" warning when a CMake project uses `find_package(Torch)`; the
  library still arrived through the Caffe2 targets before.
- `tuned/env.sh` hides MKL from CMake and `cmake/Dependencies.cmake` includes `CheckFunctionExists` explicitly
  (`6954078bb69`, `1aeaa0167ef`); MKL is x86 only, so neither changes an aarch64 build.
- `tuned/build.sh` refuses a wheel with no Python layer (`ebdf0005eb7`), a gate, not a change to the wheel.

Test-harness changes: `tuned/full_test.sh` fails on a pytest INTERNALERROR or on a module that ran no test
(`dc778050f19`), and the test venv pins `pytest<9.2` (`88169f39093`). That fixes the problem the 20261030.1
correction records: `test_cuda_primary_ctx` and `test_cuda_trace` now ran 1 and 13 tests (see Tests).

## Limitations

The same as 20261030.1; see its release notes for the details and evidence: no CPU linear algebra (LAPACK), slow CPU
float32 matmul through Eigen, no MPI, no cuDNN, matmul not batch-invariant for some bf16/fp16 shapes, single GPU only.
The build configuration that causes the first four is unchanged (`USE_MPI=OFF`, `USE_CUDNN=OFF`, no BLAS or LAPACK).
They were not rechecked by hand on this wheel; the full test run below includes the same skips.

## Tests

Scope (from the attached test log): tuned/test_modules.txt (13 modules) minus tuned/test_skips.txt + verify_abi.sh + nccl_check.sh
Modules: test_cuda test_cuda_primary_ctx test_cuda_trace test_nn test_linalg test_matmul_cuda test_tensor_creation_ops test_torch test_autograd test_reductions test_transformers test_cpp_extensions_jit distributed/test_nccl
Result: === Full test suite summary: tests, verify_abi and nccl_check passed ===

Tests run per module (passed or failed; skipped and deselected not counted), from the attached log: test_cuda 388,
test_cuda_primary_ctx 1, test_cuda_trace 13, test_nn 3261, test_linalg 2860, test_matmul_cuda 1490,
test_tensor_creation_ops 1064, test_torch 1812, test_autograd 973, test_reductions 9114, test_transformers 9613,
test_cpp_extensions_jit 37, distributed/test_nccl 2. Run on one GB10 on 2026-11-13 (about 2 h 47 min).

Smoke test of the packaged wheel passed (2026-11-13T13:06:04Z): clean-venv install and load, library identity, `test_memory_stats` (2), `test_cublas_multiple_threads_same_device` (1), `test_matmul_small_brute_force_2d_Nd` (5).

## Install

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0+gb10.cu134.20261112.0/torch-2.15.0+gb10.cu134.20261112.0-cp314-cp314-linux_aarch64.whl"
```
