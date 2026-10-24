# PyTorch 2.15.0 — gb10 cu134 (2.15.0+gb10.cu134.20261022.0)

Single-arch wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261022.0 (UTC date of the tuned commit)
- Built from commit 0e9f8218aed69b2b1aa107e24d028f55c1acd965; tested at 6a77cf367d88eaeddba3c8189ec00f755f9c3bee
- Upstream base: upstream/main merge-base 5c6918c45adefb8eaf40e7f665ccd00f9879fbef
- NCCL: 2.30.7 (v2.30.7-1-11-gbef8ea32), statically linked, from https://github.com/zbrad/nccl/releases/tag/v0.20261012.0-gb10-cu134
- Wheel sha256: f271fac71bf2a8aba0aa3ed9a991c9807b14c4d8368dff317195fd94c249a9a4

## Changes from 20261012

This wheel is a GPU-only build: it has no CPU BLAS, no CPU LAPACK and no MPI.

- **Not linked any more:** OpenBLAS, OpenMPI and libgfortran. A node no longer needs `libopenblas0-pthread` or
  `libopenmpi3t64`; libgfortran came only in through OpenBLAS. Checked: `tuned/no_cpu_blas.sh` runs in `package.sh`
  and requires `AT_BUILD_WITH_BLAS=0`, `AT_BUILD_WITH_LAPACK=0` and no BLAS/LAPACK/MPI/Fortran library in any NEEDED
  entry; it fails the 20261012 wheel and passes this one.
- Built with `-DUSE_MPI=OFF` and CMake told not to find BLAS or LAPACK.

## Limitations

What changes for a user. "Checked" means run against this wheel on one GB10; the rest is stated as such.

- **CPU linear algebra does not work.** CPU `torch.linalg.solve`, `inv`, `cholesky`, `svd`, `lu_factor` and
  `triangular_solve` raise `RuntimeError: ... requires compiling PyTorch with LAPACK` (or BLAS). The same operations on
  CUDA tensors work. Checked. Anything that calls them on CPU tensors will fail: this has not been checked for vllm,
  flashinfer or other downstream code.
- **CPU float32 matmul works but uses a generic kernel.** Checked that it runs; it is slow (a 10000x10000x10000 CPU
  reference in the test suite took about 5 minutes per case). Do not use the CPU for large matmuls.
- **No MPI.** `torch.distributed.is_mpi_available()` is False and `init_process_group("mpi")` raises "Distributed package
  doesn't have MPI built in". NCCL is unaffected (see `tuned/nccl_check.sh` in the test log). Checked: the build cache
  still lists `USE_C10D_MPI` as ON, but that is a stale internal entry; the build only compiles the MPI backend when
  `USE_MPI` and `USE_C10D_MPI` are both set (`caffe2/CMakeLists.txt:1691`) and `USE_MPI` is OFF. `ProcessGroupMPI.cpp.o` is
  952 bytes with no symbols, and the shipped `libtorch_cpu`, `libtorch_python` export no `ProcessGroupMPI` symbol and
  `torch._C._distributed_c10d` has no `ProcessGroupMPI`.
- **No cuDNN.** `PyTorch was compiled without cuDNN/MIOpen support`. The 20261012 wheel is also linked without
  libcudnn (checked by reading `libtorch_cuda.so` NEEDED), so this is not new; it is stated here because it was not
  before. cuDNN RNN, CTC and cuDNN-attention paths are not available.
- **Matmul is not batch-invariant for some shapes on this GPU.** With cuBLASLt and the reduced-precision reductions off,
  `x[:1] @ w` differs from `(x @ w)[:1]` for bf16 and fp16 (6 of 2048 elements and 23 of 2048 in the upstream test, both
  repeatable; the batched variant is invariant). Checked, alone and in the suite. Cause not established: it is cuBLAS
  from CUDA 13.4 on SM 12.1, not the BLAS change, and the guess that cuBLAS promises batch invariance only on SM 10.x is
  untested. Anyone who needs bit-identical results across batch sizes should not rely on this.
- **Single GPU only.** No multi-GPU test was run (one GPU on the test node); the multi-GPU NCCL tests skip themselves.

## Tests

Scope (from the attached test log): tuned/test_modules.txt (13 modules) minus tuned/test_skips.txt + verify_abi.sh + nccl_check.sh
Modules: test_cuda test_cuda_primary_ctx test_cuda_trace test_nn test_linalg test_matmul_cuda test_tensor_creation_ops test_torch test_autograd test_reductions test_transformers test_cpp_extensions_jit distributed/test_nccl
Result: === Full test suite summary: tests, verify_abi and nccl_check passed ===

Tests not run are listed, each with its reason, in `tuned/test_skips.txt` and in the test log header. New in this
release: 17 tests that need CPU BLAS or LAPACK (CPU variants of linalg tests, plus two lobpcg CUDA tests and
`test_preferred_linalg_library_cuda`, whose bodies build CPU tensors), the two `test_cublas_batch_invariance_blackwell`
tests (see Limitations) and `test_call_count_tunableop`, which counts profiler events by name and gave 0 in the full
run while passing alone (cause not found). One other test, `test_addmm_out_distinct_c_and_d_float_out_reduced_input`,
failed once in a rerun of `test_linalg` and passed when rerun by the harness; it is not skipped and its cause was not
investigated.

## Install

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0+gb10.cu134.20261022.0/torch-2.15.0+gb10.cu134.20261022.0-cp314-cp314-linux_aarch64.whl"
```
