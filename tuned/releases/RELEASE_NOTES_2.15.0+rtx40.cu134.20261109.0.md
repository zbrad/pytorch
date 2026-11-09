# PyTorch 2.15.0 — rtx40 cu134 (2.15.0+rtx40.cu134.20261109.0)

Single-arch wheel for RTX 4080/4090 (Ada Lovelace, SM 89) (TORCH_CUDA_ARCH_LIST=8.9), CUDA 13.4.

- Release label: 20261109.0 (UTC date of the tuned commit)
- Built from commit 1bb9060c6b099ea5011b4c3c76eedaddc10925a1; tested at 1bb9060c6b099ea5011b4c3c76eedaddc10925a1
- Upstream base: upstream/main merge-base 2cce586c7744b4f82386d55742097d4395f23f58
- NCCL: 2.32.3 (v2.32.3-1-34-gf2b26b2d), statically linked, from https://github.com/zbrad/nccl/releases/tag/v0.20261107.0-rtx40-cu134
- Wheel sha256: f732dd89bec09f3989c196dcfdc557626ec041a2a354b57839f3a2b3e75113fd

The first rtx40 release, from the same upstream base as 2.15.0+gb10.cu134.20261030.1 (merge-base above), built for
SM 89 only. Like the gb10 wheels it has no CPU BLAS/LAPACK, MPI or cuDNN (see Limitations); its NCCL is the rtx40
release named above.

Built, tested and smoke-tested on one RTX 4090 (24 GiB) under WSL2 (Ubuntu 24.04), Windows driver 621.14 (CUDA UMD
13.5, as `nvidia-smi` reports it). It has not been run on native Linux or on an RTX 4080.

## Limitations

What changes for a user. "Checked" means run against this packaged wheel (sha256 above) on that RTX 4090 on
2026-11-09.

- **CPU linear algebra does not work.** On CPU tensors, `torch.linalg.inv` and `cholesky` raise `RuntimeError:
  Calling torch.linalg.<op> on a CPU tensor requires compiling PyTorch with LAPACK`, and `svd` raises `svd: LAPACK
  library not found in compilation`; `torch.linalg.inv` on a CUDA tensor works. Checked. The build has no CPU BLAS or
  LAPACK (`torch.__config__.show()` lists neither `BLAS_INFO` nor `LAPACK_INFO`; `USE_EIGEN_FOR_BLAS=ON`).
- **CPU float32 matmul works**, through Eigen. Checked that it runs; its speed was not measured on this machine (the
  gb10 notes measured it as slow for large shapes).
- **No MPI.** `torch.distributed.is_mpi_available()` is False and `USE_MPI=OFF`. Checked. NCCL is available
  (`is_nccl_available()` True; `tuned/nccl_check.sh` in the test log).
- **No cuDNN.** `USE_CUDNN=OFF` and `torch.backends.cudnn.is_available()` is False. Checked. cuDNN RNN, CTC and
  cuDNN-attention paths are not available, and test_nn's cuDNN tests skip themselves ("cuDNN not available").
- **Single GPU only.** No multi-GPU test was run (one GPU on the test machine); the multi-GPU NCCL tests skip
  themselves.

## Tests

Scope (from the attached test log): tuned/test_modules.txt (13 modules) minus tuned/test_skips.txt + verify_abi.sh + nccl_check.sh
Modules: test_cuda test_cuda_primary_ctx test_cuda_trace test_nn test_linalg test_matmul_cuda test_tensor_creation_ops test_torch test_autograd test_reductions test_transformers test_cpp_extensions_jit distributed/test_nccl
Result: === Full test suite summary: tests, verify_abi and nccl_check passed ===

The test venv pins `pytest<9.2` (pytest 9.2 broke run_test.py's subprocess collection; upstream fix
pytorch/pytorch#202406), and the run fails if any module runs no test, so all 13 modules really ran.

Tests not run are listed, each with its reason, in `tuned/test_skips.txt` and in the test log header. The shared
entries are the ones the gb10 releases list (tests that need CPU BLAS or LAPACK, among others). New here, for rtx40
only, 15 tests. Each is an upstream test whose check assumes which kernels run; each failed alone and in the suite,
every run, and the results were checked against a float64 reference on the 20261108.0 build of this source:

- `test_addmm_out_distinct_c_and_d_float_out_reduced_input_cuda_bfloat16` (test from upstream #191706): it counts
  kernels. bf16 addmm with distinct C and D does take the fast path (no copy kernel), but cuBLASLt picks a split-K plan
  for this shape on SM 89, so the profile shows the GEMM plus `splitKreduce_kernel`: 2 kernels where the test wants 1.
  fp16 gets a single-kernel plan and passes.
- 14 `test_grouped_gemm` bf16 cases (8 `3d_3d`, plus 6 strided, column-major-a `2d_2d`/`2d_3d`/`3d_2d` regular;
  test from upstream #196075): SM 89 runs `_grouped_mm_fallback`, which differs from the `torch.mm` reference by bf16
  rounding against a bitwise tolerance. Against float64 neither is worse: for `3d_3d` the two are equally accurate
  (max error 0.062 vs 0.083 for `torch.mm`), and for the 6 other cases the fallback's `a.grad` error is 0.058-0.062
  vs 0.075-0.089 for `torch.mm`. The other grouped_gemm cases pass.

The test log's wheel hash is that of the unstamped wheel the tests ran against; the Wheel sha256 above is the packaged
wheel, whose `libtorch_cuda-tuned.so` has the build-info stamp added. The packaging commit is
dc3cf3638914cb0dda1a97ae23aae510baa3e528: it changes only `tuned/smoke_test.sh`, whose library-identity check now
accepts WSL2's libcuda loader stub plus the driver-store library it loads (this release is the first built under
WSL2).

## Install

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0+rtx40.cu134.20261109.0/torch-2.15.0+rtx40.cu134.20261109.0-cp314-cp314-linux_x86_64.whl"
```
