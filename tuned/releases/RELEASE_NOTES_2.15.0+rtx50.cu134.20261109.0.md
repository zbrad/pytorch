# PyTorch 2.15.0 — rtx50 cu134 (2.15.0+rtx50.cu134.20261109.0)

Single-arch wheel for RTX 50-series (Blackwell consumer, desktop/laptop, SM 120a) (TORCH_CUDA_ARCH_LIST=12.0a), CUDA 13.4.

- Release label: 20261109.0 (UTC date of the tuned commit)
- Built from commit 1aeaa0167efddeb16c8c9960962b8eaba2b8bab5; tested at 1aeaa0167efddeb16c8c9960962b8eaba2b8bab5
- Upstream base: upstream/main merge-base 2cce586c7744b4f82386d55742097d4395f23f58
- NCCL: 2.32.3 (v2.32.3-1-34-gf2b26b2d), statically linked, from https://github.com/zbrad/nccl/releases/tag/v0.20261107.0-rtx50-cu134
- Wheel sha256: 1d1d912db00b8789906ed88ca808f9c6053e8b6be6aa3c32eecc52099bc235d0

The first rtx50 release, from the same upstream base as 2.15.0+rtx40.cu134.20261109.0 (merge-base above), built for
SM 120a only. Like the rtx40 and gb10 wheels it has no CPU BLAS/LAPACK, MPI or cuDNN (see Limitations); its NCCL is
the rtx50 release named above.

Built, tested and smoke-tested on one RTX 5090 Laptop GPU (24 GiB) with an Intel Core Ultra 9 275HX CPU, under WSL2
(Ubuntu 24.04), Windows driver 621.14. It has not been run on native Linux, on a desktop RTX 5090 or on any other RTX
50-series card.

## Only the requested arch

The wheel contains code for `sm_120a` alone: `torch.cuda.get_arch_list()` is `['sm_120a']`, and
`libtorch_cuda-tuned.so` holds 495 `sm_120a` cubins and nothing else (`cuobjdump -lelf`). Checked. Stock PyTorch
also compiles some files for sibling arches that `TORCH_CUDA_ARCH_LIST` does not name: a 12.0a build gets an
`sm_121a` (DGX Spark) cubin, which an RTX 50 card cannot run (upstream issue pytorch/pytorch#204021). This fork
carries an opt-in `TORCH_CUDA_EXACT_ARCH_LIST` switch, set by `tuned/env.sh`, that turns those additions off. It is
proposed upstream as pytorch/pytorch#204020 (open); the fork keeps its copy until that merges.

## Limitations

What changes for a user. "Checked" means run against this packaged wheel (sha256 above) on that machine on
2026-11-09.

- **CPU bf16/fp16 backward of some layers fails on avx2_vnni_2 CPUs.** On the test machine's CPU (Core Ultra 9
  275HX, Arrow Lake), oneDNN raises `RuntimeError: DNNL does not support bf16/f16 backward on the platform with
  avx2_vnni_2` for the backward of some bf16/fp16 CPU layers; the three tests skipped below fail this way. Checked.
  This is oneDNN's behavior on that CPU class, not something this build adds: it was reported upstream on Lunar Lake
  (pytorch/pytorch#148861), and pytorch/pytorch#200055 (open) adds a fallback for GELU and Conv backward. CUDA and
  float32 CPU paths are not affected.
- **CPU linear algebra does not work.** On CPU tensors, `torch.linalg.inv` raises `RuntimeError: Calling
  torch.linalg.lu_factor on a CPU tensor requires compiling PyTorch with LAPACK`, `cholesky` the same for itself, and
  `svd` raises `svd: LAPACK library not found in compilation`; `torch.linalg.inv` on a CUDA tensor works. Checked. The
  build has no CPU BLAS or LAPACK (`torch.__config__.show()` lists neither `BLAS_INFO` nor `LAPACK_INFO`;
  `USE_EIGEN_FOR_BLAS=ON`, `USE_MKL=OFF`).
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
entries are the ones the gb10 releases list (tests that need CPU BLAS or LAPACK, among others). New here, for rtx50
only, 3 CPU tests in test_nn: the two `test_activations_bfloat16_half_cpu` tests and `TestNN::test_no_grad` (selected
exactly, by `(TestNN and test_no_grad and not test_no_grad_)`). All three fail with the oneDNN avx2_vnni_2 error
described under Limitations; they come back when this source has pytorch/pytorch#200055.

Building on a machine with Intel oneAPI installed, PyTorch's CMake finds MKL by itself even with BLAS and LAPACK off.
`tuned/env.sh` now also hides MKL from CMake (1aeaa0167ef carries a one-line `include(CheckFunctionExists)` fix that
this needed in `cmake/Dependencies.cmake`), so no library in the wheel needs an MKL library. Checked (`objdump -p`).

The test log's wheel hash is that of the unstamped wheel the tests ran against; the Wheel sha256 above is the packaged
wheel, whose `libtorch_cuda-tuned.so` has the build-info stamp added. The packaging commit is
fd0e1c25486, a merge of origin's `tuned/smoke_test.sh` WSL2 libcuda fix (dc3cf363891) and the rtx40 release notes; it
changes nothing outside `tuned/` relative to the build commit.

## Install

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0+rtx50.cu134.20261109.0/torch-2.15.0+rtx50.cu134.20261109.0-cp314-cp314-linux_x86_64.whl"
```
