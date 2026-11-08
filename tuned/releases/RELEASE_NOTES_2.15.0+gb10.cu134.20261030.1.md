# PyTorch 2.15.0 — gb10 cu134 (2.15.0+gb10.cu134.20261030.1)

Single-arch wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261030.1 (UTC date of the tuned commit)
- Built from commit 9dba13dc7b3f216204e844366a72b9065d4b3c4a; tested at 3ed59977383b423cca78e159dcc71f8be2090127
- Upstream base: upstream/main merge-base 2cce586c7744b4f82386d55742097d4395f23f58
- NCCL: 2.30.7 (v2.30.7-1-11-gbef8ea32), statically linked, from https://github.com/zbrad/nccl/releases/tag/v0.20261012.0-gb10-cu134
- Wheel sha256: f9ddcddf1cf3e5b84f5f940440a4220699c2b8156db8cb52735222182c616210

## Changes from 20261030.0

The PyTorch source is now upstream's v2.15.0 tag (merged at `5c75491a2ee`; the 20261030.0 wheel was built before that
merge). Nothing changed in the tuned build configuration: `git diff` of `tuned/` between the two build commits shows
only release notes, and the wheel's build flags still say `USE_MPI=OFF`, `USE_CUDNN=OFF`, no BLAS or LAPACK. The
packaging gates pass the same way (every shared object carries a `-tuned` name; no CPU BLAS, LAPACK or MPI; single-arch
sm_121a). The JSON build-info stamp introduced in 20261030.0 is unchanged.

Test-harness changes in this release's test commit (they do not change the wheel): `tuned/full_test.sh` now runs
`test/run_test.py` in the real `__main__` module, because a failing test module made the old wrapper hang until its
4-hour timeout; and `tuned/test_skips.txt` gains the three SVD tests that v2.15.0 added (see Tests).

## Limitations

What changes for a user. "Checked" means run against this wheel (20261030.1) on one GB10 on 2026-11-01. These
limitations are the ones recorded for 20261022.0 (where the evidence is in that release's notes); the entries below
were rechecked on this wheel, except where stated.

- **CPU linear algebra does not work.** On CPU tensors, `torch.linalg.inv`, `solve`, `cholesky`, `lu_factor` and `svd`
  raise `RuntimeError: ... requires compiling PyTorch with LAPACK` (or "LAPACK library not found"); the same operations
  on CUDA tensors work. Checked. Anything that calls them on CPU tensors will fail. A grep of vllm and flashinfer source
  on 2026-10-24 (Claude's grep, not a run) found no such call in their runtime code; HF `resize_token_embeddings` calls
  `cholesky` on the CPU and is used by a vllm test.
- **CPU float32 matmul works.** Checked that it runs on this wheel (`USE_EIGEN_FOR_BLAS=ON`). It is slow: a
  10000x10000x10000 CPU reference in the test suite took about 5 minutes per case (measured on 20261022.0, not remeasured).
  Do not use the CPU for large matmuls.
- **No MPI.** `torch.distributed.is_mpi_available()` is False, `torch._C._distributed_c10d` has no `ProcessGroupMPI`, and
  `init_process_group("mpi")` raises "Distributed package doesn't have MPI built in". Checked. NCCL is unaffected (see
  `tuned/nccl_check.sh` in the test log). The build cache still lists `USE_C10D_MPI` as ON, a stale internal entry: the
  backend is compiled only when `USE_MPI` and `USE_C10D_MPI` are both set (`caffe2/CMakeLists.txt:1699` in this source)
  and `USE_MPI` is OFF (checked on 20261022.0 that `ProcessGroupMPI.cpp.o` has no symbols; here the shipped
  `libtorch_cpu-tuned.so` exports none).
- **No cuDNN.** `torch.__config__.show()` reports `USE_CUDNN=OFF`. Checked here. The 20261012 wheel was also linked
  without libcudnn (checked on 20261022.0), so this is not new. cuDNN RNN, CTC and cuDNN-attention paths are not
  available.
- **Matmul is not batch-invariant for some shapes on this GPU.** With cuBLASLt and the reduced-precision reductions off,
  `x[:1] @ w` differs from `(x @ w)[:1]` for bf16 and fp16 (19 and 126 mismatched elements over 5 seeds in a repro of
  the upstream test, `x` 32x6144, `w` 6144x2048). Checked on this wheel with that repro. Cause not established: it is
  cuBLAS from CUDA 13.4 on SM 12.1, not the BLAS change. The guess that cuBLAS promises batch invariance only on SM 10.x
  is Claude's and untested. Anyone who needs bit-identical results across batch sizes should not rely on this.
- **Single GPU only.** No multi-GPU test was run (one GPU on the test node); the multi-GPU NCCL tests skip themselves.

## Tests

Scope (from the attached test log): tuned/test_modules.txt (13 modules) minus tuned/test_skips.txt + verify_abi.sh + nccl_check.sh
Modules: test_cuda test_cuda_primary_ctx test_cuda_trace test_nn test_linalg test_matmul_cuda test_tensor_creation_ops test_torch test_autograd test_reductions test_transformers test_cpp_extensions_jit distributed/test_nccl
Result: === Full test suite summary: tests, verify_abi and nccl_check passed ===

**Correction (2026-11-08): two of the 13 modules ran no tests.** `test_cuda_primary_ctx` and `test_cuda_trace` did not
run: the test venv resolved pytest 9.2.0, and the attached log shows `INTERNALERROR> AttributeError: 'str' object has no
attribute 'path'` for each, while collecting. pytest 9.2 (pytest-dev/pytest#14758) made `Config.cwd_relative_nodeid`
take a `NodeId`; run_test.py's `--subprocess` collection (`get_pytest_test_cases` in
`torch/testing/_internal/common_utils.py`) still passes the `str` node id. run_test.py did not count it as a failure, so
the summary above says passed. The fix upstream is pytorch/pytorch#202406 (open as of 2026-11-08). The same
holds for the attached logs of 20261027.0, 20261030.0 and 20261030.1 (all pytest 9.2.0); 20261022.0 used pytest 9.1.1
and did run both modules. The tuned test tooling now pins `pytest<9.2` and fails a run
in which any module runs no test.

**Rerun (2026-11-08):** both modules were then run against this release's published wheel (sha256 `f9ddcddf1cf3e5b84f5f940440a4220699c2b8156db8cb52735222182c616210`) on the same GB10 node, with pytest 9.1.1 and the test files of the build commit (`test/` and `torch/testing/` unchanged since 9dba13dc7b3): `test_cuda_primary_ctx` 1 passed, 3 skipped; `test_cuda_trace` 13 passed; no INTERNALERROR. The log is attached as `TEST_RESULTS_gb10_twomod_20261030.1.log`.

Tests not run are listed, each with its reason, in `tuned/test_skips.txt` and in the test log header. They are the
20261022.0 skips (17 tests that need CPU BLAS or LAPACK, the two `test_cublas_batch_invariance_blackwell` tests, and
`test_call_count_tunableop`, which counts profiler events by name and gave 0 in one full run while passing alone; cause
not found) plus, new here, `test_svd_ill_conditioned`, `test_svd_scaled_entries` and `test_svd_small_scale` (upstream
#199063, 12 CUDA variants): each builds its input or reference on the CPU (`torch.linalg.qr` or `pinv` on a CPU tensor), so
it needs the CPU LAPACK this wheel does not have. `test_addmm_out_distinct_c_and_d_float_out_reduced_input` failed once
in a rerun on 20261022.0 and passed on rerun; it is not skipped and its cause was not investigated.

The test log's wheel hash is that of the unstamped wheel the tests ran against; the Wheel sha256 above is the packaged
wheel, whose `libtorch_cuda-tuned.so` has the build-info stamp added.

## Install

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0+gb10.cu134.20261030.1/torch-2.15.0+gb10.cu134.20261030.1-cp314-cp314-linux_aarch64.whl"
```
