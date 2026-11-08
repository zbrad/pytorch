# PyTorch 2.15.0 — gb10 cu134 (2.15.0+gb10.cu134.20261027.0)

Single-arch wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261027.0 (UTC date of the tuned commit)
- Built and tested at commit b6053b6f191db4d3d8bdebd357ed13bb5f9e75ce; packaged at 80f4a34a948 (only `tuned/` changed in between)
- Upstream base: upstream/main merge-base 5c6918c45adefb8eaf40e7f665ccd00f9879fbef
- NCCL: 2.30.7 (v2.30.7-1-11-gbef8ea32), statically linked, from https://github.com/zbrad/nccl/releases/tag/v0.20261012.0-gb10-cu134
- Wheel sha256: 46cc4ed51bcf5f10dd38eeb2cbae52ce3397d5cd72a75547a5193f4c68cf47d9

## Changes from 20261022

A rebuild of the same torch source (the only non-`tuned/` change is a CI requirements pin, `.ci/docker/requirements-ci.txt`
pillow, which is not part of the wheel). It stays a GPU-only build: no CPU BLAS, no CPU LAPACK, no MPI. What changed is how it
was built and packaged, with the shared tuned-common gates:

- the build runs in a per-toolkit build venv with setuptools 84.2 or newer, checked again before packaging;
- the wheel's runtime dependencies are audited for known vulnerabilities (`pip-audit`);
- packaging and smoke test were re-run on the final commit, so the wheel's checksum differs from the one in the earlier draft.

Consumers that pin 20261022.0 are unaffected and need no change.

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

**Rerun (2026-11-08):** not rerun for this superseded release; the two modules were run against 20261030.1 instead and passed (see that release).

## Install

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0+gb10.cu134.20261027.0/torch-2.15.0+gb10.cu134.20261027.0-cp314-cp314-linux_aarch64.whl"
```
