# PyTorch 2.15.0 — rtx50-win cu134 (2.15.0+rtx50.win.cu134.20261109.1) — RTX 50-series, native Windows — pre-release

**Pre-release: a native Windows (MSVC) dev build.** It has had a smoke test and three upstream tests on one machine,
not the 13-module test suite the Linux releases run (see Tests). It is published so the other native Windows builds
(flashinfer, vLLM) can pin it by URL. It is never marked Latest; the Linux rtx50 wheel stays the Latest release.

Single-arch wheel for RTX 50-series (Blackwell consumer, desktop/laptop, SM 120a) on Windows x86_64
(TORCH_CUDA_ARCH_LIST=12.0a), CUDA 13.4, CPython 3.14 (`cp314-cp314-win_amd64`).

- Release label: 20261109.1 (UTC date of the build commit; sequence 1 because 20261109.0 is taken in this repo)
- Built from commit 29cb4830e2c9b5999d9301e2e93638a777d11fd3 (`tuned-builds`)
- Upstream base: upstream/main merge-base 2cce586c7744b4f82386d55742097d4395f23f58, the same as
  2.15.0+rtx50.cu134.20261109.0 (Linux)
- Toolchain: MSVC 14.44.35207 (Windows SDK 10.0.26100.0), CUDA 13.4 (nvcc 13.4.92)
- Wheel sha256: 7785081eb293e620a5db44ffc9596fd83ba1cfd95f8affe57118ae8e5c18156a

Built and tested on one RTX 5090 Laptop GPU (24 GiB) with an Intel Core Ultra 9 275HX CPU, Windows 11 Pro
(10.0.26100), driver 621.14, natively (not WSL). It has not been run on a desktop RTX 5090, on any other RTX 50-series
card, or with another driver.

## Install

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0%2Brtx50.win.cu134.20261109.1/torch-2.15.0%2Brtx50.win.cu134.20261109.1-cp314-cp314-win_amd64.whl"
```

No CUDA toolkit or Visual Studio is needed at run time: the wheel carries the CUDA 13.4 runtime libraries it uses
(below). Checked: the smoke test ran with neither on `PATH` and `CUDA_PATH` unset. `torch.compile` additionally needs
`triton-windows==3.9.0.post30` (PyPI), which is not a dependency of this wheel; it was checked with the dev-label
build of this wheel on 2026-11-10 (a Triton kernel and `torch.compile` of a softmax, with no compiler on `PATH`).

## How this wheel was made from the build

The wheel was built on 2026-11-09 with the dev label `2.15.0+rtx50.cu134.win.20261109.dev0`, then repacked for this
release (the built wheel itself is unchanged). The repack:

1. **Relabelled it** to the `docs/VERSIONING.md` form (zbrad/tuned-common), `+rtx50.win.cu134.<date>.<seq>`. The label
   is stored only in `torch\version.py` and the `.dist-info` (directory name, `METADATA`, `RECORD`); none of
   `torch_cpu-tuned.dll`, `torch_cuda-tuned.dll` or `torch_python-tuned.dll` contains it (checked by searching the
   files). So the compiled code is the code the dev-label smoke and tests ran on.
2. **Left out two CUDA DLLs** that the CUDA 13.4 EULA's Attachment A does not list as redistributable on Windows:
   `nvperf_host.dll` (CUPTI's performance-counter host library) and `cusolverMg64_12.dll` (cuSOLVER's multi-GPU
   API). Upstream PyTorch's Windows packaging copies both (`cmake/PostBuildSteps.cmake`, the `nvperf_host*.dll` and
   `cusolver*64_*.dll` globs). No DLL in this wheel imports either (`dumpbin /dependents` on every DLL, checked).
3. **Stamped `torch_cuda-tuned.dll`** with the tuned build info (JSON, schema `tuned-build-info/1`, the same fields as
   the Linux wheels' `.tuned_build_info` section) as its Windows version resource: the JSON is the `Comments` string,
   and `FileVersion`/`ProductVersion` are the release label. Read it with `(Get-Item torch_cuda-tuned.dll).VersionInfo`
   or Explorer's Details tab. `fork.dirty` is `null`: the build did not record the state of the tree, so it is not
   claimed either way.

The CUDA DLLs the wheel carries, all from the CUDA 13.4 toolkit and all listed in Attachment A: `cudart64_13`,
`cublas64_13`, `cublasLt64_13`, `cufft64_12`, `cufftw64_12`, `curand64_10`, `cusolver64_12`, `cusparse64_12`,
`nvrtc64_130_0`, `nvrtc-builtins64_134`, `nvJitLink_130_0`, `cupti64_2026.3.1`. The other DLLs are this build's own
(`*-tuned.dll`) and `uv.dll` (libuv 1.51.0, for gloo).

## Limitations

What changes for a user. "Checked" means run against this wheel (sha256 above) on that machine on 2026-11-12.

- **No NCCL** (there is no NCCL for Windows): `torch.distributed.is_nccl_available()` is False. gloo works:
  `is_gloo_available()` is True, and a `tcp://` rendezvous with a gloo `all_reduce` passes. Checked.
- **A raw `TCPStore()` fails** with its default `use_libuv=True`, as on upstream's Windows wheels: Windows torch has
  no libuv TCPStore backend (`TORCH_USE_LIBUV` is set only with TensorPipe, which is off on Windows).
  `init_process_group` is not affected, because `rendezvous.py` defaults `use_libuv` to 0 on win32. Seen with the
  dev-label build on 2026-11-09.
- **No cuDNN** (`USE_CUDNN=0`), **no CPU BLAS/LAPACK** (CPU `torch.linalg` that needs LAPACK raises; CPU float32
  matmul works through Eigen, checked), **no MPI**, **no MKL**: the same choices as the Linux rtx50 wheel. CUDA
  `torch.linalg` works through cuSOLVER: `inv` and `qr` of a 64x64 float64 matrix, checked.
- **Without `nvperf_host.dll`**, CUPTI features that read hardware performance counters (range profiling, PM
  sampling) are expected not to work. Not tested. Ordinary `torch.profiler` with CUDA activity works: it recorded
  CUDA kernel events, checked.
- **Without `cusolverMg64_12.dll`**, cuSOLVER's multi-GPU (`cusolverMg*`) API is unavailable. PyTorch does not call
  it (nothing in the wheel imports it).
- **Single GPU only.** No multi-GPU test was run.
- **CPython 3.14 only.**

## Tests

All on the machine above.

- **Smoke, this wheel, 2026-11-12** (attached log, `SMOKE_RESULTS_rtx50-win.log`): a new venv with this wheel and its
  PyPI dependencies, no MSVC or CUDA toolkit on `PATH`, run from an empty directory. Version, `get_arch_list() ==
  ['sm_120a']`, CUDA matmul against float64, CPU matmul, NCCL absent, gloo rendezvous and `all_reduce`, that only
  `-tuned` torch DLLs load and only from the venv; then the release checks: the label in `torch.__version__` and the
  dist-info, the two DLLs absent, CUDA `inv`/`qr`, the profiler with CUDA activity, and the build-info stamp read back
  from the installed `torch_cuda-tuned.dll`. All passed.
- **Upstream tests, dev-label build, 2026-11-09:** `test_memory_stats`, `test_cublas_multiple_threads_same_device`
  and `test_matmul_small_brute_force_2d_Nd` passed. The compiled code is identical to this wheel's (see above).
- **Not run:** the 13-module suite the Linux releases run (`tuned/test_modules.txt`), `verify_abi.sh`, and the
  `tuned/` packaging gates, which are bash and do not run natively on Windows.
