# PyTorch 2.15.0 — rtx50-win cu134 (2.15.0+rtx50.win.cu134.20261114.2) — RTX 50-series, native Windows — pre-release

**Pre-release: a native Windows (MSVC) dev build.** It has had a smoke test and the release checks on one machine,
not the 13-module test suite the Linux releases run (see Tests). It is published so the other native Windows builds
(flashinfer, vLLM) can pin it by URL. It is never marked Latest; the Linux rtx50 wheel stays the Latest release.

**It replaces 2.15.0+rtx50.win.cu134.20261109.1**, which has been withdrawn: that wheel's
`torch/share/cmake/ATen/ATenConfig.cmake` recorded the absolute path of the temporary directory it was built in, and
its binaries recorded the build machine's source and build paths. This wheel has neither (see "Build paths").

Single-arch wheel for RTX 50-series (Blackwell consumer, desktop/laptop, SM 120a) on Windows x86_64
(TORCH_CUDA_ARCH_LIST=12.0a), CUDA 13.4, CPython 3.14 (`cp314-cp314-win_amd64`).

- Release label: 20261114.2 (UTC date of the build; sequence 2 because 20261114.0 and 20261114.1 were built and
  not published: .0 before the path trimming, .1 a trimming attempt that left paths behind)
- Built from commit 69e74b44fdf8cdeed44fb40faab34dfeae358dde (`tuned-builds`), directly with the release label
  (no relabel step this time)
- Upstream base: upstream/main merge-base 2cce586c7744b4f82386d55742097d4395f23f58, the same as 20261109.1; no
  upstream merge since then
- Toolchain: MSVC 14.44.35207 (Windows SDK 10.0.26100.0), CUDA 13.4 (nvcc 13.4.92)
- Wheel sha256: 12d6004b9bf83ef1edab0ced7b9d2529a6748a769e4eca66214db4fc3d30a3d5

Built and tested on one RTX 5090 Laptop GPU (24 GiB) with an Intel Core Ultra 9 275HX CPU, Windows 11 Pro
(10.0.26100), driver 621.14, natively (not WSL). It has not been run on a desktop RTX 5090, on any other RTX 50-series
card, or with another driver.

## Install

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0%2Brtx50.win.cu134.20261114.2/torch-2.15.0%2Brtx50.win.cu134.20261114.2-cp314-cp314-win_amd64.whl"
```

No CUDA toolkit or Visual Studio is needed at run time: the wheel carries the CUDA 13.4 runtime libraries it uses
(below). Checked: the smoke test ran with neither on `PATH` and `CUDA_PATH` unset. `torch.compile` additionally needs
`triton-windows==3.9.0.post30` (PyPI), which is not a dependency of this wheel.

## Changes since 20261109.1

From `git log 29cb4830e2c..69e74b44fdf` on `tuned-builds`:

- **`torch.utils.cpp_extension` builds CUDA extensions that include CCCL on Windows** (69e74b44fdf). CUDA 13's CCCL
  headers stop with C1189 unless cl gets `/Zc:preprocessor`; torch's own CMake build passes it and
  `/Zc:__cplusplus`, but `load()` and `BuildExtension` did not pass them to nvcc's host compiler. They now do, through
  `-Xcompiler`. Test: `tuned/tests/win_cpp_extension_cccl.py` (JIT and ninja builds of a kernel that includes CCCL,
  then run on the GPU); it fails with C1189 on an earlier build without the change (20261113.0, not published) and
  passes on this wheel.
- **`TorchConfig.cmake` finds `c10` and `caffe2_nvrtc` by their `-tuned` names** (a8067cb240d), so a CMake project
  using `find_package(Torch)` links against this wheel's libraries.
- **Windows ARM64 CMake fixes** (2efd4f0feba, e6dd23dffc1, 937634b3c85): the toolkit's `lib/arm64` and the arm64
  CUPTI DLLs. They do not change an x86_64 build: this wheel has the same 22 DLLs as 20261109.1.
- Release notes and `tuned/` script syncs (no effect on the wheel).

## How this wheel was made from the build

1. **Built with the release label** (`PYTORCH_BUILD_VERSION=2.15.0+rtx50.win.cu134.20261114.2`), in a new build
   directory.
2. **Build paths** were mapped at compile time: the `CL` environment variable carried
   `/experimental:deterministic /pathmap:<source dir>=pytorch /pathmap:<build dir>=build`. cl reads `CL` on every
   invocation, including the host and device-side ones nvcc makes, and CMake does not record it, so `__FILE__` strings
   and the object paths in the debug records read `pytorch\...` and `build\...`. `/experimental:deterministic` is what
   makes cl honour `/pathmap`; its one side effect here is warning C5048 where `__DATE__`/`__TIME__` are used. Passing
   the same options through `CFLAGS`/`CUDAFLAGS` does not work: torch records its C++ flags in
   `torch/include/caffe2/core/macros.h` and in `torch_cpu-tuned.dll`, and rebuilds `CMAKE_CUDA_FLAGS`, so the
   `.cu` host code is not mapped (that was 20261114.1).
3. **`ATenConfig.cmake`'s `ATEN_INCLUDE_DIR`** is rewritten to `${CMAKE_CURRENT_LIST_DIR}/../../../include`, the
   wheel's own `torch/include`. The build had recorded the temporary directory scikit-build-core staged the wheel in.
4. **A private-path gate** searched every file in the repacked wheel, as written and with NUL bytes removed (to catch
   UTF-16 strings), for absolute paths under the build machine's source, build and user-profile directories and for
   the machine's host and user names. 0 files matched. 20261109.1 failed the same search in 14 files.
5. **Left out two CUDA DLLs** that the CUDA 13.4 EULA's Attachment A does not list as redistributable on Windows:
   `nvperf_host.dll` (CUPTI's performance-counter host library) and `cusolverMg64_12.dll` (cuSOLVER's multi-GPU
   API), as in 20261109.1. Upstream PyTorch's Windows packaging copies both (`cmake/PostBuildSteps.cmake`). No DLL in
   this wheel imports either.
6. **Stamped `torch_cuda-tuned.dll`** with the tuned build info (JSON, schema `tuned-build-info/1`) as its Windows
   version resource: the JSON is the `Comments` string, and `FileVersion`/`ProductVersion` are the release label.
   Read it with `(Get-Item torch_cuda-tuned.dll).VersionInfo`. `fork.dirty` is `true`; `git status` shows no change to
   tracked files, only three submodules marked `m` (modified content), each for its own nested `composable_kernel`
   checkout: `third_party/aiter`, `third_party/flash-attention` and `third_party/mslk`.

The CUDA DLLs the wheel carries, all from the CUDA 13.4 toolkit and all listed in Attachment A: `cudart64_13`,
`cublas64_13`, `cublasLt64_13`, `cufft64_12`, `cufftw64_12`, `curand64_10`, `cusolver64_12`, `cusparse64_12`,
`nvrtc64_130_0`, `nvrtc-builtins64_134`, `nvJitLink_130_0`, `cupti64_2026.3.1`. The other DLLs are this build's own
(`*-tuned.dll`) and `uv.dll` (libuv 1.51.0, for gloo).

## Limitations

Unchanged from 20261109.1. "Checked" means run against this wheel (sha256 above) on that machine on 2026-11-15.

- **No NCCL** (there is no NCCL for Windows): `torch.distributed.is_nccl_available()` is False. gloo works:
  `is_gloo_available()` is True, and a `tcp://` rendezvous with a gloo `all_reduce` passes. Checked.
- **A raw `TCPStore()` fails** with its default `use_libuv=True`, as on upstream's Windows wheels: Windows torch has
  no libuv TCPStore backend. `init_process_group` is not affected, because `rendezvous.py` defaults `use_libuv` to 0
  on win32. Seen with a dev build on 2026-11-09.
- **No cuDNN** (`USE_CUDNN=0`), **no CPU BLAS/LAPACK** (CPU `torch.linalg` that needs LAPACK raises; CPU float32
  matmul works through Eigen, checked), **no MPI**, **no MKL**: the same choices as the Linux rtx50 wheel. CUDA
  `torch.linalg` works through cuSOLVER: `inv` and `qr` of a 64x64 float64 matrix, checked.
- **Without `nvperf_host.dll`**, CUPTI features that read hardware performance counters (range profiling, PM
  sampling) are expected not to work. Not tested. Ordinary `torch.profiler` with CUDA activity works, checked.
- **Without `cusolverMg64_12.dll`**, cuSOLVER's multi-GPU API is unavailable. PyTorch does not call it.
- **`BuildExtension(use_ninja=False)` does not build `.cu` files on Windows** with setuptools 84.2.1: setuptools runs
  cl on them and torch's compile hook is never called. Seen 2026-11-14; it is not new in this wheel. Use the default
  `use_ninja=True` or `load()`.
- **Single GPU only.** No multi-GPU test was run.
- **CPython 3.14 only.**

## Tests

All on the machine above.

- **Smoke, this wheel, 2026-11-15** (attached: `SMOKE_RESULTS_rtx50-win.log`, and the two scripts it ran,
  `smoke_win.py` and `torch_rel_check.py`; local paths in the log are replaced by `<work>`/`<src>`): a new venv with
  this wheel and its PyPI dependencies, no MSVC or CUDA toolkit on `PATH`, `CUDA_PATH` unset, run from an empty
  directory. Version, `get_arch_list() == ['sm_120a']`, CUDA matmul against float64, CPU matmul, NCCL absent, gloo
  rendezvous and `all_reduce`, that only `-tuned` torch DLLs load and only from the venv; then the release checks:
  the label in `torch.__version__` and the dist-info, the two DLLs absent, CUDA `inv`/`qr`, the profiler with CUDA
  activity, and the build-info stamp read back from the installed `torch_cuda-tuned.dll`. All passed.
- **C++/CUDA extensions, 2026-11-15:** `tuned/tests/win_cpp_extension_cccl.py` with this wheel installed, MSVC 14.44
  and CUDA 13.4 active: the `load()` (JIT) and `BuildExtension` (ninja) builds both compiled and gave the right
  result on the GPU.
- **vLLM on this wheel, 2026-11-15:** zbrad/vllm `0.33.1.dev186+rtx50win.cu134.20261114.0` (a native Windows build,
  not published) with this torch, no toolchain on `PATH`: `pip check` clean; import of all its extensions and a
  custom-op check; greedy generation with a small model on the `FLASH_ATTN` backend, eager and with `torch.compile`;
  vLLM's qutlass kernel tests `test_mxfp4_qutlass.py` (102 passed) and `test_nvfp4_qutlass.py` (132 passed).
- **FlashInfer on this wheel, 2026-11-15:** the published `0.7.4+rtx50.win.cu134.20261113.0` wheels in a new venv
  with this torch: its release check (XQA load and decode, decode and prefill against a reference, sampling, no
  kernel compiled at run time) passed, and vLLM's FlashInfer backend generated eager and with `torch.compile` with no
  kernel compiled into the workspace.

- **Not run:** the 13-module suite the Linux releases run (`tuned/test_modules.txt`), `verify_abi.sh`, and the
  `tuned/` packaging gates, which are bash and do not run natively on Windows.
