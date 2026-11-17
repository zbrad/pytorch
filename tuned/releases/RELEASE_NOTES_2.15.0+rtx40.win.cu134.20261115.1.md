# PyTorch 2.15.0 — rtx40-win cu134 (2.15.0+rtx40.win.cu134.20261115.1) — RTX 40-series, native Windows — pre-release

**Pre-release: a native Windows (MSVC) dev build.** It has had a smoke test and the release checks on one machine,
not the 13-module test suite the Linux releases run (see Tests). It is published so the other native Windows builds
(flashinfer, vLLM) can pin it by URL. It is never marked Latest.

This is the first rtx40-win wheel. It is built the same way as `2.15.0+rtx50.win.cu134.20261114.2` (path-trimmed,
repacked, gated), for a different GPU architecture and from a later `tuned-builds` commit whose changes since that
build are all under `tuned/` (see "Changes").

Single-arch wheel for RTX 40-series (Ada Lovelace, desktop/laptop, SM 89) on Windows x86_64
(TORCH_CUDA_ARCH_LIST=8.9), CUDA 13.4, CPython 3.14 (`cp314-cp314-win_amd64`).

- Release label: 20261115.1 (UTC date of the build commit; sequence 1 because 20261115.0 was built first without
  path trimming, failed the private-path gate in 15 files, and was not published)
- Built from commit 4be85d6fc3436c1570d3ab1bbc4fddf94d99d589 (`tuned-builds`), directly with the release label
- Upstream base: upstream/main merge-base 2cce586c7744b4f82386d55742097d4395f23f58, the same as the rtx50-win
  20261114.2 wheel
- Toolchain: MSVC 14.42.34433 (Windows SDK 10.0.26100.0), CUDA 13.4 (nvcc 13.4.59)
- Wheel sha256: a5485fbb92e9be3fad1fa175b2f68ab8949c393f691b2894459d12ddd4161532

Built and tested on one desktop RTX 4090 (24 GiB) with an Intel Core i9-14900KF CPU, Windows 11 Pro (10.0.26200),
driver 621.14, natively (not WSL). It has not been run on any other RTX 40-series card, on a laptop RTX 40, or with
another driver.

## Install

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0%2Brtx40.win.cu134.20261115.1/torch-2.15.0%2Brtx40.win.cu134.20261115.1-cp314-cp314-win_amd64.whl"
```

No CUDA toolkit or Visual Studio is needed at run time: the wheel carries the CUDA 13.4 runtime libraries it uses
(below). Checked: the smoke test ran with neither on `PATH` and `CUDA_PATH` unset.

## Changes

Against the rtx50-win wheel's commit, from `git log 69e74b44fdf..4be85d6fc34` on `tuned-builds`: release notes, the
gb10 NCCL pin, and two `tuned-common` syncs (bash packaging scripts). None is in this wheel: `git diff --name-only`
lists only files under `tuned/`. Against an upstream Windows wheel, the source differences are the ones listed in
the rtx50-win 20261114.2 notes (the CCCL `/Zc` flags for `cpp_extension`, `-tuned` library names).

## How this wheel was made from the build

1. **Built with the release label** (`PYTORCH_BUILD_VERSION=2.15.0+rtx40.win.cu134.20261115.1`), in a new build
   directory.
2. **Build paths** were mapped at compile time: the `CL` environment variable carried
   `/experimental:deterministic /pathmap:<source dir>=pytorch /pathmap:<build dir>=build`, so `__FILE__` strings and
   the object paths in the debug records read `pytorch\...` and `build\...`. Why `CL` and not `CFLAGS`/`CUDAFLAGS`
   is in the rtx50-win 20261114.2 notes.
3. **`ATenConfig.cmake`'s `ATEN_INCLUDE_DIR`** is rewritten to `${CMAKE_CURRENT_LIST_DIR}/../../../include`, the
   wheel's own `torch/include`. The build had recorded the temporary directory the wheel was staged in.
4. **A private-path gate** searched every file in the repacked wheel, as written and with NUL bytes removed (to catch
   UTF-16 strings), for absolute paths under the build machine's source, build and user-profile directories and for
   the machine's host and user names. 0 files matched.
5. **Left out two CUDA DLLs** that the CUDA 13.4 EULA's Attachment A does not list as redistributable on Windows:
   `nvperf_host.dll` and `cusolverMg64_12.dll`, as in the rtx50-win wheels. No DLL in this wheel imports either.
6. **Stamped `torch_cuda-tuned.dll`** with the tuned build info (JSON, schema `tuned-build-info/1`) as its Windows
   version resource: the JSON is the `Comments` string, and `FileVersion`/`ProductVersion` are the release label.
   Read it with `(Get-Item torch_cuda-tuned.dll).VersionInfo`. `fork.dirty` is `false`.

The CUDA DLLs the wheel carries, all from the CUDA 13.4 toolkit and all listed in Attachment A: `cudart64_13`,
`cublas64_13`, `cublasLt64_13`, `cufft64_12`, `cufftw64_12`, `curand64_10`, `cusolver64_12`, `cusparse64_12`,
`nvrtc64_130_0`, `nvrtc-builtins64_134`, `nvJitLink_130_0`, `cupti64_2026.3.0`. The other DLLs are this build's own
(`*-tuned.dll`) and `uv.dll` (libuv 1.51.0, for gloo); 22 DLLs in all.

## Limitations

The same as the rtx50-win 20261114.2 wheel. "Checked" means run against this wheel (sha256 above) on that machine on
2026-11-17.

- **No NCCL** (there is no NCCL for Windows): `torch.distributed.is_nccl_available()` is False. gloo works:
  `is_gloo_available()` is True, and a `tcp://` rendezvous with a gloo `all_reduce` passes. Checked.
- **A raw `TCPStore()` fails** with its default `use_libuv=True`, as on upstream's Windows wheels;
  `init_process_group` is not affected. Seen on the rtx50-win dev build, not re-checked here.
- **No cuDNN** (`USE_CUDNN=0`), **no CPU BLAS/LAPACK**, **no MPI**, **no MKL**. CPU float32 matmul works through
  Eigen, checked. CUDA `torch.linalg` works through cuSOLVER: `inv` and `qr` of a 64x64 float64 matrix, checked.
- **Without `nvperf_host.dll`**, CUPTI features that read hardware performance counters are expected not to work.
  Not tested. Ordinary `torch.profiler` with CUDA activity works, checked.
- **`BuildExtension(use_ninja=False)` does not build `.cu` files on Windows** with setuptools 84.2.1 (seen again
  here: with ninja missing from `PATH`, the fallback compiled the `.cu` with cl and failed). Use `use_ninja=True`
  with ninja on `PATH`, or `load()`.
- **Single GPU only.** No multi-GPU test was run.
- **CPython 3.14 only.**

## Tests

All on the machine above, 2026-11-17.

- **Smoke, this wheel** (attached: `SMOKE_RESULTS_rtx40-win.log`, and the two scripts it ran, `smoke_win.py` and
  `torch_rel_check.py`; local paths in the log are replaced by `<work>`/`<src>`): a new venv with this wheel and its
  PyPI dependencies, no MSVC or CUDA toolkit on `PATH`, `CUDA_PATH` unset, run from an empty directory. Version, the
  RTX 4090 at capability (8, 9), `get_arch_list() == ['sm_89']`, CUDA matmul against float64, CPU matmul, NCCL
  absent, gloo rendezvous and `all_reduce`, that only `-tuned` torch DLLs load and only from the venv; then the
  release checks: the label in `torch.__version__` and the dist-info, the two DLLs absent, CUDA `inv`/`qr`, the
  profiler with CUDA activity, and the build-info stamp read back from the installed `torch_cuda-tuned.dll`. All
  passed.
- **C++/CUDA extensions:** `tuned/tests/win_cpp_extension_cccl.py` in that venv with MSVC 14.42 and CUDA 13.4 active
  and ninja on `PATH`: the `load()` (JIT) and `BuildExtension` (ninja) builds of a kernel that includes CCCL both
  compiled and gave the right result on the GPU. (setuptools linked the `BuildExtension` module with the newer
  Visual Studio also installed on this machine, MSVC 14.50; the compile used 14.42.)

- **Not run:** vLLM and FlashInfer on this wheel (no rtx40-win builds of either exist yet), the 13-module suite the
  Linux releases run (`tuned/test_modules.txt`), `verify_abi.sh`, and the `tuned/` packaging gates, which are bash and
  do not run natively on Windows.
