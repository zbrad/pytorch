# Tuned builds of PyTorch

Single-arch (one GPU variant, one CUDA toolkit) PyTorch wheels, published as GitHub releases of
this fork. The design this follows is `docs/ARCHITECTURE.md` in zbrad/tuned-common (requirement
IDs below, e.g. B3, refer to it). Shared logic is vendored as `tuned/common.sh`; hardware and
toolkit facts as `tuned/common-data/`; keep both current with `tuned/sync-common.sh check|sync`.

## Flow

```
CUDA_TAG=cu134 bash tuned/build.sh gb10        # compile, bake in the date label, make the wheel (hours)
CUDA_TAG=cu134 bash tuned/full_test.sh gb10    # install that wheel in a clean venv and test it
CUDA_TAG=cu134 bash tuned/package.sh gb10      # stamp a staged copy, sha256, package-info.txt, DRAFT notes
CUDA_TAG=cu134 bash tuned/smoke_test.sh gb10   # install the packaged wheel, run three repo tests
# review tuned/releases/RELEASE_NOTES_<version>.md, remove the DRAFT banner, commit and push
CUDA_TAG=cu134 bash tuned/release.sh gb10      # every gate reported together, then publish
```

With neither `CUDA_TAG` nor `CUDA_VER`, `common-data/toolkits/DEFAULT` names the toolkit (currently
`cu133`, deliberately not the highest defined). A toolkit with no row in `tuned/nccl.pins` stops the
build: pin a `zbrad/nccl` release for it, or set `GPU_TUNED_NCCL_BUILTIN=1` to build NCCL in-tree.

Everything a build makes lives in `tuned/build/<cuda-tag>/` (ignored by git): `venv/` (build tools),
`cmake/` and `ccache/`, `wheel/` (the unstamped wheel), `dist/` (the packaged wheel, its `.sha256`,
`package-info.txt`), `releases/` (test and smoke logs), `build-commit` and `release-label`.

## Identity

The wheel's version is upstream's plus `+<variant>.<cuda-tag>.<YYYYMMDD>.<seq>` (E1), e.g.
`2.15.0+gb10.cu134.20261016.0`; the date is the UTC committer date of the tuned commit and the
sequence (always written, `0` for the first build) the lowest unused among the repo's tags (E3).
The Oct 12 release `…20261012` predates the `.0` rule, is not renamed, and counts as sequence 0. The label is allocated once, in `build.sh`, because it is compiled into `torch.__version__`
(E4); `package.sh` and `release.sh` read it and never recompute it. The release tag is `v<version>`.

## Test scope (recorded decision, G3)

Decided 2026-10-16 by the maintainer: a curated subset plus the ABI and NCCL checks, not upstream's
whole `run_test.py` suite. The modules are listed, with the reasoning, in `tuned/test_modules.txt`;
`tuned/verify_abi.sh` builds a minimal out-of-tree CUDA extension against the wheel;
`tuned/nccl_check.sh` runs a single-rank NCCL all-reduce and broadcast and checks the linked version.
The log header names what is not run. `GPU_TUNED_TEST_MODULES="..."` runs a custom list and writes a
"Partial" summary that `release.sh` rejects.

## Deviations and open items

- **Tests run on the wheel, not an editable install.** `build.sh` makes the wheel with
  `python -m build`; there is no editable torch anywhere. An earlier editable build imported stale
  libraries from the source tree's `torch/lib` when run from the repo root, so every check runs from
  an empty directory against a venv.
- **H7 (JSON build info) is not adopted.** The wheel still carries the older text stamp
  (`.pytorch_build_info` in `libtorch_cuda.so`); `vllm/tuned/build.sh` reads it by that name. Migrate
  with the reader changed first.
- **The stamp's commit is HEAD at packaging**, not the build commit (`package-info.txt` has both).
- **F1/F4:** the shared objects keep their stock names (`libtorch_cuda.so`, `libc10.so`, ...). The
  wheel is told apart by its local version. Renaming them would break out-of-tree extensions; not
  decided.
- **cuDNN:** a smoke run printed "PyTorch was compiled without cuDNN/MIOpen support" for the cu134
  wheel. Not investigated here.
- **rtx40/rtx50** are defined in `common-data` but have never been built from this repo.
