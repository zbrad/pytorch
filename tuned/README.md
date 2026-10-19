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
CUDA_TAG=cu134 bash tuned/release.sh gb10      # every gate reported together; on pass writes RELEASE_READY_gb10 (publishes nothing)
CUDA_TAG=cu134 bash tuned/publish.sh gb10      # publish exactly what release.sh's gates saw (refuses if HEAD, notes or assets changed)
```

One of `CUDA_TAG` or `CUDA_VER` is required: there is no default toolkit. With neither, the scripts
fail and list the defined toolkits (from a terminal they ask which to use, for that run only). A toolkit with no row in `tuned/nccl.pins` stops the
build: pin a `zbrad/nccl` release for it, or set `GPU_TUNED_NCCL_BUILTIN=1` to build NCCL in-tree.

Everything a build makes lives in `tuned/build/<cuda-tag>/` (ignored by git): `venv/` (build tools),
`cmake/` and `ccache/`, `wheel/` (the unstamped wheel), `dist/` (the packaged wheel, its `.sha256`,
`package-info.txt`), `releases/` (test and smoke logs, and `RELEASE_READY_<variant>` once release.sh passes), `build-commit` and `release-label`.

## Identity

The wheel's version is upstream's plus `+<variant>.<cuda-tag>.<YYYYMMDD>.<seq>` (E1), e.g.
`2.15.0+gb10.cu134.20261016.0`; the date is the UTC committer date of the tuned commit and the
sequence (always written, `0` for the first build) the lowest unused among the repo's tags (E3).
The Oct 12 release `…20261012` predates the `.0` rule, is not renamed, and counts as sequence 0. The label is allocated once, in `build.sh`, because it is compiled into `torch.__version__`
(E4); `package.sh` and `release.sh` read it and never recompute it. The release tag is `v<version>`.

## Test scope (recorded decision, G3)

Decided 2026-10-16 by the maintainer: a curated subset plus the ABI and NCCL checks, not upstream's
whole `run_test.py` suite. Three files define it, and `full_test.sh` prints all of them into the log header:

- `tuned/test_modules.txt`: the modules of the release scope, with the reasoning.
- `tuned/test_skips.txt`: tests that are not run, each with its reason. *Not needed*: none of the tuned
  devices is multi-GPU, so multi-GPU modules are out (`distributed/test_c10d_nccl`: 284 of 297 tests
  skipped in 14.5 min on one GPU). *Not safe*: an OOM test that ran 25+ min on the GB10's unified memory.
  *Known failures*: tests that fail identically on the previous wheel (no cuDNN in the build, triton not
  installed, one numerics tolerance, two undiagnosed); each is a to-do and goes when its cause is fixed.
- `tuned/test_mini.txt`: the mini tier, a few named tests from each area.

Tuned builds are single-GPU, so "full" excludes multi-GPU tests two ways: whole multi-GPU modules are
listed in `test_skips.txt`, and `full_test.sh` pins the tests to one device (`CUDA_VISIBLE_DEVICES=0`, or
`GPU_TUNED_TEST_GPUS`), which makes the multi-GPU tests inside the other modules skip themselves. The
log header states the device setting.

`GPU_TUNED_TEST_SCOPE=mini CUDA_TAG=cu134 bash tuned/full_test.sh gb10` runs the mini tier plus
`verify_abi.sh` and `nccl_check.sh` in about 5 minutes (measured on the GB10, cu134) and ends
"Mini test suite summary". The default (release) scope ends "Full test suite summary"; only that is
accepted by `package.sh` and `release.sh`. `GPU_TUNED_TEST_MODULES="..."` runs a custom list and writes a
"Partial" summary, also rejected. `tuned/verify_abi.sh` builds a minimal out-of-tree CUDA extension
against the wheel; `tuned/nccl_check.sh` runs a single-rank NCCL all-reduce and broadcast and checks the
linked version.

## Deviations and open items

- **Tests run on the wheel, not an editable install.** `build.sh` makes the wheel with
  `python -m build`; there is no editable torch anywhere. An earlier editable build imported stale
  libraries from the source tree's `torch/lib` when run from the repo root, so every check runs from
  an empty directory against a venv.
- **H7 (JSON build info) is not adopted.** The wheel still carries the older text stamp
  (`.pytorch_build_info` in `libtorch_cuda.so`); `vllm/tuned/build.sh` reads it by that name. Migrate
  with the reader changed first.
- **The stamp's commit is HEAD at packaging**, not the build commit (`package-info.txt` has both).
- **F4 (decided 2026-10-16, not implemented):** the wheel's shared objects (`libtorch_cuda.so`,
  `libc10.so`, ...) still ship the stock names. They are to take the generic `-tuned` suffix, with the
  variant and toolkit staying in the version; every consumer that links torch by name is rebuilt. The
  list is in `~/.claude/cross-project-todo.md`. Until then the wheel is told apart by its local version,
  and `smoke_test.sh` fails if a `libtorch*`/`libc10*` is mapped from outside the venv.
- **cuDNN:** a smoke run printed "PyTorch was compiled without cuDNN/MIOpen support" for the cu134
  wheel. Not investigated here.
- **rtx40/rtx50** are defined in `common-data` but have never been built from this repo.
