# Packaging exception (local, GB10 fork)

This is a scoped exception to CLAUDE.md's Build-section rule ("You should
NEVER run any other command to build PyTorch"): producing a distributable
wheel from an already-built editable tree (e.g. `python -m build --wheel
--no-isolation`) is allowed, since it does not recompile anything if the
editable install is up to date. Before packaging, confirm with the user
that the tree is current (re-run the `pip install -e .` build first if
there's any doubt) — this exception covers packaging only, not a
substitute build path for compiling.

**Superseded 2026-07:** the untracked `agent_space/build_gb10.sh` this note
originally referenced no longer exists (it was never committed —
`agent_space/` is gitignored — and got lost between sessions). The real
build recipe now lives in `tuned/` (tracked, on the `tuned-builds` branch,
one variant per `tuned/devices/{gb10,rtx40,rtx50}.conf`):

- `bash tuned/build.sh <variant>` — the compile step (wraps the same
  `pip install --no-build-isolation -v -e .` CLAUDE.md already mandates,
  with `TORCH_CUDA_ARCH_LIST`/`USE_CUDA`/`USE_KLEIDIAI_SME` set per
  variant).
- `bash tuned/wheel.sh <variant>` — the packaging step this note is about.
  It computes and exports `PYTORCH_BUILD_VERSION`/`PYTORCH_BUILD_NUMBER`
  itself before calling `python -m build --wheel --no-isolation`, so the
  "packaging silently regenerates version.py with an auto-generated
  version" failure mode this note originally warned about (2026-07-06) no
  longer requires a separately-sourced, untracked env file to avoid.

If you're packaging a tree built by hand (not via `tuned/build.sh`), the
scoped exception below still applies, but set `PYTORCH_BUILD_VERSION`
yourself first -- see `tuned/wheel.sh` for the exact derivation
(`version.txt`'s base version + `.dev<date>+git<sha>.<variant>.cu<compact>`).
