# Packaging exception (local, GB10 fork)

This is a scoped exception to CLAUDE.md's Build-section rule ("You should
NEVER run any other command to build PyTorch"): producing a distributable
wheel from an already-built editable tree (e.g. `python -m build --wheel
--no-isolation`) is allowed, since it does not recompile anything if the
editable install is up to date. Before packaging, confirm with the user
that the tree is current (re-run the `pip install -e .` build first if
there's any doubt) — this exception covers packaging only, not a
substitute build path for compiling.
