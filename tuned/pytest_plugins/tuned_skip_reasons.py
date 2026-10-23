"""Record why each test was skipped, for the "skipped by reason" table in the tuned test log.

tuned/full_test.sh loads this with `-p tuned_skip_reasons` and points GPU_TUNED_SKIP_LOG at a file.
run_test.py appends its own `-rfEX` after any extra argument, so `-rs` cannot be added from outside;
a plugin is not overridden. One line per skipped test: <module>\\t<reason>. Without the variable the
plugin does nothing.
"""

from os import environ
from typing import Any, List, Optional

from _pytest.config import Config


class TunedSkipReasons:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lines: List[str] = []

    def pytest_runtest_logreport(self, report: Any) -> None:
        if not report.skipped:
            return
        longrepr = report.longrepr
        reason = longrepr[2] if isinstance(longrepr, tuple) and len(longrepr) == 3 else str(longrepr)
        module = report.nodeid.split("::", 1)[0]
        self._lines.append(f"{module}\t{' '.join(reason.split())}\n")

    def pytest_sessionfinish(self) -> None:
        if self._lines:
            with open(self._path, "a") as handle:
                handle.writelines(self._lines)


def pytest_configure(config: Config) -> None:
    path: Optional[str] = environ.get("GPU_TUNED_SKIP_LOG")
    if path:
        config.pluginmanager.register(TunedSkipReasons(path), "tuned_skip_reasons_state")
