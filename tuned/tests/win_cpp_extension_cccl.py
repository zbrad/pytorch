"""Windows check: torch.utils.cpp_extension builds a CUDA extension that includes CCCL.

CUDA 13's CCCL headers stop with C1189 ("MSVC/cl.exe with traditional preprocessor is
used") unless nvcc gives cl /Zc:preprocessor. This builds win_cpp_extension_cccl.cu
through load() (JIT) and through BuildExtension with ninja, then runs it on the GPU.

Usage, from a shell with MSVC and the CUDA toolkit active (vcvars64 + CUDA_PATH):
    python tuned/tests/win_cpp_extension_cccl.py [--cpp-extension PATH] WORK_DIR
WORK_DIR must not exist. --cpp-extension substitutes a source-tree cpp_extension.py
for the installed one, to test a change without rebuilding torch. Exit 0 = both pass.
"""

from argparse import ArgumentParser
from importlib.util import module_from_spec, spec_from_file_location
from logging import INFO, basicConfig, getLogger
from os import environ, makedirs
from os.path import abspath, dirname, exists, join
from subprocess import call
from sys import executable, exit as sys_exit, modules, path as sys_path
from types import ModuleType
from typing import Optional

logger = getLogger(__name__)

HERE = dirname(abspath(__file__))
KERNEL = join(HERE, "win_cpp_extension_cccl.cu")

SETUP_PY = """
import sys
from importlib.util import module_from_spec, spec_from_file_location
from setuptools import setup
import torch.utils.cpp_extension
src = {src!r}
if src:
    spec = spec_from_file_location("torch.utils.cpp_extension", src)
    ext = module_from_spec(spec)
    sys.modules["torch.utils.cpp_extension"] = ext
    spec.loader.exec_module(ext)
else:
    ext = torch.utils.cpp_extension
setup(
    name="cccl_ninja",
    ext_modules=[ext.CUDAExtension("cccl_ninja", [{kernel!r}])],
    cmdclass={{"build_ext": ext.BuildExtension.with_options(use_ninja=True)}},
)
"""


class win_cpp_extension_cccl:
    """Builds and runs the CCCL test extension through the JIT and ninja paths."""

    def __init__(self, work: str, cpp_extension: Optional[str]) -> None:
        self.work = work
        self.cpp_extension = cpp_extension

    def load_cpp_extension(self) -> ModuleType:
        """Return torch.utils.cpp_extension, or the given file installed in its place."""
        if self.cpp_extension is None:
            from torch.utils import cpp_extension

            return cpp_extension
        spec = spec_from_file_location("torch.utils.cpp_extension", self.cpp_extension)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load {self.cpp_extension}")
        ext = module_from_spec(spec)
        modules["torch.utils.cpp_extension"] = ext
        spec.loader.exec_module(ext)
        return ext

    def check_add_one(self, mod: ModuleType, label: str) -> bool:
        """Run the extension's add_one on the GPU and compare with torch."""
        from torch import arange, equal, float32

        x = arange(8, dtype=float32, device="cuda")
        ok = bool(equal(mod.add_one(x).cpu(), (x + 1).cpu()))
        logger.info("%s: %s", label, "PASS" if ok else "FAIL (wrong result)")
        return ok

    def run_jit(self) -> bool:
        """Build with load() and run it."""
        ext = self.load_cpp_extension()
        environ["TORCH_EXTENSIONS_DIR"] = join(self.work, "jit")
        try:
            mod = ext.load(name="cccl_jit", sources=[KERNEL], verbose=True)
        except Exception as e:
            logger.error("jit: FAIL (build): %s", e)
            return False
        return self.check_add_one(mod, "jit")

    def run_ninja(self) -> bool:
        """Build with setup.py build_ext (BuildExtension, ninja) and run it."""
        work = join(self.work, "ninja")
        makedirs(work)
        setup_py = join(work, "setup.py")
        with open(setup_py, "w") as f:
            f.write(SETUP_PY.format(src=self.cpp_extension or "", kernel=KERNEL))
        rc = call([executable, setup_py, "build_ext", "--inplace", "-v"], cwd=work)
        if rc:
            logger.error("ninja: FAIL (build rc=%d)", rc)
            return False
        sys_path.insert(0, work)
        return self.check_add_one(__import__("cccl_ninja"), "ninja")

    def run(self) -> int:
        """Run both paths; return the process exit code."""
        if exists(self.work):
            raise ValueError(f"{self.work} exists")
        makedirs(self.work)
        results = [self.run_jit(), self.run_ninja()]
        return 0 if all(results) else 1


if __name__ == "__main__":
    basicConfig(level=INFO, format="%(message)s")
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("work")
    parser.add_argument("--cpp-extension", default=None)
    args = parser.parse_args()
    sys_exit(win_cpp_extension_cccl(abspath(args.work), args.cpp_extension).run())
