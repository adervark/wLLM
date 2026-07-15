"""Build for the fused CUDA sampling extension.

Unlike native/suffix (pure C++), this module contains device code, so .cu
sources are compiled by nvcc into object files and handed to the normal
MSVC link that setuptools/pybind11 already do for the .cpp binding. The
extension is deliberately torch-free (raw pointers in, see the .cu header
comment), so it does NOT use torch.utils.cpp_extension — which would refuse
the CUDA 13.2 toolkit against a cu12.8 torch anyway. cudart is linked
statically so the .pyd has no CUDA DLL dependency beyond the driver.
"""

import os
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _cuda_home() -> Path:
    for var in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(var):
            return Path(os.environ[var])
    root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if root.is_dir():
        versions = sorted(root.glob("v*"), reverse=True)
        if versions:
            return versions[0]
    raise RuntimeError("CUDA toolkit not found: set CUDA_PATH")


# sm_89 = Ada (RTX 40xx); compute_89 PTX keeps newer GPUs working via JIT.
NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-gencode", "arch=compute_89,code=sm_89",
    "-gencode", "arch=compute_89,code=compute_89",
    "-Xcompiler", "/MD",
]


class cuda_build_ext(build_ext):
    def _msvc_bin(self) -> str | None:
        """Directory containing cl.exe — MSVC is installed but not on PATH,
        and nvcc needs -ccbin to find it."""
        self.compiler.initialize()
        cl = getattr(self.compiler, "cc", None)
        return str(Path(cl).parent) if cl else None

    def build_extensions(self):
        cuda = _cuda_home()
        nvcc = cuda / "bin" / "nvcc.exe"
        ccbin = self._msvc_bin()
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        for ext in self.extensions:
            cu_sources = [s for s in ext.sources if s.endswith(".cu")]
            ext.sources = [s for s in ext.sources if not s.endswith(".cu")]
            for cu in cu_sources:
                obj = build_temp / (Path(cu).stem + ".cu.obj")
                cmd = [str(nvcc), "-c", cu, "-o", str(obj), *NVCC_FLAGS]
                if ccbin:
                    cmd += ["-ccbin", ccbin]
                print(" ".join(cmd))
                subprocess.check_call(cmd)
                ext.extra_objects.append(str(obj))
            ext.include_dirs.append(str(cuda / "include"))
            ext.library_dirs.append(str(cuda / "lib" / "x64"))
            ext.libraries.append("cudart_static")
        super().build_extensions()


setup(
    ext_modules=[
        Pybind11Extension(
            "winllm_sampling",
            ["binding.cpp", "winllm_sampling.cu"],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": cuda_build_ext},
)
