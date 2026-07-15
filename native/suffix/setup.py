from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension("winllm_suffix", ["winllm_suffix.cpp"], cxx_std=17),
    ],
    cmdclass={"build_ext": build_ext},
)
