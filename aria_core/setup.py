from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

sources = [
    "src/cpu/kernels.cpp",
    "src/gpu/tropical.cu",
    "src/gpu/clifford.cu",
    "src/cpu/graph_validator.cpp",
    "src/cpu/shape_inference.cpp",
    "src/cpu/clifford.cpp",
    "src/cpu/hyperbolic.cpp",
    "bindings/bindings.cpp",
    "bindings/bind_kernels.cpp",
    "bindings/bind_ops.cpp",
    "bindings/bind_graph.cpp",
]

setup(
    name="aria_core",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="aria_core._C",
            sources=sources,
            include_dirs=[os.path.abspath("include"), os.path.abspath("bindings")],
            extra_compile_args={
                # Host LTO (-flto=auto) inlines across the pybind TUs and the
                # CPU kernel unity build; nvcc objects simply aren't LTO'd.
                # -fno-math-errno: kernels never read errno from libm calls.
                "cxx": [
                    "-O3",
                    "-march=native",
                    "-fopenmp",
                    "-flto=auto",
                    "-fno-math-errno",
                    "-DARIA_HAS_OPENMP",
                ],
                "nvcc": ["-O3", "-DARIA_HAS_OPENMP"],
            },
            extra_link_args=["-lgomp", "-flto=auto"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
