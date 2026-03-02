from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

sources = [
    'src/cpu/kernels.cpp',
    'src/cpu/graph_validator.cpp',
    'src/cpu/shape_inference.cpp',
    'src/cpu/clifford.cpp',
    'src/cpu/hyperbolic.cpp',
    'bindings/bindings.cpp',
]

setup(
    name='aria_core',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='aria_core._C',
            sources=sources,
            include_dirs=[os.path.abspath('include')],
            extra_compile_args=['-O3', '-march=native', '-fopenmp', '-DARIA_HAS_OPENMP'],
            extra_link_args=['-lgomp'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
