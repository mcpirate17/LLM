from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(Extension("adaptive_sampler", ["adaptive_sampler.pyx"])),
    include_dirs=[numpy.get_include()]
)
