from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

designer_src = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "aria_designer",
        "runtime",
        "src",
    )
)
aria_core_include = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "aria_core", "include"
    )
)
native_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
native_include = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "include")
)
native_build = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
native_lib = os.path.join(native_build, "libaria_native_runtime.so")

extensions = [
    Extension(
        "aria_bridge",
        sources=["aria_bridge.pyx"],
        include_dirs=[
            np.get_include(),
            designer_src,
            aria_core_include,
            native_include,
            native_src,
        ],
        extra_objects=[native_lib],
        extra_compile_args=["-O3", "-march=native", "-fPIC"],
        extra_link_args=[
            f"-Wl,-rpath,{native_build}",
        ],
        language="c",
    ),
]

setup(
    name="aria_bridge",
    ext_modules=cythonize(extensions, language_level="3"),
)
