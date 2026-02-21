
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='cpu_ops',
    ext_modules=[
        CppExtension('cpu_ops', ['cpu_ops.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
