import os
with open('/home/tim/Projects/LLM/aria_core/setup.py', 'r') as f:
    text = f.read()

text = text.replace('from torch.utils.cpp_extension import BuildExtension, CppExtension', 'from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension\nimport torch')

text = text.replace('sources = [\n    \'src/cpu/kernels.cpp\',', '''sources = [
    'src/cpu/kernels.cpp',
    'src/gpu/tropical.cu',
    'src/gpu/clifford.cu',\n''')

text = text.replace('CppExtension(', 'CUDAExtension(')

with open('/home/tim/Projects/LLM/aria_core/setup.py', 'w') as f:
    f.write(text)

