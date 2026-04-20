from __future__ import annotations

from ..runtime.native.torch_extension_loader import load_local_cpp_extension


def load_data_native():
    return load_local_cpp_extension(__file__, "_data_native.cpp", "data_native_ext_v3")
