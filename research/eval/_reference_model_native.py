from __future__ import annotations

from ..runtime.native.torch_extension_loader import load_local_cpp_extension


def load_reference_model_native():
    return load_local_cpp_extension(
        __file__,
        "_reference_model_native.cpp",
        "eval_reference_model_native_ext",
    )
