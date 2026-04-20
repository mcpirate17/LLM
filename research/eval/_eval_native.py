from __future__ import annotations

from ..runtime.native.torch_extension_loader import load_local_cpp_extension


def load_eval_native():
    return load_local_cpp_extension(__file__, "_eval_native.cpp", "eval_native_ext_v6")
