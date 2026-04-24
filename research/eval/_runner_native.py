from __future__ import annotations

from ..runtime.native.torch_extension_loader import load_local_cpp_extension


def load_runner_native():
    return load_local_cpp_extension(
        __file__,
        "_runner_native.cpp",
        "eval_runner_native_ext_v9",
    )
