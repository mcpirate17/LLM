from __future__ import annotations

from ..runtime.native.torch_extension_loader import load_local_cpp_extension


def load_loss_native():
    return load_local_cpp_extension(
        __file__,
        "_loss_native.cpp",
        "training_native_ext_v1",
    )
