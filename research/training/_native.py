from __future__ import annotations

from ..runtime.native.torch_extension_loader import load_local_cpp_extension


def load_training_native():
    """Single merged extension: data tokenization, loss kernels, curriculum."""
    return load_local_cpp_extension(
        __file__,
        "_training_native.cpp",
        "research_training_native_v1",
    )
