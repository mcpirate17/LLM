from __future__ import annotations

from ..runtime.native.torch_extension_loader import load_local_cpp_extension


def load_curriculum_native():
    return load_local_cpp_extension(
        __file__,
        "_curriculum_native.cpp",
        "training_curriculum_ext_v1",
    )
