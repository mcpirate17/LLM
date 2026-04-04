from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_reference_model_native():
    source = Path(__file__).with_name("_reference_model_native.cpp")
    return load(
        name="eval_reference_model_native_ext",
        sources=[str(source)],
        extra_cflags=["-O3"],
        verbose=False,
    )
