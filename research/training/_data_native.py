from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_data_native():
    source = Path(__file__).with_name("_data_native.cpp")
    return load(
        name="data_native_ext_v3",
        sources=[str(source)],
        extra_cflags=["-O3"],
        verbose=False,
    )
