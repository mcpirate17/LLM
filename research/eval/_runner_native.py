from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_runner_native():
    source = Path(__file__).with_name("_runner_native.cpp")
    return load(
        name="eval_runner_native_ext_v4",
        sources=[str(source)],
        extra_cflags=["-O3"],
        verbose=False,
    )
