from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load


@lru_cache(maxsize=None)
def load_local_cpp_extension(
    module_file: str,
    source_name: str,
    extension_name: str,
):
    source = Path(module_file).with_name(source_name)
    return load(
        name=extension_name,
        sources=[str(source)],
        extra_cflags=["-O3"],
        verbose=False,
    )
