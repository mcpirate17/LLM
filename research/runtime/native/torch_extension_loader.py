from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load


def _ensure_ninja_on_path() -> None:
    import ninja

    bin_dir = getattr(ninja, "BIN_DIR", None)
    if not bin_dir:
        return
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in parts:
        os.environ["PATH"] = os.pathsep.join([bin_dir, *parts])


@lru_cache(maxsize=None)
def load_local_cpp_extension(
    module_file: str,
    source_name: str,
    extension_name: str,
):
    _ensure_ninja_on_path()
    source = Path(module_file).with_name(source_name)
    return load(
        name=extension_name,
        sources=[str(source)],
        extra_cflags=["-O3"],
        verbose=False,
    )
