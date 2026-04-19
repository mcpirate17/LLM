from __future__ import annotations

from pathlib import Path

import numpy as np


def bytes_to_int64_tokens(path: Path) -> np.ndarray:
    encoded = path.read_bytes()
    return np.frombuffer(encoded, dtype=np.uint8).astype(np.int64, copy=False)
