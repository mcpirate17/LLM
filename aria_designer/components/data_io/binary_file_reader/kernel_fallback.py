"""Python fallback for binary_file_reader."""

from pathlib import Path
import numpy as np
import torch


_DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
    "int32": np.int32,
    "uint8": np.uint8,
}


class ComponentHandler:
    def validate_config(self, config):
        path = str(config.get("file_path", ""))
        return [] if path else ["file_path is required"]

    def build(self, config):
        return None

    def forward(self, inputs, config):
        file_path = Path(str(config.get("file_path", "data.bin")))
        dtype_name = str(config.get("dtype", "float32"))
        dtype = _DTYPE_MAP.get(dtype_name, np.float32)
        offset = int(config.get("offset_bytes", 0))
        shape_str = str(config.get("shape", "1,1,256"))
        shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())

        with file_path.open("rb") as f:
            if offset > 0:
                f.seek(offset)
            raw = f.read()

        arr = np.frombuffer(raw, dtype=dtype)
        if shape:
            target_size = int(np.prod(shape))
            if target_size <= arr.size:
                arr = arr[:target_size].reshape(shape)
        return {"data": torch.from_numpy(arr.copy())}
