"""Python fallback for file_loader."""
from pathlib import Path
import csv
import json
import numpy as np
import torch


class ComponentHandler:
    def validate_config(self, config):
        path = str(config.get("file_path", ""))
        return [] if path else ["file_path is required"]

    def build(self, config):
        return None

    def forward(self, inputs, config):
        file_path = Path(str(config.get("file_path", "data.csv")))
        fmt = str(config.get("file_format", "auto")).lower()
        if fmt == "auto":
            fmt = file_path.suffix.lower().lstrip(".")

        if fmt == "csv":
            delim = str(config.get("delimiter", ","))
            has_header = bool(config.get("has_header", True))
            with file_path.open("r", newline="", encoding=str(config.get("text_encoding", "utf-8"))) as f:
                reader = csv.reader(f, delimiter=delim)
                rows = list(reader)
                if has_header and rows:
                    rows = rows[1:]
            arr = np.array(rows, dtype=np.float32)
            return {"data": torch.from_numpy(arr)}

        if fmt == "json":
            with file_path.open("r", encoding=str(config.get("text_encoding", "utf-8"))) as f:
                payload = json.load(f)
            arr = np.array(payload, dtype=np.float32)
            return {"data": torch.from_numpy(arr)}

        if fmt == "npy":
            arr = np.load(file_path)
            return {"data": torch.from_numpy(arr)}

        if fmt == "pt":
            tensor = torch.load(file_path, map_location="cpu")
            if isinstance(tensor, torch.Tensor):
                return {"data": tensor}
            raise ValueError("PT file must contain a tensor")

        if fmt == "txt":
            with file_path.open("r", encoding=str(config.get("text_encoding", "utf-8"))) as f:
                lines = [line.strip() for line in f if line.strip()]
            arr = np.array([float(x) for x in lines], dtype=np.float32)
            return {"data": torch.from_numpy(arr)}

        raise ValueError(f"Unsupported file format: {fmt}")
