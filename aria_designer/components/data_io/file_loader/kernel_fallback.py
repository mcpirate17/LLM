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

    @staticmethod
    def _load_csv_array(
        file_path: Path, *, delimiter: str, has_header: bool, encoding: str
    ):
        with file_path.open("r", newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter=delimiter)
            if has_header:
                next(reader, None)
            first_row = next(reader, None)
            if first_row is None:
                return np.array([], dtype=np.float32)
            width = len(first_row)
            if width == 0:
                row_count = 1 + sum(1 for _ in reader)
                return np.empty((row_count, 0), dtype=np.float32)

            def iter_values():
                yield from (float(cell) for cell in first_row)
                for row in reader:
                    if len(row) != width:
                        raise ValueError("CSV rows must all have the same width")
                    yield from (float(cell) for cell in row)

            values = np.fromiter(iter_values(), dtype=np.float32)
            return values.reshape(-1, width)

    def forward(self, inputs, config):
        file_path = Path(str(config.get("file_path", "data.csv")))
        fmt = str(config.get("file_format", "auto")).lower()
        if fmt == "auto":
            fmt = file_path.suffix.lower().lstrip(".")

        if fmt == "csv":
            delim = str(config.get("delimiter", ","))
            has_header = bool(config.get("has_header", True))
            arr = self._load_csv_array(
                file_path,
                delimiter=delim,
                has_header=has_header,
                encoding=str(config.get("text_encoding", "utf-8")),
            )
            return {"data": torch.from_numpy(arr)}

        if fmt == "json":
            with file_path.open(
                "r", encoding=str(config.get("text_encoding", "utf-8"))
            ) as f:
                payload = json.load(f)
            arr = np.asarray(payload, dtype=np.float32)
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
            with file_path.open(
                "r", encoding=str(config.get("text_encoding", "utf-8"))
            ) as f:
                arr = np.fromiter(
                    (float(line.strip()) for line in f if line.strip()),
                    dtype=np.float32,
                )
            return {"data": torch.from_numpy(arr)}

        raise ValueError(f"Unsupported file format: {fmt}")
