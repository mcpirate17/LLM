"""Python fallback for file_writer."""
from pathlib import Path
import csv
import json
import numpy as np
import torch


class ComponentHandler:
    def validate_config(self, config):
        path = str(config.get("output_path", ""))
        return [] if path else ["output_path is required"]

    def build(self, config):
        return None

    def forward(self, inputs, config):
        if "data" not in inputs:
            raise ValueError("file_writer requires 'data' input")

        out_path = Path(str(config.get("output_path", "output.npy")))
        overwrite = bool(config.get("overwrite", False))
        fmt = str(config.get("file_format", "auto")).lower()
        if fmt == "auto":
            fmt = out_path.suffix.lower().lstrip(".")

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output path exists: {out_path}")

        tensor = inputs["data"]
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.array(tensor)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "csv":
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if bool(config.get("include_shape", True)):
                    writer.writerow(["#shape", *arr.shape])
                if arr.ndim == 1:
                    for x in arr.tolist():
                        writer.writerow([x])
                else:
                    for row in arr.reshape(arr.shape[0], -1).tolist():
                        writer.writerow(row)
        elif fmt == "json":
            payload = {"shape": list(arr.shape), "data": arr.tolist()} if bool(config.get("include_shape", True)) else arr.tolist()
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
        elif fmt == "npy":
            np.save(out_path, arr)
        elif fmt == "pt":
            torch.save(torch.from_numpy(arr), out_path)
        elif fmt == "txt":
            with out_path.open("w", encoding="utf-8") as f:
                if bool(config.get("include_shape", True)):
                    f.write(f"# shape: {list(arr.shape)}\n")
                flat = arr.reshape(-1)
                for v in flat:
                    f.write(f"{float(v)}\n")
        else:
            raise ValueError(f"Unsupported file format: {fmt}")

        return {"status": torch.tensor(1.0)}
