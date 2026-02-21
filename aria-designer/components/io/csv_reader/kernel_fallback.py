import torch
import csv
import numpy as np

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        filepath = config.get("filepath", "data.csv")
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f, delimiter=config.get("delimiter", ","))
                rows = list(reader)
                if config.get("has_header", True):
                    rows = rows[1:]
                
                # Convert to float tensor if possible
                data = np.array(rows, dtype=np.float32)
                return {"data": torch.from_numpy(data)}
        except Exception as e:
            # Return empty or error
            print(f"CSV read failed: {e}")
            return {"data": torch.empty(0)}
