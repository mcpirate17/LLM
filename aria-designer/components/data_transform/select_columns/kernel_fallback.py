"""Python fallback kernel for select_columns."""

import torch


class ComponentHandler:
    def _split_csv(self, raw):
        return [part.strip() for part in str(raw or "").split(",") if part.strip()]

    def _parse_indices(self, raw):
        parsed = []
        for token in self._split_csv(raw):
            try:
                parsed.append(int(token))
            except ValueError:
                continue
        return parsed

    def validate_config(self, config):
        errors = []
        mode = str(config.get("selection_mode", "names"))
        if mode not in {"names", "indices"}:
            errors.append("selection_mode must be one of: names, indices")

        strict = str(config.get("schema_validation", "none"))
        if strict not in {"none", "strict"}:
            errors.append("schema_validation must be one of: none, strict")

        if mode == "indices":
            tokens = self._split_csv(config.get("selected_indices", ""))
            if tokens and len(self._parse_indices(config.get("selected_indices", ""))) != len(tokens):
                errors.append("selected_indices must be comma-separated integers")

        return errors

    def build(self, config):
        return None

    def _normalize_rows(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        if data.ndim == 0:
            return data.reshape(1, 1)
        if data.ndim == 1:
            return data.reshape(-1, 1)
        if data.ndim > 2:
            return data.reshape(-1, data.shape[-1])
        return data

    def _choose_indices(self, rows, config):
        mode = str(config.get("selection_mode", "names"))
        strict = str(config.get("schema_validation", "none")) == "strict"
        keep_order = bool(config.get("keep_order", True))
        drop_invalid = bool(config.get("drop_invalid", True))

        if mode == "names":
            schema_cols = self._split_csv(config.get("schema_columns", ""))
            selected_names = self._split_csv(config.get("selected_columns", ""))

            if not selected_names:
                return list(range(rows.shape[1]))

            name_to_idx = {name: idx for idx, name in enumerate(schema_cols)}
            missing = [name for name in selected_names if name not in name_to_idx]
            if missing and strict and not drop_invalid:
                raise ValueError(f"Unknown selected_columns: {missing}")

            picked = [name_to_idx[name] for name in selected_names if name in name_to_idx]
            if strict and not schema_cols:
                raise ValueError("schema_columns is required for strict name-based selection")
        else:
            picked = self._parse_indices(config.get("selected_indices", ""))
            if not picked:
                return list(range(rows.shape[1]))

        width = int(rows.shape[1])
        invalid = [idx for idx in picked if idx < 0 or idx >= width]
        if invalid and strict and not drop_invalid:
            raise ValueError(f"selected column index out of range: {invalid}")

        picked = [idx for idx in picked if 0 <= idx < width]

        if keep_order:
            deduped = []
            seen = set()
            for idx in picked:
                if idx in seen:
                    continue
                seen.add(idx)
                deduped.append(idx)
            return deduped

        return sorted(set(picked))

    def forward(self, inputs, config):
        errors = self.validate_config(config)
        if errors:
            raise ValueError("Invalid select_columns config: " + "; ".join(errors))

        data = inputs.get("data", inputs.get("x"))
        if data is None:
            raise ValueError("select_columns requires input 'data'")

        rows = self._normalize_rows(data)
        indices = self._choose_indices(rows, config)

        if not indices:
            empty = rows[:, :0]
            return {"selected": empty}

        idx_tensor = torch.tensor(indices, dtype=torch.long, device=rows.device)
        return {"selected": rows.index_select(1, idx_tensor)}
