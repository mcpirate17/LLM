"""Python fallback kernel for data_transform/join."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch


class ComponentHandler:
    def validate_config(self, config):
        errors = []
        join_mode = str(config.get("join_mode", "inner"))
        if join_mode not in {"inner", "left", "outer", "concat_key"}:
            errors.append("join_mode must be one of: inner, left, outer, concat_key")

        for key_name in ("left_key_index", "right_key_index"):
            key_idx = int(config.get(key_name, 0))
            if key_idx < 0:
                errors.append(f"{key_name} must be >= 0")

        key_tol = float(config.get("key_tolerance", 0.0))
        if key_tol < 0.0:
            errors.append("key_tolerance must be >= 0")

        max_matches = int(config.get("max_matches_per_key", 256))
        if max_matches < 1:
            errors.append("max_matches_per_key must be >= 1")

        schema_mode = str(config.get("schema_validation", "none"))
        if schema_mode not in {"none", "strict"}:
            errors.append("schema_validation must be one of: none, strict")

        exp_left = int(config.get("expected_left_dim", -1))
        exp_right = int(config.get("expected_right_dim", -1))
        if exp_left < -1:
            errors.append("expected_left_dim must be -1 or >= 0")
        if exp_right < -1:
            errors.append("expected_right_dim must be -1 or >= 0")

        return errors

    def build(self, config):
        return None

    def _normalize_rows(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.ndim == 0:
            return x.reshape(1, 1)
        if x.ndim == 1:
            return x.reshape(-1, 1)
        if x.ndim == 2:
            return x
        return x.reshape(-1, x.shape[-1])

    def _remove_column(self, rows: torch.Tensor, col_idx: int) -> torch.Tensor:
        if rows.shape[1] <= 1:
            return rows.new_zeros((rows.shape[0], 0))
        left = rows[:, :col_idx]
        right = rows[:, col_idx + 1 :]
        return torch.cat([left, right], dim=1)

    def _match_mask(self, value: torch.Tensor, keys: torch.Tensor, tol: float) -> torch.Tensor:
        if tol <= 0.0:
            return keys == value
        return (keys - value).abs() <= tol

    def _cat_or_empty(self, rows: List[torch.Tensor], n_features: int, ref: torch.Tensor) -> torch.Tensor:
        if not rows:
            return ref.new_empty((0, n_features))
        return torch.cat(rows, dim=0)

    def _build_row(
        self,
        left_row: torch.Tensor,
        right_row: torch.Tensor,
        include_keys: bool,
        left_key_idx: int,
        right_key_idx: int,
    ) -> torch.Tensor:
        if include_keys:
            return torch.cat([left_row, right_row], dim=0)
        left_payload = self._remove_column(left_row.unsqueeze(0), left_key_idx).squeeze(0)
        right_payload = self._remove_column(right_row.unsqueeze(0), right_key_idx).squeeze(0)
        return torch.cat([left_payload, right_payload], dim=0)

    def forward(self, inputs: Dict[str, torch.Tensor], config: Dict[str, object]):
        errors = self.validate_config(config)
        if errors:
            raise ValueError("Invalid join config: " + "; ".join(errors))

        left_in = inputs.get("left", inputs.get("a", inputs.get("x")))
        right_in = inputs.get("right", inputs.get("b", inputs.get("y")))
        if left_in is None or right_in is None:
            raise ValueError("join requires both 'left' and 'right' inputs")

        left = self._normalize_rows(left_in)
        right = self._normalize_rows(right_in)

        if left.shape[0] == 0 or right.shape[0] == 0:
            # Continue with logic below so left/outer can preserve rows with fill.
            pass

        schema_mode = str(config.get("schema_validation", "none"))
        exp_left = int(config.get("expected_left_dim", -1))
        exp_right = int(config.get("expected_right_dim", -1))
        if schema_mode == "strict":
            if exp_left >= 0 and left.shape[1] != exp_left:
                raise ValueError(f"Schema validation failed: expected_left_dim={exp_left}, got {left.shape[1]}")
            if exp_right >= 0 and right.shape[1] != exp_right:
                raise ValueError(
                    f"Schema validation failed: expected_right_dim={exp_right}, got {right.shape[1]}"
                )

        left_key_idx = min(max(0, int(config.get("left_key_index", 0))), max(0, left.shape[1] - 1))
        right_key_idx = min(max(0, int(config.get("right_key_index", 0))), max(0, right.shape[1] - 1))
        join_mode = str(config.get("join_mode", "inner"))
        key_tol = float(config.get("key_tolerance", 0.0))
        include_keys = bool(config.get("include_key_columns", False))
        fill = float(config.get("missing_fill_value", 0.0))
        max_matches = int(config.get("max_matches_per_key", 256))

        left_keys = left[:, left_key_idx] if left.shape[0] > 0 else left.new_empty((0,))
        right_keys = right[:, right_key_idx] if right.shape[0] > 0 else right.new_empty((0,))

        left_payload_dim = left.shape[1] if include_keys else max(0, left.shape[1] - 1)
        right_payload_dim = right.shape[1] if include_keys else max(0, right.shape[1] - 1)
        out_dim = left_payload_dim + right_payload_dim

        right_fill = right.new_full((right_payload_dim,), fill)
        left_fill = left.new_full((left_payload_dim,), fill)

        rows_out: List[torch.Tensor] = []
        right_used = torch.zeros((right.shape[0],), dtype=torch.bool, device=right.device)

        for li in range(left.shape[0]):
            lrow = left[li]
            mask = self._match_mask(left_keys[li], right_keys, key_tol) if right.shape[0] > 0 else right_used
            match_idx = torch.nonzero(mask, as_tuple=False).flatten()
            if match_idx.numel() > max_matches:
                match_idx = match_idx[:max_matches]

            if match_idx.numel() > 0:
                for ri in match_idx.tolist():
                    rrow = right[ri]
                    out = self._build_row(lrow, rrow, include_keys, left_key_idx, right_key_idx)
                    rows_out.append(out.unsqueeze(0))
                    right_used[ri] = True
            elif join_mode in {"left", "outer", "concat_key"}:
                if include_keys:
                    left_part = lrow
                else:
                    left_part = self._remove_column(lrow.unsqueeze(0), left_key_idx).squeeze(0)
                out = torch.cat([left_part, right_fill], dim=0)
                rows_out.append(out.unsqueeze(0))

        if join_mode in {"outer", "concat_key"}:
            for ri in range(right.shape[0]):
                if right_used[ri]:
                    continue
                rrow = right[ri]
                if include_keys:
                    right_part = rrow
                else:
                    right_part = self._remove_column(rrow.unsqueeze(0), right_key_idx).squeeze(0)
                out = torch.cat([left_fill, right_part], dim=0)
                rows_out.append(out.unsqueeze(0))

        joined = self._cat_or_empty(rows_out, out_dim, left)
        return {"joined": joined}
