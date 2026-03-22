"""Python fallback kernel for split_train_val_test."""

import torch


class ComponentHandler:
    def validate_config(self, config):
        errors = []

        train_ratio = float(config.get("train_ratio", 0.8))
        val_ratio = float(config.get("val_ratio", 0.1))
        test_ratio = float(config.get("test_ratio", 0.1))

        for name, value in (
            ("train_ratio", train_ratio),
            ("val_ratio", val_ratio),
            ("test_ratio", test_ratio),
        ):
            if value < 0.0 or value > 1.0:
                errors.append(f"{name} must be in [0, 1]")

        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum <= 0.0:
            errors.append("train/val/test ratios must sum to > 0")

        if abs(ratio_sum - 1.0) > 1e-6:
            errors.append("train_ratio + val_ratio + test_ratio must equal 1.0")

        stratify_col = int(config.get("stratify_col", 0))
        if stratify_col < 0:
            errors.append("stratify_col must be >= 0")

        stratify_bins = int(config.get("stratify_bins", 8))
        if stratify_bins < 2:
            errors.append("stratify_bins must be >= 2")

        schema_validation = str(config.get("schema_validation", "none"))
        if schema_validation not in {"none", "strict"}:
            errors.append("schema_validation must be one of: none, strict")

        expected_feature_dim = int(config.get("expected_feature_dim", -1))
        if expected_feature_dim < -1:
            errors.append("expected_feature_dim must be -1 or >= 0")

        return errors

    def build(self, config):
        return None

    def _normalize_rows(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        if data.ndim == 0:
            data = data.reshape(1, 1)
        elif data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            data = data.reshape(-1, data.shape[-1])
        return data

    def _split_counts(self, n_rows, train_ratio, val_ratio):
        train_n = int(round(n_rows * train_ratio))
        val_n = int(round(n_rows * val_ratio))

        train_n = min(max(train_n, 0), n_rows)
        remaining = n_rows - train_n
        val_n = min(max(val_n, 0), remaining)
        test_n = n_rows - train_n - val_n
        return train_n, val_n, test_n

    def _contiguous_split(self, indices, train_ratio, val_ratio):
        n_rows = int(indices.numel())
        train_n, val_n, _ = self._split_counts(n_rows, train_ratio, val_ratio)
        train_idx = indices[:train_n]
        val_idx = indices[train_n : train_n + val_n]
        test_idx = indices[train_n + val_n :]
        return train_idx, val_idx, test_idx

    def _stratified_indices(self, rows, config, generator):
        stratify_col = int(config.get("stratify_col", 0))
        stratify_bins = int(config.get("stratify_bins", 8))

        n_features = rows.shape[1] if rows.ndim > 1 else 1
        stratify_col = min(max(0, stratify_col), max(0, n_features - 1))

        values = rows[:, stratify_col].detach().float()
        unique_values = torch.unique(values)

        if unique_values.numel() <= stratify_bins:
            labels = values
        else:
            quantiles = torch.linspace(
                0.0, 1.0, stratify_bins + 1, device=values.device
            )
            boundaries = torch.quantile(values, quantiles)
            boundaries = torch.unique(boundaries)
            if boundaries.numel() < 2:
                labels = torch.zeros_like(values)
            else:
                labels = torch.bucketize(values, boundaries[1:-1], right=False)

        labels = labels.to(torch.int64)
        classes = torch.unique(labels)
        return classes, labels

    def forward(self, inputs, config):
        errors = self.validate_config(config)
        if errors:
            raise ValueError(
                "Invalid split_train_val_test config: " + "; ".join(errors)
            )

        data = inputs.get("data", inputs.get("x"))
        if data is None:
            raise ValueError("split_train_val_test requires input 'data'")

        rows = self._normalize_rows(data)

        expected_feature_dim = int(config.get("expected_feature_dim", -1))
        schema_validation = str(config.get("schema_validation", "none"))
        if (
            schema_validation == "strict"
            and expected_feature_dim >= 0
            and rows.ndim == 2
            and rows.shape[1] != expected_feature_dim
        ):
            raise ValueError(
                f"Schema validation failed: expected feature dim {expected_feature_dim}, got {rows.shape[1]}"
            )

        n_rows = rows.shape[0]
        if n_rows == 0:
            empty = rows
            return {"train": empty, "val": empty, "test": empty}

        train_ratio = float(config.get("train_ratio", 0.8))
        val_ratio = float(config.get("val_ratio", 0.1))
        shuffle = bool(config.get("shuffle", True))
        seed = int(config.get("seed", 42))
        stratify = bool(config.get("stratify", False))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        if stratify and n_rows >= 3:
            classes, labels = self._stratified_indices(rows, config, generator)
            train_parts = []
            val_parts = []
            test_parts = []

            for cls in classes:
                cls_idx = torch.nonzero(labels == cls, as_tuple=False).flatten()
                if shuffle and cls_idx.numel() > 1:
                    perm = torch.randperm(cls_idx.numel(), generator=generator)
                    cls_idx = cls_idx[perm]
                t_idx, v_idx, s_idx = self._contiguous_split(
                    cls_idx, train_ratio, val_ratio
                )
                train_parts.append(t_idx)
                val_parts.append(v_idx)
                test_parts.append(s_idx)

            train_idx = (
                torch.cat(train_parts)
                if train_parts
                else torch.empty(0, dtype=torch.long)
            )
            val_idx = (
                torch.cat(val_parts) if val_parts else torch.empty(0, dtype=torch.long)
            )
            test_idx = (
                torch.cat(test_parts)
                if test_parts
                else torch.empty(0, dtype=torch.long)
            )

            if shuffle:
                if train_idx.numel() > 1:
                    train_idx = train_idx[
                        torch.randperm(train_idx.numel(), generator=generator)
                    ]
                if val_idx.numel() > 1:
                    val_idx = val_idx[
                        torch.randperm(val_idx.numel(), generator=generator)
                    ]
                if test_idx.numel() > 1:
                    test_idx = test_idx[
                        torch.randperm(test_idx.numel(), generator=generator)
                    ]
        else:
            indices = torch.arange(n_rows, dtype=torch.long)
            if shuffle and n_rows > 1:
                indices = indices[torch.randperm(n_rows, generator=generator)]
            train_idx, val_idx, test_idx = self._contiguous_split(
                indices, train_ratio, val_ratio
            )

        return {
            "train": rows.index_select(0, train_idx),
            "val": rows.index_select(0, val_idx),
            "test": rows.index_select(0, test_idx),
        }
