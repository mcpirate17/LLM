"""Python fallback kernel for filter."""
import torch


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def _compare(self, values, op, threshold):
        if op == ">":
            return values > threshold
        if op == "<":
            return values < threshold
        if op == ">=":
            return values >= threshold
        if op == "<=":
            return values <= threshold
        if op == "==":
            return values == threshold
        if op == "!=":
            return values != threshold
        return torch.ones_like(values, dtype=torch.bool)

    def forward(self, inputs, config):
        data = inputs["data"]
        scope = str(config.get("filter_scope", "row"))
        col = int(config.get("col_index", 0))
        val = float(config.get("value", 0.0))
        op = str(config.get("operator", ">"))

        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)

        # row scope: rank-2 row filtering using one feature column
        if scope == "row" and data.ndim == 2:
            col = max(0, min(data.shape[1] - 1, col))
            mask = self._compare(data[:, col], op, val)
            return {"filtered": data[mask]}

        # token scope: rank-3, score each token by feature mean, then filter sequence axis
        if scope == "token" and data.ndim >= 3:
            token_scores = data.mean(dim=-1)
            token_mask = self._compare(token_scores, op, val)
            # Keep tokens where any batch member passes.
            keep = token_mask.any(dim=0)
            return {"filtered": data[:, keep, ...]}

        # feature scope: rank-3 or rank-2, score each feature by mean over batch/sequence
        if scope == "feature" and data.ndim >= 2:
            reduce_dims = tuple(range(data.ndim - 1))
            feat_scores = data.mean(dim=reduce_dims)
            feat_mask = self._compare(feat_scores, op, val)
            return {"filtered": data[..., feat_mask]}

        # Fallback for unsupported rank/scope combos
        return {"filtered": data}
