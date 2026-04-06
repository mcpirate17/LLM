"""Python fallback kernel for hybrid_sparse_router."""

import torch
import torch.nn.functional as F

try:
    from research.scientist.native.intelligent_router import NativeSparseHybridRouter
except Exception:  # pragma: no cover - optional runtime dependency
    NativeSparseHybridRouter = None


class ComponentHandler:
    def __init__(self):
        self._router = None
        self._lane_weights = None
        self._native_router = None
        self._native_signature = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def _ensure(self, x, lane_count):
        d = x.shape[-1]
        if self._router is None or self._router.shape != (d, lane_count):
            self._router = (
                torch.randn(d, lane_count, device=x.device, dtype=x.dtype) * 0.02
            )
            self._lane_weights = [
                torch.randn(d, d, device=x.device, dtype=x.dtype) * (d**-0.5)
                for _ in range(lane_count)
            ]
            if self._native_router is not None:
                self._native_router.close()
                self._native_router = None
                self._native_signature = None

    def _token_ids_from_tensor(self, x):
        return (
            x.detach()
            .to(dtype=torch.float32)
            .abs()
            .argmax(dim=-1)
            .to(dtype=torch.int64)
        )

    def _ensure_native_router(self, token_ids, lane_count):
        if NativeSparseHybridRouter is None:
            return None
        vocab = max(int(token_ids.max().item()) + 1, 16)
        signature = (vocab, lane_count)
        if self._native_router is None or self._native_signature != signature:
            if self._native_router is not None:
                self._native_router.close()
            self._native_router = NativeSparseHybridRouter(
                vocab=vocab, lanes=lane_count
            )
            self._native_signature = signature
        return self._native_router

    def _forward_native(self, x, lane_count, confidence_threshold):
        token_ids = self._token_ids_from_tensor(x)
        router = self._ensure_native_router(token_ids, lane_count)
        if router is None:
            return None

        norms = x.detach().to(dtype=torch.float32).norm(dim=-1)
        for b in range(x.shape[0]):
            row_tokens = [int(tok) for tok in token_ids[b].tolist()]
            row_norms = norms[b]
            keep_threshold = float(torch.median(row_norms).item())
            informative = []
            for token, score in zip(row_tokens, row_norms.tolist()):
                keep = score >= keep_threshold and score > 0.0
                router.train_token_gate(token, keep, strength=1.0)
                if keep:
                    informative.append(token)
            if informative:
                router.train_span_router(
                    informative[: min(len(informative), 8)],
                    lane=sum(informative[:3]) % lane_count,
                    strength=1.0,
                )

        y = x.clone()
        for b in range(x.shape[0]):
            result = router.route([int(tok) for tok in token_ids[b].tolist()])
            if not result.spans:
                continue
            span = result.spans[0]
            if span.confidence < confidence_threshold:
                continue
            lane_id = max(0, min(int(span.lane), lane_count - 1))
            mask = torch.tensor(result.token_actions, device=x.device, dtype=torch.bool)
            if mask.any():
                y[b, mask] = F.gelu(x[b, mask] @ self._lane_weights[lane_id])
        return {"y": y}

    def _forward_python(self, x, lane_count, confidence_threshold):
        logits = x @ self._router
        probs = F.softmax(logits, dim=-1)
        lane_idx = probs.argmax(dim=-1)
        conf = probs.max(dim=-1).values
        y = x.clone()
        for lane_id in range(lane_count):
            mask = (lane_idx == lane_id) & (conf >= confidence_threshold)
            if mask.any():
                y[mask] = F.gelu(x[mask] @ self._lane_weights[lane_id])
        return {"y": y}

    def forward(self, inputs, config):
        x = inputs["x"]
        lane_count = max(2, min(int(config.get("lane_count", 3)), 8))
        confidence_threshold = float(config.get("confidence_threshold", 0.45))
        self._ensure(x, lane_count)
        native_out = self._forward_native(x, lane_count, confidence_threshold)
        if native_out is not None:
            return native_out
        return self._forward_python(x, lane_count, confidence_threshold)
