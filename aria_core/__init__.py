"""aria_core — Unified high-performance kernel library for Aria."""

import torch

from ._bootstrap import load_native_extension

load_native_extension(globals())


if "token_gate_trace_f32" not in globals():

    def token_gate_trace_f32(scores, threshold):
        flat_scores = scores.squeeze(-1) if scores.dim() == 3 else scores
        confidence = torch.sigmoid(flat_scores)
        keep_mask = (confidence >= threshold).to(torch.int64)
        return keep_mask, confidence


if "sparse_span_extract_f32" not in globals():

    def sparse_span_extract_f32(x, keep_mask, span_width):
        b, s, d = x.shape
        span_width = max(1, min(int(span_width), s))
        span_features = torch.zeros_like(x)
        span_positions = torch.full(
            (b, s, span_width), -1, dtype=torch.int64, device=x.device
        )
        span_counts = torch.zeros((b,), dtype=torch.int64, device=x.device)
        coverage = torch.zeros((b, s), dtype=torch.int64, device=x.device)
        min_kept = 1 if span_width <= 1 else 2
        for bi in range(b):
            packed = 0
            for start in range(0, s - span_width + 1):
                window = keep_mask[bi, start : start + span_width]
                if int(window.sum().item()) < min_kept or packed >= s:
                    continue
                span_features[bi, packed] = x[bi, start : start + span_width].mean(
                    dim=0
                )
                span_positions[bi, packed] = torch.arange(
                    start, start + span_width, device=x.device, dtype=torch.int64
                )
                coverage[bi, start : start + span_width] += 1
                packed += 1
            span_counts[bi] = packed
        return span_features, span_positions, span_counts, coverage


__version__ = "0.1.0"
