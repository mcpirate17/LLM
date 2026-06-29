"""Corpus-window packing for the data-pipeline search (Workstream D, inc. 4).

``pack`` decides *which* token window is sampled (vs ``order``/``fold`` which
permute an already-sampled window, and ``route`` which assigns positions to
submodules). This module owns the start-index selection so gemini's native
``CorpusTokenBatcher`` hot path is left untouched — a candidate can search packing
by choosing its starts through here instead.

Implemented:
* ``contiguous`` — uniform random windows over the flat stream (the default).
* ``doc_boundary`` — windows that never cross a document boundary, so the model
  never attends across two unrelated documents (the meaningful packing axis for a
  flat next-token stream).

``length_bucketed`` / ``best_fit`` target variable-length document packing and
need per-document materialization; they fail loud here until that lands.
"""

from __future__ import annotations

import numpy as np

# cl100k <|endoftext|>; FineFineWeb is tokenized with cl100k (see corpus notes).
DEFAULT_EOT_ID = 100257


def find_doc_boundaries(tokens: np.ndarray, eot_id: int = DEFAULT_EOT_ID) -> np.ndarray:
    """Positions of the document separator token in a flat token stream."""
    return np.flatnonzero(np.asarray(tokens) == int(eot_id))


def _doc_spans(n_tokens: int, boundaries: np.ndarray, window: int) -> np.ndarray:
    """``[start, end)`` spans (split on boundaries) long enough for ``window``."""
    cuts = np.asarray(boundaries, dtype=np.int64)
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts, [n_tokens]])
    spans = np.stack([starts, ends], axis=1)
    return spans[(spans[:, 1] - spans[:, 0]) >= window]


def pack_window_starts(
    n_tokens: int,
    batch: int,
    window: int,
    pack: str,
    rng: np.random.Generator,
    *,
    boundaries: np.ndarray | None = None,
) -> np.ndarray:
    """Start indices for ``batch`` windows of length ``window`` under ``pack``.

    ``window`` is the full slice the caller will gather (e.g. ``seq + 1`` for a
    next-token batch). ``doc_boundary`` requires ``boundaries`` (from
    :func:`find_doc_boundaries`); each returned window lies inside one document.
    """
    if window <= 0 or batch <= 0:
        raise ValueError(f"window and batch must be positive, got {window}, {batch}")
    if n_tokens <= window:
        raise ValueError(f"corpus too short ({n_tokens}) for window {window}")

    if pack == "contiguous":
        return rng.integers(0, n_tokens - window + 1, size=batch)

    if pack == "doc_boundary":
        if boundaries is None:
            raise ValueError("doc_boundary packing requires document boundaries")
        spans = _doc_spans(n_tokens, boundaries, window)
        if spans.shape[0] == 0:
            raise ValueError(
                f"no document is long enough for window {window}; "
                "lower seq_len or use contiguous packing"
            )
        doc_idx = rng.integers(0, spans.shape[0], size=batch)
        chosen = spans[doc_idx]
        # uniform start within each chosen doc such that the window stays inside
        span = chosen[:, 1] - chosen[:, 0] - window  # >= 0 by construction
        offset = (rng.random(batch) * (span + 1)).astype(np.int64)
        return chosen[:, 0] + offset

    if pack in ("length_bucketed", "best_fit"):
        raise NotImplementedError(
            f"pack {pack!r} needs per-document variable-length packing; not wired yet"
        )
    raise ValueError(f"unknown pack {pack!r}")
