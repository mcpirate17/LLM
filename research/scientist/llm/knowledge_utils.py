"""Shared knowledge-entry filtering and deduplication helpers."""

from __future__ import annotations

import re
from typing import Mapping


KNOWLEDGE_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "into",
    "when",
    "then",
    "than",
    "were",
    "been",
    "have",
    "has",
    "had",
    "are",
    "was",
    "show",
    "shows",
    "showed",
    "over",
    "under",
    "across",
    "between",
    "using",
    "use",
    "used",
    "high",
    "low",
    "very",
    "more",
    "less",
    "near",
    "around",
    "recent",
    "experiments",
    "experiment",
    "result",
    "results",
    "indicate",
    "indicates",
    "suggest",
    "suggests",
    "mode",
    "patterns",
    "pattern",
    "architecture",
    "architectures",
}


def canonical_knowledge_text(raw: str) -> str:
    text = " ".join(str(raw or "").split()).strip().lower()
    text = re.sub(r"\b\d+(?:\.\d+)?%?\b", "#", text)
    text = re.sub(r"[^a-z0-9#\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def knowledge_tokens(raw: str) -> set[str]:
    canonical = canonical_knowledge_text(raw)
    return {
        tok
        for tok in canonical.split()
        if len(tok) > 3 and tok not in KNOWLEDGE_STOPWORDS
    }


def is_semantic_duplicate(
    tokens: set[str],
    existing_tokens: set[str],
    *,
    min_intersection: int = 5,
    threshold: float = 0.18,
) -> bool:
    if not tokens or not existing_tokens:
        return False
    inter = len(tokens & existing_tokens)
    if inter < min_intersection:
        return False
    union = len(tokens | existing_tokens)
    return bool(union) and (inter / union) >= threshold


def is_prompt_low_signal(row: Mapping) -> bool:
    title = " ".join(str(row.get("title") or "").split()).strip().lower()
    content = " ".join(str(row.get("content") or "").split()).strip().lower()
    if not title or not content:
        return True
    if len(title) < 12 or len(content) < 40:
        return True
    if title.startswith("recent experiments show ") or title.startswith(
        "all recent experiments show "
    ):
        return True
    if "..." in title or "..." in content:
        return True
    if "[principle/" in title or "hybrid? no" in title:
        return True
    if "$" in content or "\\approx" in content:
        return True
    return False


def is_extracted_knowledge_low_value(title: str, content: str) -> bool:
    title_clean = " ".join(str(title or "").split()).strip()
    content_clean = " ".join(str(content or "").split()).strip()
    title_l = title_clean.lower()
    content_l = content_clean.lower()

    if len(title_clean) < 12 or len(content_clean) < 40:
        return True
    if "..." in title_clean or "..." in content_clean:
        return True
    if "1-2 sentences" in content_l or "i will now synthesize" in content_l:
        return True
    if title_l.startswith("recent experiments show ") or title_l.startswith(
        "all recent experiments show "
    ):
        return True
    if title_l.startswith("recent synthesis") and "failure" in title_l:
        return True
    if "[principle/" in title_l or "hybrid? no" in title_l:
        return True
    if "$" in content_clean or "\\approx" in content_l:
        return True

    mechanism_tokens = (
        "depth",
        "residual",
        "inverse",
        "log ",
        "frequency",
        "math_space",
        "parameter",
        "parallel",
        "routing",
        "s1",
        "loss",
        "novelty",
        "baseline",
    )
    action_tokens = (
        "improve",
        "improves",
        "degrade",
        "degrades",
        "fail",
        "fails",
        "underperform",
        "correlate",
        "correlates",
        "correlation",
        "predict",
        "predicts",
        "optimal",
        "requires",
        "avoid",
        "boost",
        "increase",
        "reduce",
        "enhance",
        "enhances",
        "outperform",
        "outperforms",
        "suggests",
        "indicates",
    )
    has_mechanism = any(tok in content_l or tok in title_l for tok in mechanism_tokens)
    has_action = any(tok in content_l for tok in action_tokens)
    has_numeric = bool(re.search(r"\d", content_clean))
    return not (has_mechanism and (has_action or has_numeric))
