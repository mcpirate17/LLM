#!/usr/bin/env python3
"""Archive low-value knowledge_base rows using deterministic heuristics."""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple


ALLOWED_CATEGORIES = {
    "principle",
    "anti_pattern",
    "sweet_spot",
    "correlation",
    "tool_insight",
}

STOPWORDS = {
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


def _is_low_value(title: str, content: str, category: str) -> Tuple[bool, str]:
    title_clean = " ".join(str(title or "").split()).strip()
    content_clean = " ".join(str(content or "").split()).strip()
    category_clean = (
        str(category or "").strip().lower().replace("-", "_").replace(" ", "_")
    )
    title_l = title_clean.lower()
    content_l = content_clean.lower()

    if category_clean not in ALLOWED_CATEGORIES:
        return True, "invalid_category"
    if len(title_clean) < 12 or len(content_clean) < 40:
        return True, "too_short"
    if "..." in title_clean or "..." in content_clean:
        return True, "ellipsis_placeholder"
    if "1-2 sentences" in content_l or "i will now synthesize" in content_l:
        return True, "template_artifact"
    if title_l.startswith("recent experiments show ") or title_l.startswith(
        "all recent experiments show "
    ):
        return True, "generic_prefix"
    if "[principle/" in title_l or "hybrid? no" in title_l:
        return True, "malformed_title"
    if "$" in content_clean or "\\approx" in content_l:
        return True, "noisy_markup"

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
    if not (has_mechanism and (has_action or has_numeric)):
        return True, "weak_signal"
    return False, ""


def _normalize_category(raw: str) -> str:
    value = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "anti_pattern": "anti_pattern",
        "anti_patterns": "anti_pattern",
        "antipattern": "anti_pattern",
        "principles": "principle",
        "sweetspot": "sweet_spot",
        "tool": "tool_insight",
        "toolinsight": "tool_insight",
        "tool_insights": "tool_insight",
    }
    return aliases.get(value, value if value in ALLOWED_CATEGORIES else "principle")


def _canonical(raw: str) -> str:
    text = " ".join(str(raw or "").split()).strip().lower()
    text = re.sub(r"\b\d+(?:\.\d+)?%?\b", "#", text)
    text = re.sub(r"[^a-z0-9#\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _semantic_tokens(title: str, content: str) -> Set[str]:
    canonical = _canonical(f"{title or ''} {content or ''}")
    return {tok for tok in canonical.split() if len(tok) > 3 and tok not in STOPWORDS}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    if not union:
        return 0.0
    return len(a & b) / union


def _is_theme_duplicate(a: Set[str], b: Set[str]) -> bool:
    inter = len(a & b)
    if inter < 5:
        return False
    return _jaccard(a, b) >= 0.16


def _effective_confidence(confidence: float, validated: int) -> float:
    bonus = min(0.18, 0.05 * math.log1p(max(validated - 1, 0)))
    return min(0.95, max(0.0, confidence) + bonus)


def curate(
    db_path: Path, apply: bool, strict: bool, weak_signal_max_validated: int
) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = cur.execute(
        "SELECT entry_id, category, title, content, times_validated, confidence, timestamp "
        "FROM knowledge_base WHERE status='active'"
    ).fetchall()

    to_archive: List[Tuple[str, str]] = []
    to_normalize: List[Tuple[str, str]] = []
    weak_signal_rows: List[Dict] = []
    seen_titles = {}
    seen_contents = {}

    for row in rows:
        entry_id = str(row["entry_id"])
        category = str(row["category"] or "")
        title = str(row["title"] or "")
        content = str(row["content"] or "")
        validated = int(row["times_validated"] or 0)

        low_value, reason = _is_low_value(title, content, category)
        if low_value:
            if reason == "weak_signal":
                if strict and validated <= weak_signal_max_validated:
                    to_archive.append((entry_id, f"weak_signal_v{validated}"))
                else:
                    weak_signal_rows.append(
                        {
                            "entry_id": entry_id,
                            "category": _normalize_category(category),
                            "title": title,
                            "content": content,
                            "times_validated": validated,
                            "confidence": float(row["confidence"] or 0.5),
                            "timestamp": float(row["timestamp"] or 0.0),
                            "tokens": _semantic_tokens(title, content),
                        }
                    )
                continue
            to_archive.append((entry_id, reason))
            continue

        normalized_cat = _normalize_category(category)
        if normalized_cat != category:
            to_normalize.append((normalized_cat, entry_id))

        title_key = _canonical(title)
        content_key = _canonical(content)
        if title_key in seen_titles:
            to_archive.append((entry_id, "duplicate_title"))
            continue
        if content_key in seen_contents:
            to_archive.append((entry_id, "duplicate_content"))
            continue
        seen_titles[title_key] = entry_id
        seen_contents[content_key] = entry_id

    # Consolidate weak-signal semantic duplicates: keep strongest representative per cluster.
    weak_by_category: Dict[str, List[Dict]] = {}
    for row in weak_signal_rows:
        weak_by_category.setdefault(str(row["category"]), []).append(row)
    weak_dedup_archives: List[Tuple[str, str]] = []
    weak_clusters = 0
    weak_cluster_kept = 0
    for category, rows_cat in weak_by_category.items():
        clusters: List[List[Dict]] = []
        for row in rows_cat:
            placed = False
            for cluster in clusters:
                if any(
                    _is_theme_duplicate(row["tokens"], other["tokens"])
                    for other in cluster
                ):
                    cluster.append(row)
                    placed = True
                    break
            if not placed:
                clusters.append([row])
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            weak_clusters += 1
            ranked = sorted(
                cluster,
                key=lambda item: (
                    _effective_confidence(
                        float(item["confidence"]), int(item["times_validated"])
                    ),
                    int(item["times_validated"]),
                    float(item["timestamp"]),
                ),
                reverse=True,
            )
            keeper = ranked[0]["entry_id"]
            weak_cluster_kept += 1
            for loser in ranked[1:]:
                weak_dedup_archives.append(
                    (
                        str(loser["entry_id"]),
                        f"weak_signal_theme_duplicate_keep:{keeper}",
                    )
                )
    if weak_dedup_archives:
        seen_archive = {entry_id for entry_id, _ in to_archive}
        for entry_id, reason in weak_dedup_archives:
            if entry_id in seen_archive:
                continue
            to_archive.append((entry_id, reason))
            seen_archive.add(entry_id)

    print(f"active_entries={len(rows)}")
    print(f"archive_candidates={len(to_archive)}")
    print(f"category_normalizations={len(to_normalize)}")
    print(f"weak_signal_candidates={len(weak_signal_rows)}")
    print(f"weak_signal_clusters={weak_clusters}")
    print(f"weak_signal_cluster_representatives={weak_cluster_kept}")
    print(f"strict_mode={str(strict).lower()}")

    if not apply:
        print("dry_run_only=true")
        for entry_id, reason in to_archive[:40]:
            print(f"archive {entry_id} reason={reason}")
        for row in weak_signal_rows[:20]:
            print(f"review {row['entry_id']} reason=weak_signal")
        return

    if to_archive:
        cur.executemany(
            "UPDATE knowledge_base SET status=? WHERE entry_id=?",
            [("archived_low_value", entry_id) for entry_id, _ in to_archive],
        )
    if to_normalize:
        cur.executemany(
            "UPDATE knowledge_base SET category=? WHERE entry_id=?",
            to_normalize,
        )
    conn.commit()
    print("applied=true")


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate knowledge_base entries.")
    parser.add_argument(
        "--db", default="research/lab_notebook.db", help="Path to lab_notebook.db"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Apply changes (default is dry run)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also archive weak-signal rows with low validation count",
    )
    parser.add_argument(
        "--weak-signal-max-validated",
        type=int,
        default=2,
        help="Strict mode threshold for weak-signal archival",
    )
    args = parser.parse_args()
    curate(
        Path(args.db),
        apply=args.apply,
        strict=bool(args.strict),
        weak_signal_max_validated=max(0, int(args.weak_signal_max_validated)),
    )


if __name__ == "__main__":
    main()
