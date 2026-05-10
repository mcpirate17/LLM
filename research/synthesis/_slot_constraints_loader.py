"""Phase 4.1 — Derive template slot motif-class allowlists from pass-cohort data.

Replaces hardcoded `ffn_classes` constants in template factories. At module
import time, queries `meta_analysis.db` (slot_observations) joined with
`lab_notebook.db` (program_results filter) for empirical pass-cohort fills
per (template, slot_index). Motif classes meeting the threshold (n>=5 +
conditional pass_rate >= 0.60) become the derived allowlist.

Caller-friendly: pass a fallback tuple in case the meta DB is unreachable or
the slot lacks sufficient data. Idempotent: results cached per process.

Pass criterion matches the rest of the pipeline:
  language_control_s05_sentence_assoc_score >= 0.95 AND failure_op != 'nano_bind'
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections import defaultdict
from pathlib import Path
from typing import Tuple

from research.defaults import RUNS_DB

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
META_DB = REPO / "research/meta_analysis.db"
LAB_DB = REPO / RUNS_DB

MIN_N_PER_CLASS = 10
MIN_PASS_RATE_PER_CLASS = 0.40
# Threshold is materially-better-than-cohort-average (cohort pass rate ~0.28).
# Originally 0.60 per the deep-dive doc, but that was too strict at the
# motif-class level — the canonical pass-cohort classes (conv_core, channel_core)
# sit at 0.55–0.71 conditional pass; 0.40 catches both reliably while still
# excluding clear losers (efficient_proj 0.00, sparse_core 0.06, math_space 0.03).

_cache_lock = threading.Lock()
_cache: dict[tuple[str, int], tuple[str, ...]] | None = None


def _query_pass_cohort_fills() -> dict[tuple[str, int], list[tuple[str, int, int]]]:
    """Return (template, slot_index) -> list of (motif_class, n_pass, n_total)."""
    if not META_DB.exists() or not LAB_DB.exists():
        logger.info("derive_slot_classes: meta or lab DB missing; deriving disabled")
        return {}
    try:
        conn = sqlite3.connect(f"file:{META_DB}?mode=ro&immutable=0", uri=True)
        conn.execute(f"ATTACH 'file:{LAB_DB}?mode=ro&immutable=0' AS lab")
    except sqlite3.Error:
        logger.exception("derive_slot_classes: DB attach failed")
        return {}
    try:
        cur = conn.execute(
            """
            SELECT so.template_name,
                   so.slot_index,
                   so.selected_motif_class,
                   SUM(CASE WHEN pr.language_control_s05_sentence_assoc_score >= 0.95
                              AND COALESCE(pr.failure_op,'') != 'nano_bind'
                            THEN 1 ELSE 0 END) AS n_pass,
                   COUNT(*) AS n_total
              FROM slot_observations so
              JOIN lab.program_results pr ON pr.result_id = so.result_id
              LEFT JOIN lab.leaderboard l ON l.result_id = pr.result_id
             WHERE pr.language_control_s05_sentence_assoc_score IS NOT NULL
               AND COALESCE(l.is_reference, 0) = 0
               AND so.selected_motif_class IS NOT NULL
             GROUP BY so.template_name, so.slot_index, so.selected_motif_class
            """
        )
        out: dict[tuple[str, int], list[tuple[str, int, int]]] = defaultdict(list)
        for tpl, slot_idx, motif_class, n_pass, n_total in cur.fetchall():
            if not tpl or slot_idx is None or not motif_class:
                continue
            out[(str(tpl), int(slot_idx))].append(
                (str(motif_class), int(n_pass), int(n_total))
            )
        return out
    finally:
        conn.close()


def _build_cache() -> dict[tuple[str, int], tuple[str, ...]]:
    """Per-(template, slot) ordered tuple of motif_classes meeting threshold."""
    fills = _query_pass_cohort_fills()
    derived: dict[tuple[str, int], tuple[str, ...]] = {}
    for key, rows in fills.items():
        accepted: list[tuple[str, float, int]] = []
        for motif_class, n_pass, n_total in rows:
            if n_total < MIN_N_PER_CLASS:
                continue
            pass_rate = n_pass / n_total if n_total else 0.0
            if pass_rate < MIN_PASS_RATE_PER_CLASS:
                continue
            accepted.append((motif_class, pass_rate, n_total))
        if not accepted:
            continue
        # Sort by pass_rate desc, then n_total desc — strongest signal first.
        accepted.sort(key=lambda t: (-t[1], -t[2]))
        derived[key] = tuple(c for c, _, _ in accepted)
    return derived


def _ensure_cache() -> dict[tuple[str, int], tuple[str, ...]]:
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = _build_cache()
            logger.info(
                "derive_slot_classes: cache built with %d (template, slot) entries",
                len(_cache),
            )
        return _cache


def derive_slot_classes(
    template_name: str,
    slot_index: int,
    fallback: Tuple[str, ...],
) -> Tuple[str, ...]:
    """Return motif_classes that pass-rate-qualify for (template, slot_index).

    Falls back to `fallback` when:
      - meta DB or lab DB is not present (e.g., test env)
      - the slot has no qualifying motif_classes (no class >= MIN_PASS_RATE)
      - the slot has fewer than MIN_N_PER_CLASS samples on every class

    Cached per process. To force re-evaluation in tests: `reset_cache()`.
    """
    cache = _ensure_cache()
    return cache.get((template_name, int(slot_index)), fallback)


def reset_cache() -> None:
    """Clear the derivation cache (used by unit tests)."""
    global _cache
    with _cache_lock:
        _cache = None


# A/B strategies for the use_derived_slot_classes flag.
# - "static":   always False; preserves the pre-2026-05-04 baseline.
# - "derived":  always True; opt-in fully data-driven slot allowlists.
# - "ab_50_50": 50/50 split keyed on a stable hash of the experiment id so
#               the assignment is reproducible per run, comparable per cohort.
SLOT_CLASS_STRATEGIES = ("static", "derived", "ab_50_50")


def resolve_slot_class_strategy(
    *,
    explicit_use_derived: bool,
    strategy: str,
    experiment_id: str | None,
) -> tuple[bool, str]:
    """Decide whether to use derived slot classes for this experiment.

    Returns (use_derived, assignment_reason). The reason string is persisted
    in graph.metadata so post-hoc analysis can compare derived vs static
    cohorts cleanly. ``explicit_use_derived=True`` always wins; otherwise the
    strategy decides.
    """
    if explicit_use_derived:
        return True, "explicit_config"
    if strategy == "derived":
        return True, "strategy_derived"
    if strategy == "ab_50_50" and experiment_id:
        # Stable per-experiment 50/50 split — same exp_id always lands on the
        # same arm, enabling reproducible cohort-level comparison.
        bit = sum(ord(ch) for ch in str(experiment_id)) & 1
        return (bool(bit), "strategy_ab_50_50")
    return False, "strategy_static"
