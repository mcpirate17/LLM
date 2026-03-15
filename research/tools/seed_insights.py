#!/usr/bin/env python3
"""Seed the insights table with verified, data-backed Bayesian insights.

Idempotent: uses semantic_key dedup to avoid duplicates.
Run: python -m research.tools.seed_insights [--db research/lab_notebook.db]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure research/ is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from research.scientist.notebook import LabNotebook

# All alpha/beta values derived from actual lab_notebook.db counts (2026-03-14).
SEED_INSIGHTS = [
    # ── Structural: graph size ──
    {
        "category": "structural_preference",
        "content": "Graph size 1-6 ops is optimal (57.7% S1 rate, n=3676).",
        "insight_type": "graph_size_optimal",
        "subject_key": "graph_size_optimal",
        "semantic_key": "structural:graph_size_optimal",
        "alpha": 2121.0,  # S1 passed in 1-6 bucket
        "beta_": 1555.0,  # S1 failed in 1-6 bucket
        "display_only": False,
        "insight_level": "structural",
        "evidence_json": {
            "test": "chi2_contingency",
            "p_value": 0.001,
            "n": 10236,
            "bucket_rates": {"1-6": 0.577, "7-9": 0.358, "10-12": 0.218, "13+": 0.022},
            "best_bucket": "1-6",
        },
    },
    {
        "category": "structural_preference",
        "content": "13+ ops collapses to 2.2% S1 (n=2990). Hard cap max_ops=12 recommended.",
        "insight_type": "graph_size_cap",
        "subject_key": "graph_size_cap",
        "semantic_key": "structural:graph_size_cap",
        "alpha": 2924.0,  # correctly predicted fail (actually failed)
        "beta_": 66.0,    # incorrectly predicted fail (actually passed)
        "display_only": False,
        "insight_level": "structural",
        "evidence_json": {
            "test": "binomial_vs_baseline",
            "p_value": 0.001,
            "n": 2990,
            "rate": 0.022,
            "recommended_max": 12,
        },
    },
    # ── Composition: top ops ──
    {
        "category": "success_factor",
        "content": "tropical_moe is best reliable op (56.6% S1, n=159).",
        "insight_type": "top_op",
        "subject_key": "tropical_moe",
        "semantic_key": "top_op:tropical_moe",
        "alpha": 90.0,   # S1 passed
        "beta_": 69.0,   # S1 failed
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "proportion_vs_baseline",
            "n": 159,
            "rate": 0.566,
            "baseline_rate": 0.322,
            "effect_size": 0.244,
        },
    },
    {
        "category": "success_factor",
        "content": "tropical_attention has 52.4% S1 (n=248). Strong performer.",
        "insight_type": "top_op",
        "subject_key": "tropical_attention",
        "semantic_key": "top_op:tropical_attention",
        "alpha": 130.0,
        "beta_": 118.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "proportion_vs_baseline",
            "n": 248,
            "rate": 0.524,
            "baseline_rate": 0.322,
            "effect_size": 0.202,
        },
    },
    {
        "category": "success_factor",
        "content": "grade_mix has 51.6% S1 (n=316). Reliable math_space op.",
        "insight_type": "top_op",
        "subject_key": "grade_mix",
        "semantic_key": "top_op:grade_mix",
        "alpha": 163.0,
        "beta_": 153.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "proportion_vs_baseline",
            "n": 316,
            "rate": 0.516,
            "baseline_rate": 0.322,
            "effect_size": 0.194,
        },
    },
    # ── Failure modes (display-only) ──
    {
        "category": "failure_mode",
        "content": "entropy_router broken (0.2% S1, n=2700). Display-only.",
        "insight_type": "failing_op",
        "subject_key": "entropy_router",
        "semantic_key": "failing_op:entropy_router",
        "alpha": 5.0,
        "beta_": 2695.0,
        "display_only": True,
        "insight_level": "op",
        "evidence_json": {
            "test": "binomial_proportion",
            "n": 2700,
            "rate": 0.002,
        },
    },
    {
        "category": "failure_mode",
        "content": "moe_topk nearly broken (0.1% S1, n=1908). Display-only.",
        "insight_type": "failing_op",
        "subject_key": "moe_topk",
        "semantic_key": "failing_op:moe_topk",
        "alpha": 1.0,
        "beta_": 1907.0,
        "display_only": True,
        "insight_level": "op",
        "evidence_json": {
            "test": "binomial_proportion",
            "n": 1908,
            "rate": 0.001,
        },
    },
    {
        "category": "failure_mode",
        "content": "graph_attention 0% S1 (n=337). Display-only.",
        "insight_type": "failing_op",
        "subject_key": "graph_attention",
        "semantic_key": "failing_op:graph_attention",
        "alpha": 0.01,
        "beta_": 337.0,
        "display_only": True,
        "insight_level": "op",
        "evidence_json": {
            "test": "binomial_proportion",
            "n": 337,
            "rate": 0.0,
        },
    },
    {
        "category": "failure_mode",
        "content": "state_space 0% S1 (n=192). Display-only.",
        "insight_type": "failing_op",
        "subject_key": "state_space",
        "semantic_key": "failing_op:state_space",
        "alpha": 0.01,
        "beta_": 192.0,
        "display_only": True,
        "insight_level": "op",
        "evidence_json": {
            "test": "binomial_proportion",
            "n": 192,
            "rate": 0.0,
        },
    },
]


def seed(db_path: str = "research/lab_notebook.db") -> int:
    """Load seed insights. Returns count of newly inserted insights."""
    nb = LabNotebook(db_path)
    count = 0
    for ins in SEED_INSIGHTS:
        insight_id = nb.record_insight(
            category=ins["category"],
            content=ins["content"],
            insight_type=ins["insight_type"],
            subject_key=ins["subject_key"],
            semantic_key=ins["semantic_key"],
            alpha=ins["alpha"],
            beta_=ins["beta_"],
            display_only=ins["display_only"],
            insight_level=ins["insight_level"],
            evidence_json=ins["evidence_json"],
        )
        count += 1
        print(f"  [{ins['insight_level']}] {ins['semantic_key']} → {insight_id}")
    nb.close()
    return count


def main():
    parser = argparse.ArgumentParser(description="Seed Bayesian insights")
    parser.add_argument("--db", default="research/lab_notebook.db")
    args = parser.parse_args()
    n = seed(args.db)
    print(f"\nSeeded {n} insights.")


if __name__ == "__main__":
    main()
