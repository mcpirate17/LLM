#!/usr/bin/env python3
"""Compress low-support failure signatures into Bayesian op priors.

Keeps DB light while retaining learning signal.

Outputs:
  - research/runtime/learning/op_priors.json
  - research/runtime/learning/op_pair_priors.json

Optionally deletes low-support rows from failure_signatures.

Usage:
  python -m research.tools.compress_failure_signatures --db research/lab_notebook.db
  python -m research.tools.compress_failure_signatures --db research/lab_notebook.db --delete-low-support
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path


def _beta_posterior_mean(n_fail: int, n_succ: int, alpha: float, beta: float) -> float:
    return (n_fail + alpha) / max(1e-9, (n_fail + n_succ + alpha + beta))


def compress_failure_signatures(
    db_path: str,
    alpha: float = 1.0,
    beta: float = 3.0,
    min_support: int = 5,
    hard_fail: float = 0.8,
    soft_fail: float = 0.6,
    delete_low_support: bool = False,
):
    import sqlite3
    from research.synthesis.primitives import PRIMITIVE_REGISTRY

    # Build set of valid ops from current registry
    valid_ops = set(PRIMITIVE_REGISTRY.keys())

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = cur.execute(
        "SELECT signature, n_failures, n_successes FROM failure_signatures"
    ).fetchall()

    op_bad_prior = defaultdict(float)
    op_bad_weight = defaultdict(float)
    op_pair_priors = {}

    keep_rows = []
    low_support_count = 0
    skipped_deprecated = 0
    for r in rows:
        sig = r["signature"]
        n_fail = int(r["n_failures"] or 0)
        n_succ = int(r["n_successes"] or 0)
        n_total = n_fail + n_succ

        # Skip signatures containing deprecated/removed ops
        parts = [p.strip() for p in sig.split("->") if p.strip()]
        if any(op not in valid_ops for op in parts):
            skipped_deprecated += 1
            continue

        if n_total < min_support:
            low_support_count += 1
            # Compress into op-level priors
            parts = [p.strip() for p in sig.split("->") if p.strip()]
            if parts:
                p_fail = _beta_posterior_mean(n_fail, n_succ, alpha, beta)
                weight = n_total / float(min_support)
                for op in parts:
                    op_bad_prior[op] += p_fail * weight
                    op_bad_weight[op] += weight
            continue

        p_fail = _beta_posterior_mean(n_fail, n_succ, alpha, beta)
        if p_fail >= soft_fail:
            keep_rows.append(sig)
            op_pair_priors[sig] = {
                "p_fail": round(p_fail, 4),
                "n_total": n_total,
            }

    # Normalize op priors to [0,1]
    op_penalties = {}
    for op, total in op_bad_prior.items():
        w = op_bad_weight.get(op, 1.0)
        score = total / max(1e-9, w)
        # clamp to [0,1]
        op_penalties[op] = max(0.0, min(1.0, float(score)))

    out_dir = Path("research/runtime/learning")
    out_dir.mkdir(parents=True, exist_ok=True)
    op_prior_path = out_dir / "op_priors.json"
    op_pair_path = out_dir / "op_pair_priors.json"

    op_prior_payload = {
        "generated_at": time.time(),
        "alpha": alpha,
        "beta": beta,
        "min_support": min_support,
        "op_penalties": op_penalties,
        "low_support_count": low_support_count,
    }
    op_pair_payload = {
        "generated_at": time.time(),
        "soft_fail": soft_fail,
        "hard_fail": hard_fail,
        "pairs": op_pair_priors,
    }

    op_prior_path.write_text(json.dumps(op_prior_payload, indent=2))
    op_pair_path.write_text(json.dumps(op_pair_payload, indent=2))

    if delete_low_support:
        cur.execute(
            "DELETE FROM failure_signatures WHERE (n_failures + n_successes) < ?",
            (min_support,),
        )
        conn.commit()

    conn.close()

    print("Compressed failure signatures")
    print(f"  DB: {db_path}")
    print(f"  total signatures: {len(rows)}")
    print(f"  skipped (deprecated ops): {skipped_deprecated}")
    print(f"  low_support (<{min_support}): {low_support_count}")
    print(f"  kept (>=soft_fail): {len(keep_rows)}")
    print(f"  wrote: {op_prior_path}")
    print(f"  wrote: {op_pair_path}")
    if delete_low_support:
        print("  deleted low-support signatures")


def main():
    parser = argparse.ArgumentParser(description="Compress failure_signatures into Bayesian priors")
    parser.add_argument("--db", default="research/lab_notebook.db")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--soft-fail", type=float, default=0.6)
    parser.add_argument("--hard-fail", type=float, default=0.8)
    parser.add_argument("--delete-low-support", action="store_true")
    args = parser.parse_args()

    compress_failure_signatures(
        db_path=args.db,
        alpha=args.alpha,
        beta=args.beta,
        min_support=args.min_support,
        hard_fail=args.hard_fail,
        soft_fail=args.soft_fail,
        delete_low_support=args.delete_low_support,
    )


if __name__ == "__main__":
    main()
