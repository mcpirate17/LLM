#!/usr/bin/env python
"""Mine data-grounded failure rules: which ops cause compile / lookahead / instability errors.

The grammar already encodes hard structural rules (forbidden prev/next pairs, residual context — see
`synthesis/_context_validation.py`). This mines the EMPIRICAL, support-weighted ones from history:
for each op, the rate at which graphs containing it die with each structural failure mode, vs the
base rate (lift). Emits a versioned JSON the cascade enforces and the grammar team can later ingest.

Failure modes (graph_runs/op_observations.error_type):
  - forward_error / RuntimeError  → compile / forward-pass failure
  - causality_violation           → lookahead leak (model sees future tokens)
  - unstable_dynamics / nan_forward → numerical blow-up

Op-PAIR grad_exploding in op_pair_profile_catalog is ~59% (over-eager profiler) so it is NOT used;
op-level rates from op_observations are the clean signal.

Usage::  python -m research.tools.mine_failure_rules [--min-support 20]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

_META_DB = "research/meta_analysis.db"
_OUT = Path("research/data/learned_failure_rules/rules.json")
_MODES = {
    "compile": ("forward_error", "RuntimeError", "forward_pass_error"),
    "lookahead": ("causality_violation",),
    "instability": ("unstable_dynamics", "nan_forward", "inflight_grad_explosion"),
    "resource": ("OutOfMemoryError", "cuda_fatal", "OutOfResources"),
}
# rate threshold above which an op is flagged "risky" for that mode (with >= min_support obs).
# NOTE: convergence/insufficient_learning are deliberately EXCLUDED — those are capability
# (handled by the measured-mechanism score), not deterministic structural rules.
_RISK_THRESHOLD = {
    "compile": 0.15,
    "lookahead": 0.10,
    "instability": 0.08,
    "resource": 0.08,
}


def mine(meta_db: str, min_support: int) -> Dict[str, Any]:
    con = sqlite3.connect(meta_db)
    rows = con.execute(
        "SELECT op_name, error_type FROM op_observations WHERE op_name IS NOT NULL"
    ).fetchall()
    con.close()
    et_to_mode = {et: m for m, ets in _MODES.items() for et in ets}
    total: Counter = Counter()
    fails: Dict[str, Counter] = {m: Counter() for m in _MODES}
    mode_total: Counter = Counter()
    n_rows = 0
    for op, et in rows:
        op = str(op)
        total[op] += 1
        n_rows += 1
        mode = et_to_mode.get(str(et))
        if mode:
            fails[mode][op] += 1
            mode_total[mode] += 1
    base = {m: mode_total[m] / max(n_rows, 1) for m in _MODES}
    rules: Dict[str, Dict[str, Any]] = {}
    for mode in _MODES:
        thr = _RISK_THRESHOLD[mode]
        entries = {}
        for op, n in total.items():
            if n < min_support:
                continue
            rate = fails[mode][op] / n
            if rate >= thr:
                entries[op] = {
                    "rate": round(rate, 4),
                    "n_obs": int(n),
                    "n_fail": int(fails[mode][op]),
                    "lift": round(rate / max(base[mode], 1e-9), 2),
                }
        rules[mode] = dict(sorted(entries.items(), key=lambda kv: -kv[1]["rate"]))
    pairs, triplets = _mine_adjacency(meta_db)
    return {
        "version": "learned_failure_rules_v1",
        "mined_ts": time.time(),
        "n_op_observations": n_rows,
        "min_support": min_support,
        "risk_threshold": _RISK_THRESHOLD,
        "base_rate": {m: round(base[m], 4) for m in _MODES},
        "rules": rules,
        "unstable_pairs": pairs,  # a->b: numerically unstable adjacency (cleaner than grad_explode)
        "unstable_triplets": triplets,  # a->b->c: true 3-op instability (diverges from pair pred)
    }


def _mine_adjacency(meta_db: str) -> tuple[list, list]:
    """Data-grounded adjacency rules from the measured pair/triplet stability catalogs.

    Pairs: extreme-kurtosis / large distribution-shift / grad-vanishing (the clean signals; raw
    grad_exploding is ~59% over-eager so excluded). Triplets: diverges_from_pair_prediction (true
    3-op interactions a profiler couldn't predict from the constituent pairs)."""
    con = sqlite3.connect(meta_db)
    try:
        pairs = [
            [str(a), str(b)]
            for a, b in con.execute(
                "SELECT op_a, op_b FROM op_pair_profile_catalog "
                "WHERE ABS(COALESCE(output_kurtosis,0))>50 OR COALESCE(distribution_shift,0)>10 "
                "OR grad_vanishing=1"
            )
        ]
        triplets = [
            [str(a), str(b), str(c)]
            for a, b, c in con.execute(
                "SELECT op_a, op_b, op_c FROM op_triplet_profile_catalog "
                "WHERE diverges_from_pair_prediction=1 OR triplet_stable=0"
            )
        ]
    except sqlite3.OperationalError:
        pairs, triplets = [], []
    finally:
        con.close()
    return pairs, triplets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default=_META_DB)
    p.add_argument("--min-support", type=int, default=20)
    p.add_argument("--out", default=str(_OUT))
    args = p.parse_args()
    report = mine(args.meta, args.min_support)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    n = {m: len(report["rules"][m]) for m in report["rules"]}
    print(
        json.dumps(
            {
                "out": out.as_posix(),
                "n_risky_ops_by_mode": n,
                **{"base_rate": report["base_rate"]},
            },
            indent=2,
            sort_keys=True,
        )
    )
    for m, ops in report["rules"].items():
        top = list(ops.items())[:5]
        print(
            f"\n{m} (base {report['base_rate'][m]}): "
            + ", ".join(f"{o}={d['rate']}(n{d['n_obs']})" for o, d in top)
        )


if __name__ == "__main__":
    main()
