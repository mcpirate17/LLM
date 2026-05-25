#!/usr/bin/env python
"""PROSPECTIVE test of the closed-book pre-probe filter — the n=1 → population hardening.

The retrospective win (`measured_descriptors.py`) showed the mechanism filter flags the one known
novel winner. This tests it FORWARD on FRESH, never-probed novel candidates: generate a new pool
(new seed), use the cheap 1-layer measured filter (`long_range_reach`, ~0.4s, no training/labels) to
predict induction-capability, then run the REAL 6-layer induction probe (`induction_score_gold`, the
expensive ground truth) on a stratified sample and check whether the cheap prediction holds.

The deployable claim under test: **the SKIP group (reach < τ) has ~0 real induction** — i.e. the
filter never throws away a capable design. Plus head-to-head vs the deployed screener's own score.

Usage::
    python -m research.tools.measured_prospective_probe --n-high 15 --n-low 15 --seed0 7000000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from research.tools.measured_descriptors import MeasuredDescriptorExtractor
from research.tools.probe_novel_candidates import _collect_pool, _probe_one

logger = logging.getLogger(__name__)
_TAU = 0.01  # validated pre-probe gate (keeps 99.3% capable, prunes 55% incapable)
_THR = 0.35  # induction-capable threshold


def _filter_pool(
    cand: List[Dict[str, Any]], ext: MeasuredDescriptorExtractor
) -> List[Dict[str, Any]]:
    """Attach the cheap measured long_range_reach to each generated candidate."""
    import json as _json

    out = []
    t0 = time.time()
    for i, c in enumerate(cand):
        d = ext.descriptors(_json.dumps(c["graph"].to_dict()))
        if d is None:
            continue
        c["reach"] = d["long_range_reach"]
        c["content_dependence"] = d["content_dependence"]
        out.append(c)
        if (i + 1) % 100 == 0:
            logger.info("  measured %d/%d (%.0fs)", i + 1, len(cand), time.time() - t0)
    return out


def _real_probe(
    sample: List[Dict[str, Any]], n_layers: int, device: str, tag: str
) -> List[Dict[str, Any]]:
    rows = []
    for i, c in enumerate(sample):
        c.setdefault("stratum", "prospective")
        c.setdefault("predicted", 0.0)
        r = _probe_one(c, n_layers, device)
        rows.append(
            {
                "fingerprint": r["fingerprint"],
                "group": tag,
                "reach": round(float(c["reach"]), 4),
                "screener_pred": round(float(c.get("predicted", 0.0)), 4),
                "real_induction": r["actual_induction_auc"],
                "novel_mixers": c.get("novel_mixers"),
            }
        )
        logger.info(
            "  [%s] %d/%d reach=%.3f real=%s",
            tag,
            i + 1,
            len(sample),
            c["reach"],
            r["actual_induction_auc"],
        )
    return rows


def _roc(scores: List[float], reals: List[float]) -> Any:
    from sklearn.metrics import roc_auc_score

    s = np.array(scores)
    y = (np.array(reals) > _THR).astype(int)
    if 0 < y.sum() < len(y):
        return round(float(roc_auc_score(y, s)), 4)
    return None


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ext = MeasuredDescriptorExtractor(n_seeds=2, device=args.device)
    logger.info("generating fresh pool (seed0=%d)…", args.seed0)
    cand = _collect_pool(args.db, args.pool, args.max_attempts, args.seed0)
    cand = _filter_pool(cand, ext)
    cand.sort(key=lambda c: -c["reach"])
    high = cand[: args.n_high]  # filter says KEEP (highest reach)
    low = [c for c in cand if c["reach"] < _TAU][: args.n_low]  # filter says SKIP
    logger.info("real-probing %d high-reach + %d skip-group…", len(high), len(low))
    rows = _real_probe(high, args.n_layers, args.device, "keep") + _real_probe(
        low, args.n_layers, args.device, "skip"
    )
    ok = [r for r in rows if r["real_induction"] is not None]
    keep = [r for r in ok if r["group"] == "keep"]
    skip = [r for r in ok if r["group"] == "skip"]
    out: Dict[str, Any] = {
        "tau": _TAU,
        "n_pool_measured": len(cand),
        "n_probed_ok": len(ok),
        "skip_group": {
            "n": len(skip),
            "max_real_induction": round(
                max((r["real_induction"] for r in skip), default=0.0), 4
            ),
            "n_capable_wrongly_skipped": sum(
                1 for r in skip if r["real_induction"] > _THR
            ),
        },
        "keep_group": {
            "n": len(keep),
            "max_real_induction": round(
                max((r["real_induction"] for r in keep), default=0.0), 4
            ),
            "n_capable": sum(1 for r in keep if r["real_induction"] > _THR),
        },
        "prospective_roc": {
            "measured_reach": _roc(
                [r["reach"] for r in ok], [r["real_induction"] for r in ok]
            ),
            "screener_pred": _roc(
                [r["screener_pred"] for r in ok], [r["real_induction"] for r in ok]
            ),
        },
        "rows": sorted(ok, key=lambda r: -(r["real_induction"] or 0.0)),
    }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="research/runs.db")
    p.add_argument("--pool", type=int, default=400)
    p.add_argument("--max-attempts", type=int, default=8000)
    p.add_argument(
        "--seed0", type=int, default=7_000_000
    )  # fresh seed ⇒ unseen candidates
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-high", type=int, default=15)
    p.add_argument("--n-low", type=int, default=15)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="research/reports/measured_prospective_probe.json")
    args = p.parse_args()
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    report = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
