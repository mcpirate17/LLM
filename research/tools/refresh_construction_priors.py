#!/usr/bin/env python3
"""Compute and optionally activate construction priors from ablation evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.scientist.construction_priors import (  # noqa: E402
    compute_construction_prior,
    record_construction_prior_snapshot,
)
from research.defaults import RUNS_DB  # noqa: E402
from research.scientist.notebook import LabNotebook  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(PROJECT_ROOT / RUNS_DB))
    parser.add_argument("--min-n", type=int, default=4)
    parser.add_argument("--min-metric-complete", type=int, default=3)
    parser.add_argument("--local-min-n", type=int, default=4)
    parser.add_argument("--local-limit", type=int, default=5000)
    parser.add_argument("--notes", default="")
    parser.add_argument("--no-activate", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nb = LabNotebook(str(args.db), use_native=False)
    try:
        prior = compute_construction_prior(
            nb,
            min_n=max(1, int(args.min_n)),
            min_metric_complete=max(1, int(args.min_metric_complete)),
            local_min_n=max(1, int(args.local_min_n)),
            local_limit=max(1, int(args.local_limit)),
        )
        if not prior["payload"]["rules"] and not prior["summary"].get(
            "n_local_edit_observations"
        ):
            print("no ablation evidence met threshold; no snapshot written")
            return 1
        version = record_construction_prior_snapshot(
            nb,
            prior,
            activate=not bool(args.no_activate),
            notes=str(args.notes or ""),
        )
        payload = {
            "status": "activated" if not args.no_activate else "recorded",
            "version": version,
            "summary": prior["summary"],
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"{payload['status']} construction prior {version}")
            print(json.dumps(prior["summary"], indent=2, sort_keys=True))
        return 0
    finally:
        nb.close()


if __name__ == "__main__":
    raise SystemExit(main())
