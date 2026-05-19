"""Real-architecture cohort audit for nano_blimp_v2 held-out metrics.

Loads champion / frontier graphs from the lab notebook, briefly trains
each on wikitext-vocab random data, then runs ``nano_blimp_score``
across multiple held_out_count settings and seeds.

Read-only: no DB writes, no leaderboard mutation. JSON output goes to
``research/reports/``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import torch

from research.eval.nano_blimp_eval import nano_blimp_score
from research.tools.nano_blimp_audit_common import load_arch, run_audit

logger = logging.getLogger(__name__)

_load_arch = load_arch


def _run_probe(
    model: torch.nn.Module,
    seed: int,
    held_out_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    res = nano_blimp_score(
        model,
        active_vocab_size=args.vocab,
        n_train_steps=args.probe_steps,
        batch_size=32,
        lr=1e-3,
        device=args.device,
        seed=seed,
        held_out_count=held_out_count,
    )
    return {
        "status": res.status,
        "metric_version": res.metric_version,
        "vocab": args.vocab,
        "probe_steps": args.probe_steps,
        "held_out_count": held_out_count,
        "seed": seed,
        "score": res.score,
        "held_out_score": res.held_out_score,
        "class_in_dist": res.class_coherence_in_dist_acc,
        "class_held_out": res.class_coherence_held_out_acc,
        "binding_in_dist": res.binding_fidelity_in_dist_acc,
        "binding_held_out": res.binding_fidelity_held_out_acc,
        "order": res.order_grammaticality_acc,
        "n_in_dist_pairs": res.n_in_dist_pairs,
        "n_held_out_pairs": res.n_held_out_pairs,
        "n_train_steps_completed": res.n_train_steps,
        "elapsed_ms": res.elapsed_ms,
    }


def _failure_fields(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "vocab": args.vocab,
        "probe_steps": args.probe_steps,
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="research/runs.db", type=Path)
    ap.add_argument("--targets", nargs="+", required=True, help="result_id list")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--held-out", nargs="+", type=int, default=[2, 3, 5])
    ap.add_argument("--vocab", type=int, default=80)
    ap.add_argument("--probe-steps", type=int, default=100)
    ap.add_argument("--base-train-steps", type=int, default=750)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(f"research/reports/nano_blimp_v2_audit_{int(time.time())}.json"),
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return run_audit(
        args=args,
        run_probe=_run_probe,
        failure_fields=_failure_fields,
        summary_title="nano_blimp_v2 cohort audit (mean +/- std across seeds)",
        logger=logger,
    )


if __name__ == "__main__":
    raise SystemExit(main())
