"""Retrain the GBM predictor and persist artifacts to disk.

After v9 trajectory metrics are wired into ``_POST_EVAL_FEATURE_NAMES``
(predictor_gbm.py:71-77), the persisted GBM artifact is stale —
``feature_names`` doesn't include the new ``fp_*_best`` columns. Run
this tool to call ``train_gbm()`` against the current corpus and write
the new model + meta to ``research/runtime/learning/``.

Usage::

    python -m research.tools.predictor_retrain
    python -m research.tools.predictor_retrain --db /custom/lab.db
    python -m research.tools.predictor_retrain --out /custom/state_dir
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="research/lab_notebook.db",
        help="Path to lab notebook (default: research/lab_notebook.db).",
    )
    parser.add_argument(
        "--out",
        default="research/runtime/learning",
        help="Artifact state dir (default: research/runtime/learning).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        default=True,
        help="Print a one-shot summary of the trained model.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from research.scientist.intelligence.predictor_gbm import train_gbm

    predictor = train_gbm(db_path=args.db)
    if not predictor.is_fitted():
        print("ERROR: predictor did not fit (insufficient data?)", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    predictor.save(out_dir)
    logger.info("wrote artifacts to %s", out_dir)

    if args.print_summary:
        gemini = [
            fn for fn in predictor.feature_names if fn.startswith("fp_")
        ]
        metrics = predictor.train_metrics or {}
        print("─" * 64)
        print(f"GBM retrain — {len(predictor.feature_names)} features, "
              f"{predictor.n_train} train rows")
        print("─" * 64)
        print(f"  gate_threshold         {predictor.gate_threshold:.4f}")
        gm = metrics.get("gate_metrics") or {}
        for k in ("roc_auc", "precision", "recall", "f1", "npv"):
            v = gm.get(k)
            if isinstance(v, (int, float)):
                print(f"  gate_{k:<19}{v:.4f}")
        print(f"  rank_spearman_ppl      {metrics.get('rank_spearman_ppl', 0):.4f}")
        print(f"  rank_spearman_composite{metrics.get('rank_spearman_composite', 0):.4f}")
        print()
        print(f"  v9 trajectory features wired in ({len(gemini)}):")
        for fn in sorted(gemini):
            print(f"    - {fn}")


if __name__ == "__main__":
    main()
