#!/usr/bin/env python3
"""Batch-promote backlog candidates to the leaderboard at screening tier.

Reads a candidate JSONL (output of ``promote_backlog_score_filter.py``) and
promotes each ``result_id`` using the same logic as the
``/api/programs/<result_id>/promote-screening`` endpoint:

1. Upsert a leaderboard row at ``tier="screening"`` with the program's
   existing metrics.
2. Tag ``trust_label`` and ``comparability_label`` (``candidate_screening`` /
   ``screening_only`` when the program has no existing labels).
3. Leave probe NULLs alone — a separate ``backfill.py`` run can fill them in.

Dry-run default. ``--apply`` executes. Writes a completion audit to
``research/reports/promote_backlog_applied_YYYY-MM-DD.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("research/reports/promote_backlog_candidates_2026-04-19.jsonl")
DEFAULT_DB = Path("research/lab_notebook.db")
DEFAULT_AUDIT = Path("research/reports/promote_backlog_applied_2026-04-19.jsonl")


def _load_candidates(path: Path, top: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if top is not None:
        rows = rows[: max(top, 0)]
    return rows


def _promote_one(nb, cand: Dict[str, Any]) -> Dict[str, Any]:
    """Replicate _api_program_promote_screening semantics (fingerprint-aware)."""
    result_id = str(cand["result_id"])
    program = nb.get_program_detail(result_id)
    if program is None:
        return {
            "result_id": result_id,
            "fp": cand.get("graph_fingerprint"),
            "status": "skipped_missing_program",
        }

    entry = nb.get_leaderboard_entry(result_id)
    # Fingerprint-level dedup: route to the existing entry for this fingerprint
    # rather than creating a duplicate leaderboard row.
    if entry is None:
        fp = str(program.get("graph_fingerprint") or "").strip()
        if fp:
            sibling = nb.get_leaderboard_entry_by_fingerprint(fp)
            if sibling and sibling.get("result_id") != result_id:
                entry = sibling
                result_id = sibling.get("result_id")
                program = nb.get_program_detail(result_id) or program
    trust_label = (
        program.get("trust_label")
        or (entry.get("trust_label") if entry else None)
        or "candidate_screening"
    )
    comparability_label = (
        program.get("comparability_label")
        or (entry.get("comparability_label") if entry else None)
        or "screening_only"
    )

    if not entry:
        entry_id = nb.upsert_leaderboard(
            result_id=result_id,
            model_source=program.get("model_source") or "batch_backlog_promotion",
            architecture_desc=str(program.get("graph_fingerprint") or "")[:40],
            screening_loss_ratio=program.get("loss_ratio"),
            screening_novelty=program.get("novelty_score"),
            screening_passed=bool(program.get("stage1_passed")),
            tier="screening",
            trust_label=trust_label,
            comparability_label=comparability_label,
            notes=(
                "Batch backlog promotion: score_max={:.2f} clears top-25 "
                "threshold per promote_backlog_score_filter".format(
                    float(cand.get("score_max", 0.0))
                )
            ),
        )
        action = "inserted"
    else:
        nb.promote_to_tier(
            entry["entry_id"],
            "screening",
            trust_label=trust_label,
            comparability_label=comparability_label,
            screening_passed=bool(program.get("stage1_passed")),
            screening_loss_ratio=program.get("loss_ratio"),
            screening_novelty=program.get("novelty_score"),
            notes="Batch backlog promotion from promote_backlog_score_filter",
        )
        entry_id = entry["entry_id"]
        action = "promoted"

    nb.conn.execute(
        """
        UPDATE program_results
        SET trust_label = ?, comparability_label = ?, timestamp = ?
        WHERE result_id = ?
        """,
        (trust_label, comparability_label, time.time(), result_id),
    )
    nb.conn.commit()

    return {
        "result_id": result_id,
        "fp": cand.get("graph_fingerprint"),
        "entry_id": str(entry_id) if entry_id is not None else None,
        "action": action,
        "trust_label": trust_label,
        "comparability_label": comparability_label,
        "score_actual": cand.get("score_actual"),
        "score_max": cand.get("score_max"),
        "status": "ok",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument(
        "--top",
        type=int,
        default=None,
        help="Cap to the top N candidates in the input list (0/None = all).",
    )
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    candidates = _load_candidates(args.input, args.top)
    print(f"Loaded {len(candidates)} candidates from {args.input}")
    print(f"{'fp':<16}  {'source':<20}  {'score_max':>9}  {'score_actual':>12}")
    for c in candidates:
        fp = (c.get("graph_fingerprint") or "")[:16]
        src = (c.get("model_source") or "-")[:20]
        sm = c.get("score_max") or 0.0
        sa = c.get("score_actual") or 0.0
        print(f"{fp:<16}  {src:<20}  {sm:>9.2f}  {sa:>12.2f}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to promote.")
        return

    from research.scientist.notebook import LabNotebook

    nb = LabNotebook(str(args.db))
    audit_rows: List[Dict[str, Any]] = []
    try:
        for c in candidates:
            try:
                result = _promote_one(nb, c)
            except Exception as e:
                logger.warning(
                    "Promote failed for %s: %s", c.get("graph_fingerprint"), e
                )
                result = {
                    "result_id": c.get("result_id"),
                    "fp": c.get("graph_fingerprint"),
                    "status": "error",
                    "error": str(e),
                }
            audit_rows.append(result)
            print(
                f"  [{(result.get('fp') or '')[:16]}] {result.get('status')}"
                f" entry_id={result.get('entry_id')}"
                f" action={result.get('action')}"
            )
        nb.flush_writes()
    finally:
        nb.close()

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w") as fh:
        for row in audit_rows:
            fh.write(json.dumps(row) + "\n")
    n_ok = sum(1 for r in audit_rows if r.get("status") == "ok")
    print(f"\nDone. {n_ok}/{len(audit_rows)} promoted. Audit: {args.audit}")


if __name__ == "__main__":
    main()
