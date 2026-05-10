"""Compare old S1.0 controlled NanoBind scores against the harder v2 probe."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from statistics import mean

import torch

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.tools._db_maintenance import connect_readonly
from research.tools.language_control_backfill import _train_base
from research.eval.language_control_probe import language_control_probe


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "research" / "reports"


def _fetch_rows(
    db: Path, *, limit: int | None, include_screened_out: bool
) -> list[sqlite3.Row]:
    conn = connect_readonly(db)
    tier_clause = (
        ""
        if include_screened_out
        else "AND COALESCE(l.tier, '') NOT IN ('screened_out', 'retired')"
    )
    limit_clause = "" if limit is None else "LIMIT ?"
    params: tuple[int, ...] = () if limit is None else (int(limit),)
    rows = conn.execute(
        f"""
        SELECT l.tier,
               l.composite_score,
               pr.result_id,
               pr.graph_fingerprint,
               pr.graph_json,
               pr.language_control_s05_binding_score AS old_s05_nb,
               pr.language_control_s10_binding_score AS old_s10_nb,
               pr.language_control_s10_binding_order_acc AS old_s10_order
        FROM leaderboard l
        JOIN program_results_compat pr ON pr.result_id = l.result_id
        WHERE pr.language_control_s10_binding_score IS NOT NULL
          AND COALESCE(l.is_reference, 0) = 0
          {tier_clause}
        ORDER BY l.composite_score DESC
        {limit_clause}
        """,
        params,
    ).fetchall()
    payloads = []
    for row in rows:
        payload = dict(row)
        payload["graph_json"] = resolve_graph_json_value(
            conn, db, payload["graph_json"]
        )
        payloads.append(payload)
    conn.close()
    return payloads


def _round_or_none(value):
    if value is None:
        return None
    return round(float(value), 4)


def _summarize(rows: list[dict], elapsed_s: float) -> dict:
    deltas = [
        r["new_s10_nb"] - r["old_s10_nb"]
        for r in rows
        if r.get("new_s10_nb") is not None and r.get("old_s10_nb") is not None
    ]
    elapsed = [r["elapsed_s"] for r in rows if r.get("elapsed_s") is not None]
    worsened = [d for d in deltas if d < 0]
    improved = [d for d in deltas if d > 0]
    return {
        "n": len(rows),
        "elapsed_s": round(elapsed_s, 1),
        "avg_elapsed_s_per_fingerprint": round(mean(elapsed), 2) if elapsed else None,
        "avg_delta_new_minus_old": round(mean(deltas), 4) if deltas else None,
        "n_worse": len(worsened),
        "n_same": len(deltas) - len(worsened) - len(improved),
        "n_better": len(improved),
        "avg_new_s10_nb": round(mean(r["new_s10_nb"] for r in rows), 4)
        if rows
        else None,
        "avg_old_s10_nb": round(mean(r["old_s10_nb"] for r in rows), 4)
        if rows
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "research/runs.db")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-screened-out", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPORTS_DIR
        / f"language_control_s10_vocab240_comparison_{int(time.time())}.json",
    )
    args = parser.parse_args()

    rows = _fetch_rows(
        args.db, limit=args.limit, include_screened_out=args.include_screened_out
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    out_rows: list[dict] = []
    for idx, row in enumerate(rows, 1):
        t0 = time.perf_counter()
        status = "ok"
        error = None
        new_nb = None
        new_order = None
        checkpoints = []
        model = None
        try:
            model = _train_base(row["graph_json"], device=args.device)
            result = language_control_probe(
                model,
                active_vocab_size=240,
                n_train_steps=2000,
                checkpoint_steps=(500, 1000, 2000),
                timeout_s=240.0,
                device=args.device,
                preserve_state=False,
            )
            payload = result.to_dict()
            new_nb = result.nano_blimp.get("nano_blimp_score")
            new_order = result.nano_blimp.get("nano_blimp_order_grammaticality_acc")
            checkpoints = payload.get("language_control_checkpoints") or []
            status = result.status
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = str(exc)
        finally:
            if model is not None:
                del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        elapsed = time.perf_counter() - t0
        out_row = {
            "index": idx,
            "result_id": row["result_id"],
            "fingerprint": row["graph_fingerprint"],
            "tier": row["tier"],
            "composite_score": _round_or_none(row["composite_score"]),
            "old_s05_nb": _round_or_none(row["old_s05_nb"]),
            "old_s10_nb": _round_or_none(row["old_s10_nb"]),
            "old_s10_order": _round_or_none(row["old_s10_order"]),
            "new_s10_nb": _round_or_none(new_nb),
            "new_s10_order": _round_or_none(new_order),
            "delta_new_minus_old": _round_or_none(
                None if new_nb is None else float(new_nb) - float(row["old_s10_nb"])
            ),
            "checkpoints": checkpoints,
            "elapsed_s": round(elapsed, 2),
            "status": status,
            "error": error,
        }
        out_rows.append(out_row)
        print(
            f"[{idx}/{len(rows)}] {row['result_id'][:8]} "
            f"old={out_row['old_s10_nb']} new={out_row['new_s10_nb']} "
            f"delta={out_row['delta_new_minus_old']} elapsed={out_row['elapsed_s']}s "
            f"status={status}",
            flush=True,
        )

    report = {
        "config": {
            "old_active_vocab_size": 200,
            "new_active_vocab_size": 240,
            "new_train_steps": 2000,
            "new_checkpoints": [500, 1000, 2000],
            "device": args.device,
            "include_screened_out": bool(args.include_screened_out),
            "limit": args.limit,
        },
        "summary": _summarize(out_rows, time.perf_counter() - started),
        "rows": out_rows,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote={args.out}")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
