"""Massive backfill: run the controlled-language probe ladder (S0.5/S1.0/Inv)
on leaderboard rows that lack data, then write 9 columns + version per row.

Order: top-N by composite_score descending. Resumable — skips rows that
already have all three tier sa_scores populated.

Cost per fingerprint: ~5s base train + 3 × ~5s probe = ~20s.
Top-200 ≈ 67 min wall.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path

import torch

from research.eval.controlled_lang_probe import (
    CONTROLLED_LANG_METRIC_VERSION,
    controlled_lang_probe,
)
from research.eval.utils import micro_train_loop
from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json
from research.tools._db_maintenance import connect_readonly

logger = logging.getLogger(__name__)

VOCAB_SIZE = 50257
BASE_TRAIN_STEPS = 750  # match screening default

# Tier configs — must match what _V14_CONFIG anchors were calibrated for.
TIERS = (
    ("s05", {"active_vocab_size": 120, "n_train_steps": 40}),
    ("s10", {"active_vocab_size": 200, "n_train_steps": 40}),
    ("inv", {"active_vocab_size": 300, "n_train_steps": 40}),
)


def _select_targets(
    db: Path, top_n: int, force: bool, required_tiers: tuple[str, ...]
) -> list[dict]:
    """Top-N leaderboard rows; skip rows that already have the required
    tier sa_scores populated (idempotent resume)."""
    conn = connect_readonly(db)
    try:
        rows = conn.execute(
            """
            SELECT l.entry_id, l.composite_score, l.tier, pr.result_id,
                   pr.graph_fingerprint, pr.graph_json,
                   pr.controlled_lang_s05_sa_score AS s05,
                   pr.controlled_lang_s10_sa_score AS s10,
                   pr.controlled_lang_inv_sa_score AS inv,
                   pgf.template_name
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id=l.result_id
            LEFT JOIN program_graph_features pgf ON pgf.result_id=l.result_id
            WHERE l.composite_score IS NOT NULL
              AND pr.graph_json IS NOT NULL AND pr.graph_json != '{}'
            ORDER BY l.composite_score DESC
            LIMIT ?
            """,
            (top_n,),
        ).fetchall()
    finally:
        conn.close()
    out = []
    skipped = 0
    for r in rows:
        d = dict(r)
        if not force and all(d.get(t) is not None for t in required_tiers):
            skipped += 1
            continue
        out.append(d)
    if skipped:
        logger.info("skipping %d rows already fully populated", skipped)
    return out


def _train_base(graph_json_str: str, *, device: str) -> torch.nn.Module:
    graph = graph_from_json(graph_json_str)
    model = compile_model([graph]).to(device)
    batches = [torch.randint(0, VOCAB_SIZE, (4, 128), device=device) for _ in range(8)]
    micro_train_loop(
        model, batches, vocab_size=VOCAB_SIZE, n_steps=BASE_TRAIN_STEPS, lr=3e-4
    )
    return model


def _run_one(fp: dict, *, device: str, tier_names: tuple[str, ...]) -> dict | None:
    """Train base, run requested tiers. Returns dict of column → value to write."""
    try:
        model = _train_base(fp["graph_json"], device=device)
    except Exception as exc:  # noqa: BLE001
        logger.error("  %s base train failed: %s", fp["entry_id"], exc)
        return None

    out: dict = {"controlled_lang_metric_version": CONTROLLED_LANG_METRIC_VERSION}
    tiers_by_name = dict(TIERS)
    for tier_name in tier_names:
        cfg = tiers_by_name.get(tier_name)
        if cfg is None:
            logger.warning("  unknown tier %s; skipping", tier_name)
            continue
        try:
            res = controlled_lang_probe(
                model,
                active_vocab_size=cfg["active_vocab_size"],
                n_train_steps=cfg["n_train_steps"],
                device=device,
            )
            sa = (res.synthetic_association or {}).get("synthetic_association_score")
            nb_order = (res.nano_blimp or {}).get("nano_blimp_order_grammaticality_acc")
            nb_score = (res.nano_blimp or {}).get("nano_blimp_score")
            out[f"controlled_lang_{tier_name}_sa_score"] = sa
            out[f"controlled_lang_{tier_name}_nb_order_acc"] = nb_order
            out[f"controlled_lang_{tier_name}_nb_score"] = nb_score
        except Exception as exc:  # noqa: BLE001
            logger.warning("  %s tier %s failed: %s", fp["entry_id"], tier_name, exc)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _write_row(con: sqlite3.Connection, result_id: str, updates: dict) -> int:
    set_clauses = []
    vals = []
    for k, v in updates.items():
        set_clauses.append(f"{k}=?")
        vals.append(v)
    if not set_clauses:
        return 0
    vals.append(result_id)
    con.execute(
        f"UPDATE program_results SET {', '.join(set_clauses)} WHERE result_id=?",
        vals,
    )
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="research/lab_notebook.db", type=Path)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="re-probe even if data exists")
    ap.add_argument(
        "--tiers",
        default="s05,s10,inv",
        help="comma-separated tier names (s05/s10/inv); defaults to all three",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            f"research/reports/controlled_lang_backfill_{int(time.time())}.jsonl"
        ),
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tier_names = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    targets = _select_targets(args.db, args.top_n, args.force, tier_names)
    logger.info(
        "selected %d targets (top-%d, tiers=%s, %s)",
        len(targets),
        args.top_n,
        ",".join(tier_names),
        args.device,
    )

    con = sqlite3.connect(str(args.db), timeout=30.0)
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA busy_timeout=15000")

    t_start = time.perf_counter()
    written = 0
    failed = 0
    with args.out.open("w") as out_fh:
        for idx, fp in enumerate(targets, 1):
            ent = fp["entry_id"]
            t0 = time.perf_counter()
            updates = _run_one(fp, device=args.device, tier_names=tier_names)
            elapsed = time.perf_counter() - t0
            if updates:
                _write_row(con, fp["result_id"], updates)
                con.commit()
                written += 1
                row = {
                    "entry_id": ent,
                    "result_id": fp["result_id"],
                    "fingerprint": fp.get("graph_fingerprint"),
                    "template": fp.get("template_name"),
                    "composite": fp.get("composite_score"),
                    "elapsed_s": round(elapsed, 1),
                    **updates,
                }
                out_fh.write(json.dumps(row) + "\n")
                out_fh.flush()
                logger.info(
                    "[%d/%d] %s: s05_sa=%s s10_sa=%s inv_sa=%s (%.1fs)",
                    idx,
                    len(targets),
                    ent,
                    updates.get("controlled_lang_s05_sa_score"),
                    updates.get("controlled_lang_s10_sa_score"),
                    updates.get("controlled_lang_inv_sa_score"),
                    elapsed,
                )
            else:
                failed += 1

    con.close()
    total = time.perf_counter() - t_start
    logger.info(
        "backfill done: %d written, %d failed in %.1fs (%.1fmin)",
        written,
        failed,
        total,
        total / 60,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
