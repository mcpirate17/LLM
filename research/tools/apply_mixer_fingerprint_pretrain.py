"""Record a mixer_fingerprint pretrain run into the local runs.db.

``mixer_fingerprint`` is a checkpointed pretrain harness that emits a JSONL
event log + ``.pt`` weights but writes NOTHING to runs.db. For novel-fingerprint
campaigns (e.g. the pq_rope / semiring nano-winners) we want the final, fully
trained eval metrics queryable alongside the screening rows.

This tool replays one finished mixer_fingerprint JSONL into ``graph_runs`` as a
single row per (run-label) keyed by a deterministic ``result_id``, so re-applying
the same log is idempotent (UPSERT). Backs up the DB before writing. The row is
stamped ``trust_label='mixer_pretrain_replay'`` so it is excluded from the
trusted/promotable cohorts (it is a from-scratch pretrain of an existing
fingerprint at a different scale, NOT a comparable screening observation).

Lane → graph_fingerprint is resolved from
``scaling_blimp_study.WINNER_LANE_FINGERPRINTS`` (single source of truth), or
overridden with ``--fingerprint``.

By default only a run whose final checkpoint carries the full ``expensive`` eval
block is applied (no partial-data writes, per project rule); pass
``--allow-partial`` to record a cheap-evals-only checkpoint.

Usage:
    python -m research.tools.apply_mixer_fingerprint_pretrain \
        --jsonl research/checkpoints/mixer_fingerprint/pq_rope_chinchilla_9be78a43_20260528.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from research.tools.scaling_blimp_study import WINNER_LANE_FINGERPRINTS

_DEFAULT_DB = Path("research/runs.db")
_DEFAULT_BACKUPS = Path("research/db_backups")

_TRUST_LABEL = "mixer_pretrain_replay"
_COMPARABILITY = "chinchilla_pretrain"
_COHORT = "mixer_pretrain"
_PROTOCOL = "mixer_fingerprint_pretrain_v1"

# Cheap-eval keys whose names differ from the graph_runs column.
_CHEAP_RENAME = {
    "blimp_overall": "blimp_overall_accuracy",
    "wikitext_ppl": "wikitext_perplexity",
    "hellaswag_acc": "hellaswag_acc",
    "induction_screening_auc": "induction_screening_auc",
}


def _load_events(jsonl: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]


def _final_checkpoint(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Last checkpoint event, preferring one that has an ``expensive`` block."""
    ckpts = [e for e in events if e.get("event") == "checkpoint"]
    if not ckpts:
        return None
    with_exp = [e for e in ckpts if e.get("expensive")]
    return (with_exp or ckpts)[-1]


def _flatten_scalars(prefix_dicts: dict[str, Any]) -> dict[str, Any]:
    """Flatten the expensive sub-dicts: scalars keep their (already column-named)
    key; nested dict/list values are JSON-encoded under ``<key>_json`` when such
    a column exists (decided later against the live schema)."""
    out: dict[str, Any] = {}
    for sub in prefix_dicts.values():
        if not isinstance(sub, dict):
            continue
        for k, v in sub.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                out.setdefault(k, v)
            elif isinstance(v, (dict, list)):
                out.setdefault(f"{k}_json", json.dumps(v))
    return out


def _build_row(
    events: list[dict[str, Any]], fingerprint: str, label: str, jsonl: Path
) -> dict[str, Any]:
    start = next((e for e in events if e.get("event") == "start"), {})
    ckpt = _final_checkpoint(events) or {}
    cheap = ckpt.get("cheap") or {}
    expensive = ckpt.get("expensive") or {}

    row: dict[str, Any] = {}
    # cheap (renamed) scalars
    for src, col in _CHEAP_RENAME.items():
        if isinstance(cheap.get(src), (int, float)):
            row[col] = cheap[src]
    # nb05 nested cheap probe (binding-order screening)
    nb05 = cheap.get("nb05") or {}
    if "max_accuracy" in nb05:
        row["language_control_s05_binding_order_acc"] = nb05.get("max_accuracy")
    # ar_legacy is the one expensive dict whose keys are bare (auc/final_acc)
    al = expensive.get("ar_legacy") or {}
    if "auc" in al:
        row["ar_legacy_auc"] = al.get("auc")
    if "final_acc" in al:
        row["ar_legacy_final_acc"] = al.get("final_acc")
    # all other expensive dicts already use column-named keys → flatten
    row.update(
        _flatten_scalars({k: v for k, v in expensive.items() if k != "ar_legacy"})
    )

    # provenance / regime
    row["param_count"] = start.get("n_params")
    row["n_train_steps"] = ckpt.get("step") or start.get("n_steps")
    row["train_budget_steps"] = start.get("n_steps")
    row["final_lr"] = start.get("min_lr")
    row["stage1_passed"] = 1
    row["trust_label"] = _TRUST_LABEL
    row["comparability_label"] = _COMPARABILITY
    row["result_cohort"] = _COHORT
    row["evaluation_protocol_version"] = _PROTOCOL
    row["model_source"] = "graph_synthesis"
    row["tokenizer_mode"] = "cl100k_base"
    row["data_provenance_json"] = json.dumps(
        {
            "source": "mixer_fingerprint_pretrain",
            "run_label": label,
            "jsonl": str(jsonl),
            "mixer": start.get("mixer"),
            "dim": start.get("dim"),
            "n_blocks": start.get("n_blocks"),
            "seq_len": start.get("seq_len"),
            "batch_size": start.get("batch_size"),
            "learning_rate": start.get("learning_rate"),
            "min_lr": start.get("min_lr"),
            "warmup_steps": start.get("warmup_steps"),
            "lr_schedule": "warmup_cosine",
        }
    )
    row["graph_fingerprint"] = fingerprint
    row["timestamp"] = time.time()
    return {k: v for k, v in row.items() if v is not None}


def _deterministic_result_id(label: str) -> str:
    # Non-cryptographic: stable short id so re-applying the same run UPSERTs.
    digest = hashlib.sha1(label.encode(), usedforsecurity=False).hexdigest()
    return "mfp_" + digest[:11]


def _backup_db(db: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = backup_dir / f"{db.stem}_pre_mfp_pretrain_{ts}.db"
    shutil.copy2(db, dst)
    return dst


def _upsert(con: sqlite3.Connection, result_id: str, row: dict[str, Any]) -> int:
    """UPSERT one graph_runs row; returns the number of columns written."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(graph_runs)").fetchall()}
    payload = {k: v for k, v in row.items() if k in cols}
    payload["result_id"] = result_id
    keys = list(payload)
    placeholders = ", ".join("?" for _ in keys)
    updates = ", ".join(f"{k}=excluded.{k}" for k in keys if k != "result_id")
    con.execute(
        f"INSERT INTO graph_runs ({', '.join(keys)}) VALUES ({placeholders}) "
        f"ON CONFLICT(result_id) DO UPDATE SET {updates}",
        [payload[k] for k in keys],
    )
    return len(payload)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument(
        "--fingerprint",
        type=str,
        default=None,
        help="Override; else resolved from the start event's mixer/lane.",
    )
    ap.add_argument("--db", type=Path, default=_DEFAULT_DB)
    ap.add_argument("--backup-dir", type=Path, default=_DEFAULT_BACKUPS)
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="Apply even without the expensive eval block.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    events = _load_events(args.jsonl)
    start = next((e for e in events if e.get("event") == "start"), {})
    lane = start.get("mixer")
    fingerprint = args.fingerprint
    if fingerprint is None:
        if lane not in WINNER_LANE_FINGERPRINTS:
            print(
                f"lane {lane!r} not in WINNER_LANE_FINGERPRINTS; pass --fingerprint",
                file=sys.stderr,
            )
            return 2
        fingerprint = WINNER_LANE_FINGERPRINTS[lane][0]

    ckpt = _final_checkpoint(events)
    if ckpt is None:
        print("no checkpoint event in JSONL — nothing to apply", file=sys.stderr)
        return 2
    if not ckpt.get("expensive") and not args.allow_partial:
        print(
            f"final checkpoint (step {ckpt.get('step')}) has no expensive eval "
            f"block; pass --allow-partial to record cheap-only",
            file=sys.stderr,
        )
        return 2

    label = args.jsonl.stem
    result_id = _deterministic_result_id(label)
    row = _build_row(events, fingerprint, label, args.jsonl)
    print(
        f"lane={lane} fp={fingerprint} result_id={result_id} "
        f"step={ckpt.get('step')} cols={len(row)}"
    )
    print(
        f"  key metrics: blimp={row.get('blimp_overall_accuracy')} "
        f"ppl={row.get('wikitext_perplexity')} "
        f"ind_val={row.get('induction_validation_auc')} "
        f"bind_v2={row.get('binding_intermediate_auc')} "
        f"ar_val={row.get('ar_validation_final_acc')}"
    )
    if args.dry_run:
        print("  [dry-run] would UPSERT the above into graph_runs")
        return 0

    backup = _backup_db(args.db, args.backup_dir)
    print(f"backed up DB to {backup}")
    con = sqlite3.connect(str(args.db))
    try:
        n = _upsert(con, result_id, row)
        con.commit()
    finally:
        con.close()
    print(f"upserted graph_runs result_id={result_id} ({n} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
