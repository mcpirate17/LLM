"""Ingest non-nano scale-run eval results (40M/100M: native, surprise, semiring,
slot_table, recip, mor, pq_rope) into runs.db.

Sources (all under research/reports/, auto-pruned at 14d — this captures them
permanently in the DB before they vanish):
  1. frontier_probes/<run>_seed<N>_post_eval.json  -> per-(run,seed) full probe battery
  2. frontier_probes/<run>_blimp.json (+ root *_blimp.json) -> BLiMP overall + by-category
  3. The Minimax normalized leaderboard table in
     research/notes/native_adaptive_reciprocal_slot_delta_rationale_2026-06-12.md

Writes four NEW tables (idempotent, keyed; safe to re-run):
  - scale_run_evals          one row per (run, seed): config + full probes_json blob
  - scale_run_probe_metrics  long/EAV: one row per (run, seed, probe_family, metric_key)
                             — lossless, every numeric leaf preserved
  - scale_run_blimp          one row per (run): blimp_overall + by_category_json
  - scale_run_leaderboard    the Minimax normalized frontier table, verbatim

These are SEPARATE from program_results/leaderboard (the synthesis pipeline's
nano-screening tables). Scale runs are full training runs, not synthesis programs,
so they get their own namespace and do NOT trip the S1 completeness enforcement.

Run:  python research/tools/ingest_scale_runs.py
"""

from __future__ import annotations

import glob
import json
import os
import sqlite3
import time
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(REPO, "research", "runs.db")
PROBES_DIR = os.path.join(REPO, "research", "reports", "frontier_probes")
REPORTS_DIR = os.path.join(REPO, "research", "reports")
NOTE_PATH = os.path.join(
    REPO,
    "research",
    "notes",
    "native_adaptive_reciprocal_slot_delta_rationale_2026-06-12.md",
)

SOURCE_NOTE = "native_adaptive_reciprocal_slot_delta_rationale_2026-06-12.md"


def _ddl(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS scale_run_evals (
            run_name TEXT NOT NULL,
            seed INTEGER NOT NULL,
            mixer TEXT,
            dim INTEGER,
            n_blocks INTEGER,
            use_ffn INTEGER,
            n_params INTEGER,
            checkpoint TEXT,
            probes_json TEXT NOT NULL,
            source_file TEXT NOT NULL,
            ingested_at REAL NOT NULL,
            PRIMARY KEY (run_name, seed)
        );

        CREATE TABLE IF NOT EXISTS scale_run_probe_metrics (
            run_name TEXT NOT NULL,
            seed INTEGER NOT NULL,
            probe_family TEXT NOT NULL,
            metric_key TEXT NOT NULL,
            value_num REAL,
            value_text TEXT,
            source_file TEXT NOT NULL,
            ingested_at REAL NOT NULL,
            PRIMARY KEY (run_name, seed, probe_family, metric_key)
        );

        CREATE TABLE IF NOT EXISTS scale_run_blimp (
            run_name TEXT PRIMARY KEY,
            lane TEXT,
            ckpt TEXT,
            step INTEGER,
            dim INTEGER,
            n_blocks INTEGER,
            use_ffn INTEGER,
            n_params_m REAL,
            blimp_overall REAL,
            n_subtasks INTEGER,
            n_per_subtask INTEGER,
            coverage REAL,
            by_category_json TEXT,
            source_file TEXT NOT NULL,
            ingested_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scale_run_leaderboard (
            rank INTEGER,
            model TEXT PRIMARY KEY,
            seq TEXT,
            tokens_m REAL,
            active_m REAL,
            tok_per_active_p REAL,
            avg_norm REAL,
            blimp_norm REAL,
            ar_cur_norm REAL,
            ar_held_norm REAL,
            bind_norm REAL,
            ms_auc_norm REAL,
            ms_all_norm REAL,
            ind_norm REAL,
            ind_val_norm REAL,
            metrics_are_normalized INTEGER NOT NULL DEFAULT 1,
            source_note TEXT NOT NULL,
            ingested_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_srpm_run ON scale_run_probe_metrics(run_name);
        CREATE INDEX IF NOT EXISTS idx_srpm_metric ON scale_run_probe_metrics(metric_key);
        """
    )


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted leaf keys. Lists/scalars become leaves."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(_flatten(v, key + "."))
            else:
                out[key] = v
    else:
        out[prefix.rstrip(".")] = obj
    return out


def ingest_post_evals(conn: sqlite3.Connection, now: float) -> tuple[int, int]:
    files = sorted(glob.glob(os.path.join(PROBES_DIR, "*_seed*_post_eval.json")))
    # Also the standalone slot_table post_eval at reports root (no _seed suffix).
    n_runs = 0
    n_metrics = 0
    for path in files:
        with open(path) as fh:
            d = json.load(fh)
        base = os.path.basename(path)
        run_name = base.split("_seed")[0]
        seed = int(d.get("seed", base.split("_seed")[1].split("_")[0]))
        probes = d.get("probes", {})
        conn.execute(
            """INSERT OR REPLACE INTO scale_run_evals
               (run_name, seed, mixer, dim, n_blocks, use_ffn, n_params,
                checkpoint, probes_json, source_file, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_name,
                seed,
                d.get("mixer"),
                d.get("dim"),
                d.get("n_blocks"),
                int(bool(d.get("use_ffn"))) if d.get("use_ffn") is not None else None,
                d.get("n_params"),
                d.get("checkpoint"),
                json.dumps(probes),
                base,
                now,
            ),
        )
        n_runs += 1
        for family, fam_val in probes.items():
            if family.startswith("_t_"):  # timing scalars
                family_name, payload = "_timing", {family: fam_val}
            elif isinstance(fam_val, dict):
                family_name, payload = family, fam_val
            else:
                family_name, payload = "_top", {family: fam_val}
            for mkey, mval in _flatten(payload).items():
                num: float | None = None
                txt: str | None = None
                if isinstance(mval, bool):
                    num = float(mval)
                elif isinstance(mval, (int, float)):
                    num = float(mval)
                elif mval is None:
                    txt = None
                else:
                    txt = str(mval)
                conn.execute(
                    """INSERT OR REPLACE INTO scale_run_probe_metrics
                       (run_name, seed, probe_family, metric_key, value_num,
                        value_text, source_file, ingested_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (run_name, seed, family_name, mkey, num, txt, base, now),
                )
                n_metrics += 1
    return n_runs, n_metrics


def ingest_blimp(conn: sqlite3.Connection, now: float) -> int:
    paths = sorted(
        set(glob.glob(os.path.join(PROBES_DIR, "*_blimp.json")))
        | set(glob.glob(os.path.join(REPORTS_DIR, "*_blimp.json")))
    )
    n = 0
    for path in paths:
        with open(path) as fh:
            d = json.load(fh)
        rec = (
            d[0] if isinstance(d, list) and d else (d if isinstance(d, dict) else None)
        )
        if not rec:
            continue
        run_name = os.path.basename(path).replace("_blimp.json", "")
        conn.execute(
            """INSERT OR REPLACE INTO scale_run_blimp
               (run_name, lane, ckpt, step, dim, n_blocks, use_ffn, n_params_m,
                blimp_overall, n_subtasks, n_per_subtask, coverage,
                by_category_json, source_file, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_name,
                rec.get("lane"),
                rec.get("ckpt"),
                rec.get("step"),
                rec.get("dim"),
                rec.get("n_blocks"),
                int(bool(rec.get("use_ffn")))
                if rec.get("use_ffn") is not None
                else None,
                rec.get("n_params_m"),
                rec.get("blimp_overall"),
                rec.get("n_subtasks"),
                rec.get("n_per_subtask"),
                rec.get("coverage"),
                json.dumps(rec.get("by_category")) if rec.get("by_category") else None,
                os.path.basename(path),
                now,
            ),
        )
        n += 1
    return n


def _parse_leaderboard_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    with open(NOTE_PATH) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
            if len(cells) != 15:
                continue
            if cells[0] in ("Rank", "") or set(cells[0]) <= set("-: "):
                continue
            if not cells[0].isdigit():
                continue
            rows.append(cells)
    return rows


def ingest_leaderboard(conn: sqlite3.Connection, now: float) -> int:
    rows = _parse_leaderboard_rows()

    def f(x: str) -> float | None:
        try:
            return float(x)
        except ValueError:
            return None

    n = 0
    for c in rows:
        conn.execute(
            """INSERT OR REPLACE INTO scale_run_leaderboard
               (rank, model, seq, tokens_m, active_m, tok_per_active_p, avg_norm,
                blimp_norm, ar_cur_norm, ar_held_norm, bind_norm, ms_auc_norm,
                ms_all_norm, ind_norm, ind_val_norm, metrics_are_normalized,
                source_note, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)""",
            (
                int(c[0]),
                c[1],
                c[2],
                f(c[3]),
                f(c[4]),
                f(c[5]),
                f(c[6]),
                f(c[7]),
                f(c[8]),
                f(c[9]),
                f(c[10]),
                f(c[11]),
                f(c[12]),
                f(c[13]),
                f(c[14]),
                SOURCE_NOTE,
                now,
            ),
        )
        n += 1
    return n


def main() -> None:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"runs.db not found at {DB_PATH}")
    now = time.time()
    conn = sqlite3.connect(DB_PATH)
    try:
        _ddl(conn)
        n_runs, n_metrics = ingest_post_evals(conn, now)
        n_blimp = ingest_blimp(conn, now)
        n_lb = ingest_leaderboard(conn, now)
        conn.commit()
    finally:
        conn.close()
    print(f"scale_run_evals:          {n_runs} (run,seed) rows")
    print(f"scale_run_probe_metrics:  {n_metrics} metric rows")
    print(f"scale_run_blimp:          {n_blimp} runs")
    print(f"scale_run_leaderboard:    {n_lb} leaderboard rows")


if __name__ == "__main__":
    main()
