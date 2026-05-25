#!/usr/bin/env python
"""Backpopulate comprehensive graph semantic features across ALL historical graphs.

Computes the math + structure + control-flow feature vector (graph_semantic_features.py)
for every graph in runs.db `graphs` and stores it, versioned, in meta_analysis.db
`graph_semantic_features` so any model can consume it without recomputation.

Re-runnable and incremental: skips fingerprints already at the current FEATURE_VERSION
(unless --force). When the extractor grows new features, bump FEATURE_VERSION and re-run.
Stored as a JSON blob (extensible: new features just appear; consumers read what they need).

Usage::

    python -m research.tools.backfill_graph_semantics                 # incremental
    python -m research.tools.backfill_graph_semantics --force         # recompute all
    python -m research.tools.backfill_graph_semantics --limit 1000    # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from typing import Dict, List, Tuple

import numpy as np

from research.defaults import RUNS_DB
from research.tools.graph_semantic_features import (
    FEATURE_VERSION,
    GraphSemanticExtractor,
)

logger = logging.getLogger(__name__)
_META_DB = "research/meta_analysis.db"
_TABLE = "graph_semantic_features"
# Table name is a fixed module constant (not user input); queries are safe.
_Q_CREATE = f"CREATE TABLE IF NOT EXISTS {_TABLE} (graph_fingerprint TEXT PRIMARY KEY, feature_version TEXT, n_features INTEGER, features_json TEXT, computed_ts REAL)"  # nosec B608  # nosemgrep: python-sql-string-formatting
_Q_DONE = f"SELECT graph_fingerprint FROM {_TABLE} WHERE feature_version=?"  # nosec B608  # nosemgrep: python-sql-string-formatting
_Q_INSERT = f"INSERT OR REPLACE INTO {_TABLE} VALUES (?,?,?,?,?)"  # nosec B608  # nosemgrep: python-sql-string-formatting
_Q_COUNT = f"SELECT COUNT(*) FROM {_TABLE} WHERE feature_version=?"  # nosec B608  # nosemgrep: python-sql-string-formatting
_Q_LOAD = (
    f"SELECT graph_fingerprint, features_json FROM {_TABLE} WHERE feature_version=?"  # nosec B608  # nosemgrep: python-sql-string-formatting
)


def _ensure_table(con: sqlite3.Connection) -> None:
    con.execute(_Q_CREATE)
    con.commit()


def run(runs_db: str, meta_db: str, force: bool, limit: int, batch: int = 2000) -> Dict:
    ext = GraphSemanticExtractor(runs_db, meta_db)
    src = sqlite3.connect(runs_db)
    dst = sqlite3.connect(meta_db)
    _ensure_table(dst)
    done = set()
    if not force:
        done = {r[0] for r in dst.execute(_Q_DONE, (FEATURE_VERSION,))}
    q = "SELECT graph_fingerprint, graph_json FROM graphs WHERE graph_json_is_placeholder=0"
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = src.execute(q).fetchall()
    src.close()

    t0 = time.time()
    pending: List[Tuple] = []
    n_done = n_skip = n_err = 0
    for fp, gj in rows:
        fp = str(fp)
        if fp in done:
            n_skip += 1
            continue
        try:
            payload = json.loads(gj)
            feats = ext.features(
                payload["nodes"], model_dim=int(payload.get("model_dim", 256))
            )
            pending.append(
                (fp, FEATURE_VERSION, len(feats), json.dumps(feats), time.time())
            )
            n_done += 1
        except Exception:
            n_err += 1
        if len(pending) >= batch:
            dst.executemany(_Q_INSERT, pending)
            dst.commit()
            pending.clear()
            logger.info(
                "  %d computed (%.0f/s)", n_done, n_done / max(time.time() - t0, 1e-9)
            )
    if pending:
        dst.executemany(_Q_INSERT, pending)
        dst.commit()
    total = dst.execute(_Q_COUNT, (FEATURE_VERSION,)).fetchone()[0]
    dst.close()
    return {
        "feature_version": FEATURE_VERSION,
        "scanned": len(rows),
        "computed": n_done,
        "skipped_existing": n_skip,
        "errors": n_err,
        "elapsed_s": round(time.time() - t0, 1),
        "total_in_table": total,
    }


def load_semantic_features(
    fps: List[str], meta_db: str = _META_DB
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load backfilled features for ``fps`` -> (X, feature_names, fps_with_features).

    Feature order is fixed (sorted names from the first row). Fingerprints without a
    stored vector are dropped from the returned fps list.
    """
    con = sqlite3.connect(meta_db)
    have = {
        str(fp): json.loads(js)
        for fp, js in con.execute(
            _Q_LOAD,
            (FEATURE_VERSION,),
        )
    }
    con.close()
    present = [fp for fp in fps if fp in have]
    if not present:
        return np.zeros((0, 0)), [], []
    names = sorted(have[present[0]].keys())
    X = np.array(
        [[have[fp].get(n, 0.0) for n in names] for fp in present], dtype=np.float64
    )
    return X, names, present


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--meta-db", default=_META_DB)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    print(
        json.dumps(
            run(args.db, args.meta_db, args.force, args.limit), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
