"""Top-10 score-stability rerun batch (codex-approved overnight scope).

Drives 10 fingerprints × 2 independent_sample replays each = 20 replays.
Captures per-fp before/after deltas + new child result_ids and posts a
compact JSON report.
"""

from __future__ import annotations

import json
import sqlite3
import time
import traceback
from pathlib import Path
from typing import Any

from research.tools.exact_graph_replay import run_exact_replay

DB = "/home/tim/Projects/LLM/research/lab_notebook.db"
RERUNS_PER_FP = 2

TOP_10: list[tuple[str, str]] = [
    ("23869ebd15bdeebe", "ebdd9a5c-8a3"),
    ("a216b7758ba7bd19", "d49297a5-3bd"),
    ("9220192dd0f33fad", "62e2e371-befd-4e10-9a4b-76d9b41baf59"),
    ("0cffa5cff90c3bc5", "80805639-fc0"),
    ("f86a6903d32c4ab6", "ec7025d7-338"),
    ("10cfec26d76e7d29", "13442e9f-aea"),
    ("9593d8b29bfaaa15", "a311fc5d-12c"),
    ("9563316e6ab93d01", "9de8edeb-e688-4ea4-b375-459e5ab22ed3"),
    ("7270410f55896bc0", "dc0d8d48-12b"),
    ("e6354a0e77b7798f", "ad096d0b-286"),
]


def snapshot(fp: str) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    n_total = conn.execute(
        "SELECT COUNT(*) FROM program_results WHERE graph_fingerprint=?", (fp,)
    ).fetchone()[0]
    n_bpe = conn.execute(
        """SELECT COUNT(*) FROM program_results WHERE graph_fingerprint=?
           AND screening_wikitext_metric_version IN
               ('bpe_eval_v1','screening_wikitext_v2_bpe')""",
        (fp,),
    ).fetchone()[0]
    lb = conn.execute(
        """SELECT entry_id, result_id, ROUND(composite_score,1) AS score,
            n_runs, replication_n, ROUND(cv_loss,3) AS cv_loss,
            ROUND(replication_loss_mean,3) AS lr_mean
           FROM leaderboard WHERE graph_fingerprint=?""",
        (fp,),
    ).fetchone()
    pr_ids = [
        r["result_id"]
        for r in conn.execute(
            "SELECT result_id FROM program_results WHERE graph_fingerprint=? ORDER BY timestamp",
            (fp,),
        ).fetchall()
    ]
    # Note any non-independent debug/orphan rows already on the fp
    debug_rows = [
        dict(r)
        for r in conn.execute(
            """SELECT result_id, intentional_rerun_reason, screening_wikitext_metric_version AS mv
               FROM program_results WHERE graph_fingerprint=?
               AND (intentional_rerun_reason IS NOT NULL
                    OR model_source='exact_graph_replay')""",
            (fp,),
        ).fetchall()
    ]
    conn.close()
    return {
        "n_total": n_total,
        "n_bpe": n_bpe,
        "lb": dict(lb) if lb else None,
        "pr_ids": pr_ids,
        "preexisting_replay_rows": debug_rows,
    }


def fetch_new_row(rid: str) -> dict[str, Any] | None:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    nm = conn.execute(
        """SELECT result_id, experiment_id,
               screening_wikitext_metric_version AS mv,
               ROUND(wikitext_perplexity,2) AS ppl,
               ROUND(loss_ratio,3) AS lr,
               n_train_steps, intentional_rerun_reason, stage1_passed
           FROM program_results WHERE result_id=?""",
        (rid,),
    ).fetchone()
    conn.close()
    return dict(nm) if nm else None


def run_one(fp: str, source_rid: str, idx: int) -> dict[str, Any]:
    err: str | None = None
    exp_id: str | None = None
    t0 = time.time()
    try:
        exp_id = run_exact_replay(
            db_path=Path(DB),
            result_ids=[source_rid],
            repeat_per_source=1,
            device="cuda",
            hypothesis=f"top-10 canary {fp[:8]} rerun_{idx}",
            fast=False,
            verbose=False,
            independent_sample=True,
        )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    return {"exp_id": exp_id, "err": err, "elapsed_s": round(time.time() - t0, 1)}


def main() -> None:
    print(f"=== top-10 canary, {RERUNS_PER_FP} reruns/fp ===")
    overall_t0 = time.time()
    per_fp: list[dict[str, Any]] = []

    for fp_idx, (fp, source_rid) in enumerate(TOP_10, start=1):
        print(f"\n[{fp_idx}/{len(TOP_10)}] {fp} (source rid {source_rid})")
        before = snapshot(fp)
        print(
            f"  BEFORE: pr_total={before['n_total']} pr_bpe={before['n_bpe']} "
            f"n_runs={before['lb']['n_runs'] if before['lb'] else None} "
            f"cv_loss={before['lb']['cv_loss'] if before['lb'] else None}"
        )
        if before["preexisting_replay_rows"]:
            print(
                f"  NOTE: {len(before['preexisting_replay_rows'])} pre-existing "
                f"replay/debug rows on this fp"
            )

        runs: list[dict[str, Any]] = []
        for i in range(RERUNS_PER_FP):
            run_meta = run_one(fp, source_rid, i + 1)
            runs.append(run_meta)
            print(
                f"  rerun {i + 1}/{RERUNS_PER_FP}: exp_id={run_meta['exp_id']} "
                f"err={run_meta['err']} elapsed={run_meta['elapsed_s']}s"
            )

        after = snapshot(fp)
        new_child_rids = [r for r in after["pr_ids"] if r not in before["pr_ids"]]
        new_rows = [fetch_new_row(r) for r in new_child_rids]
        lb_reused = (
            before["lb"] is not None
            and after["lb"] is not None
            and before["lb"]["entry_id"] == after["lb"]["entry_id"]
            and before["lb"]["result_id"] == after["lb"]["result_id"]
        )
        per_fp.append(
            {
                "fp": fp,
                "source_rid": source_rid,
                "runs": runs,
                "before": before,
                "after": after,
                "new_child_rids": new_child_rids,
                "new_rows": new_rows,
                "lb_reused": lb_reused,
            }
        )
        print(
            f"  AFTER:  pr_total={after['n_total']} pr_bpe={after['n_bpe']} "
            f"n_runs={after['lb']['n_runs']} cv_loss={after['lb']['cv_loss']}"
        )
        print(f"  NEW children: {new_child_rids}  lb_reused={lb_reused}")

    overall_elapsed = time.time() - overall_t0
    print(f"\n=== complete in {overall_elapsed:.1f}s ===")

    out = Path("/home/tim/Projects/LLM/research/perf_artifacts") / (
        f"top10_canary_{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    out.write_text(
        json.dumps(
            {
                "rerun_count_per_fp": RERUNS_PER_FP,
                "elapsed_s": round(overall_elapsed, 1),
                "per_fp": per_fp,
            },
            indent=2,
            default=str,
        )
    )
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
