"""NanoBind-S0 leaderboard backfill.

Runs `research.eval.nano_bind.nano_bind` on every leaderboard row that
has a `graph_json` and is not already `screened_out`/`retired`.

Sidecar JSON per arch at `research/reports/nano_bind_backfill/{result_id}.json`
is the resume marker. For `is_no_go=True && status=='ok'`:

    UPDATE leaderboard SET tier='screened_out' WHERE result_id=?
    UPDATE program_results SET failure_op='nano_bind',
           failure_details_json=? WHERE result_id=?

Pass rows and error/timeout rows are NEVER written.

Usage:
    python -m research.tools.nano_bind_backfill                  # full run
    python -m research.tools.nano_bind_backfill --limit 5        # smoke
    python -m research.tools.nano_bind_backfill --result-ids a,b # subset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aria_db

from research.eval.nano_bind import NANO_BIND_METRIC_VERSION, nano_bind

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "research" / "lab_notebook.db"
SIDECAR_DIR = REPO_ROOT / "research" / "reports" / "nano_bind_backfill"
SUMMARY_PATH = SIDECAR_DIR / "SUMMARY.md"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_candidates(
    mgr: aria_db.ConnectionManager,
    *,
    result_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    sql = """
        SELECT l.result_id    AS result_id,
               l.tier         AS tier_before,
               l.graph_fingerprint AS graph_fingerprint,
               p.graph_json   AS graph_json
          FROM leaderboard l
          JOIN program_results p ON p.result_id = l.result_id
         WHERE p.graph_json IS NOT NULL
           AND p.graph_json != ''
           AND p.graph_json != '{}'
           AND l.tier NOT IN ('screened_out','retired')
    """
    if result_ids:
        placeholders = ",".join("?" for _ in result_ids)
        sql += f" AND l.result_id IN ({placeholders})"
    sql += " ORDER BY l.tier, l.result_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = mgr.fetchall(sql, list(result_ids) if result_ids else [])
    return rows


def sidecar_path(result_id: str) -> Path:
    safe = result_id.replace("/", "_")
    return SIDECAR_DIR / f"{safe}.json"


def write_sidecar(
    result_id: str,
    *,
    tier_before: str,
    graph_fingerprint: str | None,
    decision: str,
    result_dict: dict | None,
    elapsed_s: float,
    error: str | None = None,
) -> None:
    payload: dict = {
        "result_id": result_id,
        "tier_before": tier_before,
        "graph_fingerprint": graph_fingerprint,
        "decision": decision,
        "ran_at_utc": utc_now_iso(),
        "elapsed_s": round(elapsed_s, 3),
    }
    if result_dict is not None:
        payload.update(result_dict)
    if error is not None:
        payload["script_error"] = error
    tmp = sidecar_path(result_id).with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f)
    tmp.replace(sidecar_path(result_id))


def apply_no_go(mgr: aria_db.ConnectionManager, result_id: str, result) -> None:
    failure_details = json.dumps(
        {
            "reason": "nano_bind_persistent_zero",
            "scores": list(result.scores),
            "metric_version": NANO_BIND_METRIC_VERSION,
            "checkpoints": list((result.sweep_metadata or {}).get("checkpoints", [])),
        },
        separators=(",", ":"),
    )
    mgr.execute(
        "UPDATE leaderboard SET tier='screened_out' WHERE result_id=?",
        [result_id],
    )
    mgr.execute(
        "UPDATE program_results SET failure_op='nano_bind', "
        "failure_details_json=? WHERE result_id=?",
        [failure_details, result_id],
    )
    mgr.commit()


def write_summary(stats: dict, samples: dict) -> None:
    lines = [
        "# NanoBind 17K backfill — summary",
        "",
        f"Generated: {utc_now_iso()}",
        "Script: `research/tools/nano_bind_backfill.py`",
        f"Metric version: `{NANO_BIND_METRIC_VERSION}`",
        "",
        "## Totals",
        "",
        f"- Eligible candidates seen: **{stats['seen']}**",
        f"- Processed this run: **{stats['processed']}**",
        f"- Resumed (sidecar already present): **{stats['resumed']}**",
        f"- NO-GO (tier→screened_out): **{stats['no_go']}**",
        f"- PASS (no leaderboard mutation): **{stats['pass']}**",
        f"- Errors / non-ok status: **{stats['error']}**",
        "",
        "## Sample no-go (10)",
        "",
        *(f"- `{rid}`" for rid in samples["no_go"][:10]),
        "",
        "## Sample pass (10)",
        "",
        *(f"- `{rid}`" for rid in samples["pass"][:10]),
        "",
        "## Errors (10)",
        "",
        *(f"- `{rid}`: {err}" for rid, err in samples["error"][:10]),
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--limit", type=int, default=None, help="Cap on candidates (smoke runs)"
    )
    p.add_argument(
        "--result-ids",
        type=str,
        default=None,
        help="Comma-separated explicit result_id allow-list",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Run NanoBind + write sidecars only; skip DB UPDATE",
    )
    p.add_argument("--progress-every", type=int, default=25)
    return p.parse_args()


def _replay_sidecar(sp: Path, rid: str, samples: dict) -> None:
    try:
        prior = json.loads(sp.read_text())
    except Exception:
        return
    d = prior.get("decision")
    if d == "no_go":
        samples["no_go"].append(rid)
    elif d == "pass":
        samples["pass"].append(rid)
    elif d == "error":
        samples["error"].append((rid, prior.get("script_error", "?")))


def _run_one(graph_json: str, device: str):
    """Returns (decision, result_or_none, err_or_none)."""
    try:
        result = nano_bind(graph_json, device=device, seed=0)
    except Exception as exc:
        return "error", None, f"{type(exc).__name__}: {exc}"
    if result.status != "ok":
        return "error", result, f"status={result.status} error={result.error}"
    return ("no_go" if result.is_no_go else "pass"), result, None


def _record_outcome(
    *, rid, decision, result, err, stats, samples, mgr, no_write
) -> None:
    if decision == "no_go":
        stats["no_go"] += 1
        samples["no_go"].append(rid)
        if not no_write:
            apply_no_go(mgr, rid, result)
    elif decision == "pass":
        stats["pass"] += 1
        samples["pass"].append(rid)
    else:
        stats["error"] += 1
        samples["error"].append((rid, err or "?"))


def _print_progress(i: int, total: int, stats: dict, t_start: float) -> None:
    wall = time.time() - t_start
    rate = stats["processed"] / wall if wall > 0 else 0.0
    remaining = (total - i) / rate if rate > 0 else float("inf")
    print(
        f"[backfill] {i}/{total} "
        f"no_go={stats['no_go']} pass={stats['pass']} "
        f"err={stats['error']} resumed={stats['resumed']} "
        f"rate={rate:.2f}/s eta={remaining / 3600:.1f}h",
        flush=True,
    )


def _process_candidate(row, args, mgr, stats, samples) -> None:
    rid = row["result_id"]
    sp = sidecar_path(rid)
    if sp.exists():
        stats["resumed"] += 1
        _replay_sidecar(sp, rid, samples)
        return

    t0 = time.time()
    decision, result, err = _run_one(row["graph_json"], args.device)
    elapsed = time.time() - t0

    write_sidecar(
        rid,
        tier_before=row["tier_before"],
        graph_fingerprint=row.get("graph_fingerprint"),
        decision=decision,
        result_dict=result.to_dict() if result is not None else None,
        elapsed_s=elapsed,
        error=err,
    )
    _record_outcome(
        rid=rid,
        decision=decision,
        result=result,
        err=err,
        stats=stats,
        samples=samples,
        mgr=mgr,
        no_write=args.no_write,
    )
    stats["processed"] += 1


def main() -> int:
    args = _parse_args()
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    mgr = aria_db.get_manager(str(DB_PATH))

    rid_filter = (
        [s.strip() for s in args.result_ids.split(",") if s.strip()]
        if args.result_ids
        else None
    )
    cands = fetch_candidates(mgr, result_ids=rid_filter, limit=args.limit)
    print(f"[backfill] candidates: {len(cands)} (db={DB_PATH})", flush=True)
    print(f"[backfill] sidecar dir: {SIDECAR_DIR}", flush=True)
    print(f"[backfill] device={args.device} no_write={args.no_write}", flush=True)

    stats = {
        "seen": len(cands),
        "processed": 0,
        "resumed": 0,
        "no_go": 0,
        "pass": 0,
        "error": 0,
    }
    samples: dict[str, list] = {"no_go": [], "pass": [], "error": []}

    t_start = time.time()
    for i, row in enumerate(cands, 1):
        _process_candidate(row, args, mgr, stats, samples)
        if (i % args.progress_every == 0) or (i == len(cands)):
            _print_progress(i, len(cands), stats, t_start)

    write_summary(stats, samples)
    print(f"[backfill] done. summary -> {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
