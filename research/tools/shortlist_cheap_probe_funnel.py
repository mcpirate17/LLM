#!/usr/bin/env python
"""Cheap-probe funnel over a CPU-cascade shortlist JSONL -> graph_runs + ranked report.

The CPU screening cascade emits a rule-clean shortlist of FULL graph dicts but
does NOT measure capability (its probe-oracle only PREDICTS it). This driver
MEASURES the real cheap probes over those graphs, cheapest-first, with a hard
no-go skip -- the funnel the handoff asks for:

  1. ar_gate          (research.eval.ar_gate)      -- permissive AR GATE. On
                                                      no-go / score < --ar-pass,
                                                      STOP (skip the rest).
  2. nano_induction_nearest (backfill.run_nano_induction_nearest_probe)
                                                   -- strong structural axis,
                                                      primary ranking signal.
  3. nb0.5 / nb1.0    (_train_base + language_control_probe at s05 / s10)
                                                   -- binding probes.

Each shortlist graph is registered into runs.db (graphs + graph_runs via the
notebook record path, quality-gate bypassed, NO stage1 claim) so probe results
have a home; every metric is persisted to graph_runs via
``backfill.store_probe_results``. This is ONLY the missing JSONL-shortlist
driver -- all probe computation reuses existing functions.

Resumable: graphs that already have ``language_control_s10_binding_score`` (or
an AR no-go verdict) are skipped. Incremental report writes. Waits for a free
GPU at startup so it can be launched behind another run.

Run from repo root:
  python -m research.tools.shortlist_cheap_probe_funnel \
      --jsonl research/reports/cpu_cascade_million_shortlist_clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from research.defaults import RUNS_DB
from research.eval.ar_gate import (
    ARGateConfig,
    ar_gate,
    ar_gate_is_no_go,
    ar_gate_score,
)
from research.eval.language_control_probe import language_control_probe
from research.scientist.notebook import LabNotebook
from research.tools.backfill import (
    clear_gpu,
    run_nano_induction_nearest_probe,
    store_probe_results,
)
from research.tools._metric_backfill_common import read_total_gpu_mib
from research.tools.language_control_backfill import TIERS, _train_base

DEV = "cuda"
AR_PASS = 0.6
_CFG = dict(TIERS)


def _wait_for_gpu(threshold_mib: int = 2500, poll_s: int = 30) -> None:
    while True:
        used = read_total_gpu_mib()
        if used < threshold_mib:
            return
        print(f"[wait] GPU busy ({used} MiB) -- sleeping {poll_s}s", flush=True)
        time.sleep(poll_s)


def _iter_shortlist(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def _existing_result_id(conn: sqlite3.Connection, fingerprint: str) -> str | None:
    row = conn.execute(
        "SELECT result_id FROM program_results_compat "
        "WHERE graph_fingerprint = ? ORDER BY result_id LIMIT 1",
        (fingerprint,),
    ).fetchone()
    return None if row is None else str(row[0])


def _row_state(conn: sqlite3.Connection, result_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT ar_gate_status, ar_gate_no_go, ar_gate_score, "
        "nano_induction_nearest_status, "
        "language_control_s10_binding_score "
        "FROM graph_runs WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    if row is None:
        return {}
    return dict(row)


def _nb_probe(model, tier: str) -> tuple[float | None, float | None, str]:
    cfg = _CFG[tier]
    checkpoints = cfg.get("checkpoints")
    res = language_control_probe(
        model,
        active_vocab_size=cfg["active_vocab_size"],
        n_train_steps=cfg["n_train_steps"],
        checkpoint_steps=(
            tuple(checkpoints) if isinstance(checkpoints, (list, tuple)) else None
        ),
        timeout_s=float(cfg.get("timeout_s", 60.0)),
        device=DEV,
        preserve_state=True,
    )
    nb = (res.nano_blimp or {}).get("nano_blimp_score")
    sa = (res.synthetic_association or {}).get("synthetic_association_score")
    return nb, sa, res.status


def _register(nb: LabNotebook, exp_id: str, fp: str, graph_json: str) -> str:
    """Insert graphs + graph_runs rows for a novel shortlist graph (no stage1 claim)."""
    rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint=fp,
        graph_json=graph_json,
        bypass_quality_gate=True,
        model_source="cpu_cascade_shortlist",
    )
    nb.flush_writes()
    return rid


def _store_ar(nb: LabNotebook, rid: str, res, score: float, nogo: bool) -> None:
    store_probe_results(
        nb,
        rid,
        {
            "ar_gate_metric_version": res.metric_version,
            "ar_gate_in_dist_pair_acc": res.in_dist_pair_acc,
            "ar_gate_held_class_acc": res.held_class_acc,
            "ar_gate_score": round(score, 4),
            "ar_gate_status": res.status,
            "ar_gate_no_go": int(nogo),
        },
        write_leaderboard=False,
    )


def _run_one(nb: LabNotebook, exp_id: str, rec: dict[str, Any], ar_pass: float) -> dict:
    fp = str(rec["fingerprint"])
    graph_json = json.dumps(rec["graph"])
    out: dict[str, Any] = {
        "fingerprint": fp,
        "label_free_probe_score": rec.get("label_free_probe_score"),
        "label_free_probe_recommendation": rec.get("label_free_probe_recommendation"),
        "mech_score": rec.get("mech_score"),
    }

    rid = _existing_result_id(nb.conn, fp)
    if rid is None:
        rid = _register(nb, exp_id, fp, graph_json)
        out["registered"] = True
    else:
        out["registered"] = False
    out["result_id"] = rid

    state = _row_state(nb.conn, rid)
    # Resumability: already fully funneled, or already AR-stopped.
    if state.get("language_control_s10_binding_score") is not None:
        out["verdict"] = "skip_have_s10"
        out["ar_gate_score"] = state.get("ar_gate_score")
        out["nano_max"] = None
        return out
    if int(state.get("ar_gate_no_go") or 0) == 1:
        out["verdict"] = "skip_ar_no_go"
        out["ar_gate_score"] = state.get("ar_gate_score")
        return out

    # 1) ar_gate (the gate).
    ar = ar_gate(
        graph_json=graph_json, device=DEV, cfg=ARGateConfig(from_s1=False, seed=0)
    )
    score = ar_gate_score(ar)
    nogo = ar_gate_is_no_go(ar)
    _store_ar(nb, rid, ar, score, nogo)
    out["ar_gate_score"] = round(score, 4)
    out["ar_gate_status"] = ar.status
    out["ar_no_go"] = nogo
    if nogo or score < ar_pass:
        out["verdict"] = "AR_FAIL_skip"
        return out

    # 2) nano_induction_nearest (strong axis / primary ranking signal).
    nano = run_nano_induction_nearest_probe(graph_json, DEV)
    store_probe_results(nb, rid, nano, write_leaderboard=False)
    out["nano_status"] = nano.get("nano_induction_nearest_status")
    out["nano_max"] = nano.get("nano_induction_nearest_max_accuracy")
    out["nano_final"] = nano.get("nano_induction_nearest_final_accuracy")

    # 3) nb0.5 + nb1.0 (binding).
    model = _train_base(graph_json, device=DEV)
    try:
        nb05, sa05, st05 = _nb_probe(model, "s05")
        nb10, sa10, st10 = _nb_probe(model, "s10")
    finally:
        del model
        clear_gpu(DEV)
    store_probe_results(
        nb,
        rid,
        {
            "language_control_metric_version": "shortlist_cheap_probe_funnel",
            "language_control_s05_binding_score": nb05,
            "language_control_s05_sentence_assoc_score": sa05,
            "language_control_s10_binding_score": nb10,
            "language_control_s10_sentence_assoc_score": sa10,
        },
        write_leaderboard=False,
    )
    out.update(nb05=nb05, nb10=nb10, nb05_status=st05, nb10_status=st10)
    out["verdict"] = "done"
    return out


def _rank_key(rec: dict[str, Any]) -> tuple:
    """nano_induction_nearest primary; nb1.0, nb0.5, ar_gate_score tiebreak."""

    def f(x: Any) -> float:
        return float(x) if isinstance(x, (int, float)) else -1.0

    return (
        f(rec.get("nano_max")),
        f(rec.get("nb10")),
        f(rec.get("nb05")),
        f(rec.get("ar_gate_score")),
    )


def _emit_report(results: list[dict], out_dir: Path, top_n: int) -> Path:
    gated = [r for r in results if r.get("verdict") == "done"]
    gated.sort(key=_rank_key, reverse=True)
    table_path = out_dir / "shortlist_cheap_probe_leaderboard.md"
    lines = [
        "# Cheap-probe funnel leaderboard",
        "",
        f"Total shortlist graphs processed: {len(results)}",
        f"AR-gated (passed gate, fully probed): {len(gated)}",
        f"AR fails/skips: {sum(1 for r in results if r.get('verdict') == 'AR_FAIL_skip')}",
        "",
        f"## Top {top_n} by nano_induction_nearest (tiebreak nb1.0 / nb0.5 / ar_gate)",
        "",
        "| rank | fingerprint | nano_max | nb1.0 | nb0.5 | ar_gate | lf_probe | lf_rec |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(gated[:top_n], 1):
        lines.append(
            f"| {i} | {r['fingerprint']} | {r.get('nano_max')} | {r.get('nb10')} | "
            f"{r.get('nb05')} | {r.get('ar_gate_score')} | "
            f"{r.get('label_free_probe_score')} | {r.get('label_free_probe_recommendation')} |"
        )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=Path("research/reports/cpu_cascade_million_shortlist_clean.jsonl"),
    )
    ap.add_argument("--db", type=Path, default=Path(RUNS_DB))
    ap.add_argument("--ar-pass", type=float, default=AR_PASS)
    ap.add_argument("--limit", type=int, default=0, help="0 = all shortlist rows")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument(
        "--results-json",
        type=Path,
        default=Path("research/reports/shortlist_cheap_probe_results.json"),
    )
    args = ap.parse_args()

    if not args.no_wait:
        _wait_for_gpu()

    records = [rec for _, rec in _iter_shortlist(args.jsonl)]
    if args.limit > 0:
        records = records[: args.limit]
    print(f"funnel over {len(records)} shortlist graphs from {args.jsonl}", flush=True)

    nb = LabNotebook(str(args.db))
    exp_id = nb.start_experiment(
        experiment_type="shortlist_cheap_probe_funnel",
        config={
            "jsonl": str(args.jsonl),
            "ar_pass": float(args.ar_pass),
            "device": DEV,
            "source_script": "shortlist_cheap_probe_funnel",
        },
        hypothesis="Cheap-probe funnel over CPU-cascade shortlist; measure what the "
        "label-free probe-oracle only predicted.",
    )

    out_dir = args.results_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        results = _funnel_loop(nb, exp_id, records, args)
        table = _emit_report(results, out_dir, args.top_n)
        counts = _verdict_counts(results)
        nb.complete_experiment(
            exp_id,
            # complete_experiment fails any run whose results["total"] == 0; the
            # processed graph count IS this run's program total, so report it
            # under "total" or a fully-successful funnel is mislabeled failed.
            results={"total": len(results), "processed": len(results), **counts},
            aria_summary=(
                f"shortlist cheap-probe funnel: probed={counts['fully_probed']} "
                f"ar_fail={counts['ar_fail']} errors={counts['errors']}; "
                f"leaderboard={table.name}"
            ),
        )
        print(
            f"\nDONE: probed={counts['fully_probed']} "
            f"skipped(s10)={counts['skipped_have_s10']} "
            f"ar_fail={counts['ar_fail']} errors={counts['errors']}\n"
            f"leaderboard -> {table}",
            flush=True,
        )
    except Exception as exc:
        nb.fail_experiment(exp_id, error=str(exc))
        raise
    finally:
        nb.close()


def _funnel_loop(
    nb: LabNotebook, exp_id: str, records: list[dict], args: argparse.Namespace
) -> list[dict]:
    results: list[dict] = []
    for i, rec in enumerate(records, 1):
        row: dict[str, Any] = {"i": i, "fingerprint": str(rec.get("fingerprint"))}
        try:
            row = _run_one(nb, exp_id, rec, args.ar_pass)
            row["i"] = i
        except Exception as exc:  # noqa: BLE001
            row["verdict"] = f"error:{type(exc).__name__}:{str(exc)[:80]}"
            clear_gpu(DEV)
        results.append(row)
        args.results_json.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"[{i}/{len(records)}] {row['fingerprint']} "
            f"ar={row.get('ar_gate_score')} nano={row.get('nano_max')} "
            f"nb10={row.get('nb10')} {row.get('verdict')}",
            flush=True,
        )
    return results


def _verdict_counts(results: list[dict]) -> dict[str, int]:
    counts = {"fully_probed": 0, "skipped_have_s10": 0, "ar_fail": 0, "errors": 0}
    for r in results:
        v = str(r.get("verdict") or "")
        if v == "done":
            counts["fully_probed"] += 1
        elif v in ("AR_FAIL_skip", "skip_ar_no_go"):
            counts["ar_fail"] += 1
        elif v == "skip_have_s10":
            counts["skipped_have_s10"] += 1
        elif v.startswith("error:"):
            counts["errors"] += 1
    return counts


if __name__ == "__main__":
    main()
