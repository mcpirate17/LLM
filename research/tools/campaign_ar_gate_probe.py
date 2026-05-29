"""Run the AR-gate probe on novel-mixer campaign survivors and persist scores.

AR-gate is the primary nano gate: score < 0.6 ≈ "can't learn". The probe is the
synthetic 3-slot binding task (corpus-independent), so it is valid regardless of
the LM corpus quality. Selects S1-passed rows for the campaign templates from
``graph_runs``, resolves each graph from the ``graphs`` table, runs the probe
(deduped by fingerprint), writes ``ar_gate_*`` back to ``graph_runs``, and prints
a per-template summary with the 0.6 verdict.

Usage:
    python -m research.tools.campaign_ar_gate_probe --since-min 180 --per-template 4
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time

from research.eval.ar_gate import ARGateConfig, ar_gate, ar_gate_score

CAMPAIGN_TEMPLATES = (
    "clifford_geometric_mixer_block",
    "tropical_maxplus_mixer_block",
    "ultrametric_hierarchical_ensemble_block",
    "reciprocal_rank_attention_block",
    "phase_lock_attention_block",
    "stdp_reciprocal_memory_block",
)

GATE_THRESHOLD = 0.6


def _template_by_experiment(conn: sqlite3.Connection) -> dict[str, str]:
    out: dict[str, str] = {}
    for eid, cfg_json in conn.execute(
        "SELECT experiment_id, config_json FROM experiments"
    ):
        if not cfg_json:
            continue
        try:
            name = json.loads(cfg_json).get("backfill_template")
        except (ValueError, TypeError):
            continue
        if name:
            out[eid] = name
    return out


def select_candidates(
    conn: sqlite3.Connection, since_min: float, per_template: int
) -> dict[str, list[dict]]:
    tmap = _template_by_experiment(conn)
    cutoff = time.time() - since_min * 60
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT result_id, experiment_id, graph_fingerprint, timestamp, stage1_passed, "
        "blimp_overall_accuracy, ar_gate_score "
        "FROM graph_runs WHERE timestamp > ? AND stage1_passed = 1 ORDER BY timestamp DESC",
        (cutoff,),
    ).fetchall()
    by_tpl: dict[str, list[dict]] = {t: [] for t in CAMPAIGN_TEMPLATES}
    seen_fp: set[str] = set()
    for r in rows:
        tpl = tmap.get(r["experiment_id"])
        if tpl not in by_tpl or r["graph_fingerprint"] in seen_fp:
            continue
        seen_fp.add(r["graph_fingerprint"])
        if len(by_tpl[tpl]) >= per_template:
            continue
        by_tpl[tpl].append(
            {
                "result_id": r["result_id"],
                "fingerprint": r["graph_fingerprint"],
                "blimp": r["blimp_overall_accuracy"],
                "existing": r["ar_gate_score"],
            }
        )
    return by_tpl


def _graph_json(conn: sqlite3.Connection, fingerprint: str) -> str | None:
    row = conn.execute(
        "SELECT graph_json FROM graphs WHERE graph_fingerprint = ? AND graph_json IS NOT NULL "
        "AND COALESCE(graph_json_is_placeholder, 0) = 0 LIMIT 1",
        (fingerprint,),
    ).fetchone()
    return row[0] if row else None


def _persist(conn: sqlite3.Connection, result_id: str, res, score: float) -> None:
    conn.execute(
        "UPDATE graph_runs SET ar_gate_score = ?, ar_gate_in_dist_pair_acc = ?, "
        "ar_gate_in_dist_class_acc = ?, ar_gate_held_pair_acc = ?, ar_gate_held_class_acc = ?, "
        "ar_gate_status = 'ok' WHERE result_id = ?",
        (
            score,
            res.in_dist_pair_acc,
            res.in_dist_class_acc,
            res.held_pair_acc,
            res.held_class_acc,
            result_id,
        ),
    )
    conn.commit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="research/runs.db")
    ap.add_argument("--since-min", type=float, default=180)
    ap.add_argument("--per-template", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--force", action="store_true", help="re-probe rows that already have a score"
    )
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cands = select_candidates(conn, args.since_min, args.per_template)
    cfg = ARGateConfig()

    results: dict[str, list[float]] = {}
    for tpl in CAMPAIGN_TEMPLATES:
        scores: list[float] = []
        for cand in cands.get(tpl, []):
            if cand["existing"] is not None and not args.force:
                scores.append(float(cand["existing"]))
                continue
            gj = _graph_json(conn, cand["fingerprint"])
            if not gj:
                print(f"  {tpl[:30]:30s} {cand['fingerprint']} — no graph_json, skip")
                continue
            t0 = time.time()
            try:
                res = ar_gate(graph_json=gj, device=args.device, cfg=cfg)
            except Exception as exc:  # noqa: BLE001 — report and continue the cohort
                print(f"  {tpl[:30]:30s} {cand['fingerprint']} — probe error: {exc}")
                continue
            score = ar_gate_score(res)
            _persist(conn, cand["result_id"], res, score)
            scores.append(score)
            print(
                f"  {tpl[:30]:30s} {cand['fingerprint']} ar_gate={score:.4f} "
                f"(pair={res.in_dist_pair_acc:.2f} heldcls={res.held_class_acc:.2f}) "
                f"blimp={cand['blimp']} [{time.time() - t0:.0f}s]"
            )
        if scores:
            results[tpl] = scores
    conn.close()

    print(
        f"\n=== AR-gate per template (threshold {GATE_THRESHOLD}: <0.6 ≈ can't learn) ==="
    )
    print(f"{'template':42s} {'n':>3s} {'median':>7s} {'max':>6s}  verdict")
    for tpl in sorted(results, key=lambda t: -statistics.median(results[t])):
        s = results[tpl]
        m = statistics.median(s)
        verdict = "LEARNS" if m >= GATE_THRESHOLD else "cannot-learn"
        print(f"{tpl:42s} {len(s):3d} {m:7.4f} {max(s):6.3f}  {verdict}")


if __name__ == "__main__":
    main()
