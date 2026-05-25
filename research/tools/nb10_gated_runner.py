#!/usr/bin/env python
"""Gated funnel -> DB: AR gate (>=0.6) then nb0.5(s05)+nb1.0(s10) for a template set.

For each distinct-fingerprint DB graph (in the target templates) missing
``language_control_s10_binding_score``: run the AR gate, persist ``ar_gate_*``;
if AR < 0.6 (no-go) skip the rest; else base-train + s05 + s10 and persist
``language_control_s05/s10`` binding + sentence-assoc. Everything lands in
``graph_runs``. Resumable (skips graphs that already have s10).

Run from repo root:  python -m research.tools.nb10_gated_runner
"""

from __future__ import annotations

import json
import sqlite3
import time

from research.eval.ar_gate import (
    ARGateConfig,
    ar_gate,
    ar_gate_is_no_go,
    ar_gate_score,
)
from research.eval.language_control_probe import language_control_probe
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.tools.language_control_backfill import TIERS, _train_base

DB = "research/runs.db"
DEV = "cuda"
AR_PASS = 0.6
CFG = dict(TIERS)
RESULTS = "research/reports/nb10_gated_results.json"
TEMPLATES = [
    "entmax_attention_block",
    "sparsemax_attention_block",
    "dplr_gated_delta_block",
    "retention_mix_block",
    "cawn_mixer_block",
    "state_space_retrieval_v2",
    "neural_symbolic_retrieval_v2",
]


def candidates(con: sqlite3.Connection, template: str):
    """One result_id per distinct fingerprint, only those missing s10."""
    return con.execute(
        """
        SELECT r.result_id, r.graph_fingerprint fp, r.graph_json,
               MAX(COALESCE(l.composite_score,0)) comp
        FROM program_graph_features f
        JOIN program_results_compat r ON r.result_id=f.result_id
        LEFT JOIN leaderboard l ON l.result_id=r.result_id
        LEFT JOIN graph_runs gr ON gr.result_id=r.result_id
        WHERE f.template_name=? AND r.graph_json IS NOT NULL
          AND gr.language_control_s10_binding_score IS NULL
        GROUP BY r.graph_fingerprint
        ORDER BY comp DESC
        """,
        (template,),
    ).fetchall()


def write(con: sqlite3.Connection, rid: str, cols: dict) -> None:
    sets = ", ".join(f"{k}=?" for k in cols)
    sql = f"UPDATE graph_runs SET {sets} WHERE result_id=?"  # nosec B608 - cols hardcoded; values parameterized
    con.execute(sql, (*cols.values(), rid))
    con.commit()


def probe(model, tier: str):
    c = CFG[tier]
    res = language_control_probe(
        model,
        active_vocab_size=c["active_vocab_size"],
        n_train_steps=c["n_train_steps"],
        checkpoint_steps=c.get("checkpoints"),
        timeout_s=float(c.get("timeout_s", 60.0)),
        device=DEV,
        preserve_state=True,
    )
    nb = (res.nano_blimp or {}).get("nano_blimp_score")
    sa = (res.synthetic_association or {}).get("synthetic_association_score")
    return nb, sa, res.status


def main() -> None:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    targets = [(t, r) for t in TEMPLATES for r in candidates(con, t)]
    print(
        f"{len(targets)} distinct graphs missing nb1.0 across {len(TEMPLATES)} templates",
        flush=True,
    )
    out: list[dict] = []
    run = gated = err = 0
    for i, (tmpl, row) in enumerate(targets, 1):
        rid, fp = row["result_id"], row["fp"][:12]
        rec = {"i": i, "template": tmpl, "fp": fp}
        try:
            gj = resolve_graph_json_value(con, DB, row["graph_json"])
        except Exception:
            gj = row["graph_json"]
        try:
            ar = ar_gate(
                graph_json=gj, device=DEV, cfg=ARGateConfig(from_s1=False, seed=0)
            )
            ars = ar_gate_score(ar)
            nogo = ar_gate_is_no_go(ar)
            rec["ar_gate"] = round(ars, 3)
            write(
                con,
                rid,
                {
                    "ar_gate_metric_version": ar.metric_version,
                    "ar_gate_in_dist_pair_acc": ar.in_dist_pair_acc,
                    "ar_gate_held_class_acc": ar.held_class_acc,
                    "ar_gate_score": round(ars, 4),
                    "ar_gate_status": ar.status,
                    "ar_gate_no_go": int(nogo),
                },
            )
            if nogo or ars < AR_PASS:
                rec["verdict"] = "AR_FAIL_skip_nb"
                gated += 1
            else:
                model = _train_base(gj, device=DEV)
                nb05, sa05, _ = probe(model, "s05")
                t0 = time.perf_counter()
                nb10, sa10, _ = probe(model, "s10")
                rec["s10_secs"] = round(time.perf_counter() - t0, 1)
                del model
                write(
                    con,
                    rid,
                    {
                        "language_control_metric_version": "lc_nb_gated_funnel",
                        "language_control_s05_binding_score": nb05,
                        "language_control_s05_sentence_assoc_score": sa05,
                        "language_control_s10_binding_score": nb10,
                        "language_control_s10_sentence_assoc_score": sa10,
                    },
                )
                rec.update(nb05=nb05, nb10=nb10)
                rec["verdict"] = "done"
                run += 1
        except Exception as exc:  # noqa: BLE001
            rec["verdict"] = f"error:{type(exc).__name__}:{str(exc)[:60]}"
            err += 1
        out.append(rec)
        json.dump(out, open(RESULTS, "w"), indent=2)
        if i % 5 == 0 or rec["verdict"] != "done":
            print(
                f"[{i}/{len(targets)}] {tmpl[:20]:20s} {fp} ar={rec.get('ar_gate')} "
                f"nb10={rec.get('nb10')} {rec['verdict']}",
                flush=True,
            )
    print(
        f"DONE: ran nb1.0 on {run}, AR-gated(skipped) {gated}, errors {err}", flush=True
    )


if __name__ == "__main__":
    main()
