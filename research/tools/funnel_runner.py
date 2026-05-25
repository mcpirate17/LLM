#!/usr/bin/env python
"""Lean AR-gate -> nb0.5 funnel over a list of templates. No screening cascade.

Per template: use existing DB graphs (up to --max-db distinct fingerprints,
highest composite first) if any exist, else build --fresh-seeds fresh graphs
via apply_template. Per graph: AR gate; on no-go STOP (no nb0.5); else nb0.5.

Waits for the GPU to be free at startup so it can be launched behind another
run. Transient (research/reports, auto-pruned).
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import subprocess
import time

from research.eval.ar_gate import (
    ARGateConfig,
    ar_gate,
    ar_gate_is_no_go,
    ar_gate_score,
)
from research.eval.nano_bind import nano_bind
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

DB = "research/runs.db"
DEV = "cuda"

# Decided order after entmax (already running separately): the user asked for
# sparsemax next, then "you decide". Priority = strong offline AR-gate score +
# novelty; DB-graph templates first (real candidates), strong zero-run ones via
# fresh build. Weak-binding (0.46) and grammar-broken (token_hodge/wavelet)
# templates are intentionally omitted.
DEFAULT_ORDER = [
    "sparsemax_attention_block",  # user: workhorse
    "state_space_retrieval_v2",
    "neural_symbolic_retrieval_v2",
    "dplr_gated_delta_block",
    "cawn_mixer_block",
    "tree_mix_attention_block",  # zero DB -> fresh
    "mlstm_sparse_ffn_block",  # zero DB -> fresh
    "conv_residual_retrieval_v2",  # zero DB -> fresh
    "product_key_memory_block",  # zero DB -> fresh
    "retention_mix_block",  # codex: unstable — last, to confirm
]


def _gpu_free_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=15,
        )
        return int(out.decode().splitlines()[0].strip())
    except Exception:
        return None


def _wait_for_gpu(threshold_mib: int = 2500, poll_s: int = 30) -> None:
    while True:
        used = _gpu_free_mib()
        if used is None or used < threshold_mib:
            return
        print(f"[wait] GPU busy ({used} MiB) — sleeping {poll_s}s", flush=True)
        time.sleep(poll_s)


def _db_graphs(con, template: str, max_db: int):
    rows = con.execute(
        """
        SELECT r.graph_fingerprint fp, r.graph_json,
               COALESCE(l.composite_score, 0) comp
        FROM program_graph_features f
        JOIN program_results_compat r ON r.result_id = f.result_id
        LEFT JOIN leaderboard l ON l.result_id = r.result_id
        WHERE f.template_name = ? AND r.graph_json IS NOT NULL
        GROUP BY r.graph_fingerprint
        ORDER BY comp DESC
        LIMIT ?
        """,
        (template, max_db),
    ).fetchall()
    out = []
    for row in rows:
        try:
            gj = resolve_graph_json_value(con, DB, row["graph_json"])
        except Exception:
            gj = row["graph_json"]
        out.append((row["fp"][:12], gj))
    return out


def _fresh_graphs(template: str, n: int, dim: int):
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.serializer import graph_to_json
    from research.synthesis.templates import apply_template

    out = []
    for s in range(n):
        rng = random.Random(s)
        g = ComputationGraph(model_dim=dim)
        inp = g.add_input()
        try:
            g.set_output(apply_template(g, inp, rng, template_name=template))
            out.append((f"fresh-s{s}", graph_to_json(g)))
        except Exception as exc:  # noqa: BLE001
            print(f"  build failed {template} s{s}: {exc}", flush=True)
    return out


def _run_one(gj: str, seed: int):
    r = ar_gate(graph_json=gj, device=DEV, cfg=ARGateConfig(from_s1=False, seed=seed))
    rec = {
        "ar_gate_score": round(ar_gate_score(r), 3),
        "ar_pair": round(r.in_dist_pair_acc, 3),
        "ar_held_class": round(r.held_class_acc, 3),
        "ar_status": r.status,
        "ar_no_go": ar_gate_is_no_go(r),
    }
    if rec["ar_no_go"]:
        rec["verdict"] = "AR_NO_GO_skip_nb05"
        return rec
    nb = nano_bind(gj, device=DEV, seed=seed)
    rec["nb05_best"] = round(max(nb.scores), 3) if nb.scores else None
    rec["nb05_held"] = round(max(nb.held_acc), 3) if nb.held_acc else None
    rec["nb05_status"] = nb.status
    rec["nb05_no_go"] = nb.is_no_go
    rec["verdict"] = "done"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--max-db", type=int, default=25)
    ap.add_argument("--fresh-seeds", type=int, default=5)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    if not args.no_wait:
        _wait_for_gpu()

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    templates = [t.strip() for t in args.templates.split(",") if t.strip()]
    print(f"funnel over {len(templates)} templates: {templates}", flush=True)

    for ti, tmpl in enumerate(templates, 1):
        graphs = _db_graphs(con, tmpl, args.max_db)
        src = "db"
        if not graphs:
            graphs = _fresh_graphs(tmpl, args.fresh_seeds, args.dim)
            src = "fresh"
        print(
            f"\n=== [{ti}/{len(templates)}] {tmpl} ({len(graphs)} graphs, {src}) ===",
            flush=True,
        )
        results = []
        for gi, (gid, gj) in enumerate(graphs, 1):
            rec = {"gid": gid}
            try:
                rec.update(_run_one(gj, seed=gi))
            except Exception as exc:  # noqa: BLE001
                rec["verdict"] = f"error:{type(exc).__name__}:{str(exc)[:60]}"
            results.append(rec)
            json.dump(
                results, open(f"research/reports/funnel_{tmpl}.json", "w"), indent=2
            )
            print(
                f"  [{gi}/{len(graphs)}] {gid} ar={rec.get('ar_gate_score')} "
                f"nogo={rec.get('ar_no_go')} nb05={rec.get('nb05_best')} {rec['verdict']}",
                flush=True,
            )
        ar = [r["ar_gate_score"] for r in results if r.get("ar_gate_score") is not None]
        nb = [r["nb05_best"] for r in results if r.get("nb05_best") is not None]
        nogo = sum(1 for r in results if r.get("ar_no_go"))
        amed = sorted(ar)[len(ar) // 2] if ar else None
        nmed = sorted(nb)[len(nb) // 2] if nb else None
        print(
            f"=== {tmpl} SUMMARY: n={len(results)} AR no-go={nogo} "
            f"AR med={amed} nb0.5 med={nmed} ===",
            flush=True,
        )
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
