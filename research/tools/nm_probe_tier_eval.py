#!/usr/bin/env python
"""O1 probe-tier evidence for the S1 passers (NM-verification plan, 2026-07-02).

Collects the 8 NM-bearing S1 passers (400-campaign experiment `14b8f2c7-c66`
+ batch-4 experiment `fbbf2566-227`, deduped on graph_fingerprint) plus the
top-N incumbents by S1 loss_ratio as the comparison population, and runs the
probe-tier battery of record on each: BLiMP, gMQAR, AR-gate, induction
(v2/intermediate), multi-slot binding. Perplexity is not computed as a goal
metric.

Reuses the existing harness rather than reinventing it:
  - `research.tools.backfill`: `reconstruct_model`, `micro_train`, `clear_gpu`,
    `store_probe_results`, `run_blimp_probe`, `run_induction_intermediate_probe`.
  - `research.tools.backfill_ar_gate`'s persisted-column shape (`ar_gate_*`)
    and `research.eval.ar_gate.ar_gate(model=..., from_s1=True)` — the model
    here is already micro-trained, so `from_s1=True` skips the redundant
    wikitext warmup (matches the probe's own "production" mode).
  - `research.eval.binding_multislot_probe.binding_multislot_probe` — the
    multi-slot binding metric of record (distinct from single-blank
    `binding_intermediate`); its own `mixed_query_*` fields are the built-in
    randomized/cross-entity query control, and `held_*` fields are the
    built-in held-out (anti-shortcut) control — reused here rather than
    inventing a new control.
  - `research.eval.gmqar.score_model_gmqar` — zero-shot, no training; the
    n_pairs=32/distractor=128 grid cell is reported as the "hardest cell"
    anti-saturation readout alongside AUDC.

`gmqar_*` has no columns on `graph_runs` anywhere in the schema (checked: no
table has them) — `ensure_gmqar_columns` adds them idempotently using the
exact `ALTER TABLE ... ADD COLUMN` idiom already used by
`research.tools.import_ar_validation_fingerprint_sweep.ensure_ar_validation_columns`.
`ar_gate_*` and `binding_multislot_*` columns already exist.

Median-of-3-seeds convention: for probes that train (ar_gate, binding
multislot), the run whose headline metric is literal-median (not a per-field
average across seeds) is persisted — the same convention
`research.eval.induction_intermediate_probe.run_induction_intermediate` /
`research.eval._probe_utils.run_probe_seeds` already use internally, so a
persisted row is always one internally-consistent run, never a franken-row.

This script only UPDATEs existing `graph_runs` rows that already carry
`stage1_passed=1` with additional probe-tier columns — it does not create
rows and never touches `stage1_passed`, so the S1-completeness write gate
(`notebook/program_writes.py:_enforce_s1_metric_completeness`) and
`trust_label` backfill conventions do not apply here (those gate *new* S1
rows, not probe enrichment of already-passed ones).

Usage:
    python -m research.tools.nm_probe_tier_eval --n-incumbents 8 --device cuda
    python -m research.tools.nm_probe_tier_eval --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DB_PATH = str(REPO / "research/runs.db")
REPORTS_DIR = REPO / "research/reports"

from research.defaults import VOCAB_SIZE
from research.tools._script_audit import (
    build_metric_backfill_context,
    complete_script_experiment,
    fail_script_experiment,
    start_script_experiment,
)
from research.tools.backfill import (
    clear_gpu,
    micro_train,
    reconstruct_model,
    run_blimp_probe,
    run_induction_intermediate_probe,
    store_probe_results,
)

# Verbatim from research/notes/autonomous_run_findings_2026-07-02.md — the
# 18-op registry list used to classify NM-bearing graphs. Do not hand-type a
# different list; re-derive from OP_IMPLS if the registry grows.
NM_OPS: tuple[str, ...] = (
    "monarch_mix",
    "butterfly_mix",
    "recurrent_depth_refine",
    "weight_dictionary_mix",
    "hypernet_layer_mix",
    "persistent_memory_refine",
    "block_sparse_mix",
    "token_merge_mix",
    "ternary_sign_mix",
    "padic_lowprec_mix",
    "subspace_mixture_mix",
    "lowrank_state_memory",
    "idempotent_oblique_memory",
    "nilpotent_lie_scan",
    "integral_control_mixer",
    "port_hamiltonian_mix",
    "cdma_slot_binding",
    "scale_equivariant_wavelet",
)

S1_EXPERIMENT_IDS: tuple[str, ...] = ("14b8f2c7-c66", "fbbf2566-227")

_NM_OPS_CTE = (
    "WITH nm_ops(op) AS (VALUES " + ", ".join(f"('{op}')" for op in NM_OPS) + ")"
)

GMQAR_COLUMNS: dict[str, str] = {
    "gmqar_audc": "REAL",
    "gmqar_d50": "INTEGER",
    "gmqar_chance": "REAL",
    "gmqar_token_pool": "INTEGER",
    "gmqar_hardest_cell_acc": "REAL",
}


def ensure_gmqar_columns(conn) -> list[str]:
    """Idempotent schema migration — copies the AR_VALIDATION_COLUMNS idiom
    from research/tools/import_ar_validation_fingerprint_sweep.py."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(graph_runs)")}
    added = []
    for name, col_type in GMQAR_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE graph_runs ADD COLUMN {name} {col_type}")
            added.append(name)
    if added:
        conn.commit()
    return added


def _which_ops(graph_json: str) -> list[str]:
    return [op for op in NM_OPS if op in graph_json]


def select_candidates(conn, n_incumbents: int) -> tuple[list[dict], list[dict]]:
    """8 NM-bearing S1 passers + top-N incumbents by loss_ratio (best first).

    Dedupes on graph_fingerprint across both experiments, keeping the lower
    loss_ratio row on a collision (mirrors the plan's dedup rule).
    """
    exp_ph = ",".join("?" for _ in S1_EXPERIMENT_IDS)
    sql = f"""
        {_NM_OPS_CTE}
        SELECT
            gr.result_id,
            gr.graph_fingerprint,
            gr.experiment_id,
            gr.loss_ratio,
            g.graph_json,
            EXISTS (
                SELECT 1 FROM nm_ops
                WHERE g.graph_json LIKE '%' || nm_ops.op || '%'
            ) AS nm_bearing
        FROM graph_runs AS gr
        JOIN graphs AS g ON g.graph_fingerprint = gr.graph_fingerprint
        WHERE gr.experiment_id IN ({exp_ph})
          AND COALESCE(gr.stage1_passed, 0) = 1
    """
    rows = [dict(r) for r in conn.execute(sql, S1_EXPERIMENT_IDS).fetchall()]

    best: dict[str, dict] = {}
    for r in rows:
        fp = r["graph_fingerprint"]
        if fp not in best or (r["loss_ratio"] or 1e9) < (best[fp]["loss_ratio"] or 1e9):
            best[fp] = r
    deduped = list(best.values())

    nm_rows = sorted(
        (r for r in deduped if r["nm_bearing"]), key=lambda x: x["loss_ratio"] or 1e9
    )
    incumbent_rows = sorted(
        (r for r in deduped if not r["nm_bearing"]),
        key=lambda x: x["loss_ratio"] or 1e9,
    )[:n_incumbents]
    return nm_rows, incumbent_rows


# ── New probe wrappers (gMQAR / AR-gate / multi-slot binding) ───────────


def run_gmqar_probe(model: torch.nn.Module, device: str) -> dict[str, Any]:
    from research.eval.gmqar import score_model_gmqar

    token_pool = 2048
    res = score_model_gmqar(
        model, vocab_size=VOCAB_SIZE, device=device, token_pool=token_pool
    )
    hardest = next(
        (
            c["acc"]
            for c in res.cells
            if c["n_pairs"] == 32 and c["distractor_tokens"] == 128
        ),
        None,
    )
    return {
        "gmqar_audc": round(res.audc, 4),
        "gmqar_d50": int(res.d50),
        "gmqar_chance": round(res.chance, 6),
        "gmqar_token_pool": token_pool,
        "gmqar_hardest_cell_acc": hardest,
        "_gmqar_cells": res.cells,
    }


def run_ar_gate_probe(
    model: torch.nn.Module, device: str, seeds: tuple[int, ...] = (0, 1, 2)
) -> dict[str, Any]:
    from research.eval.ar_gate import (
        AR_GATE_NO_GO_HELD_CLASS_THRESHOLD,
        AR_GATE_NO_GO_PAIR_THRESHOLD,
        ARGateConfig,
        ar_gate,
    )

    runs = []
    for seed in seeds:
        cfg = ARGateConfig(seed=seed, finetune_steps=400, from_s1=True, timeout_s=180.0)
        runs.append(ar_gate(model=model, device=device, cfg=cfg))
    runs_sorted = sorted(runs, key=lambda r: r.in_dist_pair_acc)
    median = runs_sorted[len(runs_sorted) // 2]
    score = round(0.6 * median.in_dist_pair_acc + 0.4 * median.held_class_acc, 4)
    no_go = (
        median.status == "ok"
        and median.in_dist_pair_acc < AR_GATE_NO_GO_PAIR_THRESHOLD
        and median.held_class_acc < AR_GATE_NO_GO_HELD_CLASS_THRESHOLD
    )
    return {
        "ar_gate_metric_version": median.metric_version,
        "ar_gate_in_dist_pair_acc": median.in_dist_pair_acc,
        "ar_gate_in_dist_class_acc": median.in_dist_class_acc,
        "ar_gate_held_pair_acc": median.held_pair_acc,
        "ar_gate_held_class_acc": median.held_class_acc,
        "ar_gate_score": score,
        "ar_gate_status": median.status,
        "ar_gate_elapsed_ms": median.elapsed_ms,
        "ar_gate_train_steps_done": median.finetune_steps_done,
        "ar_gate_no_go": int(no_go),
        "_ar_gate_seed_spread": [round(r.in_dist_pair_acc, 4) for r in runs],
        "_ar_gate_held_class_spread": [round(r.held_class_acc, 4) for r in runs],
    }


def run_binding_multislot_probe(
    model: torch.nn.Module, device: str, seeds: tuple[int, ...] = (0, 1, 2)
) -> dict[str, Any]:
    from research.eval.binding_multislot_probe import (
        BindingMultislotConfig,
        binding_multislot_probe,
    )

    runs = []
    for seed in seeds:
        cfg = BindingMultislotConfig(seed=seed, train_steps=400)
        runs.append(binding_multislot_probe(model, cfg=cfg, device=device))
    runs_sorted = sorted(runs, key=lambda r: r.held_entity_slot_acc)
    median = runs_sorted[len(runs_sorted) // 2]
    out = median.to_dict()
    out["_binding_multislot_held_slot_seed_spread"] = [
        round(r.held_entity_slot_acc, 4) for r in runs
    ]
    out["_binding_multislot_mixed_query_seed_spread"] = [
        round(r.mixed_query_acc, 4) for r in runs
    ]
    return out


# ── Orchestration ─────────────────────────────────────────────────────────


def _run_battery(
    candidate: dict, device: str, train_steps: int
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run the full probe battery on one candidate. Returns (updates, errors)."""
    updates: dict[str, Any] = {}
    errors: dict[str, str] = {}

    model, _graph = reconstruct_model(candidate["graph_json"], device)
    micro_train(model, train_steps, device)

    probes: list[tuple[str, Any]] = [
        ("blimp", lambda: run_blimp_probe(model, device, "investigation")),
        (
            "induction_intermediate",
            lambda: run_induction_intermediate_probe(model, device),
        ),
        ("gmqar", lambda: run_gmqar_probe(model, device)),
        ("ar_gate", lambda: run_ar_gate_probe(model, device)),
        ("binding_multislot", lambda: run_binding_multislot_probe(model, device)),
    ]
    for name, fn in probes:
        try:
            t0 = time.perf_counter()
            result = fn()
            updates.update(result)
            logger.info("    %s: OK (%.1fs)", name, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 — per-probe isolation, logged loud
            errors[name] = f"{type(exc).__name__}: {exc}"
            logger.error("    %s: FAILED — %s", name, errors[name])

    clear_gpu(device)
    return updates, errors


def run_eval(
    n_incumbents: int,
    device: str,
    train_steps: int,
    dry_run: bool,
    limit: int | None = None,
) -> dict[str, Any]:
    nb, exp_id = start_script_experiment(
        db_path=DB_PATH,
        experiment_type="probe_backfill",
        config={
            "n_incumbents": n_incumbents,
            "device": device,
            "train_steps": train_steps,
            "s1_experiment_ids": list(S1_EXPERIMENT_IDS),
        },
        source_script="nm_probe_tier_eval",
        hypothesis=(
            "O1: probe-tier evidence (BLiMP/gMQAR/AR-gate/induction/"
            "multi-slot binding) for the 8 NM-bearing S1 passers vs top "
            "incumbents by loss_ratio, from the 400-campaign + batch-4 "
            "S1 pools."
        ),
    )
    added_cols = ensure_gmqar_columns(nb.conn)
    if added_cols:
        logger.info("Migrated graph_runs: added columns %s", added_cols)

    nm_rows, incumbent_rows = select_candidates(nb.conn, n_incumbents)
    logger.info(
        "Candidates: %d NM-bearing, %d incumbent (requested >=%d)",
        len(nm_rows),
        len(incumbent_rows),
        n_incumbents,
    )
    for r in nm_rows:
        logger.info(
            "  NM  %s loss_ratio=%.4f ops=%s",
            r["result_id"][:12],
            r["loss_ratio"],
            _which_ops(r["graph_json"]),
        )
    for r in incumbent_rows:
        logger.info("  INC %s loss_ratio=%.4f", r["result_id"][:12], r["loss_ratio"])

    if dry_run:
        fail_script_experiment(
            nb,
            exp_id,
            error="dry-run invocation does not write results",
            results={"n_nm": len(nm_rows), "n_incumbent": len(incumbent_rows)},
        )
        nb.conn.close()
        return {"nm_rows": nm_rows, "incumbent_rows": incumbent_rows, "results": []}

    provenance_context = build_metric_backfill_context(
        kind="nm_probe_tier_eval",
        source_script="nm_probe_tier_eval",
        experiment_id=exp_id,
        device=device,
        train_steps=train_steps,
    )

    all_candidates = [dict(r, nm_bearing=True) for r in nm_rows] + [
        dict(r, nm_bearing=False) for r in incumbent_rows
    ]
    if limit is not None:
        all_candidates = all_candidates[:limit]
        logger.info(
            "--limit %d: truncated candidate list to %d", limit, len(all_candidates)
        )
    results: list[dict[str, Any]] = []
    t0 = time.time()
    n_ok = 0
    n_failed_all = 0
    try:
        for i, cand in enumerate(all_candidates, start=1):
            rid = cand["result_id"]
            fp = cand["graph_fingerprint"][:12]
            tag = "NM " if cand["nm_bearing"] else "INC"
            logger.info(
                "[%d/%d] %s %s loss_ratio=%.4f",
                i,
                len(all_candidates),
                tag,
                fp,
                cand["loss_ratio"] or -1,
            )
            try:
                updates, errors = _run_battery(cand, device, train_steps)
            except Exception as exc:  # noqa: BLE001 — reconstruction/train failure
                logger.error(
                    "  candidate FAILED entirely: %s: %s", type(exc).__name__, exc
                )
                results.append(
                    {
                        "result_id": rid,
                        "graph_fingerprint": cand["graph_fingerprint"],
                        "experiment_id": cand["experiment_id"],
                        "nm_bearing": cand["nm_bearing"],
                        "ops": _which_ops(cand["graph_json"]),
                        "loss_ratio": cand["loss_ratio"],
                        "updates": {},
                        "errors": {"_candidate": f"{type(exc).__name__}: {exc}"},
                    }
                )
                n_failed_all += 1
                clear_gpu(device)
                continue

            persisted = {k: v for k, v in updates.items() if not k.startswith("_")}
            if persisted:
                store_probe_results(
                    nb, rid, persisted, provenance_context=provenance_context
                )
                n_ok += 1
            results.append(
                {
                    "result_id": rid,
                    "graph_fingerprint": cand["graph_fingerprint"],
                    "experiment_id": cand["experiment_id"],
                    "nm_bearing": cand["nm_bearing"],
                    "ops": _which_ops(cand["graph_json"]),
                    "loss_ratio": cand["loss_ratio"],
                    "updates": updates,
                    "errors": errors,
                }
            )
            nb.conn.commit()
    except KeyboardInterrupt:
        fail_script_experiment(nb, exp_id, error="KeyboardInterrupt")
        nb.conn.close()
        raise
    except Exception as exc:
        fail_script_experiment(nb, exp_id, error=str(exc))
        nb.conn.close()
        raise

    elapsed = time.time() - t0
    complete_script_experiment(
        nb,
        exp_id,
        results={
            "n_candidates": len(all_candidates),
            "n_ok": n_ok,
            "n_failed_all": n_failed_all,
            "elapsed_s": round(elapsed, 1),
        },
        summary=(
            f"nm_probe_tier_eval: {n_ok}/{len(all_candidates)} candidates "
            f"probed OK ({n_failed_all} failed entirely) in {elapsed:.0f}s"
        ),
    )
    nb.conn.close()

    out_path = REPORTS_DIR / f"nm_probe_tier_eval_{int(time.time())}.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Wrote raw results to %s", out_path)

    return {
        "nm_rows": nm_rows,
        "incumbent_rows": incumbent_rows,
        "results": results,
        "raw_path": str(out_path),
        "elapsed_s": elapsed,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-incumbents", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--limit", type=int, default=None, help="Cap total candidates (smoke testing)."
    )
    args = p.parse_args()

    run_eval(
        n_incumbents=args.n_incumbents,
        device=args.device,
        train_steps=args.train_steps,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
