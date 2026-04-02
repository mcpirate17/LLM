#!/usr/bin/env python3
"""Unified backfill runner for all scoring probes.

Replaces 6 separate backfill_*.py scripts with a single entry point that
shares model reconstruction, candidate querying, progress reporting, GPU
cleanup, and rescoring.

Usage:
    python -m research.tools.backfill --probe binding --top 50
    python -m research.tools.backfill --probe all --tier investigation,validation
    python -m research.tools.backfill --probe blimp,hellaswag --dry-run
    python -m research.tools.backfill --probe rescore  # rescore only, no probes

Probes: binding, hellaswag, blimp, triage, fingerprint, rescore, all
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.defaults import VOCAB_SIZE
from research.scientist.leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite,
    prefetch_program_results,
)
from research.scientist.notebook import LabNotebook
from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json

DB_PATH = "research/lab_notebook.db"

# ── Probe registry ──────────────────────────────────────────────────────

_PROBE_NEEDS_TRAIN = frozenset({"binding", "hellaswag", "blimp"})
_PROBE_NEEDS_MODEL = frozenset(
    {"binding", "hellaswag", "blimp", "triage", "fingerprint"}
)
_ALL_PROBES = ("binding", "hellaswag", "blimp", "triage", "fingerprint")


# ── Model lifecycle (shared, not duplicated per probe) ──────────────────


def reconstruct_model(graph_json_str: str, device: str) -> nn.Module:
    graph = graph_from_json(graph_json_str)
    model = compile_model([graph], vocab_size=VOCAB_SIZE)
    return model.to(device).eval(), graph


def micro_train(
    model: nn.Module,
    steps: int,
    device: str,
    seq_len: int = 128,
    batch_size: int = 8,
    lr: float = 3e-4,
) -> None:
    """Random-token micro-training so binding probes run on a trained model."""
    model.train()
    vs = getattr(model, "vocab_size", VOCAB_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        data = torch.randint(0, vs, (batch_size, seq_len), device=device)
        logits = model(data)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vs), data[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()


def clear_gpu(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Candidate query (single implementation) ─────────────────────────────


@dataclass(slots=True)
class Candidate:
    entry_id: str
    result_id: str
    tier: str
    composite_score: float
    is_reference: bool
    model_source: str
    graph_json: Optional[str]
    graph_fingerprint: str


def query_candidates(
    nb: LabNotebook,
    tiers: Sequence[str],
    top_per_tier: int,
    null_column: Optional[str],
    force: bool,
) -> List[Candidate]:
    """Single candidate query for all probes. Returns top N per tier."""
    tier_ph = ",".join("?" for _ in tiers)
    where = f"l.tier IN ({tier_ph})"
    if null_column and not force:
        where += f" AND pr.{null_column} IS NULL"

    rows = nb.conn.execute(
        f"SELECT l.entry_id, l.result_id, l.tier, l.composite_score, "
        f"l.is_reference, l.model_source, "
        f"pr.graph_json, pr.graph_fingerprint "
        f"FROM leaderboard l "
        f"LEFT JOIN program_results pr ON l.result_id = pr.result_id "
        f"WHERE {where} "
        f"ORDER BY l.composite_score DESC",
        tuple(tiers),
    ).fetchall()

    # Top N per tier using dict of lists
    by_tier: dict[str, list[Candidate]] = {}
    for r in rows:
        t = r["tier"]
        tier_list = by_tier.setdefault(t, [])
        if len(tier_list) < top_per_tier:
            tier_list.append(
                Candidate(
                    entry_id=r["entry_id"],
                    result_id=r["result_id"],
                    tier=t,
                    composite_score=float(r["composite_score"] or 0),
                    is_reference=bool(r["is_reference"]),
                    model_source=r["model_source"] or "",
                    graph_json=r["graph_json"],
                    graph_fingerprint=(r["graph_fingerprint"] or "")[:12],
                )
            )

    return [c for t in tiers for c in by_tier.get(t, [])]


# ── Rescore (single implementation) ─────────────────────────────────────


def rescore_entry(
    nb: LabNotebook,
    entry_id: str,
    result_id: str,
    is_ref: bool,
    pr_cache: Dict[str, Dict],
    pr_updates: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """Recompute composite score. Returns (new_score, old_score)."""
    existing = nb.conn.execute(
        "SELECT * FROM leaderboard WHERE entry_id = ?", (entry_id,)
    ).fetchone()
    if not existing:
        return 0.0, 0.0
    d = dict(existing)
    old_score = float(d.get("composite_score") or 0)
    pr_dict = dict(pr_cache.get(result_id, {}))
    if pr_updates:
        pr_dict.update(pr_updates)
    score_kw = build_score_kwargs_from_prefetch(pr_dict, d, is_ref)
    new_score = compute_composite(**score_kw)
    if new_score != old_score:
        nb.conn.execute(
            "UPDATE leaderboard SET composite_score = ?, "
            "rescore_status = 'rescored_v7', rescore_timestamp = ?, "
            "old_composite_score = ?, rescore_reason = 'backfill_rescore' "
            "WHERE entry_id = ?",
            (new_score, time.time(), old_score, entry_id),
        )
    return new_score, old_score


def rescore_all(nb: LabNotebook) -> Tuple[int, int]:
    """Bulk rescore all leaderboard entries. Returns (total, changed)."""
    rows = nb.conn.execute(
        "SELECT entry_id, result_id, is_reference, composite_score "
        "FROM leaderboard ORDER BY composite_score DESC"
    ).fetchall()
    all_ids = [r["result_id"] for r in rows]
    pr_cache = prefetch_program_results(nb.conn, all_ids)
    changed = 0
    for row in rows:
        new, old = rescore_entry(
            nb,
            row["entry_id"],
            row["result_id"],
            bool(row["is_reference"]),
            pr_cache,
        )
        if new != old:
            changed += 1
    # Single commit at end instead of every 200 rows
    nb.conn.commit()
    return len(rows), changed


# ── Probe functions (each returns dict of columns to write) ─────────────


def run_binding_probe(
    model: nn.Module,
    device: str,
) -> Dict[str, Any]:
    from research.eval.associative_recall import associative_recall_score
    from research.eval.binding_range import binding_range_profile
    from research.eval.induction_probe import induction_score
    from research.scientist.thresholds import (
        BINDING_AR_SOFT_GATE,
        BINDING_BINDING_AUC_SOFT_GATE,
        BINDING_INDUCTION_SOFT_GATE,
    )

    ar = associative_recall_score(
        model,
        n_pairs=20,
        n_eval=200,
        n_train_steps=500,
        batch_size=16,
        device=device,
    )
    ind = induction_score(
        model,
        gaps=(4, 8, 16, 32, 64),
        n_train_steps=1000,
        n_eval=200,
        batch_size=32,
        device=device,
    )
    br = binding_range_profile(
        model,
        distances=(2, 4, 8, 16, 32, 64),
        n_eval=200,
        device=device,
    )
    bc = 0.4 * ar.auc + 0.3 * ind.auc + 0.3 * br.auc
    is_local = int(
        ar.auc < BINDING_AR_SOFT_GATE
        and ind.auc < BINDING_INDUCTION_SOFT_GATE
        and br.auc < BINDING_BINDING_AUC_SOFT_GATE
    )
    return {
        "ar_auc": ar.auc,
        "ar_final_acc": ar.final_acc,
        "ar_timed_out": int(ar.timed_out),
        "ar_above_chance": int(ar.above_chance),
        "induction_auc": ind.auc,
        "binding_auc": br.auc,
        "binding_composite": round(bc, 4),
        "local_only": is_local,
    }


def run_hellaswag_probe(
    model: nn.Module,
    device: str,
    tier: str = "investigation",
) -> Dict[str, Any]:
    from research.eval.hellaswag_eval import evaluate_hellaswag

    n_map = {
        "screening": 50,
        "investigation": 100,
        "validation": 200,
        "breakthrough": 200,
    }
    n = n_map.get(tier, 100)
    hs = evaluate_hellaswag(model, VOCAB_SIZE, device, n_examples=n)
    return {
        "hellaswag_acc": hs.get("hellaswag_acc"),
        "hellaswag_status": hs.get("hellaswag_status"),
        "hellaswag_n_examples": hs.get("hellaswag_total"),
    }


def run_blimp_probe(
    model: nn.Module,
    device: str,
    tier: str = "investigation",
) -> Dict[str, Any]:
    from research.eval.blimp_eval import evaluate_blimp

    import json

    n_map = {"investigation": 50, "validation": 200, "breakthrough": 200}
    n = n_map.get(tier, 50)
    result = evaluate_blimp(model, VOCAB_SIZE, device, n_per_subtask=n)
    return {
        "blimp_overall_accuracy": result.overall_accuracy,
        "blimp_subtask_accuracies_json": json.dumps(result.subtask_accuracies),
        "blimp_n_subtasks": result.n_subtasks,
        "blimp_status": result.status,
    }


def run_triage_probe(
    model: nn.Module,
    graph: Any,
    device: str,
    loss_ratio: float = 0.0,
    initial_loss: float = 0.0,
    final_loss: float = 0.0,
) -> Dict[str, Any]:
    from research.scientist.runner.execution_triage import run_triage

    model_dim = getattr(graph, "model_dim", 64)
    result = {
        "loss_ratio": loss_ratio,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "stage1_passed": True,
    }
    return run_triage(model, graph, result, model_dim=model_dim)


def run_fingerprint_probe(
    result_id: str,
    graph_json_str: str,
    device: str,
    timeout: int = 30,
) -> Optional[Dict[str, Any]]:
    """Run CKA fingerprint in subprocess for crash isolation."""
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(1)
    try:
        async_result = pool.apply_async(
            _fingerprint_worker, ((result_id, graph_json_str, device),)
        )
        _, updates, error = async_result.get(timeout=timeout)
        if error:
            return None
        return updates
    except mp.TimeoutError:
        return None
    finally:
        pool.terminate()
        pool.join()


def _fingerprint_worker(args_tuple):
    result_id, graph_json_str, device = args_tuple
    try:
        return (result_id, _fingerprint_one(result_id, graph_json_str, device), None)
    except Exception as e:
        return (result_id, None, str(e))


def _fingerprint_one(
    result_id: str, graph_json_str: str, device: str
) -> Dict[str, Any]:
    os.environ["ARIA_DISABLE_NATIVE_CKA"] = "1"
    from research.eval.fingerprint import compute_fingerprint
    from research.synthesis.compiler import compile_model as _compile
    from research.synthesis.serializer import graph_from_json as _from_json

    graph = _from_json(graph_json_str)
    model_dim = getattr(graph, "model_dim", 256)
    model = _compile([graph], vocab_size=256, max_seq_len=128)

    fp = compute_fingerprint(
        model,
        seq_len=64,
        model_dim=model_dim,
        vocab_size=256,
        device=device,
        n_probes=32,
        include_cka=True,
        include_behavioral_probes=True,
    )
    updates: Dict[str, Any] = {}
    if fp.cka_vs_transformer is not None:
        updates["fp_cka_vs_transformer"] = fp.cka_vs_transformer
        updates["fp_cka_vs_ssm"] = fp.cka_vs_ssm
        updates["fp_cka_vs_conv"] = fp.cka_vs_conv
    updates["cka_source"] = getattr(fp, "cka_source", None)
    updates["cka_artifact_version"] = getattr(fp, "cka_artifact_version", None)
    updates["cka_probe_protocol_hash"] = getattr(fp, "cka_probe_protocol_hash", None)
    updates["cka_reference_quality"] = (
        1 if getattr(fp, "cka_reference_quality", False) else 0
    )
    updates["novelty_score"] = fp.novelty_score
    updates["novelty_valid_for_promotion"] = 1 if fp.novelty_valid_for_promotion else 0
    updates["novelty_validity_reason"] = fp.novelty_validity_reason
    updates["novelty_reference_version"] = getattr(
        fp, "novelty_reference_version", None
    )
    updates["novelty_scoring_policy_version"] = "backfill_v3"
    for attr, col in (
        ("interaction_locality", "fp_interaction_locality"),
        ("interaction_sparsity", "fp_interaction_sparsity"),
        ("interaction_symmetry", "fp_interaction_symmetry"),
        ("interaction_hierarchy", "fp_interaction_hierarchy"),
        ("intrinsic_dim", "fp_intrinsic_dim"),
        ("isotropy", "fp_isotropy"),
        ("rank_ratio", "fp_rank_ratio"),
        ("jacobian_spectral_norm", "fp_jacobian_spectral_norm"),
        ("jacobian_effective_rank", "fp_jacobian_effective_rank"),
        ("sensitivity_uniformity", "fp_sensitivity_uniformity"),
    ):
        val = getattr(fp, attr, None)
        if val is not None:
            updates[col] = float(val)
    return updates


# ── DB write (single implementation) ────────────────────────────────────


def store_probe_results(
    nb: LabNotebook,
    result_id: str,
    updates: Dict[str, Any],
    write_leaderboard: bool = True,
) -> None:
    """Write probe results to program_results and optionally leaderboard."""
    if not updates:
        return

    # Filter to columns that exist in program_results
    pr_cols = _get_table_columns(nb, "program_results")
    pr_updates = {k: v for k, v in updates.items() if k in pr_cols}
    if pr_updates:
        cols = list(pr_updates.keys())
        vals = [pr_updates[c] for c in cols]
        set_clause = ", ".join(f"{c} = ?" for c in cols)
        nb.conn.execute(
            f"UPDATE program_results SET {set_clause} WHERE result_id = ?",
            (*vals, result_id),
        )
    if write_leaderboard:
        lb_cols = _get_table_columns(nb, "leaderboard")
        lb_updates = {k: v for k, v in updates.items() if k in lb_cols}
        if lb_updates:
            lb_col_list = list(lb_updates.keys())
            lb_vals = [lb_updates[c] for c in lb_col_list]
            lb_set = ", ".join(f"{c} = ?" for c in lb_col_list)
            nb.conn.execute(
                f"UPDATE leaderboard SET {lb_set} WHERE result_id = ?",
                (*lb_vals, result_id),
            )


_table_columns_cache: Dict[str, frozenset] = {}


def _get_table_columns(nb: LabNotebook, table: str) -> frozenset:
    if table not in _table_columns_cache:
        rows = nb.conn.execute(f"PRAGMA table_info({table})").fetchall()
        _table_columns_cache[table] = frozenset(r[1] for r in rows)
    return _table_columns_cache[table]


# ── NULL column for each probe (used for candidate filtering) ───────────

_PROBE_NULL_COLUMN: Dict[str, str] = {
    "binding": "induction_auc",
    "hellaswag": "hellaswag_acc",
    "blimp": "blimp_overall_accuracy",
    "triage": "activation_sparsity_score",
    "fingerprint": "fp_cka_vs_transformer",
}


def _probe_has_data(nb: LabNotebook, result_id: str, probe_name: str) -> bool:
    """Check if a probe's key column already has data for this entry."""
    col = _PROBE_NULL_COLUMN.get(probe_name)
    if not col:
        return False
    row = nb.conn.execute(
        f"SELECT {col} FROM program_results WHERE result_id = ?", (result_id,)
    ).fetchone()
    return row is not None and row[0] is not None


# ── Main loop ───────────────────────────────────────────────────────────


def _average_numeric_updates(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Average numeric values across multiple passes. Keep last for strings."""
    if len(runs) == 1:
        return runs[0]
    merged: Dict[str, Any] = {}
    all_keys = {k for r in runs for k in r}
    for key in all_keys:
        vals = [r[key] for r in runs if key in r]
        if not vals:
            continue
        if isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
            merged[key] = sum(vals) / len(vals)
            # Round to 4 decimals for cleanliness
            if isinstance(vals[0], float):
                merged[key] = round(merged[key], 4)
        else:
            # Non-numeric: take last value (status strings, JSON, bools)
            merged[key] = vals[-1]
    return merged


def run_backfill(
    probes: Sequence[str],
    tiers: Sequence[str],
    top_per_tier: int,
    device: str,
    train_steps: int,
    n_passes: int,
    force: bool,
    dry_run: bool,
    fp_timeout: int,
) -> None:
    nb = LabNotebook(DB_PATH)

    # Determine which null column to filter on (use first probe's column)
    null_col = _PROBE_NULL_COLUMN.get(probes[0]) if len(probes) == 1 else None

    candidates = query_candidates(nb, tiers, top_per_tier, null_col, force)
    total = len(candidates)

    # Print plan
    tier_counts: dict[str, int] = {}
    for c in candidates:
        tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
    passes_str = f", {n_passes} passes" if n_passes > 1 else ""
    print(
        f"Backfill: {total} entries, probes={','.join(probes)}, device={device}{passes_str}"
    )
    for t, n in sorted(tier_counts.items()):
        print(f"  {t}: {n}")

    if total == 0:
        print("Nothing to backfill.")
        nb.conn.close()
        return

    if dry_run:
        for c in candidates[:20]:
            print(
                f"  [{c.graph_fingerprint}] tier={c.tier} score={c.composite_score:.1f}"
            )
        if total > 20:
            print(f"  ... and {total - 20} more")
        print(f"\nDry run: would evaluate {total} entries.")
        nb.conn.close()
        return

    # Pre-fetch for rescoring
    all_ids = [c.result_id for c in candidates]
    pr_cache = prefetch_program_results(nb.conn, all_ids)

    needs_model = bool(set(probes) & _PROBE_NEEDS_MODEL)
    needs_train = bool(set(probes) & _PROBE_NEEDS_TRAIN)

    evaluated = 0
    failed = 0
    no_graph = 0
    t0 = time.time()

    for i, cand in enumerate(candidates):
        fp = cand.graph_fingerprint
        if not cand.graph_json or cand.graph_json == "{}":
            no_graph += 1
            continue

        # Determine which probes to actually run for this candidate
        active_probes = [
            p for p in probes if force or not _probe_has_data(nb, cand.result_id, p)
        ]
        if not active_probes:
            continue

        try:
            pass_results: List[Dict[str, Any]] = []

            for pass_idx in range(n_passes):
                pass_updates: Dict[str, Any] = {}
                model = None
                graph = None

                if needs_model:
                    model, graph = reconstruct_model(cand.graph_json, device)
                    if needs_train:
                        micro_train(model, train_steps, device)

                for probe_name in active_probes:
                    probe_updates = _run_single_probe(
                        probe_name,
                        model,
                        graph,
                        device,
                        cand,
                        fp_timeout,
                    )
                    if probe_updates:
                        pass_updates.update(probe_updates)

                if model is not None:
                    del model
                    clear_gpu(device)

                pass_results.append(pass_updates)

            # Average across passes
            all_updates = _average_numeric_updates(pass_results)

            # Write all results in one batch
            if all_updates:
                store_probe_results(nb, cand.result_id, all_updates)

            # Rescore with new data
            new_score, old_score = rescore_entry(
                nb,
                cand.entry_id,
                cand.result_id,
                cand.is_reference,
                pr_cache,
                all_updates,
            )
            evaluated += 1

            delta = new_score - old_score
            delta_str = f" ({delta:+.1f})" if abs(delta) > 0.1 else ""
            print(f"  [{fp}] {cand.tier} score={new_score:.1f}{delta_str}")

        except (RuntimeError, KeyError, ValueError, TypeError) as e:
            failed += 1
            print(f"  [{fp}] error: {e}")
            clear_gpu(device)

        if (i + 1) % 10 == 0:
            nb.conn.commit()

    nb.conn.commit()
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Evaluated: {evaluated}")
    print(f"  Failed:    {failed}")
    print(f"  No graph:  {no_graph}")

    nb.conn.close()


def _run_single_probe(
    name: str,
    model: Optional[nn.Module],
    graph: Any,
    device: str,
    cand: Candidate,
    fp_timeout: int,
) -> Optional[Dict[str, Any]]:
    """Dispatch to the appropriate probe function."""
    if name == "binding":
        return run_binding_probe(model, device)
    if name == "hellaswag":
        return run_hellaswag_probe(model, device, cand.tier)
    if name == "blimp":
        return run_blimp_probe(model, device, cand.tier)
    if name == "triage":
        return run_triage_probe(model, graph, device)
    if name == "fingerprint":
        return run_fingerprint_probe(
            cand.result_id,
            cand.graph_json,
            device,
            fp_timeout,
        )
    return None


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Unified backfill runner for all scoring probes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Probes: binding, hellaswag, blimp, triage, fingerprint, rescore, all",
    )
    parser.add_argument(
        "--probe",
        required=True,
        help="Comma-separated probe names, or 'all' / 'rescore'",
    )
    parser.add_argument("--top", type=int, default=50, help="Max entries per tier")
    parser.add_argument(
        "--tier",
        default="validation,investigation,breakthrough,screening",
        help="Comma-separated tiers to backfill",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Run N passes and average results (reduces variance)",
    )
    parser.add_argument(
        "--fp-timeout", type=int, default=30, help="Fingerprint subprocess timeout (s)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-evaluate even if data exists"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    probe_str = args.probe.strip().lower()
    if probe_str == "all":
        probes = list(_ALL_PROBES)
    elif probe_str == "rescore":
        # Rescore-only mode
        nb = LabNotebook(DB_PATH)
        total, changed = rescore_all(nb)
        print(f"Rescored {total} entries, {changed} changed.")
        nb.conn.close()
        return
    else:
        probes = [p.strip() for p in probe_str.split(",")]
        invalid = set(probes) - set(_ALL_PROBES)
        if invalid:
            parser.error(f"Unknown probes: {invalid}. Valid: {', '.join(_ALL_PROBES)}")

    tiers = [t.strip() for t in args.tier.split(",")]

    run_backfill(
        probes=probes,
        tiers=tiers,
        top_per_tier=args.top,
        device=args.device,
        train_steps=args.train_steps,
        n_passes=args.passes,
        force=args.force,
        dry_run=args.dry_run,
        fp_timeout=args.fp_timeout,
    )


if __name__ == "__main__":
    main()
