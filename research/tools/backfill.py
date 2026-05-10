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
import json
import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from research.defaults import VOCAB_SIZE
from research.scientist.leaderboard_scoring import (
    prefetch_program_results,
)
from research.scientist.leaderboard_rescore import (
    rescore_entry as canonical_rescore_entry,
    rescore_leaderboard,
)
from research.scientist.notebook import LabNotebook
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json
from research.training.data_pipeline import CorpusConfig, CorpusTokenBatcher
from research.training.loss_ops import clip_grad_norm_, next_token_cross_entropy
from research.tools._script_audit import (
    build_metric_backfill_context,
    complete_script_experiment,
    fail_script_experiment,
    start_script_experiment,
)

DB_PATH = "research/runs.db"
logger = logging.getLogger(__name__)
_BACKFILL_CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "corpus",
    "wikitext103_train.npy",
)
_BACKFILL_BATCHERS: dict[int, CorpusTokenBatcher | None] = {}
_BACKFILL_CORPUS_WARNED = False


def _resolve_candidate_graph_json(nb: LabNotebook, value: Any) -> str:
    return resolve_graph_json_value(nb.conn, nb.db_path, value)


# ── Probe registry ──────────────────────────────────────────────────────

_PROBE_NEEDS_TRAIN = frozenset({"binding", "hellaswag", "blimp"})
_PROBE_NEEDS_MODEL = frozenset(
    {
        "binding",
        "hellaswag",
        "blimp",
        "triage",
        "fingerprint",
        "induction_intermediate",
        "binding_intermediate",
    }
)
_ALL_PROBES = (
    "binding",
    "hellaswag",
    "blimp",
    "triage",
    "fingerprint",
    "induction_intermediate",
    "binding_intermediate",
)


# ── Model lifecycle (shared, not duplicated per probe) ──────────────────


def reconstruct_model(graph_json_str: str, device: str) -> nn.Module:
    graph = graph_from_json(graph_json_str)
    model = compile_model([graph], vocab_size=VOCAB_SIZE)
    return model.to(device).eval(), graph


def _get_backfill_batcher(vocab_size: int) -> CorpusTokenBatcher | None:
    cached = _BACKFILL_BATCHERS.get(int(vocab_size))
    if cached is not None:
        return cached
    config = CorpusConfig(
        path=_BACKFILL_CORPUS_PATH,
        fmt="auto",
        tokenizer="byte",
        max_chars=200_000,
        train_fraction=1.0,
        val_fraction=0.0,
    )
    batcher = CorpusTokenBatcher(config, int(vocab_size))
    if not batcher.ready:
        _BACKFILL_BATCHERS[int(vocab_size)] = None
        return None
    _BACKFILL_BATCHERS[int(vocab_size)] = batcher
    return batcher


def _sample_micro_train_batch(
    vocab_size: int,
    *,
    batch_size: int,
    seq_len: int,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    global _BACKFILL_CORPUS_WARNED

    batcher = _get_backfill_batcher(int(vocab_size))
    if batcher is not None:
        batch = batcher.sample_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            generator=generator,
            device=torch.device(device),
            split="train",
        )
        if batch is not None:
            return batch

    if not _BACKFILL_CORPUS_WARNED:
        logger.warning(
            "Backfill micro-train corpus unavailable; falling back to random tokens (%s)",
            _BACKFILL_CORPUS_PATH,
        )
        _BACKFILL_CORPUS_WARNED = True
    return torch.randint(0, int(vocab_size), (batch_size, seq_len), device=device)


def micro_train(
    model: nn.Module,
    steps: int,
    device: str,
    seq_len: int = 128,
    batch_size: int = 8,
    lr: float = 3e-4,
) -> None:
    """Corpus-backed micro-training so probe scores reflect learned behavior."""
    model.train()
    vs = getattr(model, "vocab_size", VOCAB_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)
    for _ in range(steps):
        data = _sample_micro_train_batch(
            vs,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            generator=generator,
        )
        logits = model(data)
        loss = next_token_cross_entropy(logits, data, int(vs))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model, 1.0)
        opt.step()
    model.eval()


def clear_gpu(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        import gc

        gc.collect()
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


_SIGNAL_WHERE_CLAUSE = (
    "(COALESCE(pr.induction_screening_auc, 0) > 0.05 "
    "OR COALESCE(pr.binding_screening_auc, 0) > 0.05 "
    "OR COALESCE(pr.ar_legacy_auc, 0) > 0.05 "
    "OR COALESCE(pr.hellaswag_acc, 0) > 0.30 "
    "OR COALESCE(pr.blimp_overall_accuracy, 0) > 0.55)"
)


def query_fingerprint_file_candidates(
    nb: LabNotebook,
    path: str,
    null_column: Optional[str],
    force: bool,
    shard: Optional[Tuple[int, int]] = None,
    limit: Optional[int] = None,
) -> List[Candidate]:
    """Candidates from a ranked JSONL file (e.g. probe_priority_next.jsonl).

    Each JSON line must carry at minimum ``result_id`` and ``fp``. Rows are
    consumed in file order (priority-descending). ``graph_json`` is joined
    from program_results. Rows are filtered against ``null_column`` unless
    ``force`` is set — same semantics as the other query helpers.
    """
    fingerprints: List[Tuple[str, str]] = []  # (result_id, fp)
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(row.get("result_id") or "").strip()
            fp = str(row.get("fp") or row.get("graph_fingerprint") or "").strip()
            if rid and fp:
                fingerprints.append((rid, fp))
    if not fingerprints:
        return []

    # Look up graph_json + leaderboard context for each result_id
    rid_to_meta: Dict[str, Dict[str, Any]] = {}
    chunk = 500
    for start in range(0, len(fingerprints), chunk):
        batch = fingerprints[start : start + chunk]
        placeholders = ",".join("?" for _ in batch)
        rows = nb.conn.execute(
            f"""
            SELECT pr.result_id, pr.graph_fingerprint, pr.graph_json,
                   l.entry_id, l.tier, l.composite_score, l.is_reference,
                   l.model_source,
                   pr.{null_column} AS probe_val
            FROM program_results_compat pr
            LEFT JOIN leaderboard l ON pr.result_id = l.result_id
            WHERE pr.result_id IN ({placeholders})
            """
            if null_column
            else f"""
            SELECT pr.result_id, pr.graph_fingerprint, pr.graph_json,
                   l.entry_id, l.tier, l.composite_score, l.is_reference,
                   l.model_source,
                   NULL AS probe_val
            FROM program_results_compat pr
            LEFT JOIN leaderboard l ON pr.result_id = l.result_id
            WHERE pr.result_id IN ({placeholders})
            """,
            tuple(r for r, _ in batch),
        ).fetchall()
        for r in rows:
            rid_to_meta[str(r["result_id"])] = dict(r)

    shard_idx, shard_n = shard if shard is not None else (0, 1)

    out: List[Candidate] = []
    seen_fp: set[str] = set()
    idx = -1
    for rid, fp in fingerprints:
        meta = rid_to_meta.get(rid)
        if meta is None:
            continue
        gj = meta.get("graph_json")
        if not gj:
            continue
        if not force and null_column and meta.get("probe_val") is not None:
            continue
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        idx += 1
        if shard_n > 1 and (idx % shard_n) != shard_idx:
            continue
        out.append(
            Candidate(
                entry_id=meta.get("entry_id") or f"priority:{fp[:12]}",
                result_id=rid,
                tier=meta.get("tier") or "priority",
                composite_score=float(meta.get("composite_score") or 0),
                is_reference=bool(meta.get("is_reference") or 0),
                model_source=meta.get("model_source") or "",
                graph_json=_resolve_candidate_graph_json(nb, gj),
                graph_fingerprint=fp,
            )
        )
        if limit and len(out) >= limit:
            break
    return out


def query_signal_candidates(
    nb: LabNotebook,
    null_column: Optional[str],
    force: bool,
    shard: Optional[Tuple[int, int]] = None,
) -> List[Candidate]:
    """Signal-only candidates across ALL tiers (and off-leaderboard).

    Returns one Candidate per unique graph_fingerprint that passes the
    capability-signal filter, ordered by composite signal strength
    (hellaswag_acc DESC, then induction_screening_auc DESC). For off-leaderboard rows,
    the Candidate has entry_id='' and composite_score=0.
    """
    where = _SIGNAL_WHERE_CLAUSE
    if null_column and not force:
        where += f" AND pr.{null_column} IS NULL"

    shard_idx, shard_n = shard if shard is not None else (0, 1)
    rows = nb.conn.execute(
        f"""
        WITH base AS (
            SELECT
                pr.result_id,
                pr.graph_fingerprint,
                pr.graph_json,
                pr.hellaswag_acc,
                l.entry_id,
                l.tier,
                l.composite_score,
                l.is_reference,
                l.model_source,
                (
                    COALESCE(pr.induction_screening_auc, 0) * 3.0
                    + COALESCE(pr.binding_screening_auc, 0) * 3.0
                    + COALESCE(pr.ar_legacy_auc, 0) * 2.0
                    + MAX(COALESCE(pr.hellaswag_acc, 0) - 0.25, 0) * 2.0
                    + MAX(COALESCE(pr.blimp_overall_accuracy, 0) - 0.50, 0) * 1.5
                ) AS signal_strength
            FROM program_results_compat pr
            LEFT JOIN leaderboard l ON pr.result_id = l.result_id
            WHERE {where}
              AND pr.graph_fingerprint IS NOT NULL
              AND pr.graph_fingerprint != ''
              AND pr.graph_json IS NOT NULL
              AND pr.graph_json != ''
              AND pr.graph_json != '{{}}'
        ),
        deduped AS (
            SELECT *
            FROM (
                SELECT
                    base.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY graph_fingerprint
                        ORDER BY signal_strength DESC,
                                 COALESCE(hellaswag_acc, 0) DESC,
                                 result_id
                    ) AS fp_rank
                FROM base
            )
            WHERE fp_rank = 1
        ),
        ranked AS (
            SELECT
                deduped.*,
                ROW_NUMBER() OVER (
                    ORDER BY signal_strength DESC,
                             COALESCE(hellaswag_acc, 0) DESC,
                             graph_fingerprint
                ) - 1 AS global_idx
            FROM deduped
        )
        SELECT
            result_id,
            graph_fingerprint,
            graph_json,
            entry_id,
            tier,
            composite_score,
            is_reference,
            model_source
        FROM ranked
        WHERE (? = 1 OR (global_idx % ?) = ?)
        ORDER BY global_idx
        """,
        (1 if shard_n <= 1 else 0, shard_n, shard_idx),
    ).fetchall()

    return [
        Candidate(
            entry_id=r["entry_id"] or f"signal:{r['graph_fingerprint'][:12]}",
            result_id=r["result_id"],
            tier=r["tier"] or "signal",
            composite_score=float(r["composite_score"] or 0),
            is_reference=bool(r["is_reference"] or 0),
            model_source=r["model_source"] or "",
            graph_json=_resolve_candidate_graph_json(nb, r["graph_json"]),
            graph_fingerprint=r["graph_fingerprint"],
        )
        for r in rows
    ]


def query_candidates(
    nb: LabNotebook,
    tiers: Sequence[str],
    top_per_tier: int,
    null_column: Optional[str],
    force: bool,
    shard: Optional[Tuple[int, int]] = None,
) -> List[Candidate]:
    """Single candidate query for all probes. Returns top N per tier.

    If `shard=(i, N)` is given, deterministically filters to rows where
    (per-tier-index % N == i). The tier-local index is computed over the
    composite-score-DESC ordering before the top-N clamp, so sharded
    workers cover disjoint fingerprints at similar score depths.
    """
    tier_ph = ",".join("?" for _ in tiers)
    where = f"l.tier IN ({tier_ph})"
    if null_column and not force:
        where += f" AND pr.{null_column} IS NULL"

    shard_idx, shard_n = shard if shard is not None else (0, 1)
    rows = nb.conn.execute(
        f"""
        WITH ranked AS (
            SELECT
                l.entry_id,
                l.result_id,
                l.tier,
                l.composite_score,
                l.is_reference,
                l.model_source,
                pr.graph_json,
                pr.graph_fingerprint,
                ROW_NUMBER() OVER (
                    PARTITION BY l.tier
                    ORDER BY l.composite_score DESC, l.result_id
                ) - 1 AS tier_idx
            FROM leaderboard l
            LEFT JOIN program_results_compat pr ON l.result_id = pr.result_id
            WHERE {where}
        ),
        sharded AS (
            SELECT
                ranked.*,
                ROW_NUMBER() OVER (
                    PARTITION BY tier
                    ORDER BY tier_idx
                ) AS shard_rank
            FROM ranked
            WHERE (? = 1 OR (tier_idx % ?) = ?)
        )
        SELECT
            entry_id,
            result_id,
            tier,
            composite_score,
            is_reference,
            model_source,
            graph_json,
            graph_fingerprint
        FROM sharded
        WHERE shard_rank <= ?
        ORDER BY tier, shard_rank
        """,
        tuple(tiers) + (1 if shard_n <= 1 else 0, shard_n, shard_idx, top_per_tier),
    ).fetchall()

    tier_order = {tier: idx for idx, tier in enumerate(tiers)}
    candidates = [
        Candidate(
            entry_id=r["entry_id"],
            result_id=r["result_id"],
            tier=r["tier"],
            composite_score=float(r["composite_score"] or 0),
            is_reference=bool(r["is_reference"]),
            model_source=r["model_source"] or "",
            graph_json=_resolve_candidate_graph_json(nb, r["graph_json"]),
            graph_fingerprint=(r["graph_fingerprint"] or ""),
        )
        for r in rows
    ]
    candidates.sort(
        key=lambda c: (
            tier_order.get(c.tier, len(tiers)),
            -float(c.composite_score),
            c.result_id,
        )
    )
    return candidates


# ── Rescore (single implementation) ─────────────────────────────────────


def rescore_entry(
    nb: LabNotebook,
    entry_id: str,
    result_id: str,
    is_ref: bool,
    pr_cache: Dict[str, Dict],
    pr_updates: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """Compatibility wrapper around the canonical leaderboard rescore helper."""
    return canonical_rescore_entry(
        nb,
        entry_id,
        result_id,
        is_ref,
        pr_cache,
        pr_updates=pr_updates,
        reason="backfill_rescore",
    )


def rescore_all(nb: LabNotebook) -> Tuple[int, int]:
    """Bulk rescore all leaderboard entries. Returns (total, changed)."""
    return rescore_leaderboard(nb, reason="backfill_rescore")


# ── Probe functions (each returns dict of columns to write) ─────────────


def run_binding_probe(
    model: nn.Module,
    device: str,
) -> Dict[str, Any]:
    from research.eval.binding_pipeline import (
        compute_binding_screening_composite,
        compute_local_only,
        run_full_binding_probes,
    )

    probe = run_full_binding_probes(model, device=device)
    bc = compute_binding_screening_composite(
        probe.ar_legacy_auc, probe.induction_screening_auc, probe.binding_screening_auc
    )
    is_local = compute_local_only(
        probe.ar_legacy_auc, probe.induction_screening_auc, probe.binding_screening_auc
    )
    result = probe.to_result_dict()
    result["binding_screening_composite"] = bc
    result["local_only"] = is_local
    return result


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
        "hellaswag_metric_version": hs.get("hellaswag_metric_version"),
        "hellaswag_tokenizer_mode": hs.get("hellaswag_tokenizer_mode"),
        "hellaswag_tiktoken_encoding": hs.get("hellaswag_tiktoken_encoding"),
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


def run_induction_intermediate_probe(model: nn.Module, device: str) -> Dict[str, Any]:
    """Investigation-tier v2 induction probe (median-of-3-seeds)."""
    from research.eval.induction_intermediate_probe import (
        run_induction_intermediate,
    )

    r = run_induction_intermediate(model, device=device)
    ok = str(r.status or "") == "ok"
    return {
        "induction_intermediate_auc": r.auc if ok else None,
        "induction_intermediate_max_gap_acc": r.max_gap_acc if ok else None,
        "induction_intermediate_steps_trained": r.steps_trained,
        "induction_intermediate_status": r.status,
        "induction_intermediate_elapsed_ms": r.elapsed_ms,
        "induction_intermediate_protocol_version": r.protocol_version,
    }


def run_binding_intermediate_probe(model: nn.Module, device: str) -> Dict[str, Any]:
    """Investigation-tier v2 binding probe (median-of-3-seeds, 2400 steps)."""
    from research.eval.binding_intermediate_probe import (
        run_binding_intermediate,
    )

    r = run_binding_intermediate(model, device=device)
    ok = str(r.status or "") == "ok"
    return {
        "binding_intermediate_auc": r.auc if ok else None,
        "binding_intermediate_max_distance_acc": (r.max_distance_acc if ok else None),
        "binding_intermediate_train_steps": r.train_steps,
        "binding_intermediate_status": r.status,
        "binding_intermediate_elapsed_ms": r.elapsed_ms,
        "binding_intermediate_protocol_version": r.protocol_version,
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
    provenance_context: Optional[Dict[str, Any]] = None,
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
        if provenance_context and "data_provenance_json" in pr_cols:
            row = nb.conn.execute(
                "SELECT data_provenance_json FROM program_results_compat WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            raw_payload = row["data_provenance_json"] if row else None
            try:
                payload = json.loads(raw_payload) if raw_payload else {}
            except (json.JSONDecodeError, TypeError, ValueError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            history = payload.get("metric_backfills")
            if not isinstance(history, list):
                history = []
            history = [entry for entry in history if isinstance(entry, dict)]
            history.append(dict(provenance_context))
            payload["metric_backfills"] = history[-5:]
            payload["last_metric_backfill"] = dict(provenance_context)
            nb.conn.execute(
                "UPDATE program_results SET data_provenance_json = ? WHERE result_id = ?",
                (json.dumps(payload, sort_keys=True, separators=(",", ":")), result_id),
            )
        fp_row = nb.conn.execute(
            "SELECT graph_fingerprint FROM program_results_compat WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        nb.upsert_induction_metric_v2(
            graph_fingerprint=str(fp_row["graph_fingerprint"] if fp_row else ""),
            result_id=str(result_id),
            row=updates,
            source_cohort="runtime_backfill",
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
    "binding": "induction_screening_auc",
    "hellaswag": "hellaswag_acc",
    "blimp": "blimp_overall_accuracy",
    "triage": "activation_sparsity_score",
    "fingerprint": "fp_cka_vs_transformer",
    "induction_intermediate": "induction_intermediate_auc",
    "binding_intermediate": "binding_intermediate_auc",
}

# Graph-deterministic probes: the result depends only on the graph, so rows
# sharing a graph_fingerprint can reuse a single computed result.
_FP_REUSABLE_PROBE_COLS: Dict[str, List[str]] = {
    "induction_intermediate": [
        "induction_intermediate_auc",
        "induction_intermediate_max_gap_acc",
        "induction_intermediate_steps_trained",
        "induction_intermediate_status",
        "induction_intermediate_elapsed_ms",
        "induction_intermediate_protocol_version",
    ],
    "binding_intermediate": [
        "binding_intermediate_auc",
        "binding_intermediate_max_distance_acc",
        "binding_intermediate_train_steps",
        "binding_intermediate_status",
        "binding_intermediate_elapsed_ms",
        "binding_intermediate_protocol_version",
    ],
}


def _seed_fingerprint_cache(
    nb: LabNotebook,
    reusable_probe_cols: Dict[str, List[str]],
    fp_cache: Dict[str, Dict[str, Any]],
) -> None:
    """Pre-populate fp_cache from existing program_results rows.

    Captures completed probe values from earlier backfill runs so the new run
    reuses them for every sibling entry that shares a graph_fingerprint.
    """
    all_cols = sorted({c for cols in reusable_probe_cols.values() for c in cols})
    if not all_cols:
        return
    existing_cols = _get_table_columns(nb, "program_results")
    present_cols = [c for c in all_cols if c in existing_cols]
    if not present_cols:
        return
    or_clause = " OR ".join(f"pr.{c} IS NOT NULL" for c in present_cols)
    sql = (
        f"SELECT pr.graph_fingerprint, "
        f"{', '.join('pr.' + c for c in present_cols)} "
        f"FROM program_results_compat pr "
        f"WHERE pr.graph_fingerprint IS NOT NULL "
        f"AND pr.graph_fingerprint != '' AND ({or_clause})"
    )
    rows = nb.conn.execute(sql).fetchall()
    for r in rows:
        fp_key = r["graph_fingerprint"]
        if not fp_key:
            continue
        entry = fp_cache.setdefault(fp_key, {})
        for _probe_name, probe_cols in reusable_probe_cols.items():
            status_col = next((c for c in probe_cols if c.endswith("_status")), None)
            if status_col in present_cols:
                status = r[status_col]
                if status is not None and str(status) != "ok":
                    continue
            for c in probe_cols:
                if c not in present_cols:
                    continue
                v = r[c]
                if v is not None and c not in entry:
                    entry[c] = v


def _prefetch_probe_state(
    nb: LabNotebook, result_ids: Sequence[str], probes: Sequence[str]
) -> dict[str, dict[str, bool]]:
    """Bulk-load probe completeness once instead of querying SQLite per probe."""
    if not result_ids:
        return {}
    columns = {
        _PROBE_NULL_COLUMN[probe_name]
        for probe_name in probes
        if probe_name in _PROBE_NULL_COLUMN
    }
    if not columns:
        return {result_id: {} for result_id in result_ids}
    placeholders = ",".join("?" for _ in result_ids)
    select_cols = ", ".join(sorted(columns))
    rows = nb.conn.execute(
        f"SELECT result_id, {select_cols} FROM program_results_compat "
        f"WHERE result_id IN ({placeholders})",
        tuple(result_ids),
    ).fetchall()
    by_result = {
        str(row["result_id"]): {
            probe_name: row[_PROBE_NULL_COLUMN[probe_name]] is not None
            for probe_name in probes
            if probe_name in _PROBE_NULL_COLUMN
        }
        for row in rows
    }
    missing_default = {
        probe_name: False for probe_name in probes if probe_name in _PROBE_NULL_COLUMN
    }
    for result_id in result_ids:
        by_result.setdefault(str(result_id), dict(missing_default))
    return by_result


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
    shard: Optional[Tuple[int, int]] = None,
    signal_only: bool = False,
    fingerprint_file: Optional[str] = None,
) -> None:
    nb, exp_id = start_script_experiment(
        db_path=DB_PATH,
        experiment_type="probe_backfill",
        config={
            "probes": list(probes),
            "tiers": list(tiers),
            "top_per_tier": top_per_tier,
            "device": device,
            "train_steps": train_steps,
            "n_passes": n_passes,
            "force": force,
            "fp_timeout": fp_timeout,
        },
        source_script="backfill",
        hypothesis=f"Backfill probes: {','.join(probes)}",
    )
    provenance_context = build_metric_backfill_context(
        kind="probe_backfill",
        source_script="backfill",
        experiment_id=exp_id,
        device=device,
        probes=list(probes),
        tiers=list(tiers),
        top_per_tier=top_per_tier,
        train_steps=train_steps,
        passes=n_passes,
        force=bool(force),
        fp_timeout=fp_timeout,
    )

    # Determine which null column to filter on (use first probe's column)
    null_col = _PROBE_NULL_COLUMN.get(probes[0]) if len(probes) == 1 else None

    if fingerprint_file:
        candidates = query_fingerprint_file_candidates(
            nb,
            fingerprint_file,
            null_col,
            force,
            shard=shard,
            limit=top_per_tier if top_per_tier > 0 else None,
        )
        print(
            f"Fingerprint-file mode: {len(candidates)} candidates from {fingerprint_file}"
        )
    elif signal_only:
        candidates = query_signal_candidates(nb, null_col, force, shard=shard)
        print(
            f"Signal-only mode: {len(candidates)} unique fingerprints with capability signal"
        )
    else:
        candidates = query_candidates(
            nb, tiers, top_per_tier, null_col, force, shard=shard
        )
    total = len(candidates)
    if shard is not None and shard[1] > 1:
        print(f"Shard {shard[0]}/{shard[1]}: {total} candidates after filter")

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
        complete_script_experiment(
            nb,
            exp_id,
            results={"total": 0, "evaluated": 0, "failed": 0, "no_graph": 0},
            summary=f"Backfill {','.join(probes)} found no candidates",
        )
        nb.conn.close()
        return

    if dry_run:
        for c in candidates[:20]:
            print(
                f"  [{c.graph_fingerprint[:12]}] tier={c.tier} score={c.composite_score:.1f}"
            )
        if total > 20:
            print(f"  ... and {total - 20} more")
        print(f"\nDry run: would evaluate {total} entries.")
        fail_script_experiment(
            nb,
            exp_id,
            error="Dry-run invocation does not write results",
            results={"total": total, "evaluated": 0, "dry_run": True},
        )
        nb.conn.close()
        return

    # Pre-fetch for rescoring
    all_ids = [c.result_id for c in candidates]
    pr_cache = prefetch_program_results(nb.conn, all_ids)
    probe_state = _prefetch_probe_state(nb, all_ids, probes)

    needs_model = bool(set(probes) & _PROBE_NEEDS_MODEL)
    needs_train = bool(set(probes) & _PROBE_NEEDS_TRAIN)

    # Graph-fingerprint-level dedup cache for graph-deterministic probes:
    # a probe that only reads the graph (not run-specific metrics) yields the
    # same result for every entry sharing a fingerprint — so run once, reuse.
    reusable_probe_cols: Dict[str, List[str]] = {
        p: cols for p, cols in _FP_REUSABLE_PROBE_COLS.items() if p in probes
    }
    fp_cache: Dict[str, Dict[str, Any]] = {}
    if reusable_probe_cols and not force:
        _seed_fingerprint_cache(nb, reusable_probe_cols, fp_cache)

    evaluated = 0
    failed = 0
    no_graph = 0
    cache_reused = 0
    t0 = time.time()

    try:
        for i, cand in enumerate(candidates):
            fp = cand.graph_fingerprint[:12]
            if not cand.graph_json or cand.graph_json == "{}":
                no_graph += 1
                continue

            # Determine which probes to actually run for this candidate
            active_probes = [
                p
                for p in probes
                if force or not probe_state.get(cand.result_id, {}).get(p, False)
            ]
            if not active_probes:
                continue

            try:
                pass_results: List[Dict[str, Any]] = []
                reused_any = False

                for pass_idx in range(n_passes):
                    pass_updates: Dict[str, Any] = {}

                    # Dedup by graph_fingerprint: split active_probes into
                    # ones we can reuse from cache vs ones we must run.
                    cached_for_fp = fp_cache.get(cand.graph_fingerprint, {})
                    probes_to_run: List[str] = []
                    for probe_name in active_probes:
                        needed = reusable_probe_cols.get(probe_name)
                        if needed and all(c in cached_for_fp for c in needed):
                            for c in needed:
                                pass_updates[c] = cached_for_fp[c]
                            reused_any = True
                        else:
                            probes_to_run.append(probe_name)

                    model = None
                    graph = None
                    if probes_to_run and needs_model:
                        model, graph = reconstruct_model(cand.graph_json, device)
                        if needs_train:
                            micro_train(model, train_steps, device)

                    for probe_name in probes_to_run:
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
                            # Populate fp_cache so siblings with same graph reuse.
                            needed_cache = reusable_probe_cols.get(probe_name)
                            if needed_cache:
                                entry = fp_cache.setdefault(cand.graph_fingerprint, {})
                                for c in needed_cache:
                                    if (
                                        c in probe_updates
                                        and probe_updates[c] is not None
                                    ):
                                        entry[c] = probe_updates[c]

                    if model is not None:
                        del model
                        clear_gpu(device)

                    pass_results.append(pass_updates)

                # Average across passes
                all_updates = _average_numeric_updates(pass_results)

                # Write all results in one batch
                if all_updates:
                    store_probe_results(
                        nb,
                        cand.result_id,
                        all_updates,
                        provenance_context=provenance_context,
                    )
                    cand_state = probe_state.setdefault(cand.result_id, {})
                    for probe_name in active_probes:
                        cand_state[probe_name] = True

                # Rescore only if this candidate is on the leaderboard;
                # off-leaderboard rows from --signal-only have synthetic
                # entry_ids (prefixed "signal:") and no leaderboard row.
                if cand.entry_id and not cand.entry_id.startswith("signal:"):
                    new_score, old_score = rescore_entry(
                        nb,
                        cand.entry_id,
                        cand.result_id,
                        cand.is_reference,
                        pr_cache,
                        all_updates,
                    )
                else:
                    new_score = old_score = 0.0
                evaluated += 1

                if reused_any:
                    cache_reused += 1
                delta = new_score - old_score
                delta_str = f" ({delta:+.1f})" if abs(delta) > 0.1 else ""
                reused_str = " [cache]" if reused_any else ""
                if cand.entry_id.startswith("signal:"):
                    # Off-leaderboard: just report probe values, no score.
                    iv2 = all_updates.get("induction_intermediate_auc", 0.0) or 0
                    bv2 = all_updates.get("binding_intermediate_auc", 0.0) or 0
                    print(f"  [{fp}] off-lb iv2={iv2:.3f} bv2={bv2:.3f}{reused_str}")
                else:
                    print(
                        f"  [{fp}] {cand.tier} score={new_score:.1f}{delta_str}{reused_str}"
                    )

            except (RuntimeError, KeyError, ValueError, TypeError) as e:
                failed += 1
                print(f"  [{fp}] error: {e}")
                clear_gpu(device)

            if (i + 1) % 10 == 0:
                nb.conn.commit()
    except KeyboardInterrupt:
        fail_script_experiment(
            nb,
            exp_id,
            error="KeyboardInterrupt",
            results={
                "total": total,
                "evaluated": evaluated,
                "failed": failed,
                "no_graph": no_graph,
            },
        )
        nb.conn.close()
        raise
    except Exception as exc:
        fail_script_experiment(
            nb,
            exp_id,
            error=str(exc),
            results={
                "total": total,
                "evaluated": evaluated,
                "failed": failed,
                "no_graph": no_graph,
            },
        )
        nb.conn.close()
        raise

    nb.conn.commit()
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Evaluated: {evaluated}")
    print(f"  Failed:    {failed}")
    print(f"  No graph:  {no_graph}")
    print(f"  Cache hits: {cache_reused}")

    complete_script_experiment(
        nb,
        exp_id,
        results={
            "total": total,
            "evaluated": evaluated,
            "failed": failed,
            "no_graph": no_graph,
            "elapsed_s": round(elapsed, 3),
            "probes": list(probes),
        },
        summary=(
            f"Backfill {','.join(probes)}: evaluated={evaluated} "
            f"failed={failed} no_graph={no_graph}"
        ),
    )
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
    if name == "induction_intermediate":
        return run_induction_intermediate_probe(model, device)
    if name == "binding_intermediate":
        return run_binding_intermediate_probe(model, device)
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
    parser.add_argument(
        "--shard",
        default=None,
        help="Shard spec 'i/N' — this worker processes rows where "
        "(per-tier index % N) == i. Use for multi-worker parallel backfills.",
    )
    parser.add_argument(
        "--signal-only",
        action="store_true",
        help="Ignore --tier/--top; select every unique graph_fingerprint with "
        "real capability signal (induction/binding/ar>0.05, hellaswag>0.30, "
        "or blimp>0.55), ordered by combined signal strength DESC.",
    )
    parser.add_argument(
        "--fingerprint-file",
        default=None,
        help="JSONL of pre-ranked candidates (e.g. probe_priority_next.jsonl). "
        "Each line must carry 'result_id' and 'fp'. Overrides --tier/--signal-only. "
        "Use --top N to cap; N=0 means consume full file.",
    )
    args = parser.parse_args()

    probe_str = args.probe.strip().lower()
    if probe_str == "all":
        probes = list(_ALL_PROBES)
    elif probe_str == "rescore":
        # Rescore-only mode
        nb, exp_id = start_script_experiment(
            db_path=DB_PATH,
            experiment_type="score_backfill",
            config={"mode": "rescore"},
            source_script="backfill",
            hypothesis="Bulk leaderboard rescore",
        )
        try:
            total, changed = rescore_all(nb)
            print(f"Rescored {total} entries, {changed} changed.")
            complete_script_experiment(
                nb,
                exp_id,
                results={"total": total, "changed": changed, "mode": "rescore"},
                summary=f"Bulk rescore complete: changed={changed}/{total}",
            )
        except KeyboardInterrupt:
            fail_script_experiment(nb, exp_id, error="KeyboardInterrupt")
            nb.conn.close()
            raise
        except Exception as exc:
            fail_script_experiment(nb, exp_id, error=str(exc))
            nb.conn.close()
            raise
        nb.conn.close()
        return
    else:
        probes = [p.strip() for p in probe_str.split(",")]
        invalid = set(probes) - set(_ALL_PROBES)
        if invalid:
            parser.error(f"Unknown probes: {invalid}. Valid: {', '.join(_ALL_PROBES)}")

    tiers = [t.strip() for t in args.tier.split(",")]

    shard: Optional[Tuple[int, int]] = None
    if args.shard:
        try:
            si, sn = args.shard.split("/")
            shard = (int(si), int(sn))
            if shard[1] < 1 or shard[0] < 0 or shard[0] >= shard[1]:
                raise ValueError
        except Exception:
            parser.error(f"Invalid --shard '{args.shard}'. Expected 'i/N' with 0<=i<N.")

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
        shard=shard,
        signal_only=args.signal_only,
        fingerprint_file=args.fingerprint_file,
    )


if __name__ == "__main__":
    main()
