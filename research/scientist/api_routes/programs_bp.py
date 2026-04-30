"""programs API route registration."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
from flask import jsonify, request
from ..runner._types import RunConfig
from ..refinement_scoring import oscillation_risk_score
from research.synthesis.workflow_converter import graph_to_workflow
from ._helpers import get_runner
from ..json_utils import json_safe
from ._strategy_recommendations import (
    annotate_qkv_usage,
    enrich_program_detail,
    program_lineage_chain,
)
from .deps import ApiRouteContext
from ._utils import register_notebook_routes, with_notebook_context

logger = logging.getLogger(__name__)

_TRUST_LABEL_RANK = {
    "": 0,
    "candidate_screening": 1,
    "candidate_grade": 2,
    "reference": 3,
}

_COMPARABILITY_LABEL_RANK = {
    "": 0,
    "screening_only": 1,
    "candidate_comparable": 2,
    "reference_comparable": 3,
}


def _preserve_stronger_label(*values: Any, ranks: Dict[str, int], fallback: str) -> str:
    best = fallback
    best_rank = ranks.get(str(fallback).strip().lower(), 0)
    for value in values:
        normalized = str(value or "").strip().lower()
        if ranks.get(normalized, 0) > best_rank:
            best = normalized
            best_rank = ranks[normalized]
    return best


def _leaderboard_backed_program_detail(nb, result_id: str) -> Optional[Dict[str, Any]]:
    """Synthesize a program-detail payload from leaderboard/reference data."""
    lb = nb.conn.execute(
        "SELECT * FROM leaderboard WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    if not lb:
        return None

    merged: Dict[str, Any] = dict(lb)
    pr = nb.conn.execute(
        "SELECT * FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    if pr:
        merged.update(dict(pr))
        merged = nb._parse_program_json_fields(merged)

    merged.setdefault("result_id", result_id)
    merged["is_reference"] = bool(merged.get("is_reference"))
    if merged["is_reference"]:
        merged["model_source"] = "reference"
    merged["loss_ratio"] = (
        merged.get("loss_ratio")
        if merged.get("loss_ratio") is not None
        else merged.get("screening_loss_ratio")
    )
    merged["novelty_score"] = (
        merged.get("novelty_score")
        if merged.get("novelty_score") is not None
        else merged.get("screening_novelty")
    )
    if not merged.get("graph_fingerprint"):
        ref = next(
            (row for row in nb.get_references() if row.get("result_id") == result_id),
            None,
        )
        if ref:
            merged["graph_fingerprint"] = ref.get("graph_fingerprint")
            merged["architecture_family"] = ref.get("architecture_family")
            merged["param_count"] = merged.get("param_count") or ref.get("param_count")

    if not merged.get("architecture_family"):
        merged["architecture_family"] = nb._classify_architecture_family(
            graph_json=merged.get("graph_json"),
            routing_mode=merged.get("routing_mode"),
        )
    if merged.get("architecture_family") == "Unknown":
        merged["architecture_family"] = nb._reference_family_fallback(
            merged.get("reference_name")
        )

    if merged.get("graph_json") and isinstance(merged.get("graph_json"), str):
        try:
            merged["graph_json_parsed"] = json.loads(merged["graph_json"])
        except json.JSONDecodeError as exc:
            logger.debug(
                "Failed to parse graph_json for result_id=%s in leaderboard-backed detail: %s",
                result_id,
                exc,
            )

    merged.setdefault("stage1_passed", 1 if merged.get("is_reference") else 0)
    merged.setdefault("has_training_curve", False)
    merged.setdefault("experiment_id", None)
    merged.setdefault("reference_like", bool(merged.get("is_reference")))
    merged.setdefault(
        "most_similar_to",
        merged.get("reference_name") or merged.get("architecture_family"),
    )
    return merged


def _get_cached_program_explanation(nb, result_id: str) -> Optional[str]:
    row = nb.conn.execute(
        "SELECT llm_explanation FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    if not row:
        return None
    explanation = row[0] if isinstance(row, (tuple, list)) else row["llm_explanation"]
    return explanation or None


def _generate_program_explanation(
    nb, result_id: str, program: Dict[str, Any]
) -> Optional[str]:
    from ..llm.context_experiment import build_program_context
    from ._helpers import get_aria_for_notebook

    aria = get_aria_for_notebook(str(nb.db_path))
    explanation = aria.explain_fingerprint(build_program_context(program))
    if not explanation:
        return None
    nb.conn.execute(
        "UPDATE program_results SET llm_explanation = ? WHERE result_id = ?",
        (explanation, result_id),
    )
    nb.conn.commit()
    return explanation


def _api_program_detail(result_id, nb=None):
    """Full program detail with parsed graph JSON + fingerprint + all metrics."""
    requested_result_id = str(result_id or "").strip()
    canonical_result_id = nb.resolve_canonical_result_id(requested_result_id)
    result_id = canonical_result_id or requested_result_id

    program = nb.get_program_detail(result_id)
    if program is None:
        program = _leaderboard_backed_program_detail(nb, result_id)
    if program is None:
        # Fallback: resolve fingerprint (architecture_desc) to result_id
        row = nb.conn.execute(
            "SELECT result_id FROM leaderboard WHERE architecture_desc = ? LIMIT 1",
            (result_id,),
        ).fetchone()
        if row:
            resolved_id = row[0] if isinstance(row, (tuple, list)) else row["result_id"]
            program = nb.get_program_detail(resolved_id)
            if program is None:
                program = _leaderboard_backed_program_detail(nb, resolved_id)
    if program is None:
        return jsonify({"error": "Not found"}), 404

    program["requested_result_id"] = requested_result_id
    program["canonical_result_id"] = result_id
    program["superseded_requested_result"] = requested_result_id != result_id

    try:
        curve = nb.get_training_curve(result_id)
        program["has_training_curve"] = len(curve) > 0
    except Exception as exc:
        logger.debug(
            "Failed to load training curve for result_id=%s: %s", result_id, exc
        )
        program["has_training_curve"] = False

    cached_explanation = _get_cached_program_explanation(nb, result_id)
    if cached_explanation:
        program["llm_explanation"] = cached_explanation

    program = enrich_program_detail(nb, program)

    try:
        program["lineage_chain"] = program_lineage_chain(nb, result_id)
    except Exception as exc:
        logger.debug(
            "Failed to load lineage chain for result_id=%s: %s", result_id, exc
        )
        program["lineage_chain"] = []

    try:
        evidence_rows = nb.get_causal_rule_evidence(
            result_id=result_id,
            limit=20,
        )
        for item in evidence_rows:
            evidence_id = item.get("evidence_id")
            if evidence_id:
                children = nb.get_causal_ablation_child_observations(
                    evidence_id=evidence_id,
                    limit=8,
                )
                item["child_observation_count"] = len(children)
                item["child_observations"] = children
        program["causal_rule_evidence"] = evidence_rows
    except Exception as exc:
        logger.debug(
            "Failed to load causal evidence for result_id=%s: %s", result_id, exc
        )
        program["causal_rule_evidence"] = []

    return jsonify(json_safe(program))


def _api_program_explanation(result_id, nb=None):
    """Generate or fetch cached LLM explanation for a program."""
    requested_result_id = str(result_id or "").strip()
    canonical_result_id = nb.resolve_canonical_result_id(requested_result_id)
    result_id = canonical_result_id or requested_result_id
    force = bool((request.get_json(silent=True) or {}).get("force", False))

    program = nb.get_program_detail(result_id)
    if program is None:
        program = _leaderboard_backed_program_detail(nb, result_id)
    if program is None:
        return jsonify({"error": "Not found"}), 404

    if not force:
        cached_explanation = _get_cached_program_explanation(nb, result_id)
        if cached_explanation:
            return jsonify(
                json_safe(
                    {
                        "result_id": result_id,
                        "requested_result_id": requested_result_id,
                        "canonical_result_id": result_id,
                        "superseded_requested_result": requested_result_id != result_id,
                        "llm_explanation": cached_explanation,
                        "source": "cached",
                    }
                )
            )

    try:
        explanation = _generate_program_explanation(nb, result_id, program)
    except Exception as exc:
        logger.debug(
            "LLM fingerprint explanation failed for result_id=%s: %s",
            result_id,
            exc,
        )
        return (
            jsonify(
                {
                    "result_id": result_id,
                    "requested_result_id": requested_result_id,
                    "canonical_result_id": result_id,
                    "superseded_requested_result": requested_result_id != result_id,
                    "llm_explanation": None,
                    "source": "unavailable",
                    "error": str(exc),
                }
            ),
            503,
        )

    if not explanation:
        return jsonify(
            {
                "result_id": result_id,
                "requested_result_id": requested_result_id,
                "canonical_result_id": result_id,
                "superseded_requested_result": requested_result_id != result_id,
                "llm_explanation": None,
                "source": "unavailable",
            }
        )

    return jsonify(
        json_safe(
            {
                "result_id": result_id,
                "requested_result_id": requested_result_id,
                "canonical_result_id": result_id,
                "superseded_requested_result": requested_result_id != result_id,
                "llm_explanation": explanation,
                "source": "generated",
            }
        )
    )


def _api_program_lineage(result_id: str, nb=None):
    """Program lineage chain for refinement traceability."""
    program = nb.get_program_detail(result_id)
    if program is None:
        return jsonify({"error": "Not found"}), 404
    chain = program_lineage_chain(nb, result_id)
    return jsonify(
        json_safe(
            {
                "result_id": result_id,
                "lineage_chain": chain,
                "depth": len(chain),
            }
        )
    )


def _api_program_refine_analysis(result_id, nb=None):
    from ..analytics import ExperimentAnalytics, RefinementAnalyzer

    program = nb.get_program_detail(result_id)
    if program is None:
        return jsonify({"error": "Not found"}), 404

    analytics = ExperimentAnalytics(nb)
    analyzer = RefinementAnalyzer(analytics)
    analysis = analyzer.analyze_program_for_refinement(result_id, program)
    return jsonify(json_safe(analysis))


def _api_program_morph(result_id, nb=None):
    """Generate scored mutation candidates for a program."""
    import math as _math
    import random as _random
    from research.synthesis.grammar import GrammarConfig
    from research.synthesis.serializer import graph_from_json, graph_to_json
    from research.synthesis.validator import validate_graph
    from ..search.evolution import _mutate_graph

    body = request.get_json(silent=True) or {}
    intent = str(body.get("intent", "balanced")).lower()
    n_candidates = min(20, max(1, int(body.get("n_candidates", 5))))

    if intent not in ("quality", "compression", "sparsity", "novelty", "balanced"):
        return jsonify({"error": f"Invalid intent: {intent}"}), 400

    program = nb.get_program_detail(result_id)
    if program is None:
        return jsonify({"error": "Not found"}), 404

    graph_json_str = program.get("graph_json")
    if not graph_json_str:
        return jsonify({"error": "No graph JSON for this program"}), 400

    try:
        parent_graph = graph_from_json(graph_json_str)
    except Exception as e:
        return jsonify({"error": f"Could not reconstruct graph: {e}"}), 400

    grammar = GrammarConfig()
    op_success: dict = {}
    try:
        for row in nb.get_op_success_rates():
            n_used = float(row.get("n_used") or 0)
            n_s1 = float(row.get("n_stage1_passed") or 0)
            if n_used > 0:
                op_success[str(row.get("op_name"))] = n_s1 / n_used
    except Exception as exc:
        logger.debug("Failed to load op success rates for morph suggestions: %s", exc)

    if body.get("use_analysis"):
        try:
            from ..analytics import ExperimentAnalytics, RefinementAnalyzer

            analytics = ExperimentAnalytics(nb)
            analyzer = RefinementAnalyzer(analytics)
            analysis_data = analyzer.analyze_program_for_refinement(result_id, program)
            recipe = analysis_data.get("recipe", {})
            hints = recipe.get("grammar_hints", {})
            for op_name, mult in hints.get("boost_ops", {}).items():
                current = grammar.op_weights.get(op_name, 1.0)
                grammar.op_weights[op_name] = min(3.0, current * mult)
        except Exception as e:
            logger.warning("Morph: analysis hint application failed: %s", e)

    rng = _random.Random(hash((result_id, intent, time.time())))
    pool_size = n_candidates * 4
    candidates = []
    seen_fps = set()
    parent_ops = sorted(
        set(str(n.op_name) for n in parent_graph.nodes.values() if not n.is_input)
    )

    for _ in range(pool_size):
        try:
            child = _mutate_graph(parent_graph, grammar, rng)
        except Exception as exc:
            logger.debug(
                "Morph candidate mutation failed for result_id=%s intent=%s: %s",
                result_id,
                intent,
                exc,
            )
            continue
        child.prune_unreachable_nodes()
        validation = validate_graph(child, max_ops=30, max_depth=20)
        if not validation.valid:
            continue
        fp = child.fingerprint()
        if fp in seen_fps:
            continue
        seen_fps.add(fp)

        child_ops_list = [
            str(n.op_name) for n in child.nodes.values() if not n.is_input
        ]
        n_ops = max(1, int(child.n_ops()))
        depth = max(1, int(child.depth()))
        params = max(1.0, float(child.n_params_estimate()))
        unique_ops = len(set(child_ops_list))

        learned_quality = 0.5
        if child_ops_list:
            learned_quality = sum(
                op_success.get(op, 0.5) for op in child_ops_list
            ) / len(child_ops_list)
        compression_proxy = 1.0 / (
            1.0 + _math.log1p(params) + 0.25 * n_ops + 0.15 * depth
        )
        novelty_proxy = min(
            1.0, (unique_ops / max(1, n_ops)) + (0.1 if depth >= 4 else 0.0)
        )
        sparse_hint_ops = (
            "sparse",
            "gate",
            "topk",
            "mask",
            "threshold",
            "skip",
            "mixture",
        )
        sparse_op_bonus = 0.0
        if child_ops_list:
            sparse_op_bonus = sum(
                1.0
                for op in child_ops_list
                if any(t in op.lower() for t in sparse_hint_ops)
            ) / len(child_ops_list)
        sparsity_proxy = min(1.0, 0.7 * compression_proxy + 0.3 * sparse_op_bonus)
        oscillation_risk, stability = oscillation_risk_score(child)
        parent_novelty = float(program.get("novelty_score") or 0.0)
        parent_quality = 1.0 - float(program.get("loss_ratio") or 1.0)

        if intent == "quality":
            score = (
                0.60 * learned_quality
                + 0.25 * parent_quality
                + 0.15 * compression_proxy
                - 0.10 * oscillation_risk
            )
        elif intent == "compression":
            score = (
                0.60 * compression_proxy
                + 0.25 * learned_quality
                + 0.15 * parent_quality
                - 0.10 * oscillation_risk
            )
        elif intent == "sparsity":
            score = (
                0.60 * sparsity_proxy
                + 0.25 * learned_quality
                + 0.15 * compression_proxy
                - 0.10 * oscillation_risk
            )
        elif intent == "novelty":
            score = (
                0.55 * novelty_proxy
                + 0.25 * learned_quality
                + 0.20 * parent_novelty
                - 0.06 * oscillation_risk
            )
        else:
            score = (
                0.35 * learned_quality
                + 0.25 * compression_proxy
                + 0.20 * novelty_proxy
                + 0.20 * max(parent_quality, parent_novelty)
                - 0.10 * oscillation_risk
            )

        child_ops = sorted(set(child_ops_list))
        added_ops = [op for op in child_ops if op not in parent_ops]
        removed_ops = [op for op in parent_ops if op not in child_ops]

        workflow_json = None
        try:
            workflow_json = graph_to_workflow(
                child, workflow_id=fp[:12], name=f"morph_{fp[:8]}"
            )
        except Exception as exc:
            logger.debug(
                "Failed to convert morph candidate to workflow for fingerprint=%s: %s",
                fp[:12],
                exc,
            )

        candidates.append(
            {
                "fingerprint": fp,
                "score": round(float(score), 4),
                "n_ops": n_ops,
                "depth": depth,
                "params_estimate": int(params),
                "unique_ops": unique_ops,
                "ops": child_ops,
                "added_ops": added_ops,
                "removed_ops": removed_ops,
                "graph_json": graph_to_json(child),
                "workflow_json": workflow_json,
                "score_breakdown": {
                    "learned_quality": round(float(learned_quality), 4),
                    "compression_proxy": round(float(compression_proxy), 4),
                    "novelty_proxy": round(float(novelty_proxy), 4),
                    "sparsity_proxy": round(float(sparsity_proxy), 4),
                    "oscillation_risk": round(float(oscillation_risk), 4),
                    "has_residual": int(stability.get("has_residual", 0.0) > 0.5),
                    "norm_count": int(stability.get("norm_count", 0.0)),
                },
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    top = candidates[:n_candidates]

    return jsonify(
        {
            "result_id": result_id,
            "intent": intent,
            "source_ops": parent_ops,
            "source_fingerprint": parent_graph.fingerprint(),
            "n_generated": len(seen_fps),
            "candidates": top,
        }
    )


def _api_program_external_benchmarks(result_id, nb=None):
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, (dict, list)):
        return jsonify({"error": "Payload must be a JSON object or list."}), 400
    ok = nb.set_external_benchmarks(result_id, payload)
    if not ok:
        return jsonify({"error": "Program result not found or payload invalid."}), 404
    return jsonify({"status": "ok", "result_id": result_id})


def _api_program_backfill_metrics(notebook_path: str, result_id, nb=None):
    from ..screening_recompute import recompute_screening_metrics

    program = nb.get_program_detail(result_id)
    if not program:
        return jsonify({"error": "Program not found"}), 404
    body = request.get_json(silent=True) or {}
    device = str(body.get("device") or "cpu").strip().lower()
    mode = str(body.get("mode") or "full_screening").strip().lower()
    allow_insufficient_learning_metrics = bool(
        body.get("allow_insufficient_learning_metrics", True)
    )
    if mode == "probe_only":
        from research.tools.backfill import _fingerprint_one, store_probe_results

        result = _fingerprint_one(
            result_id=str(result_id),
            graph_json_str=str(program.get("graph_json") or ""),
            device=device,
        )
        store_probe_results(
            nb=nb,
            result_id=str(result_id),
            updates=result,
            write_leaderboard=True,
            provenance_context={
                "kind": "program_detail_backfill",
                "source": "api_program_backfill_metrics_probe_only",
                "device": device,
            },
        )
        nb.conn.commit()
        return jsonify({"status": "ok", "result_id": result_id, "backfill": result})

    result = recompute_screening_metrics(
        nb=nb,
        notebook_path=Path(notebook_path),
        result_id=str(result_id),
        device=device,
        allow_insufficient_learning_metrics=allow_insufficient_learning_metrics,
        provenance_source="api_program_backfill_metrics",
    )
    return jsonify({"status": "ok", "result_id": result_id, "backfill": result})


def _api_program_backfill_loss(notebook_path: str, result_id, nb=None):
    program = nb.get_program_detail(result_id)
    if not program:
        return jsonify({"error": "Program not found"}), 404
    graph_json = program.get("graph_json")
    if not graph_json:
        return jsonify({"error": "No graph_json for this program"}), 400
    initial_loss = program.get("initial_loss")

    exp_id = program.get("experiment_id")
    config_json = None
    if exp_id:
        exp_row = nb.conn.execute(
            "SELECT config_json FROM experiments WHERE experiment_id = ?", (exp_id,)
        ).fetchone()
        if exp_row:
            config_json = exp_row["config_json"]

    import dataclasses as _dc

    config_dict = json.loads(config_json) if config_json else {}
    valid_fields = {f.name for f in _dc.fields(RunConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = RunConfig(**filtered)

    import torch

    body = request.get_json(silent=True) or {}
    device = str(body.get("device", "cpu"))
    dev = torch.device(device)

    from research.synthesis.serializer import graph_from_json as _gfj

    graph = _gfj(graph_json)
    graph_dim = getattr(graph, "model_dim", None)
    if graph_dim and config.model_dim != graph_dim:
        config.model_dim = int(graph_dim)

    from ..native_runner import compile_model_native_first as _compile

    layer_graphs = [graph] * config.n_layers
    model = _compile(
        layer_graphs, vocab_size=config.vocab_size, max_seq_len=config.max_seq_len
    )
    model = model.to(dev).eval()

    seq_len = min(128, config.max_seq_len)
    updates = {}

    try:
        losses = []
        with torch.no_grad():
            for i in range(2):
                ids = torch.randint(0, config.vocab_size, (4, seq_len), device=dev)
                logits = model(ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    ids[:, 1:].reshape(-1),
                )
                if torch.isfinite(loss):
                    losses.append(loss.item())
        if losses:
            disc_loss = sum(losses) / len(losses)
            updates["discovery_loss"] = disc_loss
            if initial_loss:
                disc_ratio = disc_loss / max(float(initial_loss), 1e-6)
                updates["discovery_loss_ratio"] = disc_ratio
            else:
                updates["discovery_loss_ratio"] = None
                updates["discovery_loss_ratio_note"] = "initial_loss_missing"
    except Exception as e:
        updates["discovery_loss_error"] = str(e)

    data_mode = str(config.data_mode or "random").strip().lower()
    if data_mode in ("corpus", "huggingface"):
        try:
            runner = get_runner(notebook_path)
            if data_mode == "huggingface":
                batcher = runner._get_hf_batcher(config)
            else:
                batcher = runner._get_corpus_batcher(config)
            if batcher and batcher.ready:
                losses = []
                gen = torch.Generator(device=dev)
                gen.manual_seed(9999)
                with torch.no_grad():
                    for i in range(2):
                        batch = batcher.sample_batch(
                            batch_size=4,
                            seq_len=seq_len,
                            generator=gen,
                            device=dev,
                            split="val",
                        )
                        if batch is None:
                            continue
                        logits = model(batch)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        loss = torch.nn.functional.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            batch[:, 1:].reshape(-1),
                        )
                        if torch.isfinite(loss):
                            losses.append(loss.item())
                if losses:
                    val_loss = sum(losses) / len(losses)
                    updates["validation_loss"] = val_loss
                    if initial_loss:
                        val_ratio = val_loss / max(float(initial_loss), 1e-6)
                        updates["validation_loss_ratio"] = val_ratio
                    else:
                        updates["validation_loss_ratio"] = None
                        updates["validation_loss_ratio_note"] = "initial_loss_missing"
                    final_loss = program.get("final_loss")
                    if final_loss:
                        updates["generalization_gap"] = val_loss - float(final_loss)
        except Exception as e:
            updates["validation_loss_error"] = str(e)

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    if updates:
        db_updates = {k: v for k, v in updates.items() if not k.endswith("_error")}
        if db_updates:
            set_parts = [f"{k} = ?" for k in db_updates]
            vals = list(db_updates.values()) + [result_id]
            nb.conn.execute(
                f"UPDATE program_results SET {', '.join(set_parts)} WHERE result_id = ?",
                vals,
            )
            lb_cols = {
                c[1]
                for c in nb.conn.execute("PRAGMA table_info(leaderboard)").fetchall()
            }
            lb_updates = {k: v for k, v in db_updates.items() if k in lb_cols}
            if lb_updates:
                lb_set = [f"{k} = ?" for k in lb_updates]
                lb_vals = list(lb_updates.values()) + [result_id]
                nb.conn.execute(
                    f"UPDATE leaderboard SET {', '.join(lb_set)} WHERE result_id = ?",
                    lb_vals,
                )
            nb.conn.commit()

    return jsonify({"status": "ok", "result_id": result_id, "updates": updates})


def _api_program_rescreen(notebook_path: str, result_id, nb=None):
    from research.tools.exact_graph_replay import start_exact_replay_async

    program = nb.get_program_detail(result_id)
    if program is None:
        program = _leaderboard_backed_program_detail(nb, result_id)
    if program is None:
        return jsonify({"error": "Program not found"}), 404
    if not program.get("graph_json"):
        return jsonify({"error": "No graph_json for this program"}), 400

    body = request.get_json(silent=True) or {}
    device = str(body.get("device") or "cuda").strip().lower()
    if device not in {"cpu", "cuda"}:
        return jsonify({"error": "device must be 'cpu' or 'cuda'"}), 400
    repeat_per_source = int(body.get("repeat_per_source") or 1)
    repeat_per_source = max(1, min(repeat_per_source, 8))
    fast = bool(body.get("fast", True))
    hypothesis = (
        body.get("hypothesis") or f"UI-triggered exact replay for {result_id[:8]}"
    )

    try:
        exp_id = start_exact_replay_async(
            db_path=Path(notebook_path),
            result_ids=[result_id],
            repeat_per_source=repeat_per_source,
            device=device,
            fast=fast,
            verbose=False,
            hypothesis=str(hypothesis),
        )
    except Exception as exc:
        logger.exception("Failed to start rescreen for %s", result_id)
        return jsonify({"error": f"Failed to start screening replay: {exc}"}), 500

    return jsonify(
        {
            "status": "started",
            "mode": "exact_graph_replay",
            "experiment_id": exp_id,
            "result_id": result_id,
            "repeat_per_source": repeat_per_source,
            "device": device,
            "fast": fast,
        }
    )


# Stage budget defaults match research/defaults.py.  Each "queue X
# rerun" button uses its tier's natural budget so the new sample is
# in the same regime as the existing tier rows.
_STAGE_DEFAULT_STEPS = {
    "screening": 750,  # STAGE1_STEPS
    "investigation": 2500,  # INVESTIGATION_STEPS
    "validation": 10000,  # VALIDATION_STEPS
}
_STAGE_QUEUE_NAMES = {
    "screening": "replay",  # S1 reruns go through the exact_graph_replay path
    "investigation": "investigation",
    "validation": "validation",
}


def _api_program_queue_validation_rerun(result_id, nb=None):
    """Queue N reruns at a chosen stage for a program.

    Each rerun is a row in ``followup_tasks``; the runner claims them
    sequentially and re-runs the stage's pipeline (S1 screening replay
    via exact_graph_replay, investigation via start_investigation, or
    validation via start_validation).  Each completed rerun produces a
    new ``program_results`` row; the leaderboard aggregator means the
    metrics across rows of the same fingerprint+tier.

    Body (optional):
        stage  str   "screening" | "investigation" | "validation"
                     (default "validation").
        n      int   number of reruns (default 1, max 5).
        n_seeds int  seeds per rerun (default 1; only used at validation).
        n_steps int  step budget per rerun (default depends on stage).
        reason str   free-text shown in evidence_pack.
    """
    program = nb.get_program_detail(result_id)
    if program is None:
        program = _leaderboard_backed_program_detail(nb, result_id)
    if program is None:
        return jsonify({"error": "Program not found"}), 404

    body = request.get_json(silent=True) or {}
    stage_in = str(body.get("stage") or "validation").strip().lower()
    if stage_in not in _STAGE_DEFAULT_STEPS:
        return (
            jsonify({"error": f"stage must be one of {sorted(_STAGE_DEFAULT_STEPS)}"}),
            400,
        )
    try:
        n_req = int(body.get("n", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "n must be an integer"}), 400
    if n_req < 1 or n_req > 5:
        return jsonify({"error": "n must be in [1, 5]"}), 400
    reason = str(body.get("reason") or "").strip()[:500]
    try:
        n_seeds = max(1, min(5, int(body.get("n_seeds", 1))))
    except (TypeError, ValueError):
        n_seeds = 1
    try:
        n_steps = int(body.get("n_steps", _STAGE_DEFAULT_STEPS[stage_in]))
    except (TypeError, ValueError):
        n_steps = _STAGE_DEFAULT_STEPS[stage_in]
    n_steps = max(50, min(50000, n_steps))

    fp = (program.get("graph_fingerprint") or "").strip() or None
    queue_stage = _STAGE_QUEUE_NAMES[stage_in]

    if queue_stage == "replay":
        # S1 screening replay path: exact_graph_replay reads
        # ``repeat_per_source``, ``device``, ``fast`` from config_json.
        # CRITICAL: fast=False for stability reruns.  fast=True triggers
        # _apply_fast_replay_budget, which clamps stage1_steps to 80
        # regardless of the user's request — at that budget most archs
        # fail S0/S05 gates and produce zero new rows.  We want the full
        # user-specified budget so the rerun is comparable to the
        # original sample.
        config_payload = {
            "repeat_per_source": 1,
            "device": "cuda",
            "fast": False,
            "stage1_steps": n_steps,
        }
    else:
        config = RunConfig()
        config.gbm_prescreener_enabled = False
        config.allow_unproven_ml_influence = False
        if queue_stage == "investigation":
            config.investigation_steps = n_steps
        else:  # validation
            config.validation_n_seeds = n_seeds
            config.validation_steps = n_steps
        config_payload = config.to_dict()

    task_ids: list[str] = []
    for i in range(n_req):
        tid = nb.enqueue_followup_task(
            stage=queue_stage,
            result_ids=[str(result_id)],
            hypothesis=(
                f"User-triggered {stage_in} rerun: add a sample at "
                f"{n_steps}-step budget for mean/CV aggregation within "
                f"the {stage_in} pool."
            ),
            config=config_payload,
            evidence_pack={
                "reason": reason,
                "rerun_index": i + 1,
                "rerun_total": n_req,
                "stage": stage_in,
                "n_steps": n_steps,
                "n_seeds": n_seeds,
                "fingerprint": fp,
            },
            source_context="program_detail_rerun",
            priority_score=float(program.get("composite_score") or 0.0),
            priority_reasons={
                "policy": "user_triggered_program_detail",
                "stage": stage_in,
                "n_steps": n_steps,
                "reason": reason or None,
                "fingerprint": fp,
            },
            metadata={
                "source": "ui_program_detail",
                "stage": stage_in,
                "n_steps": n_steps,
                "n_seeds": n_seeds,
                "rerun_index": i + 1,
                "rerun_total": n_req,
            },
            bypass_dedup=True,
        )
        if tid:
            task_ids.append(tid)
    return jsonify(
        {
            "status": "queued",
            "result_id": str(result_id),
            "graph_fingerprint": fp,
            "stage": stage_in,
            "n_steps": n_steps,
            "n_seeds": n_seeds,
            "n_requested": n_req,
            "task_ids": task_ids,
            "queued_count": len(task_ids),
        }
    )


def _api_program_pending_reruns(result_id, nb=None):
    """List queued/running reruns for a program across all stages.

    Filters ``followup_tasks`` for stage in (replay, investigation,
    validation) — i.e. the three stages exposed by the rerun panel.
    Returns the most recent 50 with status, queued time, source
    context, and the stage label inferred from evidence_pack.
    """
    rows = nb.conn.execute("""SELECT task_id, stage, status, source_context,
                  result_ids_json, timestamp,
                  started_timestamp, completed_timestamp,
                  outcome, priority_score, evidence_pack_json
           FROM followup_tasks
           WHERE stage IN ('replay', 'investigation', 'validation')
             AND status IN ('queued','running')
           ORDER BY timestamp DESC
           LIMIT 300""").fetchall()
    rid = str(result_id)
    out: list[Dict[str, Any]] = []
    for r in rows:
        try:
            ids = json.loads(r["result_ids_json"] or "[]") or []
        except (json.JSONDecodeError, TypeError):
            ids = []
        if rid not in [str(x) for x in ids]:
            continue
        try:
            evidence = json.loads(r["evidence_pack_json"] or "{}") or {}
        except (json.JSONDecodeError, TypeError):
            evidence = {}
        # Map runner-stage back to user-facing label: 'replay' = S1
        # screening rerun.
        runner_stage = r["stage"]
        ui_stage = evidence.get("stage") or (
            "screening" if runner_stage == "replay" else runner_stage
        )
        out.append(
            {
                "task_id": r["task_id"],
                "status": r["status"],
                "stage": ui_stage,
                "runner_stage": runner_stage,
                "n_steps": evidence.get("n_steps"),
                "n_seeds": evidence.get("n_seeds"),
                "source_context": r["source_context"],
                "queued_at": r["timestamp"],
                "started_at": r["started_timestamp"],
                "completed_at": r["completed_timestamp"],
                "outcome": r["outcome"],
                "priority_score": r["priority_score"],
                "rerun_index": evidence.get("rerun_index"),
                "rerun_total": evidence.get("rerun_total"),
                "reason": evidence.get("reason"),
            }
        )
        if len(out) >= 50:
            break
    return jsonify({"result_id": rid, "tasks": out})


def _api_drain_pending_validation_rerun(notebook_path: str, nb=None):
    """Pop one queued rerun (any stage) and start it now.

    Stage priority: replay (S1) > investigation > validation.  Mirrors
    what continuous mode does on each cycle tick.  Refuses if an
    experiment is already running.

    Returns the runner-stage that was launched and the task_id, or
    ``idle`` if all queues are empty / ``no_op`` if the runner refused.
    """
    runner = get_runner(notebook_path)
    if runner.is_running:
        running_id = getattr(runner, "current_experiment_id", None)
        return (
            jsonify(
                {
                    "status": "busy",
                    "running_experiment_id": running_id,
                    "message": "An experiment is already running; queue will drain when it finishes.",
                }
            ),
            409,
        )

    drain_stages = (
        ("replay", runner._run_pending_replay),
        ("investigation", runner._run_pending_investigation),
        ("validation", runner._run_pending_validation),
    )
    for stage_name, drain_fn in drain_stages:
        pre = {
            row["task_id"]
            for row in nb.conn.execute(
                "SELECT task_id FROM followup_tasks WHERE stage = ? AND status='queued'",
                (stage_name,),
            ).fetchall()
        }
        if not pre:
            continue
        try:
            drain_fn()
        except Exception as exc:
            logger.exception("Failed to drain pending %s rerun", stage_name)
            return jsonify({"error": f"drain failed: {exc}"}), 500
        post = {
            row["task_id"]
            for row in nb.conn.execute(
                "SELECT task_id FROM followup_tasks WHERE stage = ? AND status='queued'",
                (stage_name,),
            ).fetchall()
        }
        launched = list(pre - post)
        if launched:
            return jsonify(
                {
                    "status": "launched",
                    "stage": stage_name,
                    "task_ids": launched,
                    "running_experiment_id": getattr(
                        runner, "current_experiment_id", None
                    ),
                }
            )
    return jsonify({"status": "idle", "message": "no queued rerun tasks"})


def _api_program_cancel_rerun(result_id, task_id, nb=None):
    """Cancel a queued validation rerun task.

    Refuses to cancel if the task is already running — at that point
    the runner owns it.
    """
    row = nb.conn.execute(
        "SELECT status, result_ids_json FROM followup_tasks WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return jsonify({"error": "task not found"}), 404
    try:
        ids = json.loads(row["result_ids_json"] or "[]") or []
    except (json.JSONDecodeError, TypeError):
        ids = []
    if str(result_id) not in [str(x) for x in ids]:
        return jsonify({"error": "task does not belong to this program"}), 400
    if row["status"] != "queued":
        return (
            jsonify({"error": f"cannot cancel task in status {row['status']!r}"}),
            409,
        )
    nb.conn.execute(
        """UPDATE followup_tasks
              SET status = 'cancelled',
                  completed_timestamp = ?,
                  outcome = 'user_cancelled'
            WHERE task_id = ? AND status = 'queued'""",
        (time.time(), task_id),
    )
    nb._maybe_commit()
    return jsonify({"status": "cancelled", "task_id": task_id})


def _api_program_promote_screening(result_id, nb=None):
    program = nb.get_program_detail(result_id)
    if program is None:
        program = _leaderboard_backed_program_detail(nb, result_id)
    if program is None:
        return jsonify({"error": "Program not found"}), 404

    entry = nb.get_leaderboard_entry(result_id)
    # Fingerprint-level dedup: if an entry already exists for this fingerprint
    # under a different result_id, route the promotion to that entry instead
    # of creating a duplicate leaderboard row.
    if entry is None:
        fp = str(program.get("graph_fingerprint") or "").strip()
        if fp:
            sibling_entry = nb.get_leaderboard_entry_by_fingerprint(fp)
            if sibling_entry and sibling_entry.get("result_id") != result_id:
                entry = sibling_entry
                result_id = sibling_entry.get("result_id")
                program = nb.get_program_detail(result_id) or program
    trust_label = _preserve_stronger_label(
        program.get("trust_label"),
        entry.get("trust_label") if entry else None,
        ranks=_TRUST_LABEL_RANK,
        fallback="candidate_screening",
    )
    comparability_label = _preserve_stronger_label(
        program.get("comparability_label"),
        entry.get("comparability_label") if entry else None,
        ranks=_COMPARABILITY_LABEL_RANK,
        fallback="screening_only",
    )
    if not entry:
        entry_id = nb.upsert_leaderboard(
            result_id=result_id,
            model_source=program.get("model_source") or "manual_screening_promotion",
            architecture_desc=str(program.get("graph_fingerprint") or "")[:40],
            screening_loss_ratio=program.get("loss_ratio"),
            screening_novelty=program.get("novelty_score"),
            screening_passed=bool(program.get("stage1_passed")),
            tier="screening",
            trust_label=trust_label,
            comparability_label=comparability_label,
            notes="Manual screening promotion from Discoveries",
        )
        entry = nb.get_leaderboard_entry(result_id) or {"entry_id": entry_id}
    else:
        existing_notes = str(entry.get("notes") or "").strip()
        note_prefix = "Manual screening promotion from Discoveries"
        note_value = (
            existing_notes
            if note_prefix in existing_notes
            else (
                f"{existing_notes}\n{note_prefix}".strip()
                if existing_notes
                else note_prefix
            )
        )
        nb.promote_to_tier(
            entry["entry_id"],
            "screening",
            trust_label=trust_label,
            comparability_label=comparability_label,
            screening_passed=bool(program.get("stage1_passed")),
            screening_loss_ratio=program.get("loss_ratio"),
            screening_novelty=program.get("novelty_score"),
            notes=note_value,
        )

    nb.conn.execute(
        """
        UPDATE program_results
        SET trust_label = ?, comparability_label = ?, timestamp = ?
        WHERE result_id = ?
        """,
        (trust_label, comparability_label, time.time(), result_id),
    )
    nb.conn.commit()

    updated = nb.get_leaderboard_entry(result_id)
    return jsonify(
        {
            "status": "ok",
            "result_id": result_id,
            "entry": updated or entry,
        }
    )


def _api_programs(nb=None):
    n = request.args.get("n", 20, type=int)
    sort_by = request.args.get("sort", "novelty_score")
    from ..analytics import ExperimentAnalytics

    analytics = ExperimentAnalytics(nb)
    programs = nb.get_top_programs(n, sort_by)
    annotate_qkv_usage(programs, analytics)
    return jsonify(json_safe(programs))


def _api_training_curve(result_id, nb=None):
    curve = nb.get_training_curve(result_id)
    return jsonify(curve)


def _api_program_causal_evidence(result_id, nb=None):
    requested_result_id = str(result_id or "").strip()
    canonical_result_id = nb.resolve_canonical_result_id(requested_result_id)
    result_id = canonical_result_id or requested_result_id
    rows = nb.get_causal_rule_evidence(result_id=result_id, limit=50)
    for item in rows:
        evidence_id = item.get("evidence_id")
        if evidence_id:
            item["child_observations"] = nb.get_causal_ablation_child_observations(
                evidence_id=evidence_id,
                limit=200,
            )
    return jsonify(json_safe({"result_id": result_id, "evidence": rows}))


def _api_program_causal_ablation(notebook_path: str, result_id, nb=None):
    requested_result_id = str(result_id or "").strip()
    canonical_result_id = nb.resolve_canonical_result_id(requested_result_id)
    result_id = canonical_result_id or requested_result_id
    if nb.get_program_detail(result_id) is None:
        return jsonify({"error": "Program not found"}), 404
    body = request.get_json(silent=True) or {}
    config = RunConfig.from_dict(body if isinstance(body, dict) else {})
    config.enable_causal_ablation = True
    config.causal_ablation_top_k = max(
        1, int(body.get("top_k", body.get("causal_ablation_top_k", 1)) or 1)
    )
    config.causal_ablation_max_signals = max(
        1,
        int(body.get("max_signals", body.get("causal_ablation_max_signals", 2)) or 2),
    )
    config.causal_ablation_max_graphs = max(
        1, int(body.get("max_graphs", body.get("causal_ablation_max_graphs", 4)) or 4)
    )
    runner = get_runner(notebook_path, start_projector=True)
    try:
        run_id = runner.start_causal_ablation(result_id, config)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"status": "started", "run_id": run_id, "result_id": result_id})


def _api_bulk_causal_ablation_start(notebook_path: str, nb=None):
    body = request.get_json(silent=True) or {}
    config = RunConfig.from_dict(body if isinstance(body, dict) else {})
    config.continuous = True
    config.enable_causal_ablation = True
    config.causal_ablation_interval = max(
        1, int(body.get("interval", body.get("causal_ablation_interval", 3)) or 3)
    )
    config.causal_ablation_top_k = max(
        1, int(body.get("top_k", body.get("causal_ablation_top_k", 1)) or 1)
    )
    config.causal_ablation_max_signals = max(
        1,
        int(body.get("max_signals", body.get("causal_ablation_max_signals", 2)) or 2),
    )
    config.causal_ablation_max_graphs = max(
        1, int(body.get("max_graphs", body.get("causal_ablation_max_graphs", 4)) or 4)
    )
    if config.max_experiments <= 0:
        config.max_experiments = max(
            1, int(body.get("max_experiments", body.get("n_cycles", 5)) or 5)
        )
    if config.n_programs <= 0:
        config.n_programs = 40
    runner = get_runner(notebook_path, start_projector=True)
    try:
        run_id = runner.start_continuous(config)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify(
        {
            "status": "started",
            "run_id": run_id,
            "mode": "continuous",
            "causal_ablation": {
                "interval": config.causal_ablation_interval,
                "top_k": config.causal_ablation_top_k,
                "max_signals": config.causal_ablation_max_signals,
                "max_graphs": config.causal_ablation_max_graphs,
            },
        }
    )


def _api_causal_ablation_summary(nb=None):
    limit = request.args.get("limit", 50, type=int)
    rows = nb.get_causal_component_interaction_summary(limit=limit)
    evidence_total = nb.conn.execute(
        "SELECT COUNT(*) AS n FROM causal_rule_evidence"
    ).fetchone()
    observation_total = nb.conn.execute(
        "SELECT COUNT(*) AS n FROM causal_ablation_child_observations"
    ).fetchone()
    outcome_rows = nb.conn.execute(
        """
        SELECT outcome, COUNT(*) AS n, AVG(confidence) AS avg_confidence,
               AVG(effect_size) AS avg_effect_size
        FROM causal_rule_evidence
        GROUP BY outcome
        ORDER BY n DESC
        """
    ).fetchall()
    source_rows = nb.conn.execute(
        """
        SELECT source, COUNT(*) AS n,
               SUM(CASE WHEN COALESCE(stage1_passed, 0) = 1 THEN 1 ELSE 0 END)
                   AS stage1_count,
               AVG(loss_ratio) AS avg_loss_ratio
        FROM causal_ablation_child_observations
        GROUP BY source
        ORDER BY n DESC
        """
    ).fetchall()
    backfill_gap = nb.conn.execute(
        """
        SELECT COUNT(*) AS total_ablation_rows,
               SUM(CASE WHEN COALESCE(pr.stage1_passed, 0) = 1 THEN 1 ELSE 0 END)
                   AS s1_ablation_rows,
               SUM(CASE WHEN COALESCE(pr.stage1_passed, 0) = 1
                         AND (
                            pr.hellaswag_acc IS NULL
                            OR pr.blimp_overall_accuracy IS NULL
                            OR pr.induction_auc IS NULL
                            OR pr.binding_auc IS NULL
                            OR pr.binding_composite IS NULL
                            OR pr.ar_auc IS NULL
                            OR pr.wikitext_perplexity IS NULL
                            OR pr.wikitext_score IS NULL
                            OR pr.fp_jacobian_erf_density IS NULL
                            OR pr.fp_icld_delta_loss IS NULL
                            OR pr.fp_logit_margin_delta IS NULL
                         )
                        THEN 1 ELSE 0 END) AS s1_missing_core_metrics
        FROM program_results pr
        JOIN experiments e ON e.experiment_id = pr.experiment_id
        WHERE e.experiment_type = 'ablation'
        """
    ).fetchone()
    recent_24h = nb.conn.execute(
        """
        SELECT COUNT(*) AS evidence_count,
               COUNT(DISTINCT ablation_experiment_id) AS experiments,
               SUM(CASE WHEN outcome = 'supported' THEN 1 ELSE 0 END)
                   AS supported_count,
               SUM(CASE WHEN outcome LIKE 'refuted%' THEN 1 ELSE 0 END)
                   AS refuted_count
        FROM causal_rule_evidence
        WHERE timestamp >= ?
        """,
        (time.time() - 86400,),
    ).fetchone()
    totals = {
        "evidence_count": int(evidence_total["n"] or 0) if evidence_total else 0,
        "observation_count": (
            int(observation_total["n"] or 0) if observation_total else 0
        ),
        "recent_24h": dict(recent_24h) if recent_24h else {},
        "outcomes": [dict(row) for row in outcome_rows],
        "sources": [dict(row) for row in source_rows],
        "backfill_gap": dict(backfill_gap) if backfill_gap else {},
    }
    return jsonify(json_safe({"summary": rows, "totals": totals}))


def _api_causal_ablation_champions(nb=None):
    """Per-champion ablation rollup: how many children, support/refute counts,
    per-metric mean Δ, and metric coverage. Powers the 'By Champion' tab.
    """
    limit = max(1, min(int(request.args.get("limit", 50, type=int) or 50), 500))
    rows = nb.conn.execute(
        """
        WITH children AS (
            SELECT obs.parent_result_id,
                   obs.parent_fingerprint,
                   COUNT(*) AS evidence_count,
                   COUNT(DISTINCT obs.child_fingerprint) AS child_fingerprint_count,
                   SUM(CASE WHEN cp.stage1_passed = 1 THEN 1 ELSE 0 END)
                       AS s1_pass_count,
                   SUM(CASE WHEN
                       cp.hellaswag_acc IS NOT NULL
                       AND cp.blimp_overall_accuracy IS NOT NULL
                       AND cp.induction_auc IS NOT NULL
                       AND cp.binding_auc IS NOT NULL
                       AND cp.binding_composite IS NOT NULL
                       AND cp.ar_auc IS NOT NULL
                       AND cp.wikitext_perplexity IS NOT NULL
                       THEN 1 ELSE 0 END) AS metric_complete_count,
                   AVG(CASE WHEN cp.loss_ratio IS NOT NULL AND pp.loss_ratio IS NOT NULL
                            THEN cp.loss_ratio - pp.loss_ratio END)
                       AS avg_loss_delta,
                   AVG(CASE WHEN cp.induction_auc IS NOT NULL AND pp.induction_auc IS NOT NULL
                            THEN pp.induction_auc - cp.induction_auc END)
                       AS avg_induction_drop,
                   AVG(CASE WHEN cp.binding_composite IS NOT NULL AND pp.binding_composite IS NOT NULL
                            THEN pp.binding_composite - cp.binding_composite END)
                       AS avg_binding_drop,
                   AVG(CASE WHEN cp.ar_auc IS NOT NULL AND pp.ar_auc IS NOT NULL
                            THEN pp.ar_auc - cp.ar_auc END)
                       AS avg_ar_drop,
                   AVG(CASE WHEN cp.hellaswag_acc IS NOT NULL AND pp.hellaswag_acc IS NOT NULL
                            THEN pp.hellaswag_acc - cp.hellaswag_acc END)
                       AS avg_hellaswag_drop,
                   AVG(CASE WHEN cp.blimp_overall_accuracy IS NOT NULL
                                 AND pp.blimp_overall_accuracy IS NOT NULL
                            THEN pp.blimp_overall_accuracy - cp.blimp_overall_accuracy END)
                       AS avg_blimp_drop,
                   AVG(CASE WHEN cp.wikitext_perplexity IS NOT NULL
                                 AND pp.wikitext_perplexity IS NOT NULL
                                 AND pp.wikitext_perplexity > 0
                            THEN (cp.wikitext_perplexity - pp.wikitext_perplexity)
                                 / pp.wikitext_perplexity END)
                       AS avg_ppl_pct_change
            FROM causal_ablation_child_observations obs
            LEFT JOIN program_results cp ON cp.result_id = obs.child_result_id
            LEFT JOIN program_results pp ON pp.result_id = obs.parent_result_id
            GROUP BY obs.parent_result_id, obs.parent_fingerprint
        ),
        rules AS (
            SELECT parent_result_id,
                   COUNT(*) AS rule_count,
                   SUM(CASE WHEN outcome = 'supported' THEN 1 ELSE 0 END)
                       AS supported_count,
                   SUM(CASE WHEN outcome LIKE 'refuted%' THEN 1 ELSE 0 END)
                       AS refuted_count
            FROM causal_rule_evidence
            GROUP BY parent_result_id
        )
        SELECT children.parent_result_id   AS result_id,
               children.parent_fingerprint AS graph_fingerprint,
               children.evidence_count,
               children.child_fingerprint_count,
               children.s1_pass_count,
               children.metric_complete_count,
               CASE WHEN children.evidence_count > 0
                    THEN CAST(children.metric_complete_count AS REAL)
                         / children.evidence_count
                    ELSE 0.0 END AS metric_complete_rate,
               COALESCE(rules.rule_count, 0) AS rule_count,
               COALESCE(rules.supported_count, 0) AS supported_count,
               COALESCE(rules.refuted_count, 0) AS refuted_count,
               children.avg_loss_delta,
               children.avg_induction_drop,
               children.avg_binding_drop,
               children.avg_ar_drop,
               children.avg_hellaswag_drop,
               children.avg_blimp_drop,
               children.avg_ppl_pct_change,
               l.composite_score,
               l.tier,
               pp.loss_ratio AS parent_loss_ratio,
               pp.wikitext_perplexity AS parent_wikitext_perplexity,
               pp.induction_auc AS parent_induction_auc,
               pp.binding_composite AS parent_binding_composite,
               pp.hellaswag_acc AS parent_hellaswag_acc,
               pp.blimp_overall_accuracy AS parent_blimp,
               pp.ar_auc AS parent_ar_auc
        FROM children
        LEFT JOIN rules ON rules.parent_result_id = children.parent_result_id
        LEFT JOIN program_results pp ON pp.result_id = children.parent_result_id
        LEFT JOIN leaderboard l ON l.result_id = children.parent_result_id
        ORDER BY children.evidence_count DESC, children.child_fingerprint_count DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return jsonify(json_safe({"champions": [dict(r) for r in rows]}))


def _api_causal_ablation_components(nb=None):
    """Per-component (op / op_pair / motif) summary across all champions.
    Surfaces what an op does in different contexts. Powers 'By Component'.
    """
    limit = max(1, min(int(request.args.get("limit", 200, type=int) or 200), 1000))
    rule_type = request.args.get("rule_type", "")
    where = ""
    params: list = []
    if rule_type:
        where = "WHERE rule_type = ?"
        params.append(rule_type)
    rows = nb.conn.execute(
        f"""
        WITH metric_rows AS (
            SELECT obs.rule_type,
                   obs.rule_key,
                   obs.parent_result_id,
                   CASE WHEN cp.loss_ratio IS NOT NULL AND pp.loss_ratio IS NOT NULL
                        THEN cp.loss_ratio - pp.loss_ratio END AS d_loss,
                   CASE WHEN cp.induction_auc IS NOT NULL AND pp.induction_auc IS NOT NULL
                        THEN pp.induction_auc - cp.induction_auc END AS d_induction,
                   CASE WHEN cp.binding_composite IS NOT NULL AND pp.binding_composite IS NOT NULL
                        THEN pp.binding_composite - cp.binding_composite END AS d_binding,
                   CASE WHEN cp.ar_auc IS NOT NULL AND pp.ar_auc IS NOT NULL
                        THEN pp.ar_auc - cp.ar_auc END AS d_ar,
                   CASE WHEN cp.hellaswag_acc IS NOT NULL AND pp.hellaswag_acc IS NOT NULL
                        THEN pp.hellaswag_acc - cp.hellaswag_acc END AS d_hellaswag,
                   CASE WHEN cp.blimp_overall_accuracy IS NOT NULL
                             AND pp.blimp_overall_accuracy IS NOT NULL
                        THEN pp.blimp_overall_accuracy - cp.blimp_overall_accuracy END AS d_blimp,
                   CASE WHEN cp.wikitext_perplexity IS NOT NULL
                             AND pp.wikitext_perplexity IS NOT NULL
                             AND pp.wikitext_perplexity > 0
                        THEN (cp.wikitext_perplexity - pp.wikitext_perplexity)
                             / pp.wikitext_perplexity END AS d_ppl_pct
            FROM causal_ablation_child_observations obs
            LEFT JOIN program_results cp ON cp.result_id = obs.child_result_id
            LEFT JOIN program_results pp ON pp.result_id = obs.parent_result_id
        )
        SELECT rule_type,
               rule_key,
               COUNT(*) AS observation_count,
               COUNT(DISTINCT parent_result_id) AS parent_count,
               AVG(d_loss) AS avg_d_loss,
               AVG(d_induction) AS avg_d_induction,
               AVG(d_binding) AS avg_d_binding,
               AVG(d_ar) AS avg_d_ar,
               AVG(d_hellaswag) AS avg_d_hellaswag,
               AVG(d_blimp) AS avg_d_blimp,
               AVG(d_ppl_pct) AS avg_d_ppl_pct,
               SUM(CASE WHEN d_loss IS NOT NULL THEN 1 ELSE 0 END) AS n_loss,
               SUM(CASE WHEN d_induction IS NOT NULL THEN 1 ELSE 0 END) AS n_induction,
               SUM(CASE WHEN d_binding IS NOT NULL THEN 1 ELSE 0 END) AS n_binding,
               SUM(CASE WHEN d_ar IS NOT NULL THEN 1 ELSE 0 END) AS n_ar,
               SUM(CASE WHEN d_hellaswag IS NOT NULL THEN 1 ELSE 0 END) AS n_hellaswag,
               SUM(CASE WHEN d_blimp IS NOT NULL THEN 1 ELSE 0 END) AS n_blimp,
               SUM(CASE WHEN d_ppl_pct IS NOT NULL THEN 1 ELSE 0 END) AS n_ppl
        FROM metric_rows
        {where}
        GROUP BY rule_type, rule_key
        ORDER BY observation_count DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return jsonify(json_safe({"components": [dict(r) for r in rows]}))


def _api_causal_ablation_recommendations(nb=None):
    """Construction recommendations: ✓ USE / ✗ AVOID / ⚠ MIXED rules with
    n, contexts, and per-metric average impact. The 'do this' surface.
    """
    limit = max(1, min(int(request.args.get("limit", 80, type=int) or 80), 400))
    min_n = max(2, int(request.args.get("min_n", 4, type=int) or 4))
    rows = nb.conn.execute(
        f"""
        WITH metric_rows AS (
            SELECT obs.rule_type,
                   obs.rule_key,
                   obs.parent_result_id,
                   CASE WHEN cp.loss_ratio IS NOT NULL AND pp.loss_ratio IS NOT NULL
                        THEN cp.loss_ratio - pp.loss_ratio END AS d_loss,
                   CASE WHEN cp.induction_auc IS NOT NULL AND pp.induction_auc IS NOT NULL
                        THEN pp.induction_auc - cp.induction_auc END AS d_induction,
                   CASE WHEN cp.binding_composite IS NOT NULL AND pp.binding_composite IS NOT NULL
                        THEN pp.binding_composite - cp.binding_composite END AS d_binding,
                   CASE WHEN cp.ar_auc IS NOT NULL AND pp.ar_auc IS NOT NULL
                        THEN pp.ar_auc - cp.ar_auc END AS d_ar,
                   CASE WHEN cp.hellaswag_acc IS NOT NULL AND pp.hellaswag_acc IS NOT NULL
                        THEN pp.hellaswag_acc - cp.hellaswag_acc END AS d_hellaswag,
                   CASE WHEN cp.blimp_overall_accuracy IS NOT NULL
                             AND pp.blimp_overall_accuracy IS NOT NULL
                        THEN pp.blimp_overall_accuracy - cp.blimp_overall_accuracy END AS d_blimp,
                   CASE WHEN cp.wikitext_perplexity IS NOT NULL
                             AND pp.wikitext_perplexity IS NOT NULL
                             AND pp.wikitext_perplexity > 0
                        THEN (cp.wikitext_perplexity - pp.wikitext_perplexity)
                             / pp.wikitext_perplexity END AS d_ppl_pct,
                   CASE WHEN cp.hellaswag_acc IS NOT NULL
                             AND cp.blimp_overall_accuracy IS NOT NULL
                             AND cp.induction_auc IS NOT NULL
                             AND cp.binding_composite IS NOT NULL
                             AND cp.ar_auc IS NOT NULL
                             AND cp.wikitext_perplexity IS NOT NULL
                        THEN 1 ELSE 0 END AS metric_complete
            FROM causal_ablation_child_observations obs
            LEFT JOIN program_results cp ON cp.result_id = obs.child_result_id
            LEFT JOIN program_results pp ON pp.result_id = obs.parent_result_id
        ),
        agg AS (
            SELECT rule_type,
                   rule_key,
                   COUNT(*) AS n,
                   COUNT(DISTINCT parent_result_id) AS contexts,
                   SUM(metric_complete) AS metric_complete_count,
                   AVG(d_loss) AS avg_d_loss,
                   AVG(d_induction) AS avg_d_induction,
                   AVG(d_binding) AS avg_d_binding,
                   AVG(d_ar) AS avg_d_ar,
                   AVG(d_hellaswag) AS avg_d_hellaswag,
                   AVG(d_blimp) AS avg_d_blimp,
                   AVG(d_ppl_pct) AS avg_d_ppl_pct,
                   SUM(CASE WHEN d_induction IS NOT NULL THEN 1 ELSE 0 END) AS n_induction,
                   SUM(CASE WHEN d_binding IS NOT NULL THEN 1 ELSE 0 END) AS n_binding,
                   SUM(CASE WHEN d_blimp IS NOT NULL THEN 1 ELSE 0 END) AS n_blimp,
                   SUM(CASE WHEN d_hellaswag IS NOT NULL THEN 1 ELSE 0 END) AS n_hellaswag,
                   SUM(CASE WHEN d_ar IS NOT NULL THEN 1 ELSE 0 END) AS n_ar,
                   SUM(CASE WHEN d_ppl_pct IS NOT NULL THEN 1 ELSE 0 END) AS n_ppl
            FROM metric_rows
            GROUP BY rule_type, rule_key
            HAVING COUNT(*) >= {int(min_n)} AND SUM(metric_complete) >= 3
        )
        SELECT * FROM agg
        ORDER BY n DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return jsonify(json_safe({"recommendations": [dict(r) for r in rows]}))


def _api_causal_ablation_children_for_rule(nb=None):
    """Drill-down: child observations for a given rule_type/rule_key with
    full per-metric numbers. Used by the rule detail drawer."""
    rule_type = request.args.get("rule_type", "")
    rule_key = request.args.get("rule_key", "")
    parent_result_id = request.args.get("parent_result_id", "")
    if not rule_type or not rule_key:
        return jsonify({"error": "rule_type and rule_key required"}), 400
    limit = max(1, min(int(request.args.get("limit", 100, type=int) or 100), 500))
    where = "obs.rule_type = ? AND obs.rule_key = ?"
    params: list = [rule_type, rule_key]
    if parent_result_id:
        where += " AND obs.parent_result_id = ?"
        params.append(parent_result_id)
    rows = nb.conn.execute(
        f"""
        SELECT obs.parent_result_id,
               obs.parent_fingerprint,
               obs.child_result_id,
               obs.child_fingerprint,
               obs.source,
               cp.loss_ratio AS child_loss_ratio,
               cp.wikitext_perplexity AS child_ppl,
               cp.hellaswag_acc AS child_hellaswag,
               cp.blimp_overall_accuracy AS child_blimp,
               cp.induction_auc AS child_induction,
               cp.binding_composite AS child_binding,
               cp.ar_auc AS child_ar,
               cp.fp_jacobian_erf_density AS child_erf_density,
               cp.fp_icld_delta_loss AS child_icld_delta,
               cp.trust_label AS child_trust_label,
               cp.comparability_label AS child_comparability_label,
               pp.loss_ratio AS parent_loss_ratio,
               pp.wikitext_perplexity AS parent_ppl,
               pp.hellaswag_acc AS parent_hellaswag,
               pp.blimp_overall_accuracy AS parent_blimp,
               pp.induction_auc AS parent_induction,
               pp.binding_composite AS parent_binding,
               pp.ar_auc AS parent_ar,
               pp.fp_jacobian_erf_density AS parent_erf_density
        FROM causal_ablation_child_observations obs
        LEFT JOIN program_results cp ON cp.result_id = obs.child_result_id
        LEFT JOIN program_results pp ON pp.result_id = obs.parent_result_id
        WHERE {where}
        ORDER BY obs.timestamp DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return jsonify(json_safe({"children": [dict(r) for r in rows]}))


def _api_construction_prior_active(nb=None):
    """Return the currently active construction prior snapshot."""
    from research.scientist.construction_priors import (
        get_active_construction_prior,
        list_construction_prior_snapshots,
    )
    active = get_active_construction_prior(nb)
    snapshots = list_construction_prior_snapshots(nb, limit=20)
    return jsonify(json_safe({"active": active, "snapshots": snapshots}))


def _api_construction_prior_refresh(nb=None):
    """Compute a fresh prior from current evidence and activate it."""
    from research.scientist.construction_priors import (
        compute_construction_prior,
        record_construction_prior_snapshot,
    )
    body = (request.is_json and request.json) or {}
    min_n = max(2, int(body.get("min_n", 4) or 4))
    notes = str(body.get("notes") or "")
    prior = compute_construction_prior(nb, min_n=min_n)
    if not prior["payload"]["rules"]:
        return jsonify({"error": "no rules met threshold; nothing to snapshot"}), 400
    version = record_construction_prior_snapshot(nb, prior, activate=True, notes=notes)
    return jsonify(json_safe({
        "status": "activated",
        "version": version,
        "summary": prior["summary"],
    }))


def _api_purge_junk_programs(nb=None):
    dry_run = True
    if request.is_json and request.json:
        dry_run = request.json.get("dry_run", True)
    result = nb.purge_junk_programs(dry_run=dry_run)
    return jsonify(result)


def register_programs_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)
    register_notebook_routes(
        app,
        wnb,
        (
            ("/api/programs/<result_id>", "api_program_detail", _api_program_detail),
            (
                "/api/programs/<result_id>/explanation",
                "api_program_explanation",
                _api_program_explanation,
                ("POST",),
            ),
            (
                "/api/programs/<result_id>/lineage",
                "api_program_lineage",
                _api_program_lineage,
            ),
            (
                "/api/programs/<result_id>/refine-analysis",
                "api_program_refine_analysis",
                _api_program_refine_analysis,
            ),
            (
                "/api/programs/<result_id>/morph",
                "api_program_morph",
                _api_program_morph,
                ("POST",),
            ),
            (
                "/api/programs/<result_id>/external-benchmarks",
                "api_program_external_benchmarks",
                _api_program_external_benchmarks,
                ("POST",),
            ),
            (
                "/api/programs/<result_id>/backfill-metrics",
                "api_program_backfill_metrics",
                _api_program_backfill_metrics,
                ("POST",),
                (notebook_path,),
            ),
            (
                "/api/programs/<result_id>/backfill-loss",
                "api_program_backfill_loss",
                _api_program_backfill_loss,
                ("POST",),
                (notebook_path,),
            ),
            (
                "/api/programs/<result_id>/rescreen",
                "api_program_rescreen",
                _api_program_rescreen,
                ("POST",),
                (notebook_path,),
            ),
            (
                "/api/programs/<result_id>/promote-screening",
                "api_program_promote_screening",
                _api_program_promote_screening,
                ("POST",),
            ),
            (
                "/api/programs/<result_id>/queue-validation-rerun",
                "api_program_queue_validation_rerun",
                _api_program_queue_validation_rerun,
                ("POST",),
            ),
            (
                "/api/programs/<result_id>/pending-reruns",
                "api_program_pending_reruns",
                _api_program_pending_reruns,
            ),
            (
                "/api/programs/<result_id>/pending-reruns/<task_id>/cancel",
                "api_program_cancel_rerun",
                _api_program_cancel_rerun,
                ("POST",),
            ),
            (
                "/api/programs/<result_id>/causal-evidence",
                "api_program_causal_evidence",
                _api_program_causal_evidence,
            ),
            (
                "/api/programs/<result_id>/causal-ablation",
                "api_program_causal_ablation",
                _api_program_causal_ablation,
                ("POST",),
                (notebook_path,),
            ),
            (
                "/api/ablations/bulk/start",
                "api_bulk_causal_ablation_start",
                _api_bulk_causal_ablation_start,
                ("POST",),
                (notebook_path,),
            ),
            (
                "/api/ablations/causal-summary",
                "api_causal_ablation_summary",
                _api_causal_ablation_summary,
            ),
            (
                "/api/ablations/champions",
                "api_causal_ablation_champions",
                _api_causal_ablation_champions,
            ),
            (
                "/api/ablations/components",
                "api_causal_ablation_components",
                _api_causal_ablation_components,
            ),
            (
                "/api/ablations/recommendations",
                "api_causal_ablation_recommendations",
                _api_causal_ablation_recommendations,
            ),
            (
                "/api/ablations/children-for-rule",
                "api_causal_ablation_children_for_rule",
                _api_causal_ablation_children_for_rule,
            ),
            (
                "/api/ablations/construction-prior",
                "api_construction_prior_active",
                _api_construction_prior_active,
            ),
            (
                "/api/ablations/construction-prior/refresh",
                "api_construction_prior_refresh",
                _api_construction_prior_refresh,
                ("POST",),
            ),
            (
                "/api/runner/drain-pending-validation-rerun",
                "api_drain_pending_validation_rerun",
                _api_drain_pending_validation_rerun,
                ("POST",),
                (notebook_path,),
            ),
            ("/api/programs", "api_programs", _api_programs),
            (
                "/api/programs/<result_id>/training-curve",
                "api_training_curve",
                _api_training_curve,
            ),
            (
                "/api/programs/purge-junk",
                "api_purge_junk_programs",
                _api_purge_junk_programs,
                ("POST",),
            ),
        ),
    )
