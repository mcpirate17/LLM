"""Program mutating actions: morph, external benchmarks, backfills, rescreen, promote, purge."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from flask import jsonify, request

from research.synthesis.workflow_converter import graph_to_workflow

from .._helpers import get_runner
from ...refinement_scoring import oscillation_risk_score
from ...runner._types import RunConfig

from ._shared import (
    _COMPARABILITY_LABEL_RANK,
    _TRUST_LABEL_RANK,
    _leaderboard_backed_program_detail,
    _preserve_stronger_label,
)

logger = logging.getLogger(__name__)


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
            from ...analytics import ExperimentAnalytics, RefinementAnalyzer

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
    from ...screening_recompute import recompute_screening_metrics

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
                f"UPDATE graph_runs SET {', '.join(set_parts)} WHERE result_id = ?",
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
    candidate_confirmation = bool(body.get("candidate_confirmation"))
    try:
        stage1_steps = (
            int(body.get("stage1_steps"))
            if body.get("stage1_steps") is not None
            else None
        )
    except (TypeError, ValueError):
        return jsonify({"error": "stage1_steps must be an integer"}), 400
    if stage1_steps is not None:
        stage1_steps = max(50, min(50000, stage1_steps))
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
            independent_sample=candidate_confirmation,
            candidate_confirmation=candidate_confirmation,
            stage1_steps=stage1_steps,
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
            "candidate_confirmation": candidate_confirmation,
            "stage1_steps": stage1_steps,
        }
    )


# Stage budget defaults match research/defaults.py.  Each "queue X
# rerun" button uses its tier's natural budget so the new sample is
# in the same regime as the existing tier rows.


def _resolve_screening_promotion_target(result_id, nb):
    program = nb.get_program_detail(result_id)
    if program is None:
        program = _leaderboard_backed_program_detail(nb, result_id)
    if program is None:
        return result_id, None, None

    entry = nb.get_leaderboard_entry(result_id)
    if entry is None:
        fp = str(program.get("graph_fingerprint") or "").strip()
        if fp:
            sibling_entry = nb.get_leaderboard_entry_by_fingerprint(fp)
            if sibling_entry and sibling_entry.get("result_id") != result_id:
                entry = sibling_entry
                result_id = sibling_entry.get("result_id")
                program = nb.get_program_detail(result_id) or program
    return result_id, program, entry


def _is_backfill_program(program: dict, entry: dict | None) -> bool:
    current_result_cohort = (
        str(
            program.get("result_cohort")
            or (entry.get("result_cohort") if entry else "")
        )
        .strip()
        .lower()
    )
    current_trust_label = (
        str(program.get("trust_label") or (entry.get("trust_label") if entry else ""))
        .strip()
        .lower()
    )
    current_comparability_label = (
        str(
            program.get("comparability_label")
            or (entry.get("comparability_label") if entry else "")
        )
        .strip()
        .lower()
    )
    return (
        current_result_cohort == "backfill"
        or current_trust_label == "backfill_observation"
        or current_comparability_label == "reconstructed_init_variant"
    )


def _screening_promotion_labels(program: dict, entry: dict | None) -> tuple[str, str]:
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
    return trust_label, comparability_label


def _upsert_manual_screening_promotion(
    nb, result_id: str, program: dict, entry: dict | None
) -> dict:
    trust_label, comparability_label = _screening_promotion_labels(program, entry)
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
        return nb.get_leaderboard_entry(result_id) or {"entry_id": entry_id}
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
        return entry


def _api_program_promote_screening(result_id, nb=None):
    result_id, program, entry = _resolve_screening_promotion_target(result_id, nb)
    if program is None:
        return jsonify({"error": "Program not found"}), 404

    if _is_backfill_program(program, entry):
        return (
            jsonify(
                {
                    "error": (
                        "Backfill rows cannot be manually promoted into candidate "
                        "provenance. Run candidate confirmation instead."
                    )
                }
            ),
            409,
        )

    entry = _upsert_manual_screening_promotion(nb, result_id, program, entry)
    trust_label, comparability_label = _screening_promotion_labels(program, entry)

    nb.conn.execute(
        """
        UPDATE graph_runs
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


def _api_purge_junk_programs(nb=None):
    dry_run = True
    if request.is_json and request.json:
        dry_run = request.json.get("dry_run", True)
    result = nb.purge_junk_programs(dry_run=dry_run)
    return jsonify(result)
