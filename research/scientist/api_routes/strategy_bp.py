"""Decision packet, fingerprint, and strategy briefing route registration."""

from __future__ import annotations

import json
import logging
import os
from flask import jsonify, request
from ..json_utils import json_safe as _json_safe
from ..notebook.graph_artifacts import resolve_graph_json_value
from ..runner._types import RunConfig
from ._helpers import (
    get_aria_for_notebook,
    get_runner,
    with_native_runner_progress,
    get_run_trigger_snapshot,
)
from ._strategy_decision_packet import build_decision_packet
from ._strategy_briefing import (
    gather_briefing_data,
    try_llm_briefing,
    build_deterministic_briefing,
    determine_recommended_action,
)
from .deps import ApiRouteContext
from ._utils import register_notebook_routes, register_routes, with_notebook_context

logger = logging.getLogger(__name__)


def _register_decision_routes(app, notebook_path: str, wnb) -> None:
    """Decision: decision-packet, reproducibility-manifest, workflow-export."""

    def api_decision_packet(result_id, nb=None):
        """One-click evidence bundle for promotion decisions."""
        packet = build_decision_packet(nb, result_id)
        if packet is None:
            return jsonify({"error": "Not found"}), 404
        return jsonify(packet)

    def api_reproducibility_manifest(result_id, nb=None):
        """Exportable reproducibility manifest for a program result."""
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        program = nb.get_program_detail(result_id)
        if program is None:
            return jsonify({"error": "Not found"}), 404

        experiment_id = program.get("experiment_id")
        experiment = None
        if experiment_id:
            try:
                experiment = nb.get_experiment(experiment_id)
            except Exception as exc:
                logger.debug("Suppressed error: %s", exc)

        config = (experiment or {}).get("config", {}) or {}
        training = {}
        try:
            tp = json.loads(program.get("training_program_json") or "{}")
            training = tp
        except (json.JSONDecodeError, TypeError):
            pass

        # Grammar weights snapshot from experiment config
        grammar_weights = config.get("applied_grammar_weights") or config.get(
            "grammar_weights"
        )
        grammar_config = config.get("grammar_config", {})

        manifest = {
            "result_id": result_id,
            "graph_fingerprint": program.get("graph_fingerprint"),
            "experiment_id": experiment_id,
            "experiment_type": (experiment or {}).get("experiment_type"),
            "timestamp": program.get("timestamp"),
            "code_version": config.get("code_version"),
            "seeds": {
                "experiment_seed": config.get("seed"),
                "training_seed": training.get("seed"),
            },
            "data": {
                "data_mode": config.get("data_mode"),
                "dataset": config.get("dataset"),
                "seq_len": training.get("seq_len") or config.get("seq_len"),
                "batch_size": training.get("batch_size") or config.get("batch_size"),
                "vocab_size": training.get("vocab_size") or config.get("vocab_size"),
            },
            "grammar": {
                "max_ops": grammar_config.get("max_ops"),
                "max_depth": grammar_config.get("max_depth"),
                "weights_snapshot": grammar_weights,
            },
            "training": {
                "learning_rate": training.get("learning_rate") or training.get("lr"),
                "steps": training.get("steps") or training.get("n_steps"),
                "warmup_steps": training.get("warmup_steps"),
            },
            "architecture": {
                "param_count": program.get("param_count"),
                "graph_json": program.get("graph_json"),
            },
            "outcomes": {
                "stage0_passed": bool(program.get("stage0_passed")),
                "stage05_passed": bool(program.get("stage05_passed")),
                "stage1_passed": bool(program.get("stage1_passed")),
                "loss_ratio": program.get("loss_ratio"),
                "discovery_loss_ratio": program.get("discovery_loss_ratio"),
                "validation_loss_ratio": program.get("validation_loss_ratio"),
                "novelty_score": program.get("novelty_score"),
                "baseline_loss_ratio": program.get("baseline_loss_ratio"),
                "validation_baseline_ratio": program.get("validation_baseline_ratio"),
            },
            "canonical_metrics": {
                "compression": analytics.canonical_compression_metrics(program),
            },
            "packet_status": analytics.reproducibility_packet_status(program),
        }
        return jsonify(manifest)

    def api_workflow_export(result_id: str, nb=None):
        """Export a program result as an aria_designer workflow JSON."""
        row = nb.conn.execute(
            "SELECT graph_json, model_dim FROM program_results_compat WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        if not row or not row["graph_json"]:
            return jsonify({"error": "Program not found or has no graph"}), 404

        from research.synthesis.serializer import graph_from_json
        from research.synthesis.workflow_converter import graph_to_workflow

        graph_json = resolve_graph_json_value(nb.conn, nb.db_path, row["graph_json"])
        graph = graph_from_json(graph_json, model_dim=row["model_dim"])
        workflow = graph_to_workflow(
            graph,
            workflow_id=f"aria_{result_id[:8]}",
            name=f"Aria Discovery {result_id[:8]}",
            metadata={"result_id": result_id},
        )
        return jsonify(workflow)

    register_notebook_routes(
        app,
        wnb,
        (
            (
                "/api/decision-packet/<result_id>",
                "api_decision_packet",
                api_decision_packet,
            ),
            (
                "/api/reproducibility-manifest/<result_id>",
                "api_reproducibility_manifest",
                api_reproducibility_manifest,
            ),
            (
                "/api/reproducibility-manifest/<result_id>/workflow",
                "api_workflow_export",
                api_workflow_export,
                ("GET",),
            ),
        ),
    )


def _register_fingerprint_routes(app, notebook_path: str, wnb) -> None:
    """Fingerprint: references, fingerprint/resolve, fingerprint/history."""

    def api_references(nb=None):
        """Get pinned reference architectures."""
        from ..naming import annotate_display_names

        refs = [
            entry
            for entry in nb.get_leaderboard(
                limit=500,
                sort_by="composite_score",
                include_references=True,
                trusted_only=True,
            )
            if entry.get("is_reference")
        ]
        annotate_display_names(refs)
        return jsonify(
            {
                "entries": _json_safe(refs),
                "total": len(refs),
            }
        )

    def api_fingerprint_resolve(nb=None):
        """Resolve a result_id or fingerprint prefix to a concrete program result.

        Preference order for fingerprint prefixes:
        1) Best leaderboard-backed run (highest composite score)
        2) Best surviving run by loss ratio
        """
        value = str(request.args.get("value") or "").strip()
        if not value:
            return jsonify({"error": "value query param required"}), 400
        direct = nb.conn.execute(
            "SELECT result_id, graph_fingerprint FROM program_results_compat WHERE result_id = ?",
            (value,),
        ).fetchone()
        if direct:
            return jsonify(
                {
                    "result_id": direct["result_id"],
                    "graph_fingerprint": direct.get("graph_fingerprint"),
                    "resolved_from": "result_id",
                    "candidates": [],
                }
            )
        rows = nb.conn.execute(
            """
            SELECT
                pr.result_id,
                pr.graph_fingerprint,
                pr.experiment_id,
                pr.stage1_passed,
                pr.loss_ratio,
                pr.timestamp,
                lb.tier,
                lb.composite_score,
                lb.screening_loss_ratio,
                lb.investigation_loss_ratio,
                lb.validation_loss_ratio
            FROM program_results_compat pr
            LEFT JOIN leaderboard lb ON lb.result_id = pr.result_id
            WHERE pr.graph_fingerprint LIKE ?
            ORDER BY
                CASE WHEN lb.result_id IS NULL THEN 1 ELSE 0 END ASC,
                COALESCE(lb.composite_score, -1e9) DESC,
                pr.stage1_passed DESC,
                (pr.loss_ratio IS NULL) ASC,
                pr.loss_ratio ASC,
                pr.timestamp DESC
            LIMIT 50
            """,
            (f"{value}%",),
        ).fetchall()
        if rows:
            chosen_row = dict(rows[0])
            candidates = []
            for row in rows:
                candidates.append(
                    {
                        "result_id": row["result_id"],
                        "graph_fingerprint": row["graph_fingerprint"],
                        "experiment_id": row["experiment_id"],
                        "stage1_passed": bool(row["stage1_passed"]),
                        "loss_ratio": row["loss_ratio"],
                        "timestamp": row["timestamp"],
                        "tier": row["tier"],
                        "composite_score": row["composite_score"],
                        "screening_loss_ratio": row["screening_loss_ratio"],
                        "investigation_loss_ratio": row["investigation_loss_ratio"],
                        "validation_loss_ratio": row["validation_loss_ratio"],
                    }
                )
            return jsonify(
                {
                    "result_id": chosen_row.get("result_id"),
                    "graph_fingerprint": chosen_row.get("graph_fingerprint"),
                    "resolved_from": "graph_fingerprint",
                    "candidate_count": len(candidates),
                    "selection_policy": "leaderboard_composite_then_loss",
                    "candidates": candidates,
                }
            )
        return jsonify({"error": "No matching fingerprint or result_id found."}), 404

    def api_fingerprint_history(nb=None):
        """Return chronological run history for a fingerprint prefix/result_id."""
        value = str(request.args.get("value") or "").strip()
        limit = int(request.args.get("limit", 100) or 100)
        limit = max(1, min(limit, 500))
        if not value:
            return jsonify({"error": "value query param required"}), 400
        direct = nb.conn.execute(
            "SELECT graph_fingerprint FROM program_results_compat WHERE result_id = ? LIMIT 1",
            (value,),
        ).fetchone()
        fingerprint_like = (direct["graph_fingerprint"] if direct else value) + "%"
        rows = nb.conn.execute(
            """
            SELECT
                pr.result_id,
                pr.graph_fingerprint,
                pr.experiment_id,
                pr.timestamp,
                pr.stage0_passed,
                pr.stage05_passed,
                pr.stage1_passed,
                pr.loss_ratio,
                pr.discovery_loss_ratio,
                pr.validation_loss_ratio,
                lb.tier,
                lb.composite_score,
                lb.screening_loss_ratio,
                lb.investigation_loss_ratio,
                lb.validation_loss_ratio AS lb_validation_loss_ratio,
                lb.investigation_passed,
                lb.validation_passed
            FROM program_results_compat pr
            LEFT JOIN leaderboard lb ON lb.result_id = pr.result_id
            WHERE pr.graph_fingerprint LIKE ?
            ORDER BY pr.timestamp DESC
            LIMIT ?
            """,
            (fingerprint_like, limit),
        ).fetchall()
        history = [dict(r) for r in rows]
        best_row = nb.conn.execute(
            """
            SELECT
                pr.result_id,
                pr.graph_fingerprint,
                pr.experiment_id,
                pr.timestamp,
                pr.loss_ratio,
                lb.tier,
                lb.composite_score,
                lb.validation_loss_ratio
            FROM program_results_compat pr
            JOIN leaderboard lb ON lb.result_id = pr.result_id
            WHERE pr.graph_fingerprint LIKE ?
              AND lb.composite_score IS NOT NULL
            ORDER BY lb.composite_score DESC, pr.timestamp DESC
            LIMIT 1
            """,
            (fingerprint_like,),
        ).fetchone()
        best_by_composite = dict(best_row) if best_row else None
        return jsonify(
            {
                "query": value,
                "resolved_graph_fingerprint": history[0]["graph_fingerprint"]
                if history
                else None,
                "total": len(history),
                "best_leaderboard_run": best_by_composite,
                "runs": history,
            }
        )

    register_notebook_routes(
        app,
        wnb,
        (
            ("/api/references", "api_references", api_references),
            (
                "/api/fingerprint/resolve",
                "api_fingerprint_resolve",
                api_fingerprint_resolve,
            ),
            (
                "/api/fingerprint/history",
                "api_fingerprint_history",
                api_fingerprint_history,
            ),
        ),
    )


def _register_execution_routes(app, notebook_path: str) -> None:
    """Execution: worker/evaluate, progress."""

    def api_worker_evaluate():
        """Z12: Distributed worker endpoint for evaluating a computation graph."""
        runner = get_runner(notebook_path)
        body = request.get_json(silent=True) or {}

        graph_json = body.get("graph_json")
        config_dict = body.get("config")
        seed = body.get("seed", 42)

        if not graph_json or not config_dict:
            return jsonify({"error": "Missing graph_json or config"}), 400

        try:
            from research.synthesis.graph import json_to_graph
            from research.synthesis.compiler import compile_model
            import torch

            graph = json_to_graph(graph_json)
            config = RunConfig.from_dict(config_dict)

            dev_str = config.device if torch.cuda.is_available() else "cpu"
            dev = torch.device(dev_str)

            # Compile model locally on worker
            layer_graphs = [graph] * config.n_layers
            model = compile_model(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            ).to(dev)

            # Use the runner's async-friendly training method
            result = runner._micro_train_async(model, config, seed, dev)

            return jsonify(
                {
                    "status": "ok",
                    "result": result,
                    "device": dev_str,
                    "worker_id": os.environ.get("ARIA_WORKER_ID", "anonymous"),
                }
            )

        except Exception as e:
            logger.error("Worker evaluation failed: %s", e)
            return jsonify({"error": str(e), "passed": False}), 500

    def api_progress():
        """Get current experiment progress (poll-based alternative to SSE)."""
        runner = get_runner(notebook_path)
        progress_payload = with_native_runner_progress(runner.progress.to_dict())
        trigger = get_run_trigger_snapshot(progress_payload.get("experiment_id"))
        progress_payload["run_trigger_source"] = trigger.get("source")
        progress_payload["run_trigger"] = trigger
        return jsonify(
            {
                "is_running": runner.is_running,
                "progress": progress_payload,
                "native_runner": progress_payload.get("native_runner"),
                "run_trigger_source": trigger.get("source"),
                "run_trigger": trigger,
            }
        )

    register_routes(
        app,
        (
            (
                "/api/worker/evaluate",
                "api_worker_evaluate",
                api_worker_evaluate,
                ("POST",),
            ),
            ("/api/progress", "api_progress", api_progress),
        ),
    )


def _register_briefing_routes(app, notebook_path: str, wnb) -> None:
    """Briefing: strategy/briefing."""

    def api_strategy_briefing(nb=None):
        """Data-driven strategy briefing for the overview page.

        Tries LLM-powered briefing first (via Aria), falls back to
        deterministic rules.  Always returns a valid response.
        """
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        recent = nb.get_recent_experiments(10)
        data = gather_briefing_data(nb, analytics, recent)

        # Optional: highlight a just-completed experiment
        just_completed_id = request.args.get("just_completed")
        just_completed_exp = None
        if just_completed_id:
            for e in recent:
                if (e.get("experiment_id") or "").startswith(just_completed_id):
                    just_completed_exp = e
                    break
            aria_inst = get_aria_for_notebook(notebook_path)
            if hasattr(aria_inst, "_briefing_cache"):
                aria_inst._briefing_cache = None

        # Try LLM-powered briefing first
        aria = get_aria_for_notebook(notebook_path)
        llm_response = try_llm_briefing(
            nb, aria, analytics, data, recent, just_completed_exp
        )
        if llm_response:
            return jsonify(llm_response)

        # Deterministic fallback
        briefing = build_deterministic_briefing(nb, data)
        action_result = determine_recommended_action(nb, data)

        return jsonify(
            {
                "briefing": briefing,
                "action": action_result["action"],
                "action_label": action_result["action_label"],
                "action_rationale": action_result["action_rationale"],
                "ai_powered": False,
                "fallback_reason": "llm_unavailable",
                "suggested_config": action_result["suggested_config"],
                "evidence": data["recommendation_evidence"],
                "data": data["data_block"],
                "compression_opportunities": data["compression_opportunities"],
                "ref_comparison": data["ref_comparison"],
            }
        )

    register_notebook_routes(
        app,
        wnb,
        (
            (
                "/api/strategy/briefing",
                "api_strategy_briefing",
                api_strategy_briefing,
            ),
        ),
    )


def register_strategy_bp_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)
    _register_decision_routes(app, notebook_path, wnb)
    _register_fingerprint_routes(app, notebook_path, wnb)
    _register_execution_routes(app, notebook_path)
    _register_briefing_routes(app, notebook_path, wnb)
