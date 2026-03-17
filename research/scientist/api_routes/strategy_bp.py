"""Decision packet, fingerprint, and strategy briefing route registration."""

from __future__ import annotations

import json
import logging
import os
import traceback
from typing import Optional

from flask import jsonify, request
from ..json_utils import json_safe as _json_safe
from ..notebook import LabNotebook
from ..runner import RunConfig
from ..persona import get_aria
from ._helpers import (
    get_runner,
    with_native_runner_progress,
    get_run_trigger_snapshot,
    normalize_result_ids,
)
from ._strategy_preflight import (
    build_start_mode_eligibility,
    normalize_briefing_mode,
    briefing_action_from_mode,
    briefing_action_label,
    augment_sparse_action_config,
)
from ._strategy_recommendations import (
    compute_cross_run_stability,
    compute_recommendation,
    compute_compression_opportunities,
    compute_sparse_evidence,
    sparse_coverage_summary,
)
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_strategy_bp_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path

    @app.route("/api/decision-packet/<result_id>")
    def api_decision_packet(result_id):
        """One-click evidence bundle for promotion decisions."""
        nb = LabNotebook(notebook_path)
        try:
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            fingerprint = program.get("graph_fingerprint", "")
            experiment_id = program.get("experiment_id")

            # Leaderboard entry (targeted)
            leaderboard_entry = None
            try:
                leaderboard_entry = nb.get_leaderboard_entry(result_id)
            except Exception:
                leaderboard_entry = None

            # Experiment data + failure analysis
            experiment = None
            failure_analysis = {"funnel": {}, "errors": {}, "stage_deaths": {}}
            if experiment_id:
                try:
                    experiment = nb.get_experiment(experiment_id)
                except Exception:
                    pass
                try:
                    failure_analysis = nb.get_failure_analysis(experiment_id)
                except Exception:
                    pass

            # Hypothesis chain — find hypothesis linked to this experiment
            hypothesis_chain = []
            if experiment_id:
                try:
                    hyp_row = nb.conn.execute(
                        "SELECT hypothesis_id FROM hypotheses WHERE experiment_id = ?",
                        (experiment_id,),
                    ).fetchone()
                    if hyp_row:
                        hypothesis_chain = nb.get_hypothesis_chain(
                            hyp_row["hypothesis_id"]
                            if isinstance(hyp_row, dict)
                            else hyp_row[0]
                        )
                except Exception:
                    pass

            # Cross-run stability for this specific result
            cross_run = {"trend": "unknown", "seen_runs": 0}
            try:
                top = nb.get_top_programs(20, sort_by="loss_ratio")
                stability = compute_cross_run_stability(nb, top)
                for c in stability.get("candidates", []):
                    if c.get("result_id") == result_id:
                        cross_run = {
                            "trend": c.get("trend", "unknown"),
                            "seen_runs": c.get("seen_runs", 0),
                        }
                        break
            except Exception:
                pass

            # Build outcomes by phase
            tier = (leaderboard_entry or {}).get("tier", "screening")
            outcomes = {
                "screening": {
                    "loss_ratio": program.get("loss_ratio"),
                    "novelty": program.get("novelty_score"),
                },
                "investigation": None,
                "validation": None,
            }
            if leaderboard_entry:
                inv_lr = leaderboard_entry.get("investigation_loss_ratio")
                if inv_lr is not None:
                    outcomes["investigation"] = {
                        "loss_ratio": inv_lr,
                        "robustness": leaderboard_entry.get("investigation_robustness"),
                        "passed": bool(leaderboard_entry.get("investigation_passed")),
                    }
                val_lr = leaderboard_entry.get("validation_loss_ratio")
                if val_lr is not None:
                    outcomes["validation"] = {
                        "loss_ratio": val_lr,
                        "baseline_ratio": leaderboard_entry.get(
                            "validation_baseline_ratio"
                        ),
                        "multi_seed_std": leaderboard_entry.get(
                            "validation_multi_seed_std"
                        ),
                        "passed": bool(leaderboard_entry.get("validation_passed")),
                    }

            # Baseline comparison
            bl_ratio = program.get("baseline_loss_ratio")
            baseline_comparison = {"ratio": bl_ratio, "interpretation": "unknown"}
            if bl_ratio is not None:
                if bl_ratio < 0.95:
                    baseline_comparison["interpretation"] = "outperforms"
                elif bl_ratio <= 1.05:
                    baseline_comparison["interpretation"] = "comparable"
                else:
                    baseline_comparison["interpretation"] = "underperforms"

            # Failure context
            failure_context = {
                "stage_at_death": program.get("stage_at_death"),
                "error_type": program.get("error_type"),
                "experiment_errors": failure_analysis.get("errors", {}),
                "experiment_funnel": failure_analysis.get("funnel", {}),
            }

            # Recommendation
            recommendation = compute_recommendation(program, leaderboard_entry)

            # Evidence flags
            from ..analytics import ExperimentAnalytics

            analytics = ExperimentAnalytics(nb)
            packet_status = analytics.reproducibility_packet_status(
                leaderboard_entry if leaderboard_entry else program
            )
            evidence_flags = {
                "has_baseline": bl_ratio is not None,
                "has_cka_artifact": program.get("cka_source") == "artifact",
                "has_multi_seed": outcomes["validation"] is not None,
                "has_hypothesis": len(hypothesis_chain) > 0,
                "repro_packet_ready": packet_status.get("status") == "ready",
            }

            return jsonify(
                {
                    "result_id": result_id,
                    "fingerprint": fingerprint,
                    "experiment_id": experiment_id,
                    "hypothesis_chain": hypothesis_chain,
                    "outcomes": outcomes,
                    "baseline_comparison": baseline_comparison,
                    "failure_context": failure_context,
                    "cross_run_stability": cross_run,
                    "recommendation": recommendation,
                    "evidence_flags": evidence_flags,
                    "compression_metrics": analytics.canonical_compression_metrics(
                        leaderboard_entry if leaderboard_entry else program
                    ),
                    "reproducibility_packet": packet_status,
                }
            )
        except Exception as e:
            logger.error(
                f"Error in /api/decision-packet/{result_id}: {e}\n"
                f"{traceback.format_exc()}"
            )
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/reproducibility-manifest/<result_id>")
    def api_reproducibility_manifest(result_id):
        """Exportable reproducibility manifest for a program result."""
        nb = LabNotebook(notebook_path)
        try:
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
                except Exception:
                    pass

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
                    "batch_size": training.get("batch_size")
                    or config.get("batch_size"),
                    "vocab_size": training.get("vocab_size")
                    or config.get("vocab_size"),
                },
                "grammar": {
                    "max_ops": grammar_config.get("max_ops"),
                    "max_depth": grammar_config.get("max_depth"),
                    "weights_snapshot": grammar_weights,
                },
                "training": {
                    "learning_rate": training.get("learning_rate")
                    or training.get("lr"),
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
                    "validation_baseline_ratio": program.get(
                        "validation_baseline_ratio"
                    ),
                },
                "canonical_metrics": {
                    "compression": analytics.canonical_compression_metrics(program),
                },
                "packet_status": analytics.reproducibility_packet_status(program),
            }
            return jsonify(manifest)
        except Exception as e:
            logger.error(
                f"Error in /api/reproducibility-manifest/{result_id}: {e}\n"
                f"{traceback.format_exc()}"
            )
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/reproducibility-manifest/<result_id>/workflow", methods=["GET"])
    def api_workflow_export(result_id: str):
        """Export a program result as an aria_designer workflow JSON."""
        nb = LabNotebook(notebook_path)
        try:
            row = nb.conn.execute(
                "SELECT graph_json, model_dim FROM program_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if not row or not row["graph_json"]:
                return jsonify({"error": "Program not found or has no graph"}), 404

            from ..synthesis.serializer import graph_from_json
            from ..synthesis.workflow_converter import graph_to_workflow

            graph = graph_from_json(row["graph_json"], model_dim=row["model_dim"])
            workflow = graph_to_workflow(
                graph,
                workflow_id=f"aria_{result_id[:8]}",
                name=f"Aria Discovery {result_id[:8]}",
                metadata={"result_id": result_id},
            )
            return jsonify(workflow)
        except Exception as e:
            logger.error(f"Error exporting workflow for {result_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/references")
    def api_references():
        """Get pinned reference architectures."""
        nb = LabNotebook(notebook_path)
        try:
            from ..naming import annotate_display_names

            refs = nb.get_references()
            annotate_display_names(refs)
            return jsonify(
                {
                    "entries": _json_safe(refs),
                    "total": len(refs),
                }
            )
        except Exception as e:
            logger.error(f"Error in /api/references: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/fingerprint/resolve")
    def api_fingerprint_resolve():
        """Resolve a result_id or fingerprint prefix to a concrete program result.

        Preference order for fingerprint prefixes:
        1) Best leaderboard-backed run (highest composite score)
        2) Best surviving run by loss ratio
        """
        value = str(request.args.get("value") or "").strip()
        if not value:
            return jsonify({"error": "value query param required"}), 400
        nb = LabNotebook(notebook_path)
        try:
            direct = nb.conn.execute(
                "SELECT result_id, graph_fingerprint FROM program_results WHERE result_id = ?",
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
                FROM program_results pr
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
            return jsonify(
                {"error": "No matching fingerprint or result_id found."}
            ), 404
        except Exception as e:
            logger.error(f"Error in /api/fingerprint/resolve: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/fingerprint/history")
    def api_fingerprint_history():
        """Return chronological run history for a fingerprint prefix/result_id."""
        value = str(request.args.get("value") or "").strip()
        limit = int(request.args.get("limit", 100) or 100)
        limit = max(1, min(limit, 500))
        if not value:
            return jsonify({"error": "value query param required"}), 400
        nb = LabNotebook(notebook_path)
        try:
            direct = nb.conn.execute(
                "SELECT graph_fingerprint FROM program_results WHERE result_id = ? LIMIT 1",
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
                FROM program_results pr
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
                FROM program_results pr
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
        except Exception as e:
            logger.error(f"Error in /api/fingerprint/history: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/worker/evaluate", methods=["POST"])
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
            from ..synthesis.graph import json_to_graph
            from ..synthesis.compiler import compile_model
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

    @app.route("/api/progress")
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

    @app.route("/api/strategy/briefing")
    def api_strategy_briefing():
        """Data-driven strategy briefing for the overview page.

        Tries LLM-powered briefing first (via Aria), falls back to
        deterministic rules.  Always returns a valid response.
        """
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics

            analytics = ExperimentAnalytics(nb)
            summary = nb.get_dashboard_summary()
            recent = nb.get_recent_experiments(10)
            trajectory = analytics.learning_trajectory() or {}
            compression_coverage = analytics.compression_coverage() or {}
            compression_opportunities = compute_compression_opportunities(
                compression_coverage
            )
            primitive_effectiveness = (
                analytics.compression_primitive_effectiveness() or {}
            )
            sparse_evidence = compute_sparse_evidence(nb)
            sparse_coverage_data = analytics.sparse_coverage() or {}
            sparse_coverage_overview = sparse_coverage_summary(sparse_coverage_data)

            # Optional: highlight a just-completed experiment
            just_completed_id = request.args.get("just_completed")
            just_completed_exp = None
            if just_completed_id:
                for e in recent:
                    if (e.get("experiment_id") or "").startswith(just_completed_id):
                        just_completed_exp = e
                        break
                # Clear briefing cache so LLM sees the new context
                aria_inst = get_aria()
                if hasattr(aria_inst, "_briefing_cache"):
                    aria_inst._briefing_cache = None

            # --- Pipeline counts (exclude pinned reference architectures) ---
            leaderboard_rows = nb.conn.execute(
                "SELECT tier, COUNT(*) as cnt FROM leaderboard "
                "WHERE COALESCE(is_reference, 0) = 0 GROUP BY tier"
            ).fetchall()
            tiers = {r["tier"]: r["cnt"] for r in leaderboard_rows}
            screening = tiers.get("screening", 0)
            investigation = tiers.get("investigation", 0)
            validation = tiers.get("validation", 0)
            breakthrough = tiers.get("breakthrough", 0)

            # --- Recent outcomes ---
            completed = [e for e in recent if e.get("status") == "completed"]
            recent_s1_rates = []
            for e in completed[:5]:
                gen = e.get("n_programs_generated") or 0
                passed = e.get("n_stage1_passed") or 0
                if gen > 0:
                    recent_s1_rates.append(passed / gen)

            avg_recent_s1 = (
                sum(recent_s1_rates) / len(recent_s1_rates) if recent_s1_rates else None
            )

            # --- Learning trend ---
            trend = trajectory.get("trend", "insufficient_data")
            slope = trajectory.get("slope")

            # --- Common data block (used by both LLM and deterministic) ---
            total_exp = summary.get("total_experiments", 0)
            total_progs = summary.get("total_programs_evaluated", 0)
            s1_survivors = summary.get("stage1_survivors", 0)

            pipeline_data = {
                "screening": screening,
                "investigation": investigation,
                "validation": validation,
                "breakthrough": breakthrough,
            }
            compression_summary = compression_opportunities.get("summary") or {}
            data_block = {
                "total_experiments": total_exp,
                "total_programs": total_progs,
                "s1_survivors": s1_survivors,
                "avg_recent_s1_rate": avg_recent_s1,
                "learning_trend": trend,
                "learning_slope": slope,
                "pipeline": pipeline_data,
                "compression": compression_summary,
                "compression_primitives": primitive_effectiveness.get("primitives", []),
                "sparse": sparse_evidence,
            }

            recent_window = recent[:10]
            recent_cancelled = 0
            recent_failed = 0
            for exp in recent_window:
                status = str(exp.get("status") or "").strip().lower()
                if status in {"cancelled", "canceled"}:
                    recent_cancelled += 1
                elif status == "failed":
                    recent_failed += 1

            recent_completed_window = completed[:5]
            recent_zero_s1_runs = 0
            for exp in recent_completed_window:
                gen = exp.get("n_programs_generated") or 0
                passed = exp.get("n_stage1_passed") or 0
                if gen > 0 and passed == 0:
                    recent_zero_s1_runs += 1

            recommendation_evidence = {
                "learning_trend": trend,
                "learning_slope": slope,
                "avg_recent_s1_rate": avg_recent_s1,
                "recent_completed_runs": len(recent_completed_window),
                "recent_zero_s1_runs": recent_zero_s1_runs,
                "recent_cancelled_runs": recent_cancelled,
                "recent_failed_runs": recent_failed,
                "pipeline": pipeline_data,
                "compression": compression_summary,
                "compression_primitives": primitive_effectiveness.get("primitives", []),
                "sparse": sparse_evidence,
                "sparse_coverage": sparse_coverage_overview,
            }

            # --- Try LLM-powered briefing first ---
            aria = get_aria()
            fallback_reason: Optional[str] = None
            llm = aria._get_llm()
            llm_reachable = False
            if llm is None:
                fallback_reason = "llm_not_configured"
            else:
                try:
                    llm_reachable = (
                        bool(llm.is_available())
                        if hasattr(llm, "is_available")
                        else True
                    )
                except Exception:
                    llm_reachable = False
                if not llm_reachable:
                    fallback_reason = "llm_unreachable"
            ref_comparison = None
            try:
                from ..llm.context_briefing import build_briefing_context

                # Gather extra context for LLM
                try:
                    active_campaigns = nb.get_active_campaigns()
                    campaign = active_campaigns[0] if active_campaigns else None
                except Exception:
                    campaign = None

                try:
                    dw = analytics.get_current_grammar_weights() or {}
                except Exception:
                    dw = {}

                try:
                    gw = analytics.compute_grammar_weights() or {}
                except Exception:
                    gw = {}

                try:
                    top_programs = nb.conn.execute(
                        "SELECT graph_fingerprint, loss_ratio, novelty_score, tier "
                        "FROM leaderboard WHERE COALESCE(is_reference, 0) = 0 "
                        "ORDER BY composite_score DESC LIMIT 3"
                    ).fetchall()
                    top_progs = (
                        [dict(r) for r in top_programs] if top_programs else None
                    )
                except Exception:
                    top_progs = None

                # --- Reference comparison: surface when synthesized models beat references ---
                try:
                    ref_rows = nb.conn.execute(
                        "SELECT reference_name, composite_score, loss_ratio "
                        "FROM leaderboard WHERE COALESCE(is_reference, 0) = 1 "
                        "ORDER BY composite_score DESC"
                    ).fetchall()
                    best_ref_score = max(
                        (r["composite_score"] for r in ref_rows), default=None
                    )
                    if best_ref_score and top_progs:
                        best_synth_score = nb.conn.execute(
                            "SELECT composite_score FROM leaderboard "
                            "WHERE COALESCE(is_reference, 0) = 0 "
                            "ORDER BY composite_score DESC LIMIT 1"
                        ).fetchone()
                        if (
                            best_synth_score
                            and best_synth_score["composite_score"] > best_ref_score
                        ):
                            ref_comparison = {
                                "beats_all_references": True,
                                "best_synthesized_score": float(
                                    best_synth_score["composite_score"]
                                ),
                                "best_reference_score": float(best_ref_score),
                                "margin_pct": round(
                                    100.0
                                    * (
                                        best_synth_score["composite_score"]
                                        - best_ref_score
                                    )
                                    / best_ref_score,
                                    1,
                                ),
                                "references": [
                                    {
                                        "name": r["reference_name"],
                                        "score": float(r["composite_score"]),
                                    }
                                    for r in ref_rows
                                ],
                            }
                        else:
                            ref_comparison = {
                                "beats_all_references": False,
                                "best_reference_score": float(best_ref_score),
                                "references": [
                                    {
                                        "name": r["reference_name"],
                                        "score": float(r["composite_score"]),
                                    }
                                    for r in ref_rows
                                ],
                            }
                    else:
                        ref_comparison = None
                except Exception:
                    ref_comparison = None

                try:
                    scaling_summary_data = None
                    try:
                        scaling_summary_data = nb.get_scaling_summary()
                    except Exception:
                        pass
                    briefing_context = build_briefing_context(
                        recent_experiments=recent,
                        pipeline_tiers=tiers,
                        learning_trajectory=trajectory,
                        campaign=campaign,
                        grammar_weights=gw,
                        default_weights=dw,
                        top_programs=top_progs,
                        just_completed=just_completed_exp,
                        sparse_coverage=sparse_coverage_data,
                        scaling_summary=scaling_summary_data,
                        ref_comparison=ref_comparison,
                    )
                except Exception:
                    briefing_context = {
                        "pipeline": pipeline_data,
                        "learning": {
                            "trend": trend,
                            "slope": slope,
                            "avg_recent_s1_rate": avg_recent_s1,
                        },
                        "recent_experiments": recent[:5],
                        "campaign": campaign,
                    }

                ai_briefing = aria.generate_briefing(context=briefing_context)
                if ai_briefing and ai_briefing.get("briefing_text"):
                    suggested = ai_briefing.get("suggested_action") or {}
                    normalized_mode = normalize_briefing_mode(suggested.get("mode"))
                    action_key = briefing_action_from_mode(normalized_mode)
                    suggested_config = dict(suggested.get("config") or {})
                    hypothesis = suggested.get("hypothesis")
                    if normalized_mode:
                        suggested_config["mode"] = normalized_mode
                    if hypothesis:
                        suggested_config["hypothesis"] = hypothesis
                    # Modes that require result_ids — resolve them automatically
                    if normalized_mode in (
                        "investigation",
                        "validation",
                    ) and not suggested_config.get("result_ids"):
                        _tier = (
                            "screening"
                            if normalized_mode == "investigation"
                            else "investigation"
                        )
                        _TIER_SQL = {
                            "screening": "SELECT result_id FROM leaderboard WHERE tier = ? AND screening_passed = 1 ORDER BY screening_loss_ratio ASC LIMIT 20",
                            "investigation": "SELECT result_id FROM leaderboard WHERE tier = ? AND investigation_passed = 1 ORDER BY investigation_loss_ratio ASC LIMIT 20",
                        }
                        _tier_rows = nb.conn.execute(
                            _TIER_SQL[_tier], (_tier,)
                        ).fetchall()
                        _rids = [r["result_id"] for r in _tier_rows if r["result_id"]]
                        suggested_config["result_ids"] = _rids

                    if normalized_mode in ("investigation", "validation"):
                        _requested = normalize_result_ids(
                            suggested_config.get("result_ids", [])
                        )
                        _eligibility = build_start_mode_eligibility(
                            nb, normalized_mode, _requested
                        )
                        _eligible = _eligibility.get("eligible_result_ids") or []
                        if _eligible:
                            suggested_config["result_ids"] = _eligible
                        else:
                            # No actionable candidates under start-mode guardrails — downgrade to continuous
                            normalized_mode = "continuous"
                            action_key = "continuous"
                            _hypothesis = suggested_config.get("hypothesis")
                            suggested_config = {
                                "mode": "continuous",
                                "model_source": "mixed",
                            }
                            if _hypothesis:
                                suggested_config["hypothesis"] = _hypothesis

                    suggested_config = augment_sparse_action_config(
                        suggested_config,
                        normalized_mode,
                        sparse_coverage_data,
                    )
                    return jsonify(
                        {
                            "briefing": ai_briefing["briefing_text"],
                            "action": action_key or normalized_mode or "continuous",
                            "action_label": briefing_action_label(
                                normalized_mode, hypothesis
                            ),
                            "action_rationale": suggested.get("reasoning", ""),
                            "ai_powered": True,
                            "confidence": ai_briefing.get("confidence", 0.5),
                            "suggested_config": suggested_config or None,
                            "evidence": recommendation_evidence,
                            "data": data_block,
                            "compression_opportunities": compression_opportunities,
                            "ref_comparison": ref_comparison,
                        }
                    )
                if fallback_reason is None:
                    fallback_reason = "llm_empty_response"
            except Exception as e:
                logger.warning(f"LLM briefing unavailable, using deterministic: {e}")
                err_msg = str(e)[:120]
                fallback_reason = f"llm_error:{type(e).__name__}: {err_msg}"

            # --- Deterministic fallback: build briefing sentences ---
            sentences = []
            if total_exp > 0:
                sentences.append(
                    f"Across {total_exp} experiments, {total_progs:,} architectures "
                    f"have been evaluated with {s1_survivors} stage-1 survivors "
                    f"({s1_survivors / max(total_progs, 1) * 100:.1f}% overall pass rate)."
                )

            # 2. Recent performance
            if avg_recent_s1 is not None:
                n_recent = len(recent_s1_rates)
                sentences.append(
                    f"The last {n_recent} completed experiment{'s' if n_recent != 1 else ''} "
                    f"averaged a {avg_recent_s1 * 100:.1f}% S1 pass rate."
                )

            # 3. Learning trajectory
            if trend == "improving" and slope is not None:
                sentences.append(
                    f"The system is learning — S1 rate is improving at "
                    f"+{abs(slope) * 100:.2f} percentage points per experiment."
                )
            elif trend == "declining" and slope is not None:
                sentences.append(
                    f"S1 rate is declining ({slope * 100:.2f} pp/experiment). "
                    f"Consider switching search strategy or trying evolution mode."
                )
            elif trend == "plateaued":
                sentences.append(
                    "S1 rate has plateaued — a novelty search or evolution run "
                    "could help escape the current local optimum."
                )

            # 4. Pipeline state
            pipeline_parts = []
            if screening > 0:
                pipeline_parts.append(f"{screening} at screening")
            if investigation > 0:
                pipeline_parts.append(f"{investigation} under investigation")
            if validation > 0:
                pipeline_parts.append(f"{validation} in validation")
            if breakthrough > 0:
                pipeline_parts.append(
                    f"{breakthrough} breakthrough{'s' if breakthrough != 1 else ''}"
                )
            if pipeline_parts:
                sentences.append(f"Candidate pipeline: {', '.join(pipeline_parts)}.")

            compressed_share = float(
                compression_summary.get("compressed_test_share") or 0.0
            )
            compressed_survival = float(
                compression_summary.get("compressed_survival_rate") or 0.0
            )
            if compression_summary:
                sentences.append(
                    "Compression coverage: "
                    f"{compressed_share * 100:.1f}% of tested candidates use compact techniques; "
                    f"compressed survival is {compressed_survival * 100:.1f}%."
                )

            sparse_n = int(sparse_evidence.get("n_sparse_programs") or 0)
            if sparse_n > 0:
                sparse_density = float(sparse_evidence.get("avg_density_mean") or 0.0)
                sparse_nm = sparse_evidence.get("avg_nm_compliance")
                sparse_fragment = f"Sparse telemetry: {sparse_n} runs with mean density {sparse_density * 100:.1f}%"
                if sparse_nm is not None:
                    sparse_fragment += f", N:M compliance {float(sparse_nm) * 100:.1f}%"
                sparse_fragment += "."
                sentences.append(sparse_fragment)

            # 5. Last experiment outcome
            if completed:
                last = completed[0]
                last_s1 = last.get("n_stage1_passed") or 0
                last_gen = last.get("n_programs_generated") or 0
                last_loss = last.get("best_loss_ratio")
                last_id = last.get("experiment_id", "")[:8]
                parts = [f"Last experiment ({last_id}): {last_s1}/{last_gen} passed S1"]
                if last_loss is not None:
                    parts.append(f"best loss {last_loss:.4f}")
                aria_sum = last.get("aria_summary")
                if aria_sum:
                    parts.append(f"— {aria_sum}")
                sentences.append(". ".join(parts) + ".")

            # 6. Data-driven diversity analysis
            try:
                # Op category distribution from learning log
                op_rows = nb.conn.execute(
                    "SELECT op_name, s1_passes, total_uses FROM op_success_rates "
                    "WHERE total_uses >= 5 ORDER BY "
                    "CAST(s1_passes AS REAL) / CAST(total_uses AS REAL) DESC LIMIT 3"
                ).fetchall()
                if op_rows:
                    top_ops = [
                        f"{r['op_name']} ({r['s1_passes']}/{r['total_uses']})"
                        for r in op_rows
                    ]
                    sentences.append(f"Top-performing operators: {', '.join(top_ops)}.")

                # Failure mode analysis
                failure_rows = nb.conn.execute(
                    "SELECT stage_at_death, COUNT(*) as cnt FROM program_results "
                    "WHERE stage1_passed = 0 AND stage_at_death IS NOT NULL "
                    "GROUP BY stage_at_death ORDER BY cnt DESC LIMIT 2"
                ).fetchall()
                if failure_rows:
                    failure_parts = [
                        f"{r['stage_at_death']} ({r['cnt']})" for r in failure_rows
                    ]
                    sentences.append(
                        f"Dominant failure stages: {', '.join(failure_parts)}."
                    )

                # Architecture diversity check
                unique_fps = nb.conn.execute(
                    "SELECT COUNT(DISTINCT SUBSTR(graph_fingerprint, 1, 8)) "
                    "FROM leaderboard"
                ).fetchone()[0]
                total_leaderboard = (
                    screening + investigation + validation + breakthrough
                )
                if unique_fps is not None and total_leaderboard > 0:
                    diversity_ratio = unique_fps / total_leaderboard
                    if diversity_ratio < 0.5:
                        sentences.append(
                            f"Warning: only {unique_fps} unique architecture "
                            f"families in {total_leaderboard} "
                            f"leaderboard entries — search may be converging."
                        )
            except Exception:
                pass  # Analytics are optional enhancements

            # 7. Reference architecture comparison
            try:
                if ref_comparison and ref_comparison.get("beats_all_references"):
                    margin = ref_comparison.get("margin_pct", 0)
                    sentences.append(
                        f"Milestone: a synthesized architecture now beats ALL "
                        f"reference baselines by {margin}%."
                    )
                elif ref_comparison and ref_comparison.get("references"):
                    best_ref = ref_comparison["best_reference_score"]
                    sentences.append(
                        f"Best reference baseline score: {best_ref:.1f}. "
                        f"No synthesized model has surpassed it yet."
                    )
            except Exception:
                pass

            briefing = " ".join(sentences)

            # --- Determine recommended action ---
            action = None
            action_label = None
            action_rationale = None
            screening_result_ids = []

            if breakthrough > 0:
                action = "export_breakthrough"
                action_label = "Export Breakthrough Report"
                action_rationale = (
                    f"{breakthrough} candidate{'s have' if breakthrough != 1 else ' has'} "
                    f"reached breakthrough tier — ready for publication review."
                )
            elif compressed_share < 0.2 and total_exp >= 3:
                action = "compact_synthesis"
                action_label = "Run Compactness-Focused Synthesis"
                action_rationale = (
                    "Compression techniques are underexplored in this campaign. "
                    "Run a compactness-focused synthesis batch to improve model efficiency coverage."
                )
            elif sparse_coverage_overview.get("below_target") and total_exp >= 3:
                sparse_share = float(
                    sparse_coverage_overview.get("sparse_share") or 0.0
                )
                sparse_survival = float(
                    sparse_coverage_overview.get("sparse_survival_rate") or 0.0
                )
                target_share = float(
                    sparse_coverage_overview.get("target_share") or 0.15
                )
                action = "novelty_search"
                action_label = "Run Sparse-Focused Novelty Search"
                action_rationale = (
                    f"Sparse coverage is below target ({sparse_share * 100:.1f}% < {target_share * 100:.0f}%) "
                    f"with {sparse_survival * 100:.1f}% sparse survival. "
                    "Run novelty search with sparse-focused morphological sampling to explore high-upside sparse candidates."
                )
            elif validation > 0 and screening == 0 and investigation == 0:
                action = "monitor_validation"
                action_label = "Review Validation Progress"
                action_rationale = (
                    f"{validation} candidate{'s are' if validation != 1 else ' is'} "
                    f"in validation. Monitor results before starting new experiments."
                )
            elif screening > 0:
                inv_failed = nb.conn.execute(
                    "SELECT COUNT(*) FROM leaderboard "
                    "WHERE tier = 'investigation' AND investigation_passed = 0"
                ).fetchone()[0]
                # Fetch actual result_ids for screening survivors
                screening_rows = nb.conn.execute(
                    "SELECT result_id FROM leaderboard "
                    "WHERE tier = 'screening' AND screening_passed = 1 "
                    "AND COALESCE(is_reference, 0) = 0 "
                    "ORDER BY composite_score DESC LIMIT 20"
                ).fetchall()
                screening_candidate_ids = [
                    r["result_id"] for r in screening_rows if r["result_id"]
                ]
                screening_result_ids = []
                if screening_candidate_ids:
                    screening_eligibility = build_start_mode_eligibility(
                        nb,
                        "investigation",
                        screening_candidate_ids,
                    )
                    screening_result_ids = (
                        screening_eligibility.get("eligible_result_ids") or []
                    )
                if not screening_result_ids:
                    # No actionable screening survivors — fall through to default
                    action = "continuous"
                    action_label = "Continue Research"
                    action_rationale = (
                        "Screening survivors exist but are not currently eligible for investigation reruns. "
                        "Continue generating new architectures."
                    )
                else:
                    action = "investigate"
                    action_label = (
                        f"Investigate {len(screening_result_ids)} Screening "
                        f"Survivor{'s' if len(screening_result_ids) != 1 else ''}"
                    )
                    rationale_parts = [
                        f"{len(screening_result_ids)} candidate{'s' if len(screening_result_ids) != 1 else ''} passed "
                        f"screening and "
                        f"{'are' if len(screening_result_ids) != 1 else 'is'} awaiting deeper investigation"
                    ]
                    if inv_failed > 0:
                        rationale_parts.append(
                            f"({inv_failed} prior investigation"
                            f"{'s' if inv_failed != 1 else ''} "
                            f"failed — fresh candidates may outperform)"
                        )
                    if avg_recent_s1 is not None:
                        rationale_parts.append(
                            f"with recent {avg_recent_s1 * 100:.0f}% hit rate"
                        )
                    action_rationale = ", ".join(rationale_parts) + "."
            elif total_exp == 0:
                action = "start_first"
                action_label = "Run First Experiment"
                action_rationale = (
                    "No experiments yet. Start a mixed continuous run to begin "
                    "exploring the architecture space."
                )
            elif trend == "declining" or (
                len(recent_s1_rates) >= 3 and all(r == 0 for r in recent_s1_rates[:3])
            ):
                action = "novelty_search"
                action_label = "Try Evolution / Novelty Search"
                action_rationale = (
                    "Recent experiments are underperforming. An evolution or "
                    "novelty-driven search can escape the current local minimum."
                )
            else:
                action = "continuous"
                action_label = "Continue Research"
                action_rationale = (
                    "The pipeline is active and the system is "
                    + ("learning" if trend == "improving" else "exploring")
                    + ". Continue generating and evaluating new architectures."
                )

            # Build deterministic suggested_config from action
            det_mode_map = {
                "investigate": "investigation",
                "continuous": "continuous",
                "start_first": "continuous",
                "novelty_search": "novelty",
                "compact_synthesis": "synthesis",
                "export_breakthrough": None,
                "monitor_validation": None,
            }
            det_mode = det_mode_map.get(action, "continuous")
            if action == "compact_synthesis":
                det_config = {
                    "mode": "synthesis",
                    "model_source": "mixed",
                    "morph_ratio": 0.85,
                    "max_depth": 5,
                    "max_ops": 8,
                    "math_space_weight": 1.8,
                    "residual_prob": 0.85,
                    "n_programs": 80,
                }
            elif action == "novelty_search" and sparse_coverage_overview.get(
                "below_target"
            ):
                det_config = {
                    "mode": "novelty",
                    "model_source": "mixed",
                    "morph_ratio": 0.8,
                    "morph_focus_sparse": True,
                    "morph_sparse_weight_storage": "semi_structured_2_4",
                    "use_synthesized_training": True,
                    "math_space_weight": 2.2,
                    "max_depth": 6,
                    "max_ops": 10,
                    "n_programs": 120,
                }
            elif action == "investigate" and screening_result_ids:
                det_config = {
                    "mode": "investigation",
                    "model_source": "mixed",
                    "result_ids": screening_result_ids,
                }
            else:
                det_config = (
                    {"mode": det_mode, "model_source": "mixed"} if det_mode else None
                )

            det_config = (
                augment_sparse_action_config(
                    det_config,
                    det_config.get("mode")
                    if isinstance(det_config, dict)
                    else det_mode,
                    sparse_coverage_data,
                )
                if isinstance(det_config, dict)
                else det_config
            )

            return jsonify(
                {
                    "briefing": briefing,
                    "action": action,
                    "action_label": action_label,
                    "action_rationale": action_rationale,
                    "ai_powered": False,
                    "fallback_reason": fallback_reason,
                    "suggested_config": det_config,
                    "evidence": recommendation_evidence,
                    "data": data_block,
                    "compression_opportunities": compression_opportunities,
                    "ref_comparison": ref_comparison,
                }
            )
        except Exception as e:
            logger.error(f"Error in /api/strategy/briefing: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()
