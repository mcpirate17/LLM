"""programs API route registration."""
from __future__ import annotations

import functools
import time
import datetime
from flask import jsonify, request, Response
from ..json_utils import json_safe as _json_safe
from ..notebook import LabNotebook
from .deps import ApiRouteContext, install_legacy_symbols

def register_programs_routes(app, context: ApiRouteContext):
    install_legacy_symbols(globals(), context)

    @app.route("/api/programs/<result_id>")
    def api_program_detail(result_id):
        """Full program detail with parsed graph JSON + fingerprint + all metrics."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            # Include training curve availability flag
            try:
                curve = nb.get_training_curve(result_id)
                program["has_training_curve"] = len(curve) > 0
            except Exception:
                program["has_training_curve"] = False

            # Try LLM explanation of fingerprint (non-critical)
            try:
                ctx = build_program_context(program)
                explanation = aria.explain_fingerprint(ctx)
                if explanation:
                    program["llm_explanation"] = explanation
            except Exception as e:
                logger.debug(f"LLM fingerprint explanation failed for {result_id}: {e}")

            program = _enrich_program_detail(nb, program)

            try:
                program["lineage_chain"] = _program_lineage_chain(nb, result_id)
            except Exception:
                program["lineage_chain"] = []

            return jsonify(_json_safe(program))
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/programs/<result_id>/lineage")
    def api_program_lineage(result_id: str):
        """Program lineage chain for refinement traceability."""
        nb = LabNotebook(notebook_path)
        try:
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404
            chain = _program_lineage_chain(nb, result_id)
            return jsonify(_json_safe({
                "result_id": result_id,
                "lineage_chain": chain,
                "depth": len(chain),
            }))
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}/lineage: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/programs/<result_id>/refine-analysis")
    def api_program_refine_analysis(result_id):
        """Data-driven refinement analysis for a program."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics, RefinementAnalyzer

            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            analytics = ExperimentAnalytics(nb)
            analyzer = RefinementAnalyzer(analytics)
            analysis = analyzer.analyze_program_for_refinement(result_id, program)
            return jsonify(_json_safe(analysis))
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}/refine-analysis: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/programs/<result_id>/morph", methods=["POST"])
    def api_program_morph(result_id):
        """Generate scored mutation candidates for a program.

        Request JSON:
            intent: str — quality|compression|sparsity|novelty|balanced (default: balanced)
            n_candidates: int — number of candidates to return (default: 5, max: 20)

        Returns top-N mutation candidates ranked by intent score, with op diffs
        and score breakdowns. No training or eval — fast, synchronous, <2s.
        """
        nb = LabNotebook(notebook_path)
        try:
            import random as _random
            from ..synthesis.grammar import GrammarConfig
            from ..synthesis.serializer import graph_from_json, graph_to_json
            from ..synthesis.validator import validate_graph
            from ..search.evolution import _mutate_graph

            # Import workflow converter for aria_designer format
            try:
                import sys as _sys
                _designer_root = str(Path(__file__).resolve().parents[2] / "aria_designer")
                if _designer_root not in _sys.path:
                    _sys.path.insert(0, _designer_root)
                from runtime.importer import graph_to_workflow as _graph_to_workflow
            except ImportError:
                _graph_to_workflow = None

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

            # Get grammar and op success rates for scoring
            grammar = GrammarConfig()
            op_success: dict = {}
            try:
                for row in nb.get_op_success_rates():
                    n_used = float(row.get("n_used") or 0)
                    n_s1 = float(row.get("n_stage1_passed") or 0)
                    if n_used > 0:
                        op_success[str(row.get("op_name"))] = n_s1 / n_used
            except Exception:
                pass

            # Optionally apply analysis-driven grammar hints
            analysis_data = None
            if body.get("use_analysis"):
                try:
                    from .analytics import ExperimentAnalytics, RefinementAnalyzer
                    analytics = ExperimentAnalytics(nb)
                    analyzer = RefinementAnalyzer(analytics)
                    analysis_data = analyzer.analyze_program_for_refinement(result_id, program)
                    recipe = analysis_data.get("recipe", {})
                    hints = recipe.get("grammar_hints", {})
                    for op_name in hints.get("exclude_ops", []):
                        grammar.excluded_ops = grammar.excluded_ops | {op_name}
                    for op_name, mult in hints.get("boost_ops", {}).items():
                        current = grammar.op_weights.get(op_name, 1.0)
                        grammar.op_weights[op_name] = min(3.0, current * mult)
                except Exception as e:
                    logger.warning("Morph: analysis hint application failed: %s", e)

            # Generate mutation pool
            rng = _random.Random(hash((result_id, intent, time.time())))
            pool_size = n_candidates * 4  # oversample then pick top-N
            candidates = []
            seen_fps = set()
            parent_ops = sorted(set(
                str(n.op_name) for n in parent_graph.nodes.values() if not n.is_input
            ))

            for _ in range(pool_size):
                try:
                    child = _mutate_graph(parent_graph, grammar, rng)
                except Exception:
                    continue
                
                # Z15: Prune dead branches (unreachable nodes) before validation 
                # to prevent redundant complexity from bloat mutations.
                child.prune_dead_branches()
                
                validation = validate_graph(child, max_ops=30, max_depth=20)
                if not validation.valid:
                    continue
                fp = child.fingerprint()
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)

                # Score using runner's scoring logic (inline for speed)
                child_ops_list = [
                    str(n.op_name) for n in child.nodes.values() if not n.is_input
                ]
                n_ops = max(1, int(child.n_ops()))
                depth = max(1, int(child.depth()))
                params = max(1.0, float(child.n_params_estimate()))
                unique_ops = len(set(child_ops_list))

                import math as _math
                learned_quality = 0.5
                if child_ops_list:
                    learned_quality = sum(op_success.get(op, 0.5) for op in child_ops_list) / len(child_ops_list)
                compression_proxy = 1.0 / (1.0 + _math.log1p(params) + 0.25 * n_ops + 0.15 * depth)
                novelty_proxy = min(1.0, (unique_ops / max(1, n_ops)) + (0.1 if depth >= 4 else 0.0))
                sparse_hint_ops = ("sparse", "gate", "topk", "mask", "threshold", "skip", "mixture")
                sparse_op_bonus = 0.0
                if child_ops_list:
                    sparse_op_bonus = sum(
                        1.0 for op in child_ops_list if any(t in op.lower() for t in sparse_hint_ops)
                    ) / len(child_ops_list)
                sparsity_proxy = min(1.0, 0.7 * compression_proxy + 0.3 * sparse_op_bonus)
                parent_novelty = float(program.get("novelty_score") or 0.0)
                parent_quality = 1.0 - float(program.get("loss_ratio") or 1.0)

                if intent == "quality":
                    score = 0.60 * learned_quality + 0.25 * parent_quality + 0.15 * compression_proxy
                elif intent == "compression":
                    score = 0.60 * compression_proxy + 0.25 * learned_quality + 0.15 * parent_quality
                elif intent == "sparsity":
                    score = 0.60 * sparsity_proxy + 0.25 * learned_quality + 0.15 * compression_proxy
                elif intent == "novelty":
                    score = 0.55 * novelty_proxy + 0.25 * learned_quality + 0.20 * parent_novelty
                else:
                    score = 0.35 * learned_quality + 0.25 * compression_proxy + 0.20 * novelty_proxy + 0.20 * max(parent_quality, parent_novelty)

                child_ops = sorted(set(child_ops_list))
                added_ops = [op for op in child_ops if op not in parent_ops]
                removed_ops = [op for op in parent_ops if op not in child_ops]

                # Convert to workflow format for designer loading
                workflow_json = None
                if _graph_to_workflow:
                    try:
                        wf = _graph_to_workflow(child, workflow_id=fp[:12], name=f"morph_{fp[:8]}")
                        workflow_json = wf
                    except Exception:
                        pass

                candidates.append({
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
                    },
                })

            candidates.sort(key=lambda c: c["score"], reverse=True)
            top = candidates[:n_candidates]

            return jsonify({
                "result_id": result_id,
                "intent": intent,
                "source_ops": parent_ops,
                "source_fingerprint": parent_graph.fingerprint(),
                "n_generated": len(seen_fps),
                "candidates": top,
            })
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}/morph: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/programs/<result_id>/external-benchmarks", methods=["POST"])
    def api_program_external_benchmarks(result_id):
        """Attach external benchmark scores to a program result."""
        nb = LabNotebook(notebook_path)
        try:
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, (dict, list)):
                return jsonify({"error": "Payload must be a JSON object or list."}), 400
            ok = nb.set_external_benchmarks(result_id, payload)
            if not ok:
                return jsonify({"error": "Program result not found or payload invalid."}), 404
            return jsonify({"status": "ok", "result_id": result_id})
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}/external-benchmarks: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/programs/<result_id>/backfill-metrics", methods=["POST"])
    def api_program_backfill_metrics(result_id):
        """Recompute missing metrics (fingerprint, novelty, spectral, quantization, etc.) for a program."""
        nb = LabNotebook(notebook_path)
        try:
            program = nb.get_program_detail(result_id)
            if not program:
                return jsonify({"error": "Program not found"}), 404

            # Build the row shape that backfill_entry expects
            leaderboard = nb.conn.execute(
                "SELECT entry_id, screening_loss_ratio FROM leaderboard WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if not leaderboard:
                return jsonify({"error": "No leaderboard entry for this result_id"}), 404

            lb = dict(leaderboard)
            row = {
                "result_id": result_id,
                "graph_json": program.get("graph_json"),
                "entry_id": lb["entry_id"],
                "screening_loss_ratio": lb.get("screening_loss_ratio"),
            }

            from ..tools.backfill_metrics import backfill_entry
            body = request.get_json(silent=True) or {}
            device = str(body.get("device", "cpu"))
            result = backfill_entry(row, device=device)
            return jsonify({"status": "ok", "result_id": result_id, "backfill": result})
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}/backfill-metrics: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/programs/<result_id>/backfill-loss", methods=["POST"])
    def api_program_backfill_loss(result_id):
        """Recompute discovery_loss and validation_loss for a program by rebuilding + evaluating."""
        nb = LabNotebook(notebook_path)
        try:
            program = nb.get_program_detail(result_id)
            if not program:
                return jsonify({"error": "Program not found"}), 404
            graph_json = program.get("graph_json")
            if not graph_json:
                return jsonify({"error": "No graph_json for this program"}), 400
            initial_loss = program.get("initial_loss")
            if not initial_loss:
                return jsonify({"error": "No initial_loss recorded — cannot compute ratios"}), 400

            # Load experiment config for model params
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
            import torch.nn.functional as F

            body = request.get_json(silent=True) or {}
            device = str(body.get("device", "cpu"))
            dev = torch.device(device)

            from ..synthesis.serializer import graph_from_json as _gfj
            graph = _gfj(graph_json)
            graph_dim = getattr(graph, "model_dim", None)
            if graph_dim and config.model_dim != graph_dim:
                config.model_dim = int(graph_dim)

            from .native_runner import compile_model_native_first as _compile
            layer_graphs = [graph] * config.n_layers
            model = _compile(layer_graphs, vocab_size=config.vocab_size, max_seq_len=config.max_seq_len)
            model = model.to(dev).eval()

            seq_len = min(128, config.max_seq_len)
            updates = {}

            # Discovery loss (random tokens)
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
                    disc_ratio = disc_loss / max(float(initial_loss), 1e-6)
                    updates["discovery_loss"] = disc_loss
                    updates["discovery_loss_ratio"] = disc_ratio
            except Exception as e:
                updates["discovery_loss_error"] = str(e)

            # Validation loss (heldout corpus split, if corpus/HF mode)
            data_mode = str(config.data_mode or "random").strip().lower()
            if data_mode in ("corpus", "huggingface"):
                try:
                    runner = _get_runner(notebook_path)
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
                                    batch_size=4, seq_len=seq_len,
                                    generator=gen, device=dev, split="val",
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
                            val_ratio = val_loss / max(float(initial_loss), 1e-6)
                            updates["validation_loss"] = val_loss
                            updates["validation_loss_ratio"] = val_ratio
                            final_loss = program.get("final_loss")
                            if final_loss:
                                updates["generalization_gap"] = val_loss - float(final_loss)
                except Exception as e:
                    updates["validation_loss_error"] = str(e)

            del model
            if device != "cpu":
                torch.cuda.empty_cache()

            # Write updates to DB (both program_results and leaderboard)
            if updates:
                db_updates = {k: v for k, v in updates.items() if not k.endswith("_error")}
                if db_updates:
                    set_parts = [f"{k} = ?" for k in db_updates]
                    vals = list(db_updates.values()) + [result_id]
                    nb.conn.execute(
                        f"UPDATE program_results SET {', '.join(set_parts)} WHERE result_id = ?",
                        vals,
                    )
                    # Mirror to leaderboard table (only cols that exist there)
                    lb_cols = {c[1] for c in nb.conn.execute("PRAGMA table_info(leaderboard)").fetchall()}
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
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}/backfill-loss: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/programs")
    def api_programs():
        """List top programs."""
        n = request.args.get("n", 20, type=int)
        sort_by = request.args.get("sort", "novelty_score")
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            programs = nb.get_top_programs(n, sort_by)
            _annotate_qkv_usage(programs, analytics)
            return jsonify(_json_safe(programs))
        except Exception as e:
            logger.error(f"Error in /api/programs: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/programs/<result_id>/training-curve")
    def api_training_curve(result_id):
        """Per-step training data for a program."""
        nb = LabNotebook(notebook_path)
        try:
            curve = nb.get_training_curve(result_id)
            return jsonify(curve)
        except Exception as e:
            logger.error(f"Error in training-curve: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/programs/purge-junk", methods=["POST"])
    def api_purge_junk_programs():
        """Purge Stage 0 failure program results that carry no useful data."""
        dry_run = True
        if request.is_json and request.json:
            dry_run = request.json.get("dry_run", True)
        nb = LabNotebook(notebook_path)
        try:
            result = nb.purge_junk_programs(dry_run=dry_run)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error purging junk programs: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


