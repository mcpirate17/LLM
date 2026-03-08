"""Read/data API route registration for scientist API."""
from __future__ import annotations

from ..json_utils import json_safe as _json_safe

from .deps import ApiRouteContext, install_legacy_symbols

def register_read_routes(app, context: ApiRouteContext):
    install_legacy_symbols(globals(), context)
    @app.route("/api/native-profile/v2/data")
    def api_native_runner_profile():
        """Return per-node profiling data from the most recent native execution."""
        try:
            from ..native_runner import get_native_profile, _try_import_rust_scheduler

            rust = _try_import_rust_scheduler()
            profiling_enabled = bool(
                rust is not None
                and hasattr(rust, "profiler_enabled")
                and rust.profiler_enabled()
            )

            profile = get_native_profile()
            if profile is not None:
                node_profiles = list(profile.get("node_profiles", []))
                total_duration_us = sum(
                    float(p.get("duration_us", 0)) for p in node_profiles
                )
                return jsonify({
                    "status": "ok",
                    "enabled": profiling_enabled,
                    "node_profiles": node_profiles,
                    "peak_memory_bytes": int(profile.get("peak_memory_bytes", 0)),
                    "total_duration_us": total_duration_us,
                })
            else:
                return jsonify({
                    "status": "ok",
                    "enabled": profiling_enabled,
                    "node_profiles": [],
                    "peak_memory_bytes": 0,
                    "total_duration_us": 0.0,
                })
        except Exception as e:
            logger.error(f"Error in /api/native-profile/v2/data: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/native-profile/v2/enable", methods=["POST"])
    def api_native_runner_profile_enable():
        """Toggle native kernel profiling on or off."""
        try:
            from ..native_runner import enable_native_profiling, _try_import_rust_scheduler

            body = request.get_json(silent=True) or {}
            enable = bool(body.get("enable", True))

            result = enable_native_profiling(enable)

            rust = _try_import_rust_scheduler()
            now_enabled = bool(
                rust is not None
                and hasattr(rust, "profiler_enabled")
                and rust.profiler_enabled()
            )

            return jsonify({
                "status": "ok",
                "requested": enable,
                "enabled": now_enabled,
                "accepted": result,
            })
        except Exception as e:
            logger.error(f"Error in /api/native-profile/v2/enable: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/status")
    def api_status():
        """Get Aria's current status and dashboard summary."""
        nb = LabNotebook(notebook_path)
        runner = _get_runner(notebook_path)
        aria = get_aria()
        try:
            summary = nb.get_dashboard_summary()
            
            # Check if ANY experiment is marked as running in the DB
            db_running = nb.conn.execute(
                "SELECT experiment_id FROM experiments WHERE status = 'running' LIMIT 1"
            ).fetchone()
            is_running = runner.is_running or (db_running is not None)
            
            progress_payload = _with_native_runner_progress(runner.progress.to_dict())
            # If DB says running but local runner is idle, try to pull latest progress from DB
            if db_running and not runner.is_running:
                exp_id = db_running[0]
                db_stats = nb.conn.execute(
                    "SELECT n_programs_generated, n_stage1_passed FROM experiments WHERE experiment_id = ?",
                    (exp_id,)
                ).fetchone()
                if db_stats:
                    progress_payload["experiment_id"] = exp_id
                    progress_payload["current_program"] = db_stats[0]
                    progress_payload["stage1_passed"] = db_stats[1]
                    progress_payload["status"] = "running (background)"

            trigger = _get_run_trigger_snapshot(progress_payload.get("experiment_id"))
            progress_payload["run_trigger_source"] = trigger.get("source")
            progress_payload["run_trigger"] = trigger
            return jsonify({
                "aria": aria.get_status(db_summary=summary),
                "summary": summary,
                "is_running": is_running,
                "progress": progress_payload,
                "native_runner": progress_payload.get("native_runner"),
                "run_trigger_source": trigger.get("source"),
                "run_trigger": trigger,
            })
        except Exception as e:
            logger.error(f"Error in /api/status: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments")
    def api_experiments():
        """List experiments (newest first)."""
        n = request.args.get("n", type=int)
        if n is None:
            n = request.args.get("limit", type=int)
        if n is None:
            n = 200
        n = max(1, min(n, 5000))
        offset = request.args.get("offset", 0, type=int)
        offset = max(0, min(offset, 1_000_000))
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_recent_experiments(n, offset=offset))
        except Exception as e:
            logger.error(f"Error in /api/experiments: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>")
    def api_experiment_detail(experiment_id):
        """Get experiment details with entries and per-experiment programs."""
        nb = LabNotebook(notebook_path)
        try:
            exp = nb.get_experiment(experiment_id)
            if exp is None:
                return jsonify({"error": "Not found"}), 404
            entries = nb.get_entries(experiment_id=experiment_id)
            programs = nb.get_program_results(experiment_id)
            prereg = nb.get_preregistration_for_experiment(experiment_id)
            deviations = nb.get_preregistration_deviations(experiment_id)
            payload = {
                "experiment": exp,
                "entries": entries,
                "programs": programs,
                "preregistration": prereg,
                "preregistration_deviations": deviations,
            }
            return jsonify(_json_safe(payload))
        except Exception as e:
            logger.error(f"Error in /api/experiments/{experiment_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/programs")
    def api_experiment_programs(experiment_id):
        """All programs for an experiment (not just S1 survivors)."""
        nb = LabNotebook(notebook_path)
        try:
            programs = nb.get_program_results(experiment_id)
            return jsonify(_json_safe(programs))
        except Exception as e:
            logger.error(f"Error in /api/experiments/{experiment_id}/programs: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

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
            from ..analytics import ExperimentAnalytics, RefinementAnalyzer

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
                    from ..analytics import ExperimentAnalytics, RefinementAnalyzer
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

            body = request.get_json(silent=True) or {}
            device = str(body.get("device", "cpu"))
            dev = torch.device(device)

            from ..synthesis.serializer import graph_from_json as _gfj
            graph = _gfj(graph_json)
            graph_dim = getattr(graph, "model_dim", None)
            if graph_dim and config.model_dim != graph_dim:
                config.model_dim = int(graph_dim)

            from ..native_runner import compile_model_native_first as _compile
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

    @app.route("/api/recompute-failure-signatures", methods=["POST"])
    def api_recompute_failure_signatures():
        """Delete and rebuild failure_signatures using S1-only failures."""
        nb = LabNotebook(notebook_path)
        try:
            count = nb.recompute_failure_signatures()
            return jsonify({"status": "ok", "signatures_created": count})
        except Exception as e:
            logger.error(f"Error in /api/recompute-failure-signatures: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/reset-op-stats", methods=["POST"])
    def api_reset_op_stats():
        """Reset op_success_rates for specific ops so they get a fresh start.

        POST body: {"ops": ["op1", "op2", ...]}
        If no ops specified, resets all ops with 0 S1 passes.
        """
        nb = LabNotebook(notebook_path)
        try:
            data = request.get_json(silent=True) or {}
            ops = data.get("ops")
            if ops:
                for op_name in ops:
                    nb.conn.execute(
                        "DELETE FROM op_success_rates WHERE op_name = ?",
                        (op_name,),
                    )
                nb.conn.commit()
                count = len(ops)
            else:
                cur = nb.conn.execute(
                    "DELETE FROM op_success_rates WHERE n_stage1_passed = 0 AND n_used >= 5"
                )
                nb.conn.commit()
                count = cur.rowcount
            return jsonify({"status": "ok", "ops_reset": count})
        except Exception as e:
            logger.error(f"Error in /api/reset-op-stats: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/healer/tasks")
    def api_healer_tasks():
        """List recent Code Healer tasks."""
        nb = LabNotebook(notebook_path)
        try:
            limit = request.args.get("limit", 20, type=int)
            return jsonify(nb.get_recent_healer_tasks(limit=max(1, min(limit, 200))))
        except Exception as e:
            logger.error(f"Error in /api/healer/tasks: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/healer/tasks/<task_id>")
    def api_healer_task_detail(task_id: str):
        """Get one healer task with state history."""
        nb = LabNotebook(notebook_path)
        try:
            task = nb.get_healer_task(task_id)
            if task is None:
                return jsonify({"error": "Not found"}), 404
            return jsonify({
                "task": task,
                "events": nb.get_healer_events(task_id, limit=200),
            })
        except Exception as e:
            logger.error(f"Error in /api/healer/tasks/{task_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/failures")
    def api_failure_analysis(experiment_id):
        """Failure analysis: error distribution, stage funnel."""
        nb = LabNotebook(notebook_path)
        try:
            analysis = nb.get_failure_analysis(experiment_id)
            return jsonify(analysis)
        except Exception as e:
            logger.error(f"Error in failure analysis: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/analysis")
    def api_experiment_analysis(experiment_id):
        """LLM-generated analysis (stored or on-demand)."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            exp = nb.get_experiment(experiment_id)
            if exp is None:
                return jsonify({"error": "Not found"}), 404

            # Return stored analysis if available
            stored = exp.get("llm_analysis")
            if stored:
                return jsonify({"analysis": stored, "source": "stored"})

            # Try generating on-demand
            results = exp.get("results") or {}
            from ..llm.context import build_experiment_context
            ctx = build_experiment_context(results)
            analysis = aria.analyze_results(results, context=ctx)

            if analysis:
                # Cache it
                try:
                    nb.conn.execute(
                        "UPDATE experiments SET llm_analysis = ? WHERE experiment_id = ?",
                        (analysis, experiment_id),
                    )
                    nb.conn.commit()
                except Exception as e:
                    logger.warning("Failed caching llm_analysis for %s: %s",
                                   experiment_id, e)
                return jsonify({"analysis": analysis, "source": "generated"})

            return jsonify({"analysis": None, "source": "unavailable",
                            "reason": "No LLM backend configured"})
        except Exception as e:
            logger.error(f"Error in experiment analysis: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/trends")
    def api_trends():
        """Cross-experiment trend data for charts."""
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_experiment_trends())
        except Exception as e:
            logger.error(f"Error in /api/trends: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/trends/context")
    def api_trends_context():
        """Trend data plus adaptation-event deltas for inline linkage UI."""
        nb = LabNotebook(notebook_path)

        def _event_delta_payload(trends: List[Dict[str, Any]], event: Dict[str, Any]) -> Dict[str, Any]:
            timestamp = float(event.get("timestamp") or 0.0)
            previous = [row for row in trends if float(row.get("timestamp") or 0.0) < timestamp]
            following = [row for row in trends if float(row.get("timestamp") or 0.0) >= timestamp]

            before = previous[-3:]
            after = following[:3]

            before_ids = [str(row.get("experiment_id")) for row in before if row.get("experiment_id")]
            after_ids = [str(row.get("experiment_id")) for row in after if row.get("experiment_id")]

            def _avg(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
                values = [float(row[key]) for row in rows if row.get(key) is not None]
                if not values:
                    return None
                return sum(values) / len(values)

            before_adj_s1 = _avg(before, "adjusted_s1_pass_rate")
            after_adj_s1 = _avg(after, "adjusted_s1_pass_rate")
            before_novelty = _avg(before, "best_novelty_score")
            after_novelty = _avg(after, "best_novelty_score")
            before_loss = _avg(before, "best_loss_ratio")
            after_loss = _avg(after, "best_loss_ratio")

            return {
                "timestamp": timestamp,
                "event_type": event.get("event_type"),
                "description": event.get("description") or "Grammar weights adjusted",
                "before_window": {
                    "n_experiments": len(before),
                    "experiment_ids": before_ids,
                    "adjusted_s1_rate": before_adj_s1,
                    "best_novelty": before_novelty,
                    "best_loss_ratio": before_loss,
                },
                "after_window": {
                    "n_experiments": len(after),
                    "experiment_ids": after_ids,
                    "adjusted_s1_rate": after_adj_s1,
                    "best_novelty": after_novelty,
                    "best_loss_ratio": after_loss,
                },
                "delta": {
                    "adjusted_s1_rate": (
                        after_adj_s1 - before_adj_s1
                        if after_adj_s1 is not None and before_adj_s1 is not None
                        else None
                    ),
                    "best_novelty": (
                        after_novelty - before_novelty
                        if after_novelty is not None and before_novelty is not None
                        else None
                    ),
                    "best_loss_ratio": (
                        after_loss - before_loss
                        if after_loss is not None and before_loss is not None
                        else None
                    ),
                },
            }

        try:
            trends = nb.get_experiment_trends()
            learning_log = nb.get_learning_log(limit=300)
            adaptation_events = [
                _event_delta_payload(trends, event)
                for event in learning_log
                if event.get("event_type") == "grammar_weights_applied"
            ]
            return jsonify({
                "trends": trends,
                "adaptation_events": adaptation_events,
                "generated_at": time.time(),
            })
        except Exception as e:
            logger.error(f"Error in /api/trends/context: {e}")
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
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            programs = nb.get_top_programs(n, sort_by)
            _annotate_qkv_usage(programs, analytics)
            return jsonify(_json_safe(programs))
        except Exception as e:
            logger.error(f"Error in /api/programs: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/insights")
    def api_insights():
        """List active insights, deduplicated by content (keeps latest)."""
        category = request.args.get("category")
        nb = LabNotebook(notebook_path)
        try:
            raw = nb.get_insights(category=category, limit=200)
            return jsonify(_deduplicate_insights(raw))
        except Exception as e:
            logger.error(f"Error in /api/insights: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/insights/boost", methods=["POST"])
    def api_insights_boost():
        """Record a request to boost an insight in future experiment selection."""
        payload = request.get_json(silent=True) or {}
        insight_id = str(payload.get("insight_id") or "").strip()
        content = str(payload.get("content") or "").strip()
        category = str(payload.get("category") or "").strip()
        confidence = payload.get("confidence")
        if not insight_id:
            return jsonify({"error": "insight_id required"}), 400
        nb = LabNotebook(notebook_path)
        try:
            evidence = json.dumps({
                "insight_id": insight_id,
                "category": category or None,
                "confidence": confidence,
                "content": content[:400] if content else None,
            }, sort_keys=True)
            desc = f"Boost requested for insight {insight_id}"
            if category:
                desc += f" ({category})"
            nb.log_learning_event(
                "insight_boost",
                desc,
                evidence=evidence,
            )
            return jsonify({"status": "ok", "insight_id": insight_id})
        except Exception as e:
            logger.error(f"Error in /api/insights/boost: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/entries")
    def api_entries():
        """List notebook entries."""
        exp_id = request.args.get("experiment_id")
        entry_type = request.args.get("type")
        n = request.args.get("n", 50, type=int)
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.get_entries(
                experiment_id=exp_id, entry_type=entry_type, limit=n
            )
            return jsonify(_normalize_entries(entries))
        except Exception as e:
            logger.error(f"Error in /api/entries: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/live-feed")
    def api_live_feed():
        """List persisted live-feed events for replay in the dashboard."""
        exp_id = request.args.get("experiment_id")
        n = request.args.get("n", 100, type=int)
        nb = LabNotebook(notebook_path)
        try:
            query_limit = max(n, 1000)
            entries = nb.get_entries(
                experiment_id=exp_id,
                entry_type="live_feed",
                limit=query_limit,
            )

            # Default behavior should show a coherent experiment stream.
            # Without this, mixed cross-experiment rows can look like broken
            # generation timelines (e.g., Gen 3 -> Gen 13 with unrelated runs).
            if not exp_id:
                latest_exp_id = next(
                    (
                        entry.get("experiment_id")
                        for entry in entries
                        if entry.get("experiment_id")
                    ),
                    None,
                )
                if latest_exp_id:
                    entries = [
                        entry
                        for entry in entries
                        if entry.get("experiment_id") == latest_exp_id
                    ]

            events = []
            for entry in reversed(entries):
                evt = _entry_to_live_feed_event(entry)
                if evt is not None:
                    events.append(evt)
            if len(events) > n:
                events = events[-n:]
            return jsonify(events)
        except Exception as e:
            logger.error(f"Error in /api/live-feed: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/live-loss-curve")
    def api_live_loss_curve():
        """Return the in-memory training loss curve for the live chart."""
        if _runner is None:
            return jsonify([])
        try:
            return jsonify(_runner.get_live_loss_curve())
        except Exception as e:
            logger.error("Error in /api/live-loss-curve: %s", e)
            return jsonify([])

    @app.route("/api/metrics/<metric_name>")
    def api_metrics(metric_name):
        """Get time-series metrics."""
        exp_id = request.args.get("experiment_id")
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_metrics(metric_name, experiment_id=exp_id))
        except Exception as e:
            logger.error(f"Error in /api/metrics/{metric_name}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/dashboard")
    def api_dashboard():
        """Get all dashboard data in one call."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            summary = nb.get_dashboard_summary()

            # Add campaign/hypothesis/knowledge counts
            try:
                active_campaigns = nb.get_active_campaigns()
                total_hypotheses = nb.conn.execute(
                    "SELECT COUNT(*) FROM hypotheses"
                ).fetchone()[0]
                knowledge_entries = nb.conn.execute(
                    "SELECT COUNT(*) FROM knowledge_base WHERE status = 'active'"
                ).fetchone()[0]
                summary["active_campaigns"] = len(active_campaigns)
                summary["total_hypotheses"] = total_hypotheses
                summary["knowledge_entries"] = knowledge_entries
            except Exception as e:
                logger.warning("Failed enriching dashboard campaign metadata: %s", e)

            recent_experiments = nb.get_recent_experiments(30)
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            top_programs = nb.get_top_programs(10)
            _annotate_qkv_usage(top_programs, analytics)
            production_readiness = _compute_breakthrough_production_readiness(nb, analytics)

            data = {
                "aria": aria.get_status(db_summary=summary),
                "summary": summary,
                "recent_experiments": recent_experiments,
                "top_programs": top_programs,
                "production_readiness": production_readiness,
                "insights": _deduplicate_insights(nb.get_insights(limit=50)),
                "recent_entries": _normalize_entries(nb.get_entries(limit=20)),
                "is_running": runner.is_running,
                "progress": _with_native_runner_progress(runner.progress.to_dict()),
            }

            # Compute deltas from latest completed experiment
            try:
                completed = [e for e in recent_experiments
                             if e.get("status") == "completed"]
                if len(completed) >= 2:
                    latest = completed[0]
                    previous = completed[1]
                    data["deltas"] = {
                        "experiment_id": latest.get("experiment_id"),
                        "programs": (latest.get("n_programs_generated") or 0)
                                    - (previous.get("n_programs_generated") or 0),
                        "stage1": (latest.get("n_stage1_passed") or 0)
                                  - (previous.get("n_stage1_passed") or 0),
                        "best_loss": round(
                            (latest.get("best_loss_ratio") or 1)
                            - (previous.get("best_loss_ratio") or 1), 4
                        ) if latest.get("best_loss_ratio") else None,
                        "best_novelty": round(
                            (latest.get("best_novelty_score") or 0)
                            - (previous.get("best_novelty_score") or 0), 4
                        ) if latest.get("best_novelty_score") else None,
                    }
            except Exception:
                pass

            # Include learning trajectory trend in summary
            try:
                trajectory = analytics.learning_trajectory()
                if trajectory and trajectory.get("trend") != "insufficient_data":
                    summary["learning_trend"] = trajectory.get("trend")
                    summary["learning_slope"] = trajectory.get("slope")
                    summary["recent_s1_rate"] = trajectory.get("recent_s1_rate")
            except Exception:
                pass

            # Include latest auto-recommendation if experiment just completed
            last_rec = runner.last_recommendation
            if last_rec:
                data["last_recommendation"] = last_rec

            return jsonify(data)
        except Exception as e:
            logger.error(f"Error in /api/dashboard: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Report endpoint ──

    @app.route("/api/report")
    def api_report():
        """Consolidated research report with all data."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            fast_mode = _parse_bool_query(request.args.get("fast"), default=False)
            include_heavy = _parse_bool_query(
                request.args.get("include_heavy"),
                default=not fast_mode,
            )
            include_narrative = _parse_bool_query(
                request.args.get("include_narrative"),
                default=not fast_mode,
            )

            top_limit = 20 if not fast_mode else 12
            expanded_limit = 80 if include_heavy else 0
            recent_limit = 100 if include_heavy else 30

            data = {
                "summary": nb.get_dashboard_summary(),
                "top_programs": nb.get_report_top_programs_grouped_by_fingerprint(top_limit, sort_by="loss_ratio"),
                "top_programs_expanded": nb.get_top_programs(expanded_limit, sort_by="loss_ratio") if include_heavy else [],
                "recent_experiments": nb.get_recent_experiments(recent_limit),
                "op_success_rates": analytics.op_success_rates(),
                "failure_patterns": analytics.failure_patterns(),
                "grammar_weights": {
                    "learned": analytics.compute_grammar_weights(),
                    "default": analytics.get_current_grammar_weights(),
                    "control_comparison": analytics.control_experiment_comparison(),
                    "holdout_validation": analytics.holdout_validation(),
                    "learning_diagnostics": analytics.grammar_weight_learning_diagnostics(),
                },
                "learning_log": nb.get_learning_log(limit=20 if fast_mode else 50),
                "insights": nb.get_insights(),
                "report_mode": {
                    "fast": fast_mode,
                    "include_heavy": include_heavy,
                    "include_narrative": include_narrative,
                },
            }
            if include_heavy:
                data.update({
                    "math_family_coverage": analytics.math_family_coverage(),
                    "mathspace_operator_impact": analytics.mathspace_operator_impact(),
                    "routing_mode_comparison": analytics.routing_mode_comparison(),
                    "gating_behavior_diagnostics": analytics.gating_behavior_diagnostics(),
                    "structural_correlations": analytics.structural_correlations(),
                    "top_op_combinations": analytics.top_op_combinations(10),
                    "efficiency_frontier": analytics.efficiency_frontier(),
                    "experiment_clusters": analytics.experiment_clusters(),
                })
            learning_diagnostics = data["grammar_weights"].get("learning_diagnostics") or {}
            data["architecture_rerun_telemetry"] = {
                "unique_fingerprint_count": int(learning_diagnostics.get("unique_fingerprints") or 0),
                "total_result_rows": int(learning_diagnostics.get("total_rows") or 0),
                "repeat_result_rows": int(learning_diagnostics.get("repeat_rows") or 0),
                "rerun_ratio": float(learning_diagnostics.get("rerun_ratio") or 0.0),
                "top_fingerprint_concentration": float(learning_diagnostics.get("top_fingerprint_concentration") or 0.0),
                "weighting_mode": str(learning_diagnostics.get("mode") or "unknown"),
            }
            data["action_eligibility"] = _build_report_action_eligibility(
                nb,
                [
                    row.get("result_id")
                    for row in [*(data["top_programs"] or []), *(data["top_programs_expanded"] or [])]
                    if row.get("result_id")
                ],
            )
            _annotate_qkv_usage(data["top_programs"], analytics)
            _annotate_qkv_usage(data["top_programs_expanded"], analytics)

            expanded_by_fingerprint: Dict[str, List[Dict[str, Any]]] = {}
            for row in data["top_programs_expanded"]:
                fp = row.get("graph_fingerprint")
                if not fp:
                    continue
                expanded_by_fingerprint.setdefault(fp, []).append(row)

            grouped_rank_by_fingerprint = {
                row.get("graph_fingerprint"): index
                for index, row in enumerate(data["top_programs"], start=1)
                if row.get("graph_fingerprint")
            }
            for fp, rows in expanded_by_fingerprint.items():
                repeat_count = len(rows)
                grouped_rank = grouped_rank_by_fingerprint.get(fp)
                for repeat_index, row in enumerate(rows, start=1):
                    row["group_repeat_count"] = repeat_count
                    row["group_repeat_index"] = repeat_index
                    row["grouped_fingerprint_rank"] = grouped_rank

            data["cross_run_stability"] = _compute_cross_run_stability(
                nb, data["top_programs"]
            )
            stability_by_result = {
                candidate.get("result_id"): candidate
                for candidate in data["cross_run_stability"].get("candidates", [])
                if candidate.get("result_id")
            }
            stability_by_fingerprint = {
                candidate.get("graph_fingerprint"): candidate
                for candidate in data["cross_run_stability"].get("candidates", [])
                if candidate.get("graph_fingerprint")
            }

            fallback_stability = {
                "trend": "unknown",
                "seen_runs": 0,
                "latest_rank": None,
                "previous_rank": None,
                "rank_delta": None,
            }
            for program in [*(data["top_programs"] or []), *(data["top_programs_expanded"] or [])]:
                by_result = stability_by_result.get(program.get("result_id"))
                by_fingerprint = stability_by_fingerprint.get(program.get("graph_fingerprint"))
                program["cross_run_stability"] = by_result or by_fingerprint or fallback_stability

            # Generate narrative only when explicitly enabled
            data["narrative"] = None
            if include_narrative:
                try:
                    narrative = aria.generate_report_narrative(data)
                    data["narrative"] = narrative
                except Exception as e:
                    logger.debug(f"Report narrative generation failed: {e}")
                    data["narrative"] = None

            return jsonify(data)
        except Exception as e:
            logger.error(f"Error in /api/report: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/report/query")
    def api_report_query():
        """Scoped report payload for date/theme/trend report generation."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            start_ts = _parse_report_date(request.args.get("start_date"), end_of_day=False)
            end_ts = _parse_report_date(request.args.get("end_date"), end_of_day=True)
            theme = str(request.args.get("theme") or "all").strip().lower()
            trend = str(request.args.get("trend") or "all").strip().lower()
            include_narrative = _parse_bool_query(
                request.args.get("include_narrative"),
                default=False,
            )
            try:
                limit = int(request.args.get("limit") or 20)
            except Exception:
                limit = 20
            limit = max(5, min(120, limit))

            snapshot_query = {
                "start_date": request.args.get("start_date"),
                "end_date": request.args.get("end_date"),
                "theme": theme,
                "trend": trend,
                "limit": limit,
                "include_narrative": bool(include_narrative),
            }
            latest_completed_ts = nb.get_latest_completed_experiment_timestamp()
            snapshot_key = _build_report_snapshot_key("report_query", snapshot_query)

            if not include_narrative:
                cached = nb.get_report_snapshot(
                    snapshot_key=snapshot_key,
                    scope="report_query",
                    min_latest_completed_ts=latest_completed_ts,
                )
                if isinstance(cached, dict):
                    cached["snapshot_cache"] = {
                        "enabled": True,
                        "hit": True,
                        "key": snapshot_key,
                        "latest_completed_ts": latest_completed_ts,
                    }
                    return jsonify(cached)

            experiments = nb.get_recent_experiments(500)
            filtered_experiments = []
            for exp in experiments:
                ts = exp.get("timestamp")
                if isinstance(ts, (int, float)):
                    if start_ts is not None and ts < start_ts:
                        continue
                    if end_ts is not None and ts > end_ts:
                        continue
                if not _report_experiment_matches_trend(exp, trend):
                    continue
                filtered_experiments.append(exp)

            sort_by = "novelty_score" if trend == "high_novelty" else "loss_ratio"
            expanded = nb.get_top_programs(max(limit * 3, 120), sort_by=sort_by)
            filtered_programs: List[Dict[str, Any]] = []
            for program in expanded:
                ts = program.get("timestamp")
                if isinstance(ts, (int, float)):
                    if start_ts is not None and ts < start_ts:
                        continue
                    if end_ts is not None and ts > end_ts:
                        continue
                if not _report_program_matches_theme(program, theme):
                    continue
                filtered_programs.append(program)

            grouped = []
            seen = set()
            for row in filtered_programs:
                fp = row.get("graph_fingerprint")
                if fp and fp in seen:
                    continue
                if fp:
                    seen.add(fp)
                grouped.append(row)
                if len(grouped) >= limit:
                    break

            base_summary = nb.get_dashboard_summary()
            summary = _build_filtered_report_summary(base_summary, filtered_experiments)

            data = {
                "summary": summary,
                "top_programs": grouped,
                "top_programs_expanded": filtered_programs[: max(limit * 2, 40)],
                "recent_experiments": filtered_experiments[: max(limit * 5, 40)],
                "op_success_rates": analytics.op_success_rates(),
                "failure_patterns": analytics.failure_patterns(),
                "insights": nb.get_insights(),
                "learning_log": nb.get_learning_log(limit=30),
                "narrative": None,
                "query": {
                    "start_date": request.args.get("start_date"),
                    "end_date": request.args.get("end_date"),
                    "theme": theme,
                    "trend": trend,
                    "limit": limit,
                    "matched_experiments": len(filtered_experiments),
                    "matched_programs": len(filtered_programs),
                },
                "snapshot_cache": {
                    "enabled": True,
                    "hit": False,
                    "key": snapshot_key,
                    "latest_completed_ts": latest_completed_ts,
                },
            }

            if include_narrative:
                try:
                    data["narrative"] = aria.generate_report_narrative(data)
                except Exception as e:
                    logger.debug(f"Scoped report narrative generation failed: {e}")
                    data["narrative"] = None

            if not include_narrative:
                try:
                    nb.save_report_snapshot(
                        snapshot_key=snapshot_key,
                        scope="report_query",
                        query=snapshot_query,
                        payload=data,
                        latest_completed_ts=latest_completed_ts,
                    )
                except Exception as e:
                    logger.debug(f"Scoped report snapshot save failed: {e}")

            return jsonify(data)
        except Exception as e:
            logger.error(f"Error in /api/report/query: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Analytics endpoints ──

    @app.route("/api/analytics/op-success")
    def api_op_success():
        """Op success rate table."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.op_success_rates())
        except Exception as e:
            logger.error(f"Error in op-success: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/recommendation-signals")
    def api_recommendation_signals():
        """Aggregate compact, data-driven recommendation signals for Designer."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics

            analytics = ExperimentAnalytics(nb)
            op_rates = analytics.op_success_rates() or {}
            op_priors = []
            for op_name, stats in op_rates.items():
                n_used = int(stats.get("n_used") or 0)
                if n_used < 5:
                    continue
                op_priors.append({
                    "op_name": op_name,
                    "s1_rate": round(_to_safe_float(stats.get("s1_rate")), 6),
                    "n_used": n_used,
                })
            op_priors.sort(key=lambda r: (r.get("s1_rate", 0.0), r.get("n_used", 0)), reverse=True)

            toxic_pairs = nb.get_failure_signature_blocklist(min_seen=5, max_fail_rate=0.85)
            toxic_op_names: set[str] = set()
            toxic_signatures = []
            for signature, penalty in toxic_pairs.items():
                toks = [tok.strip() for tok in str(signature).split("->") if tok.strip()]
                toxic_op_names.update(toks)
                toxic_signatures.append({
                    "signature": signature,
                    "penalty": round(_to_safe_float(penalty, 1.0), 6),
                })
            toxic_signatures.sort(key=lambda r: r.get("penalty", 1.0))

            compression_coverage = analytics.compression_coverage() or {}
            compression_opportunities = _compute_compression_opportunities(compression_coverage)
            top_techniques = (compression_opportunities or {}).get("top_techniques") or []
            compression_techniques = [
                str(item.get("technique") or "").strip()
                for item in top_techniques
                if str(item.get("technique") or "").strip()
            ]

            interactions_raw = nb.get_selection_insight_interactions(limit=120)
            interactions = []
            for row in interactions_raw:
                n_trials = int(row.get("n_trials") or 0)
                if n_trials < 2:
                    continue
                interactions.append({
                    "insight_a": row.get("insight_a"),
                    "insight_b": row.get("insight_b"),
                    "mean_reward": round(_to_safe_float(row.get("mean_reward"), 0.0), 6),
                    "n_trials": n_trials,
                    "n_supported": int(row.get("n_supported") or 0),
                    "n_not_supported": int(row.get("n_not_supported") or 0),
                })
            interactions.sort(
                key=lambda r: (abs(r.get("mean_reward", 0.0) - 0.5), r.get("n_trials", 0)),
                reverse=True,
            )

            insights = _deduplicate_insights(nb.get_insights(limit=120))
            compressed_insights = [
                {
                    "insight_id": row.get("insight_id"),
                    "category": row.get("category"),
                    "insight_type": row.get("insight_type"),
                    "subject_key": row.get("subject_key"),
                    "semantic_key": row.get("semantic_key"),
                    "content": row.get("content"),
                    "confidence": round(_to_safe_float(row.get("confidence"), 0.0), 6),
                }
                for row in insights
            ]

            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "research.analytics",
                "summary": nb.get_dashboard_summary(),
                "op_priors": op_priors[:80],
                "toxic_signatures": toxic_signatures[:80],
                "toxic_ops": sorted(toxic_op_names),
                "compression_opportunities": compression_opportunities,
                "compression_techniques": compression_techniques[:20],
                "insights": compressed_insights[:80],
                "insight_interactions": interactions[:60],
                "native_runner": _native_runner_canary_status_payload(force_refresh=False),
                "aria_core": {
                    "available": bool(importlib.util.find_spec("aria_core")),
                    "proactive_gating_in_research_runner": True,
                },
            }
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in recommendation-signals: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/failure-patterns")
    def api_failure_patterns():
        """Failure analysis by error type and stage."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.failure_patterns())
        except Exception as e:
            logger.error(f"Error in failure-patterns: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/grammar-weights")
    def api_grammar_weights():
        """Current vs learned grammar weights."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            defaults = analytics.get_current_grammar_weights()
            learned = analytics.compute_grammar_weights()
            control_comparison = analytics.control_experiment_comparison()
            holdout = analytics.holdout_validation()
            explanation = aria.explain_grammar_weights(defaults, learned)
            diagnostics = analytics.grammar_weight_learning_diagnostics()
            return jsonify({
                "default": defaults,
                "learned": learned,
                "control_comparison": control_comparison,
                "holdout_validation": holdout,
                "learning_diagnostics": diagnostics,
                "architecture_rerun_telemetry": {
                    "unique_fingerprint_count": int(diagnostics.get("unique_fingerprints") or 0),
                    "total_result_rows": int(diagnostics.get("total_rows") or 0),
                    "repeat_result_rows": int(diagnostics.get("repeat_rows") or 0),
                    "rerun_ratio": float(diagnostics.get("rerun_ratio") or 0.0),
                    "top_fingerprint_concentration": float(diagnostics.get("top_fingerprint_concentration") or 0.0),
                    "weighting_mode": str(diagnostics.get("mode") or "unknown"),
                },
                "explanation": explanation,
            })
        except Exception as e:
            logger.error(f"Error in grammar-weights: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/efficiency-frontier")
    def api_efficiency_frontier():
        """Pareto-optimal programs on loss vs FLOPs."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.efficiency_frontier())
        except Exception as e:
            logger.error(f"Error in efficiency-frontier: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/efficiency-frontier-3d")
    def api_efficiency_frontier_3d():
        """Pareto-optimal programs on loss vs FLOPs vs compression (3D)."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.efficiency_frontier_3d())
        except Exception as e:
            logger.error(f"Error in efficiency-frontier-3d: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/regression-vs-baseline")
    def api_regression_vs_baseline():
        """Accuracy/speed tradeoff view based on baseline ratio vs throughput."""
        limit = request.args.get("limit", 200, type=int)
        nb = LabNotebook(notebook_path)
        try:
            rows = nb.conn.execute(
                """
                SELECT
                    result_id,
                    experiment_id,
                    timestamp,
                    loss_ratio,
                    baseline_loss_ratio,
                    throughput_tok_s,
                    flops_per_token,
                    novelty_score
                FROM program_results
                WHERE stage1_passed = 1
                  AND baseline_loss_ratio IS NOT NULL
                  AND throughput_tok_s IS NOT NULL
                  AND throughput_tok_s > 0
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (max(20, int(limit)),),
            ).fetchall()

            points = []
            for row in rows:
                item = dict(row)
                item["baseline_beats_reference"] = float(item.get("baseline_loss_ratio") or 0.0) < 1.0
                points.append(item)

            # Pareto frontier for (maximize throughput, minimize baseline ratio)
            frontier = []
            best_ratio = float("inf")
            for item in sorted(points, key=lambda p: float(p.get("throughput_tok_s") or 0.0), reverse=True):
                ratio = float(item.get("baseline_loss_ratio") or float("inf"))
                if ratio <= best_ratio:
                    frontier.append(item)
                    best_ratio = ratio

            summary = {
                "n_points": len(points),
                "n_beating_baseline": sum(1 for p in points if p["baseline_beats_reference"]),
                "best_baseline_ratio": min(
                    (float(p.get("baseline_loss_ratio") or float("inf")) for p in points),
                    default=None,
                ),
                "best_throughput_tok_s": max(
                    (float(p.get("throughput_tok_s") or 0.0) for p in points),
                    default=0.0,
                ),
                "frontier_count": len(frontier),
            }

            return jsonify({
                "points": points,
                "pareto_frontier": frontier,
                "summary": summary,
            })
        except Exception as e:
            logger.error(f"Error in regression-vs-baseline: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/experiment-clusters")
    def api_experiment_clusters():
        """Deterministic experiment clustering summary and stability signal."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.experiment_clusters())
        except Exception as e:
            logger.error(f"Error in experiment-clusters: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/routing-health")
    def api_routing_health():
        """Routing telemetry health summary grouped by routing mode."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.routing_health() or {}
            payload.setdefault("available", False)
            payload.setdefault("by_mode", [])
            payload.setdefault("explanation", "Routing telemetry is unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in routing-health: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/routing-comparison")
    def api_routing_comparison():
        """Consolidated routing-mode comparison with confidence/sample labels."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.routing_mode_comparison() or {}
            payload.setdefault("available", False)
            payload.setdefault("by_mode", [])
            payload.setdefault("n_modes", 0)
            payload.setdefault("total_programs", 0)
            payload.setdefault("routed_programs", 0)
            payload.setdefault("uniform_programs", 0)
            payload.setdefault("explanation", "Routing comparison data is unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in routing-comparison: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/gating-diagnostics")
    def api_gating_diagnostics():
        """Canonical gating behavior diagnostics (entropy/collapse/retention)."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.gating_behavior_diagnostics() or {}
            payload.setdefault("available", False)
            payload.setdefault("total_routed_programs", 0)
            payload.setdefault("avg_gate_entropy", None)
            payload.setdefault("collapse_risk_counts", {"low": 0, "medium": 0, "high": 0, "unknown": 0})
            payload.setdefault("by_mode", [])
            payload.setdefault("token_retention_curve_overall", [])
            payload.setdefault("explanation", "Gating diagnostics are unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in gating-diagnostics: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/gate-health")
    def api_gate_health():
        """Causality gate performance: daily failure rate + loss correlation."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            n_days = request.args.get("days", 14, type=int)
            return jsonify(analytics.gate_health_daily(n_days=n_days))
        except Exception as e:
            logger.error(f"Error in gate-health: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/math-family-coverage")
    def api_math_family_coverage():
        """Coverage of evaluated/surviving programs by mathematical family."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.math_family_coverage())
        except Exception as e:
            logger.error(f"Error in math-family-coverage: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/mathspace-impact")
    def api_mathspace_impact():
        """Impact of math-space operators/families on S1/validation/novelty."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.mathspace_operator_impact() or {}
            payload.setdefault("available", False)
            payload.setdefault("totals", {
                "n_programs_with_graph": 0,
                "n_programs_with_mathspace": 0,
                "n_mathspace_ops_observed": 0,
            })
            payload.setdefault("by_operator", [])
            payload.setdefault("by_family", [])
            payload.setdefault("top_trustworthy_operators", [])
            payload.setdefault("explanation", "Math-space impact data is unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in mathspace-impact: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/compression-coverage")
    def api_compression_coverage():
        """Coverage of compression techniques across tested and surviving programs."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.compression_coverage())
        except Exception as e:
            logger.error(f"Error in compression-coverage: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/compression-opportunities")
    def api_compression_opportunities():
        """Ranked compactness opportunities with actionable next-run suggestions."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            coverage = analytics.compression_coverage() or {}
            return jsonify(_compute_compression_opportunities(coverage))
        except Exception as e:
            logger.error(f"Error in compression-opportunities: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/negative-results")
    def api_negative_results():
        """Aggregated negative results: failed ops, error types, anti-patterns."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.negative_results_synthesis())
        except Exception as e:
            logger.error(f"Error in negative-results: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-trajectory")
    def api_learning_trajectory():
        """S1 rate trend over time with regression analysis."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.learning_trajectory())
        except Exception as e:
            logger.error(f"Error in learning-trajectory: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/strategy-backtest")
    def api_strategy_backtest():
        """Aggregate cross-experiment outcomes by search intent."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.strategy_backtest())
        except Exception as e:
            logger.error(f"Error in strategy-backtest: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/control-comparison")
    def api_control_comparison():
        """Compare control (default weights) vs learned-weight experiments."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            result = analytics.control_experiment_comparison()
            if result is None:
                return jsonify({"status": "insufficient_data",
                                "message": "Need at least 2 control and 2 learned experiments"})
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in control-comparison: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-summary")
    def api_learning_summary():
        """Aria-generated 3-5 bullet summary of what the system has learned."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            payload = aria.summarize_learning_bullets({
                "summary": nb.get_dashboard_summary(),
                "grammar_default": analytics.get_current_grammar_weights(),
                "grammar_learned": analytics.compute_grammar_weights(),
                "frontier": analytics.efficiency_frontier(),
                "clusters": analytics.experiment_clusters(),
                "recent_experiments": nb.get_recent_experiments(10),
                "trajectory": analytics.learning_trajectory(),
            })
            payload.setdefault("bullets", [])
            payload.setdefault("source", "rule-based")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in learning-summary: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-log")
    def api_learning_log():
        """Audit trail of grammar weight changes."""
        n = request.args.get("n", 100, type=int)
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_learning_log(limit=n))
        except Exception as e:
            logger.error(f"Error in learning-log: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/insight-interactions")
    def api_insight_interactions():
        """Pairwise insight synergy/antagonism learned from selection outcomes."""
        nb = LabNotebook(notebook_path)
        try:
            limit = request.args.get("limit", 80, type=int)
            min_trials = request.args.get("min_trials", 1, type=int)
            rows = nb.get_selection_insight_interactions(limit=max(1, min(limit, 500)))
            rows = [
                row for row in rows
                if int(row.get("n_trials") or 0) >= max(1, int(min_trials))
            ]

            insight_rows = nb.get_insights(limit=500)
            insight_by_id = {
                str(row.get("insight_id")): row for row in insight_rows if row.get("insight_id")
            }

            enriched: List[Dict[str, Any]] = []
            for row in rows:
                a_id = str(row.get("insight_a") or "")
                b_id = str(row.get("insight_b") or "")
                a = insight_by_id.get(a_id, {})
                b = insight_by_id.get(b_id, {})
                mean_reward = _to_safe_float(row.get("mean_reward"), 0.0)
                n_trials = int(row.get("n_trials") or 0)
                supported = int(row.get("n_supported") or 0)
                int(row.get("n_not_supported") or 0)
                support_rate = (supported / n_trials) if n_trials > 0 else 0.0
                label = "synergistic" if mean_reward >= 0.55 else ("antagonistic" if mean_reward <= 0.45 else "mixed")
                confidence = "high" if n_trials >= 8 else ("medium" if n_trials >= 4 else "low")
                enriched.append({
                    **row,
                    "support_rate": round(support_rate, 6),
                    "interaction_label": label,
                    "confidence_label": confidence,
                    "insight_a_content": a.get("content"),
                    "insight_b_content": b.get("content"),
                    "insight_a_category": a.get("category"),
                    "insight_b_category": b.get("category"),
                    "is_singleton": a_id == b_id,
                })

            synergistic = [
                row for row in enriched
                if not row.get("is_singleton") and row.get("interaction_label") == "synergistic"
            ][:10]
            antagonistic = [
                row for row in enriched
                if not row.get("is_singleton") and row.get("interaction_label") == "antagonistic"
            ][:10]
            singleton = [
                row for row in enriched
                if row.get("is_singleton")
            ][:10]
            return jsonify({
                "available": len(enriched) > 0,
                "total_interactions": len(enriched),
                "synergistic_pairs": synergistic,
                "antagonistic_pairs": antagonistic,
                "singleton_insights": singleton,
                "interactions": enriched,
                "explanation": (
                    "Interaction score is learned from downstream outcomes of selection decisions "
                    "(supported/not_supported with reward aggregation)."
                ),
            })
        except Exception as e:
            logger.error(f"Error in insight-interactions: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

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
            failure_analysis = {"funnel": {}, "errors": {}, "stage_deaths": {}}
            if experiment_id:
                try:
                    nb.get_experiment(experiment_id)
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
                            hyp_row["hypothesis_id"] if isinstance(hyp_row, dict)
                            else hyp_row[0]
                        )
                except Exception:
                    pass

            # Cross-run stability for this specific result
            cross_run = {"trend": "unknown", "seen_runs": 0}
            try:
                top = nb.get_top_programs(20, sort_by="loss_ratio")
                stability = _compute_cross_run_stability(nb, top)
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
            (leaderboard_entry or {}).get("tier", "screening")
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
                        "baseline_ratio": leaderboard_entry.get("validation_baseline_ratio"),
                        "multi_seed_std": leaderboard_entry.get("validation_multi_seed_std"),
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
            recommendation = _compute_recommendation(program, leaderboard_entry)

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

            return jsonify({
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
            })
        except Exception as e:
            logger.error(f"Error in /api/decision-packet/{result_id}: {e}\n"
                         f"{traceback.format_exc()}")
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
            grammar_weights = config.get("applied_grammar_weights") or config.get("grammar_weights")
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
        except Exception as e:
            logger.error(f"Error in /api/reproducibility-manifest/{result_id}: {e}\n"
                         f"{traceback.format_exc()}")
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
                metadata={"result_id": result_id}
            )
            return jsonify(workflow)
        except Exception as e:
            logger.error(f"Error exporting workflow for {result_id}: {e}")
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

    # ── Leaderboard endpoints ──

    @app.route("/api/references")
    def api_references():
        """Get pinned reference architectures."""
        nb = LabNotebook(notebook_path)
        try:
            from ..naming import annotate_display_names
            refs = nb.get_references()
            annotate_display_names(refs)
            return jsonify({
                "entries": _json_safe(refs),
                "total": len(refs),
            })
        except Exception as e:
            logger.error(f"Error in /api/references: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/leaderboard")
    def api_leaderboard():
        """Get leaderboard entries, optionally filtered by tier."""
        tier = request.args.get("tier")
        limit = request.args.get("limit", 50, type=int)
        sort_by = request.args.get("sort", "screening_loss_ratio")
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            entries = nb.get_leaderboard(tier=tier, limit=limit, sort_by=sort_by)
            _attach_long_context_breakdown(nb, entries)
            stability = _compute_cross_run_stability(
                nb, nb.get_top_programs(20, sort_by="loss_ratio")
            )
            stability_by_result = {
                c.get("result_id"): c
                for c in stability.get("candidates", [])
                if c.get("result_id")
            }
            for entry in entries:
                if not entry.get("architecture_family"):
                    entry["architecture_family"] = nb._classify_architecture_family(
                        entry.get("result_id")
                    )
                entry["cross_run_stability"] = stability_by_result.get(
                    entry.get("result_id"),
                    {
                        "trend": "unknown",
                        "seen_runs": 0,
                        "latest_rank": None,
                        "previous_rank": None,
                        "rank_delta": None,
                    },
                )
            _annotate_qkv_usage(entries, analytics)
            # Group by tier for the dashboard
            tiers = {}
            for entry in entries:
                t = entry.get("tier", "screening")
                if t not in tiers:
                    tiers[t] = []
                tiers[t].append(entry)
            return jsonify({
                "entries": entries,
                "by_tier": tiers,
                "total": len(entries),
                "cross_run_stability_summary": stability.get("summary", {}),
                "cross_run_stability_window": stability.get("window_size", 0),
            })
        except Exception as e:
            logger.error(f"Error in /api/leaderboard: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/leaderboard/status", methods=["POST"])
    def api_leaderboard_update_status():
        """Update status (tier) for an existing leaderboard record."""
        body = request.get_json(silent=True) or {}
        tier = str(body.get("tier") or "").strip().lower()
        entry_id = str(body.get("entry_id") or "").strip()
        result_id = str(body.get("result_id") or "").strip()

        valid_tiers = {"screening", "investigation", "validation", "breakthrough"}
        if tier not in valid_tiers:
            return jsonify({"error": "tier must be one of screening, investigation, validation, breakthrough"}), 400
        if not entry_id and not result_id:
            return jsonify({"error": "entry_id or result_id is required"}), 400

        nb = LabNotebook(notebook_path)
        try:
            row = None
            if entry_id:
                row = nb.conn.execute(
                    "SELECT entry_id, result_id, tier FROM leaderboard WHERE entry_id = ?",
                    (entry_id,),
                ).fetchone()
            if row is None and result_id:
                row = nb.conn.execute(
                    "SELECT entry_id, result_id, tier FROM leaderboard WHERE result_id = ?",
                    (result_id,),
                ).fetchone()
            if row is None:
                return jsonify({"error": "Leaderboard entry not found"}), 404

            resolved_entry_id = row["entry_id"]
            nb.promote_to_tier(resolved_entry_id, tier)

            updated = nb.conn.execute(
                "SELECT entry_id, result_id, tier, timestamp FROM leaderboard WHERE entry_id = ?",
                (resolved_entry_id,),
            ).fetchone()

            return jsonify({
                "success": True,
                "entry": dict(updated) if updated else {"entry_id": resolved_entry_id, "tier": tier},
            })
        except Exception as e:
            logger.error(f"Error in /api/leaderboard/status: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/leaderboard/pin", methods=["POST"])
    def api_leaderboard_pin():
        """Pin or unpin a leaderboard entry."""
        body = request.get_json(silent=True) or {}
        entry_id = str(body.get("entry_id") or "").strip()
        result_id = str(body.get("result_id") or "").strip()
        pinned = bool(body.get("pinned", False))

        if not entry_id and not result_id:
            return jsonify({"error": "entry_id or result_id is required"}), 400

        nb = LabNotebook(notebook_path)
        try:
            resolved_entry_id = entry_id
            if not resolved_entry_id and result_id:
                row = nb.conn.execute(
                    "SELECT entry_id FROM leaderboard WHERE result_id = ?",
                    (result_id,),
                ).fetchone()
                if row:
                    resolved_entry_id = row["entry_id"]
            
            if not resolved_entry_id:
                return jsonify({"error": "Leaderboard entry not found"}), 404

            nb.set_leaderboard_pin(resolved_entry_id, pinned)
            return jsonify({"success": True, "entry_id": resolved_entry_id, "pinned": pinned})
        except Exception as e:
            logger.error(f"Error in /api/leaderboard/pin: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/discoveries")
    def api_discoveries():
        """Unified discoveries endpoint merging leaderboard + raw candidates.

        Query params:
          tier: filter by tier (screening/investigation/validation/breakthrough)
          limit: max results (default 100)
          sort: sort key (default composite_score)
          view: 'all' for raw candidates, 'ranked' for leaderboard (default ranked)
        """
        from ..naming import annotate_display_names

        tier = request.args.get("tier")
        limit = request.args.get("limit", 100, type=int)
        sort_by = request.args.get("sort", "composite_score")
        view = request.args.get("view", "ranked")
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            if view == "all":
                # Raw S1 survivors from program_results
                programs = nb.get_top_programs(limit, sort_by="loss_ratio")
                _attach_long_context_breakdown(nb, programs)
                _annotate_qkv_usage(programs, analytics)
                # Add family classification + display names
                for p in programs:
                    p["architecture_family"] = nb._classify_architecture_family(
                        graph_json=p.get("graph_json"),
                        routing_mode=p.get("routing_mode"),
                    )
                    p["tier"] = _infer_tier_for_program(nb, p)
                annotate_display_names(programs)
                # Strip large fields from response
                for p in programs:
                    p.pop("graph_json", None)
                    p.pop("_graph_json", None)
                    p.pop("loss_curve", None)

                # Compute tier counts from all S1 survivors
                tier_counts = _count_discovery_tiers(nb)

                return jsonify({
                    "entries": _json_safe(programs),
                    "total": len(programs),
                    "tier_counts": tier_counts,
                    "view": "all",
                })

            # Default: ranked leaderboard view
            entries = nb.get_leaderboard(tier=tier, limit=limit, sort_by=sort_by)
            _attach_long_context_breakdown(nb, entries)
            stability = _compute_cross_run_stability(
                nb, nb.get_top_programs(20, sort_by="loss_ratio")
            )
            stability_by_result = {
                c.get("result_id"): c
                for c in stability.get("candidates", [])
                if c.get("result_id")
            }
            for entry in entries:
                entry["cross_run_stability"] = stability_by_result.get(
                    entry.get("result_id"),
                    {"trend": "unknown", "seen_runs": 0,
                     "latest_rank": None, "previous_rank": None, "rank_delta": None},
                )
            _annotate_qkv_usage(entries, analytics)
            annotate_display_names(entries)

            # Summary counts
            tier_counts = _count_discovery_tiers(nb)

            return jsonify({
                "entries": _json_safe(entries),
                "total": len(entries),
                "tier_counts": tier_counts,
                "cross_run_stability_summary": stability.get("summary", {}),
                "cross_run_stability_window": stability.get("window_size", 0),
                "view": "ranked",
            })
        except Exception as e:
            logger.error(f"Error in /api/discoveries: {e}")
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
                return jsonify({
                    "result_id": direct["result_id"],
                    "graph_fingerprint": direct.get("graph_fingerprint"),
                    "resolved_from": "result_id",
                    "candidates": [],
                })
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
                    candidates.append({
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
                    })
                return jsonify({
                    "result_id": chosen_row.get("result_id"),
                    "graph_fingerprint": chosen_row.get("graph_fingerprint"),
                    "resolved_from": "graph_fingerprint",
                    "candidate_count": len(candidates),
                    "selection_policy": "leaderboard_composite_then_loss",
                    "candidates": candidates,
                })
            return jsonify({"error": "No matching fingerprint or result_id found."}), 404
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
            return jsonify({
                "query": value,
                "resolved_graph_fingerprint": history[0]["graph_fingerprint"] if history else None,
                "total": len(history),
                "best_leaderboard_run": best_by_composite,
                "runs": history,
            })
        except Exception as e:
            logger.error(f"Error in /api/fingerprint/history: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Control endpoints ──
