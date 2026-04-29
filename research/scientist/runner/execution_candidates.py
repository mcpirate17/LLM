"""Execution mixin: candidate generation, grammar config, pending escalation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional


from ...synthesis.grammar import GrammarConfig, batch_generate
from ..native_runner import compile_model_native_first as compile_model
from ...synthesis.validator import validate_graph
from ...synthesis.serializer import graph_to_json, graph_summary
from ..shared_utils import resolve_device

import logging

logger = logging.getLogger(__name__)

from ._types import ModelCandidate, RunConfig

if TYPE_CHECKING:
    from ..notebook import LabNotebook


class _ExecutionCandidatesMixin:
    """Candidate generation, grammar config building, pending escalation."""

    __slots__ = ()

    def _claim_pending_followup_from_notebook(self, stage: str):
        try:
            nb = self._make_notebook()
        except Exception as e:
            logger.warning(
                "Failed to open notebook for %s follow-up claim: %s", stage, e
            )
            return None
        try:
            task = nb.claim_followup_task(stage)
        except Exception as e:
            logger.warning("Failed to claim %s follow-up task: %s", stage, e)
            nb.close()
            return None
        nb.close()
        if not task:
            return None
        config_payload = task.get("config_json") or {}
        config = (
            RunConfig.from_dict(config_payload)
            if isinstance(config_payload, dict) and config_payload
            else RunConfig()
        )
        return {
            "task_id": task.get("task_id"),
            "result_ids": list(task.get("result_ids_json") or []),
            "config": config,
            "hypothesis": str(task.get("hypothesis") or ""),
            "evidence_pack": task.get("evidence_pack_json") or {},
            "source_context": task.get("source_context"),
        }

    def _finalize_followup_task(
        self,
        task_id: str | None,
        *,
        success: bool,
        stage: str,
        error: str | None = None,
        permanent_failure: bool = False,
    ) -> None:
        if not task_id:
            return
        try:
            nb = self._make_notebook()
        except Exception as e:
            logger.warning("Failed to open notebook to finalize %s task: %s", stage, e)
            return
        try:
            if success:
                nb.complete_followup_task(
                    task_id,
                    outcome="launched",
                    metadata={"stage": stage},
                )
            elif permanent_failure:
                # Don't requeue — the task can never succeed (e.g. the
                # candidate was promoted past this stage). Mark completed
                # with a distinct outcome so audits can find these.
                nb.complete_followup_task(
                    task_id,
                    outcome="permanent_failure",
                    metadata={"stage": stage, "error": str(error or "")[:500]},
                )
            else:
                nb.requeue_followup_task(
                    task_id,
                    outcome="launch_failed",
                    metadata={"stage": stage, "error": str(error or "")[:500]},
                )
        except Exception as e:
            logger.warning(
                "Failed to finalize %s follow-up task %s: %s", stage, task_id, e
            )
        finally:
            nb.close()

    @staticmethod
    def _is_tier_guard_failure(exc: BaseException) -> bool:
        if not isinstance(exc, ValueError):
            return False
        msg = str(exc)
        return (
            "Cannot investigate" in msg
            or "Cannot validate" in msg
            or "already at or beyond" in msg
        )

    def _run_pending_replay(self) -> bool:
        """Run one queued exact replay task through the canonical replay pipeline."""
        if self.is_running:
            return False
        try:
            nb = self._make_notebook()
        except Exception as e:
            logger.warning("Failed to open notebook for replay task claim: %s", e)
            return False
        try:
            task = nb.claim_followup_task("replay")
        except Exception as e:
            logger.warning("Failed to claim replay task: %s", e)
            nb.close()
            return False
        nb.close()
        if not task:
            return False

        config_payload = task.get("config_json") or {}
        repeat_per_source = max(
            1,
            min(3, int(config_payload.get("repeat_per_source") or 1)),
        )
        device = str(config_payload.get("device") or "cuda")
        fast = bool(config_payload.get("fast", True))
        result_ids = [
            str(rid).strip()
            for rid in (task.get("result_ids_json") or [])
            if str(rid).strip()
        ]
        hypothesis = str(task.get("hypothesis") or "Active-learning exact replay")

        # Score-stability reruns explicitly want a NEW program_results row
        # per replay (each rerun is a child experiment under the same
        # graph_fingerprint parent), not a patch on the source row.
        # Detect these by source_context.  Original "rescreen / triage"
        # uses still patch the source row by default.
        source_context = str(task.get("source_context") or "").strip()
        independent_sample = source_context in {"program_detail_rerun", "queue_rerun"}

        try:
            from research.tools.exact_graph_replay import run_exact_replay

            exp_id = run_exact_replay(
                db_path=Path(self.notebook_path),
                result_ids=result_ids,
                repeat_per_source=repeat_per_source,
                device=device,
                hypothesis=hypothesis,
                fast=fast,
                verbose=False,
                independent_sample=independent_sample,
            )
            nb_done = None
            try:
                nb_done = self._make_notebook()
                nb_done.complete_followup_task(
                    str(task.get("task_id") or ""),
                    outcome="completed",
                    metadata={"stage": "replay", "replay_experiment_id": exp_id},
                )
            except Exception as e:
                logger.warning("Failed to attach replay completion metadata: %s", e)
            finally:
                try:
                    nb_done.close()
                except Exception:
                    pass
            return True
        except Exception as e:
            logger.warning("Failed to run exact replay task: %s", e)
            self._finalize_followup_task(
                str(task.get("task_id") or ""),
                success=False,
                stage="replay",
                error=str(e),
            )
            return False

    def _generate_candidates(
        self,
        config: RunConfig,
        n: int,
        source: str = "graph_synthesis",
        nb: "Optional[LabNotebook]" = None,
    ) -> List["ModelCandidate"]:
        """Generate candidate models from the specified source.

        source: "graph_synthesis", "morphological_box", or "mixed"
        Returns candidates that pass Stage 0 smoke test.
        """
        candidates: List[ModelCandidate] = []
        dev_str = str(resolve_device(config.device))

        if source == "mixed":
            n_morph = int(n * config.morph_ratio)
            n_graph = n - n_morph
            candidates.extend(
                self._generate_candidates(config, n_graph, "graph_synthesis", nb=nb)
            )
            candidates.extend(
                self._generate_candidates(config, n_morph, "morphological_box", nb=nb)
            )
            return candidates

        if source == "morphological_box":
            try:
                from ...morphological_box import roll, describe_spec
                from ...arch_builder import build_model, BuildConfig

                sparse_weight_options = (
                    "structured_sparse",
                    "semi_structured_2_4",
                    "block_sparse",
                )

                build_cfg = BuildConfig(
                    dim=config.model_dim,
                    n_layers=config.n_layers,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )

                for i in range(n):
                    if self._stop_event.is_set():
                        break
                    try:
                        fixed_choices: Dict[str, str] = {}
                        if bool(getattr(config, "morph_focus_sparse", False)):
                            explicit_sparse = str(
                                getattr(config, "morph_sparse_weight_storage", "") or ""
                            ).strip()
                            if explicit_sparse in sparse_weight_options:
                                fixed_choices["weight_storage"] = explicit_sparse
                            else:
                                fixed_choices["weight_storage"] = sparse_weight_options[
                                    i % len(sparse_weight_options)
                                ]
                        fixed_routing = str(
                            getattr(config, "morph_compute_routing", "") or ""
                        ).strip()
                        if fixed_routing:
                            fixed_choices["compute_routing"] = fixed_routing
                        fixed_channel = str(
                            getattr(config, "morph_channel_mixing", "") or ""
                        ).strip()
                        if fixed_channel:
                            fixed_choices["channel_mixing"] = fixed_channel

                        spec = roll(
                            seed=i + int(time.time() * 1000) % 100000,
                            generation=0,
                            fixed=fixed_choices or None,
                        )
                        model = build_model(spec, build_cfg)
                        desc = describe_spec(spec)

                        # Quick smoke test
                        sandbox_result = self._safe_eval_for_stage(
                            model,
                            stage_tag="morph_candidate_gen",
                            batch_size=2,
                            seq_len=min(128, config.max_seq_len),
                            vocab_size=config.vocab_size,
                            device=dev_str,
                        )
                        if sandbox_result.passed:
                            import json as _json

                            candidates.append(
                                ModelCandidate(
                                    source="morphological_box",
                                    model=model,
                                    description=desc,
                                    arch_spec=spec,
                                    arch_spec_json=_json.dumps(spec.to_dict()),
                                    fingerprint=spec.id,
                                )
                            )
                        else:
                            del model
                    except (RuntimeError, ValueError, TypeError) as e:
                        logger.debug(f"Morphological candidate {i} failed: {e}")
                        continue
            except ImportError:
                logger.warning("morphological_box or arch_builder not available")
            return candidates

        # Default: graph_synthesis — optionally blend notebook-derived weights.
        from ..ml_influence_policy import component_is_allowed

        if component_is_allowed("learned_candidate_weights", config):
            op_weights, template_weights, motif_weights, category_weights = (
                self._load_learned_weights(nb=nb)
            )
        else:
            op_weights, template_weights, motif_weights, category_weights = (
                {},
                {},
                {},
                {},
            )
            logger.info(
                "Candidate generation learned weights disabled or blocked for this run"
            )
        grammar = self._build_grammar_config(
            config,
            op_weights=op_weights,
            category_weights=category_weights,
        )
        if template_weights:
            if grammar.routing_mandatory:
                for k, v in template_weights.items():
                    grammar.template_weights.setdefault(k, v)
            else:
                grammar.template_weights.update(template_weights)
        if motif_weights:
            grammar.motif_weights.update(motif_weights)

        graphs = batch_generate(n, grammar).graphs
        for graph in graphs:
            if self._stop_event.is_set():
                break
            validation = validate_graph(
                graph,
                max_ops=max(1, int(config.max_ops)),
                max_depth=max(1, int(config.max_depth)),
                min_splits=config.min_splits,
            )
            if not validation.valid:
                continue
            try:
                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
                sandbox_result = self._safe_eval_for_stage(
                    model,
                    stage_tag="graph_candidate_gen",
                    batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=config.vocab_size,
                    device=dev_str,
                )
                if sandbox_result.passed:
                    candidates.append(
                        ModelCandidate(
                            source="graph_synthesis",
                            model=model,
                            description=graph_summary(graph),
                            graph=graph,
                            graph_json=graph_to_json(graph),
                            fingerprint=graph.fingerprint(),
                        )
                    )
                else:
                    del model
            except (RuntimeError, ValueError, TypeError) as e:
                logger.debug("Graph candidate compilation failed: %s", e)
                continue

        return candidates

    def _load_learned_weights(
        self,
        nb: "Optional[LabNotebook]" = None,
    ) -> tuple:
        """Load analytics-derived op/template/motif/category weights from notebook history.

        Returns (op_weights, template_weights, motif_weights, category_weights) dicts.
        Falls back to empty dicts on any failure — never blocks candidate generation.
        """
        op_weights: Dict[str, float] = {}
        template_weights: Dict[str, float] = {}
        motif_weights: Dict[str, float] = {}
        category_weights: Dict[str, float] = {}
        try:
            from ..analytics import ExperimentAnalytics

            if nb is None:
                nb = getattr(self, "nb", None) or getattr(self, "_notebook", None)
            if nb is None:
                nb = self._make_notebook()
            if nb is None:
                return op_weights, template_weights, motif_weights, category_weights
            analytics = ExperimentAnalytics(nb)
            _window_cutoff = time.time() - 604800  # 7-day window
            try:
                learned_op = analytics.compute_op_weights(since_ts=_window_cutoff)
                op_weights.update(learned_op)
            except (KeyError, ValueError, TypeError) as e:
                logger.debug("compute_op_weights failed: %s", e)
            try:
                learned_tpl = analytics.compute_template_weights(
                    since_ts=_window_cutoff
                )
                if learned_tpl:
                    template_weights.update(learned_tpl)
            except (KeyError, ValueError, TypeError) as e:
                logger.debug("compute_template_weights failed: %s", e)
            # Template selection scheduler — Thompson sampling or UCB1
            try:
                db_path = (
                    str(nb.db_path)
                    if hasattr(nb, "db_path")
                    else "research/lab_notebook.db"
                )
                _use_thompson = getattr(
                    getattr(self, "_current_config", None),
                    "use_thompson_sampling",
                    False,
                )
                if _use_thompson:
                    from ...search.scheduler import ThompsonScheduler

                    sched_weights = ThompsonScheduler(db_path=db_path).sample()
                else:
                    from ...search.scheduler import ExplorationScheduler

                    sched_weights = ExplorationScheduler(db_path=db_path).step()

                if sched_weights:
                    for tpl, w in sched_weights.items():
                        # Blend: existing analytics weight × scheduler weight (geometric mean)
                        if tpl in template_weights:
                            template_weights[tpl] = (template_weights[tpl] * w) ** 0.5
                        else:
                            template_weights[tpl] = w
            except (ImportError, KeyError, ValueError, RuntimeError) as e:
                logger.debug("Template scheduler failed: %s", e)
            try:
                learned_motif = analytics.compute_motif_weights(since_ts=_window_cutoff)
                if learned_motif:
                    motif_weights.update(learned_motif)
            except (KeyError, ValueError, TypeError) as e:
                logger.debug("compute_motif_weights failed: %s", e)
            try:
                syn_motif, syn_tpl = analytics.compute_synergy_boosts()
                for name, boost in syn_motif.items():
                    motif_weights[name] = motif_weights.get(name, 1.0) * boost
                for name, boost in syn_tpl.items():
                    template_weights[name] = template_weights.get(name, 1.0) * boost
            except (KeyError, ValueError, TypeError) as e:
                logger.debug("compute_synergy_boosts failed: %s", e)
            # ── Temporal Bayesian tracker: temporal-decay-aware weights ──
            # Overrides analytics weights for ops with sufficient evidence,
            # respects code-fix resets (ops fixed recently get higher weights).
            try:
                from ..intelligence.temporal_bayesian import TemporalBayesianTracker

                db_path = (
                    str(nb.db_path)
                    if hasattr(nb, "db_path")
                    else "research/lab_notebook.db"
                )
                tracker = TemporalBayesianTracker.from_db(
                    db_path=db_path,
                    apply_decay=True,
                    detect_fixes=True,
                )
                _use_thompson = getattr(
                    getattr(self, "_current_config", None),
                    "use_thompson_sampling",
                    False,
                )
                bayes_mode = "thompson" if _use_thompson else "mean"
                bayes_op = tracker.op_weights(mode=bayes_mode)
                bayes_tpl = tracker.template_weights(mode=bayes_mode)
                bayes_motif = tracker.motif_weights(mode=bayes_mode)
                # Blend: geometric mean of analytics × Bayesian
                for op, bw in bayes_op.items():
                    if op in op_weights:
                        op_weights[op] = (op_weights[op] * bw) ** 0.5
                    else:
                        op_weights[op] = bw
                for tpl, bw in bayes_tpl.items():
                    if tpl in template_weights:
                        template_weights[tpl] = (template_weights[tpl] * bw) ** 0.5
                    else:
                        template_weights[tpl] = bw
                for mot, bw in bayes_motif.items():
                    if mot in motif_weights:
                        motif_weights[mot] = (motif_weights[mot] * bw) ** 0.5
                    else:
                        motif_weights[mot] = bw
                logger.debug(
                    "Bayesian tracker: blended %d op, %d template, %d motif weights (mode=%s)",
                    len(bayes_op),
                    len(bayes_tpl),
                    len(bayes_motif),
                    bayes_mode,
                )
            except Exception as exc:
                logger.debug("Bayesian tracker unavailable: %s", exc)

            # InteractionModel motif adjustment DISABLED — no holdout evidence.
            # Ensemble calibration gave it weight -0.06 (harmful).  Re-enable
            # only after adding holdout ROC/PPV evaluation to the model.

            # Penalize ops with 0% S1 pass rate and sufficient sample size
            try:
                neg = analytics.negative_results_synthesis()
                for op_info in neg.get("failed_ops", []):
                    if (
                        op_info.get("s1_rate", 1) == 0
                        and op_info.get("n_used", 0) >= 5
                        and op_info.get("confidence", 0) >= 0.7
                    ):
                        op_name = op_info["op_name"]
                        if op_info.get("failure_stage") == "compilation":
                            op_weights[op_name] = 0.15
                        else:
                            op_weights[op_name] = 0.1
                for op_info in neg.get("weak_ops", []):
                    op_name = op_info.get("op_name", "")
                    penalty = op_info.get("penalty_weight", 1.0)
                    if op_name:
                        op_weights[op_name] = penalty
            except (KeyError, ValueError, TypeError) as e:
                logger.debug("negative_results_synthesis failed: %s", e)
            # Category-level weights from historical success data
            try:
                last_cat = getattr(self, "_last_category_weights", None)
                learned_cat = analytics.compute_grammar_weights(last_applied=last_cat)
                if learned_cat:
                    category_weights.update(learned_cat)
                    self._last_category_weights = dict(learned_cat)
            except (KeyError, ValueError, TypeError) as e:
                logger.debug("compute_grammar_weights failed: %s", e)
            if op_weights or category_weights:
                logger.info(
                    "Loaded %d op weights, %d category weights for candidate generation",
                    len(op_weights),
                    len(category_weights),
                )
        except Exception as exc:
            logger.debug("Analytics weight loading unavailable: %s", exc)
        return op_weights, template_weights, motif_weights, category_weights

    # ── Training with synthesized programs ──

    def _build_grammar_config(
        self,
        config: RunConfig,
        op_weights: Optional[Dict[str, float]] = None,
        category_weights: Optional[Dict[str, float]] = None,
    ) -> GrammarConfig:
        """Create a GrammarConfig from a RunConfig with standardized defaults."""
        from ...synthesis.grammar import GrammarConfig

        _forced = getattr(config, "forced_template", None)

        # Capability-first mode: dispatch before routing_first so the stricter
        # preset wins when both flags are set. Promotes role-slot templates
        # (trunk+sidecar topology) and turns on gate8_retrieval_dead via
        # binding_capable_required.
        if getattr(config, "_capability_first_mode", False):
            grammar = GrammarConfig.capability_first(model_dim=config.model_dim)
            if getattr(config, "composition_depth", 0) > 0:
                grammar.composition_depth = config.composition_depth
            grammar.max_ops = getattr(config, "max_ops", 20)
            if op_weights:
                for op_name, w in op_weights.items():
                    if w < 1.0:
                        grammar.op_weights[op_name] = (
                            grammar.op_weights.get(op_name, 1.0) * w
                        )
                    else:
                        grammar.op_weights.setdefault(op_name, w)
            if category_weights:
                grammar.category_weights.update(category_weights)
            grammar.forced_template = _forced
            return grammar

        # exploit_mode implies routing-first: mandate routing/splits in every graph
        if getattr(config, "exploit_mode", False) or getattr(
            config, "_routing_first_mode", False
        ):
            grammar = GrammarConfig.routing_first(model_dim=config.model_dim)
            if getattr(config, "composition_depth", 0) > 0:
                grammar.composition_depth = config.composition_depth
            grammar.max_ops = getattr(config, "max_ops", 20)
            if op_weights:
                for op_name, w in op_weights.items():
                    if w < 1.0:
                        grammar.op_weights[op_name] = (
                            grammar.op_weights.get(op_name, 1.0) * w
                        )
                    else:
                        grammar.op_weights.setdefault(op_name, w)
            if category_weights:
                grammar.category_weights.update(category_weights)
            grammar.forced_template = _forced
            return grammar

        # Exotic mode: use the exotic preset as base, then layer on learned op_weights
        if getattr(config, "_exotic_mode", False):
            grammar = GrammarConfig.exotic(model_dim=config.model_dim)
            # Merge learned op_weights (exotic preset weights take precedence
            # only when learned weight is default 1.0)
            if op_weights:
                for op_name, w in op_weights.items():
                    existing = grammar.op_weights.get(op_name, 1.0)
                    # If the learned system penalizes an op, respect it even in exotic mode
                    if w < 1.0:
                        grammar.op_weights[op_name] = existing * w
                    else:
                        grammar.op_weights.setdefault(op_name, w)
            if category_weights:
                grammar.category_weights.update(category_weights)
            grammar.forced_template = _forced
            return grammar

        # Efficiency mode: use the efficient preset as base
        if getattr(config, "_efficiency_mode", False):
            grammar = GrammarConfig.efficient(model_dim=config.model_dim)
            if op_weights:
                for op_name, w in op_weights.items():
                    if w < 1.0:
                        grammar.op_weights[op_name] = (
                            grammar.op_weights.get(op_name, 1.0) * w
                        )
                    else:
                        grammar.op_weights.setdefault(op_name, w)
            if category_weights:
                grammar.category_weights.update(category_weights)
            grammar.forced_template = _forced
            return grammar

        # Pick up structured_sparsity_bias from mode recommendation or config
        sparsity_bias = getattr(
            self,
            "_structured_sparsity_bias_override",
            getattr(config, "structured_sparsity_bias", 0.0),
        )

        # Merge API-provided op_weights with learned op_weights
        merged_op_weights = dict(op_weights or {})
        if config.op_weights:
            # API overrides take precedence over learned weights
            merged_op_weights.update(config.op_weights)

        grammar_kwargs: Dict[str, Any] = dict(
            model_dim=config.model_dim,
            max_depth=config.max_depth,
            max_ops=config.max_ops,
            residual_prob=config.residual_prob,
            split_prob=config.grammar_split_prob,
            merge_prob=config.grammar_merge_prob,
            risky_op_prob=config.grammar_risky_op_prob,
            freq_domain_prob=config.grammar_freq_domain_prob,
            structured_sparsity_bias=sparsity_bias,
            op_weights=merged_op_weights,
            min_splits=config.min_splits,
            three_way_split_prob=config.three_way_split_prob,
            branch_depth=config.branch_depth,
            max_recursion_depth=config.max_recursion_depth,
            composition_depth=getattr(config, "composition_depth", 0),
            template_weights=config.template_weights or {},
            use_db_weights=config.template_weights
            is None,  # skip DB weights when explicit
            routing_mandatory=config.routing_mandatory,
            forced_template=getattr(config, "forced_template", None),
        )
        if config.category_weights:
            grammar_kwargs["category_weights"] = dict(config.category_weights)
        grammar = GrammarConfig(**grammar_kwargs)
        # Baseline routing/difficulty op boosts (always applied unless overridden)
        _routing_defaults = {
            "token_entropy": 2.5,
            "route_topk": 2.0,
            "route_lanes": 2.0,
            "moe_topk": 2.0,
            "moe_2expert": 2.0,
            "token_merging": 1.5,
            "learned_token_gate": 1.5,
            "depth_weighted_proj": 1.5,
        }
        for op_name, default_w in _routing_defaults.items():
            grammar.op_weights.setdefault(op_name, default_w)

        # Apply specialized weights
        grammar.category_weights["math_space"] = config.math_space_weight
        # Apply analytics-derived category weights from historical success data
        if category_weights:
            grammar.category_weights.update(category_weights)
        # Apply custom category weights from API (overrides analytics + defaults)
        if config.category_weights:
            grammar.category_weights.update(config.category_weights)

        # Apply Bayesian op priors from compressed learning (optional)
        try:
            from pathlib import Path
            import json as _json

            priors_path = Path("research/runtime/learning/op_priors.json")
            if priors_path.exists():
                payload = _json.loads(priors_path.read_text())
                op_penalties = (
                    payload.get("op_penalties", {}) if isinstance(payload, dict) else {}
                )
                if isinstance(op_penalties, dict):
                    for op_name, penalty in op_penalties.items():
                        try:
                            p = float(penalty)
                        except (ValueError, TypeError):
                            continue
                        # Convert penalty (0..1) into weight multiplier (1..0.5)
                        mult = max(0.5, 1.0 - 0.5 * max(0.0, min(1.0, p)))
                        grammar.op_weights[op_name] = (
                            grammar.op_weights.get(op_name, 1.0) * mult
                        )
        except (OSError, ValueError, KeyError) as e:
            logger.debug("Op priors loading failed: %s", e)

        # Apply cluster-based suggestions (optional)
        try:
            from pathlib import Path
            import json as _json

            sugg_path = Path("research/runtime/learning/cluster_suggestions.json")
            if sugg_path.exists():
                payload = _json.loads(sugg_path.read_text())
                if isinstance(payload, dict):
                    op_weight_suggestions = (
                        payload.get("op_weight_suggestions")
                        or payload.get("op_weights")
                        or {}
                    )
                    op_penalties = payload.get("op_penalties") or {}
                    op_promotions = payload.get("op_promotions") or {}
                    avoid_patterns = payload.get("avoid_patterns") or []
                    promote_patterns = payload.get("promote_patterns") or []

                    def _apply_mult(op_name: str, mult: float):
                        if not op_name:
                            return
                        m = max(0.2, min(3.0, float(mult)))
                        grammar.op_weights[op_name] = (
                            grammar.op_weights.get(op_name, 1.0) * m
                        )

                    for op_name, mult in op_weight_suggestions.items():
                        try:
                            _apply_mult(op_name, float(mult))
                        except (ValueError, TypeError):
                            continue

                    for op_name, p in op_penalties.items():
                        try:
                            penalty = max(0.0, min(1.0, float(p)))
                        except (ValueError, TypeError):
                            continue
                        _apply_mult(op_name, 1.0 - 0.4 * penalty)

                    for op_name, p in op_promotions.items():
                        try:
                            promo = max(0.0, min(1.0, float(p)))
                        except (ValueError, TypeError):
                            continue
                        _apply_mult(op_name, 1.0 + 0.4 * promo)

                    def _ops_from_pattern(pat: str):
                        if "->" in pat:
                            parts = [p.strip() for p in pat.split("->", 1)]
                        elif "," in pat:
                            parts = [p.strip() for p in pat.split(",", 1)]
                        else:
                            parts = [pat.strip()]
                        return [p for p in parts if p]

                    for pat in avoid_patterns:
                        for op_name in _ops_from_pattern(str(pat)):
                            _apply_mult(op_name, 0.85)
                    for pat in promote_patterns:
                        for op_name in _ops_from_pattern(str(pat)):
                            _apply_mult(op_name, 1.1)
        except (OSError, ValueError, KeyError) as e:
            logger.debug("Cluster suggestions loading failed: %s", e)

        # Apply designer feedback signals (from Aria suggestion outcomes)
        try:
            from pathlib import Path
            import json as _json

            feedback_path = Path("research/runtime/learning/designer_feedback.json")
            if feedback_path.exists():
                payload = _json.loads(feedback_path.read_text())
                if isinstance(payload, dict):
                    # accepted_ops: ops the user applied in designer → boost
                    for op_name, count in (payload.get("accepted_ops") or {}).items():
                        try:
                            n = int(count)
                        except (TypeError, ValueError):
                            continue
                        if n >= 2:
                            boost = min(1.5, 1.0 + 0.1 * n)
                            grammar.op_weights[op_name] = (
                                grammar.op_weights.get(op_name, 1.0) * boost
                            )
                    # rejected_ops: ops the user rejected → penalize
                    for op_name, count in (payload.get("rejected_ops") or {}).items():
                        try:
                            n = int(count)
                        except (TypeError, ValueError):
                            continue
                        if n >= 2:
                            penalty = max(0.6, 1.0 - 0.08 * n)
                            grammar.op_weights[op_name] = (
                                grammar.op_weights.get(op_name, 1.0) * penalty
                            )
        except (OSError, ValueError, KeyError) as e:
            logger.debug("Designer feedback loading failed: %s", e)

        # Gradient stability penalty: penalize ops with severely exploding
        # gradients from component profiling data (measured, not speculative).
        # Only targets truly pathological ops — grad_norm ~2000 is normal for
        # parameterized ops at d=128, so we use high thresholds.
        try:
            from research.profiling.schema import ComponentDB

            with ComponentDB() as cdb:
                rows = cdb.query(
                    "SELECT op_name, grad_norm FROM op_profiles "
                    "WHERE grad_exploding = 1 AND error IS NULL "
                    "AND grad_norm > 50000"
                )
                for row in rows:
                    op_name = row["op_name"]
                    grad_norm = float(row["grad_norm"])
                    if grad_norm > 1_000_000:
                        mult = 0.3  # catastrophic (reciprocal, div_safe)
                    else:
                        mult = 0.6  # severe (log, state_space, conv_only)
                    grammar.op_weights[op_name] = (
                        grammar.op_weights.get(op_name, 1.0) * mult
                    )
        except (ImportError, OSError, ValueError, RuntimeError) as e:
            logger.debug("Gradient stability penalty loading failed: %s", e)

        return grammar

    def _run_pending_investigation(self):
        """Launch pending auto-investigation if queued."""
        if self.is_running:
            return
        pending = getattr(self, "_pending_investigation", None)
        if pending is None:
            pending = self._claim_pending_followup_from_notebook("investigation")
            if pending is None:
                return
        self._pending_investigation = None

        # Anything in the followup_tasks queue is a deliberate request
        # (manual UI rerun, stale-screening recovery, backfill sweep,
        # score-stability variance probing).  Bypass the tier guard so
        # candidates already promoted past investigation can still be
        # re-run for variance / score-stability — that's the whole point
        # of queueing them.
        try:
            self.start_investigation(
                result_ids=pending["result_ids"],
                config=pending["config"],
                hypothesis=pending["hypothesis"],
                force=True,
            )
            self._finalize_followup_task(
                pending.get("task_id"),
                success=True,
                stage="investigation",
            )
        except Exception as e:
            logger.warning(f"Failed to launch auto-investigation: {e}")
            self._finalize_followup_task(
                pending.get("task_id"),
                success=False,
                stage="investigation",
                error=str(e),
                permanent_failure=self._is_tier_guard_failure(e),
            )

    def _run_pending_validation(self):
        """Launch pending auto-validation if queued."""
        if self.is_running:
            return
        pending = getattr(self, "_pending_validation", None)
        if pending is None:
            pending = self._claim_pending_followup_from_notebook("validation")
            if pending is None:
                return
        self._pending_validation = None

        # Anything in the followup_tasks queue is a deliberate request
        # — score-stability variance probes, manual reruns, recovery
        # backfills.  Bypass the tier guard so already-validated
        # candidates can be re-run for variance.
        try:
            self.start_validation(
                result_ids=pending["result_ids"],
                config=pending["config"],
                hypothesis=pending["hypothesis"],
                trigger="score_stability_rerun",
                force=True,
            )
            self._finalize_followup_task(
                pending.get("task_id"),
                success=True,
                stage="validation",
            )
        except Exception as e:
            logger.warning(f"Failed to launch auto-validation: {e}")
            self._finalize_followup_task(
                pending.get("task_id"),
                success=False,
                stage="validation",
                error=str(e),
                permanent_failure=self._is_tier_guard_failure(e),
            )
