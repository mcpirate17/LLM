"""Execution mixin: candidate generation, grammar config, pending escalation."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn

from ...synthesis.grammar import GrammarConfig, batch_generate
from ..native_runner import compile_model_native_first as compile_model
from ...synthesis.validator import validate_graph
from ...synthesis.serializer import graph_to_json, graph_summary
from ..shared_utils import resolve_device

import logging
logger = logging.getLogger(__name__)

from ._types import RunConfig


class _ExecutionCandidatesMixin:
    """Candidate generation, grammar config building, pending escalation."""

    __slots__ = ()

    def _generate_candidates(self, config: RunConfig, n: int,
                             source: str = "graph_synthesis") -> List['ModelCandidate']:
        """Generate candidate models from the specified source.

        source: "graph_synthesis", "morphological_box", or "mixed"
        Returns candidates that pass Stage 0 smoke test.
        """
        from ._types import ModelCandidate

        candidates: List[ModelCandidate] = []
        dev_str = str(resolve_device(config.device))

        if source == "mixed":
            n_morph = int(n * config.morph_ratio)
            n_graph = n - n_morph
            candidates.extend(
                self._generate_candidates(config, n_graph, "graph_synthesis"))
            candidates.extend(
                self._generate_candidates(config, n_morph, "morphological_box"))
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
                            explicit_sparse = str(getattr(config, "morph_sparse_weight_storage", "") or "").strip()
                            if explicit_sparse in sparse_weight_options:
                                fixed_choices["weight_storage"] = explicit_sparse
                            else:
                                fixed_choices["weight_storage"] = sparse_weight_options[i % len(sparse_weight_options)]
                        fixed_routing = str(getattr(config, "morph_compute_routing", "") or "").strip()
                        if fixed_routing:
                            fixed_choices["compute_routing"] = fixed_routing
                        fixed_channel = str(getattr(config, "morph_channel_mixing", "") or "").strip()
                        if fixed_channel:
                            fixed_choices["channel_mixing"] = fixed_channel

                        spec = roll(seed=i + int(time.time() * 1000) % 100000,
                                    generation=0,
                                    fixed=fixed_choices or None)
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
                            candidates.append(ModelCandidate(
                                source="morphological_box",
                                model=model,
                                description=desc,
                                arch_spec=spec,
                                arch_spec_json=_json.dumps(spec.to_dict()),
                                fingerprint=spec.id,
                            ))
                        else:
                            del model
                    except Exception as e:
                        logger.debug(f"Morphological candidate {i} failed: {e}")
                        continue
            except ImportError:
                logger.warning("morphological_box or arch_builder not available")
            return candidates

        # Default: graph_synthesis
        grammar = self._build_grammar_config(config)

        graphs = batch_generate(n, grammar)
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
                    candidates.append(ModelCandidate(
                        source="graph_synthesis",
                        model=model,
                        description=graph_summary(graph),
                        graph=graph,
                        graph_json=graph_to_json(graph),
                        fingerprint=graph.fingerprint(),
                    ))
                else:
                    del model
            except Exception:
                continue

        return candidates

    # ── Training with synthesized programs ──

    def _build_grammar_config(self, config: RunConfig,
                              excluded_ops: Optional[Set[str]] = None,
                              op_weights: Optional[Dict[str, float]] = None) -> GrammarConfig:
        """Create a GrammarConfig from a RunConfig with standardized defaults."""
        from ...synthesis.grammar import GrammarConfig

        # Exotic mode: use the exotic preset as base, then layer on
        # excluded_ops and learned op_weights
        if getattr(config, "_exotic_mode", False):
            grammar = GrammarConfig.exotic(model_dim=config.model_dim)
            # Merge with defaults (non-causal ops) rather than replacing
            grammar.excluded_ops = grammar.excluded_ops | (excluded_ops or set())
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
            return grammar

        # Efficiency mode: use the efficient preset as base
        if getattr(config, "_efficiency_mode", False):
            grammar = GrammarConfig.efficient(model_dim=config.model_dim)
            grammar.excluded_ops = grammar.excluded_ops | (excluded_ops or set())
            if op_weights:
                for op_name, w in op_weights.items():
                    if w < 1.0:
                        grammar.op_weights[op_name] = grammar.op_weights.get(op_name, 1.0) * w
                    else:
                        grammar.op_weights.setdefault(op_name, w)
            return grammar

        # Pick up structured_sparsity_bias from mode recommendation or config
        sparsity_bias = getattr(self, "_structured_sparsity_bias_override",
                                getattr(config, "structured_sparsity_bias", 0.0))

        # Merge API-provided op_weights with learned op_weights
        merged_op_weights = dict(op_weights or {})
        if config.op_weights:
            # API overrides take precedence over learned weights
            merged_op_weights.update(config.op_weights)

        grammar = GrammarConfig(
            model_dim=config.model_dim,
            min_depth=config.min_depth,
            max_depth=min(config.max_depth, 12),
            max_ops=min(config.max_ops, 20),
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
        )
        # Merge learned excluded_ops with defaults (non-causal ops)
        if excluded_ops:
            grammar.excluded_ops = grammar.excluded_ops | excluded_ops
        # Apply specialized weights
        grammar.category_weights["math_space"] = config.math_space_weight
        # Apply custom category weights from API (overrides defaults)
        if config.category_weights:
            grammar.category_weights.update(config.category_weights)

        # Apply Bayesian op priors from compressed learning (optional)
        try:
            from pathlib import Path
            import json as _json
            priors_path = Path("research/runtime/learning/op_priors.json")
            if priors_path.exists():
                payload = _json.loads(priors_path.read_text())
                op_penalties = payload.get("op_penalties", {}) if isinstance(payload, dict) else {}
                if isinstance(op_penalties, dict):
                    for op_name, penalty in op_penalties.items():
                        try:
                            p = float(penalty)
                        except Exception:
                            continue
                        # Convert penalty (0..1) into weight multiplier (1..0.5)
                        mult = max(0.5, 1.0 - 0.5 * max(0.0, min(1.0, p)))
                        grammar.op_weights[op_name] = grammar.op_weights.get(op_name, 1.0) * mult
        except Exception:
            pass

        # Apply cluster-based suggestions (optional)
        try:
            from pathlib import Path
            import json as _json
            sugg_path = Path("research/runtime/learning/cluster_suggestions.json")
            if sugg_path.exists():
                payload = _json.loads(sugg_path.read_text())
                if isinstance(payload, dict):
                    op_weight_suggestions = payload.get("op_weight_suggestions") or payload.get("op_weights") or {}
                    op_penalties = payload.get("op_penalties") or {}
                    op_promotions = payload.get("op_promotions") or {}
                    avoid_patterns = payload.get("avoid_patterns") or []
                    promote_patterns = payload.get("promote_patterns") or []

                    def _apply_mult(op_name: str, mult: float):
                        if not op_name:
                            return
                        m = max(0.2, min(3.0, float(mult)))
                        grammar.op_weights[op_name] = grammar.op_weights.get(op_name, 1.0) * m

                    for op_name, mult in op_weight_suggestions.items():
                        try:
                            _apply_mult(op_name, float(mult))
                        except Exception:
                            continue

                    for op_name, p in op_penalties.items():
                        try:
                            penalty = max(0.0, min(1.0, float(p)))
                        except Exception:
                            continue
                        _apply_mult(op_name, 1.0 - 0.4 * penalty)

                    for op_name, p in op_promotions.items():
                        try:
                            promo = max(0.0, min(1.0, float(p)))
                        except Exception:
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
        except Exception:
            pass

        return grammar

    def _run_pending_investigation(self):
        """Launch pending auto-investigation if queued."""
        pending = getattr(self, "_pending_investigation", None)
        if pending is None:
            return
        self._pending_investigation = None

        if self.is_running:
            return

        try:
            self.start_investigation(
                result_ids=pending["result_ids"],
                config=pending["config"],
                hypothesis=pending["hypothesis"],
            )
        except Exception as e:
            logger.warning(f"Failed to launch auto-investigation: {e}")

    def _run_pending_validation(self):
        """Launch pending auto-validation if queued."""
        pending = getattr(self, "_pending_validation", None)
        if pending is None:
            return
        self._pending_validation = None

        if self.is_running:
            return

        try:
            self.start_validation(
                result_ids=pending["result_ids"],
                config=pending["config"],
                hypothesis=pending["hypothesis"],
                trigger="auto_escalate",
            )
        except Exception as e:
            logger.warning(f"Failed to launch auto-validation: {e}")
