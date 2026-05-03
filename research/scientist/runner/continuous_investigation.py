"""Continuous investigation methods (pre-inv gate + inline investigation), split from continuous.py."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import List, Optional

from ..json_utils import json_safe


from ...eval.perf_budget import evaluate_perf_budget_gate
from ...training.checkpointing import CheckpointManager
from ...training.training_program import synthesize_training_program_batch
from ..notebook import LabNotebook
from ._helpers import (
    _build_source_map,
    _submit_benchmark_eval,
    _submit_v2_probe_eval,
    clear_gpu_memory,
)
from ._lifecycle import _LifecycleMixin
from ..llm.context_experiment import (
    build_investigation_context,
)
from ..shared_utils import resolve_device

import logging

logger = logging.getLogger(__name__)

from ._types import RunConfig, LiveProgress


def _fail_loud(phase: str, message: str, exc: BaseException) -> None:
    logger.exception("%s: %s", phase, message)
    raise RuntimeError(f"{phase}: {message}") from exc


class _ContinuousInvestigationMixin:
    """Pre-investigation gate and inline investigation execution."""

    __slots__ = ()
    _publish_terminal_event = _LifecycleMixin._publish_terminal_event
    _publish_continuous_investigation_terminal_event = (
        _LifecycleMixin._publish_terminal_event
    )
    _fail_experiment_compat = _LifecycleMixin._fail_experiment_compat
    _complete_experiment_compat = _LifecycleMixin._complete_experiment_compat
    _log_learning_event_compat = _LifecycleMixin._log_learning_event_compat

    def _pre_inv_probe(
        self, config: RunConfig, nb: LabNotebook, result_id: str
    ) -> Optional[float]:
        """Stage C: single-seed probe at reduced step count.

        Runs 1 training program at probe_steps_fraction of investigation_steps.
        Returns loss_ratio or None on failure.
        """
        try:
            details = nb.get_program_details([result_id])
            if not details or not details[0]:
                return None
            source = details[0]
            graph_json = source.get("graph_json")
            if not graph_json:
                return None

            probe_config = config.copy()
            probe_config.stage1_steps = max(
                50,
                int(config.investigation_steps * config.pre_inv_probe_steps_fraction),
            )
            probe_config.stage1_batch_size = config.investigation_batch_size
            probe_config.n_programs = 1

            dev = resolve_device(config.device)
            str(dev)

            from research.synthesis.compiler import compile_model

            model = compile_model(graph_json, probe_config, device=dev)
            if model is None:
                return None

            result = self._micro_train(model, probe_config, dev, graph_json=graph_json)
            lr = result.get("loss_ratio") if result else None
            return float(lr) if lr is not None else None
        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            logger.warning("Pre-inv probe failed for %s: %s", result_id[:8], e)
            return None

    def _apply_predictor_filter(
        self, config: RunConfig, nb: LabNotebook, eligible: list
    ) -> list:
        """Filter candidates using the lightweight performance predictor.

        Removes candidates whose predicted investigation loss_ratio exceeds
        config.investigation_predictor_max_lr.
        """
        from ..intelligence.predictor import (
            train as train_predictor,
            predict as predict_lr,
        )
        from ..ml_influence_policy import component_is_allowed

        if not component_is_allowed("investigation_predictor", config):
            logger.info(
                "Investigation predictor requested but blocked by ML trust policy"
            )
            return eligible

        try:
            model = train_predictor(nb)
        except (RuntimeError, ValueError) as e:
            logger.warning("Predictor training failed, skipping filter: %s", e)
            return eligible

        if not model.is_fitted():
            return eligible

        max_lr = config.investigation_predictor_max_lr
        kept = []
        filtered_count = 0
        for row in eligible:
            fp_json = row.get("fingerprint_json")
            if not fp_json:
                kept.append(row)
                continue

            fp_dict = fp_json if isinstance(fp_json, dict) else None
            if fp_dict is None:
                try:
                    fp_dict = json.loads(fp_json)
                except (json.JSONDecodeError, TypeError):
                    kept.append(row)
                    continue

            predicted = predict_lr(
                model,
                fp_dict,
                novelty_score=float(row.get("novelty_score") or 0),
                structural_novelty=float(row.get("structural_novelty") or 0),
            )

            if predicted > max_lr:
                filtered_count += 1
                logger.debug(
                    "Predictor filtered %s: predicted_lr=%.4f > %.4f",
                    str(row.get("result_id", ""))[:8],
                    predicted,
                    max_lr,
                )
            else:
                kept.append(row)

        if filtered_count:
            logger.info(
                "Predictor filter: removed %d/%d candidates (predicted_lr > %.2f)",
                filtered_count,
                len(eligible),
                max_lr,
            )

        return kept

    def _pre_investigation_gate(
        self, config: RunConfig, nb: LabNotebook, leaderboard: list
    ) -> List[str]:
        """Orchestrate three-stage pre-investigation gate.

        Stage A: SQL hard reject (numerical health, stability, gradient path)
        Stage B: Composite readiness score, rank and take top-N
        Stage C: Optional single-seed probe

        Returns filtered, ranked result_ids ready for investigation.
        Falls back to legacy behavior when pre_inv_gate_enabled=False.
        """
        if not config.pre_inv_gate_enabled:
            # Legacy behavior: filter by loss_ratio threshold only
            investigated_fps = nb.get_investigated_fingerprints()
            candidates = [
                e
                for e in leaderboard
                if e.get("tier") == "screening"
                and e.get("screening_loss_ratio") is not None
                and e["screening_loss_ratio"]
                < config.investigation_loss_ratio_threshold
                and "provisional_random_tokens" not in (e.get("tags") or "")
            ]
            if investigated_fps:
                candidates = [
                    c
                    for c in candidates
                    if c.get("graph_fingerprint", c.get("architecture_desc", ""))
                    not in investigated_fps
                ]
            return [
                c["result_id"]
                for c in candidates[: config.auto_investigate_top_n]
                if c.get("result_id")
            ]

        # ── Stage A: Hard reject via SQL ──
        # Uses composite_score as primary gate (not loss_ratio) — models
        # with strong efficiency/novelty/stability deserve investigation
        # even if loss is only moderate.
        eligible = nb.get_investigation_eligible(
            _max_lr=config.pre_inv_max_lr,
            min_stability=config.pre_inv_min_stability,
            min_spectral_norm=config.pre_inv_min_spectral_norm,
            max_spectral_norm=config.pre_inv_max_spectral_norm,
            min_improvement_rate=config.pre_inv_min_improvement_rate,
            _ref_lr_ceiling=self._reference_margin_ceiling(config, nb),
        )

        # Filter out already-investigated fingerprints
        investigated_fps = nb.get_investigated_fingerprints()
        if investigated_fps:
            before = len(eligible)
            eligible = [
                e
                for e in eligible
                if e.get("graph_fingerprint") not in investigated_fps
            ]
            skipped = before - len(eligible)
            if skipped:
                logger.info(
                    "Pre-inv gate: skipped %d already-investigated candidates", skipped
                )

        if not eligible:
            logger.info("Pre-inv gate Stage A: no eligible candidates")
            # Even with zero eligible, reprieve candidates may exist
            if not config.slope_reprieve_enabled:
                return []
            # Fall through to reprieve check below

        # ── Capability-first filter: when the user has opted into
        # capability-first mode, only investigate graphs that contain at
        # least one content-addressed retrieval op. Without this, the
        # investigation queue is dominated by ~19K old simple graphs
        # from before the capability-first templates existed, which
        # score higher on screening composite (they converge faster in
        # 200 steps) but will never produce the binding/induction scores
        # we're targeting. See the 2026-04-17 diagnosis.
        if getattr(config, "_capability_first_mode", False) and eligible:
            _RETRIEVAL_OPS = frozenset(
                {
                    "matmul",
                    "gather_topk",
                    "outer_product",
                    "cosine_similarity",
                    "graph_attention",
                    "softmax_attention",
                    "diff_attention",
                    "latent_attention_compressor",
                    "linear_attention",
                    "gated_linear_attention",
                    "associative_memory",
                }
            )
            before_capfirst = len(eligible)
            capfirst_eligible = []
            for e in eligible:
                gj = e.get("graph_json") or ""
                # Parse the graph and check actual op_name fields —
                # string matching produces false positives from metadata.
                try:
                    g = json.loads(gj) if gj else {}
                    nodes = g.get("nodes") or {}
                    if isinstance(nodes, dict):
                        graph_ops = {
                            n.get("op_name") or n.get("op")
                            for n in nodes.values()
                            if isinstance(n, dict)
                        }
                    else:
                        graph_ops = set()
                    if graph_ops & _RETRIEVAL_OPS:
                        capfirst_eligible.append(e)
                except (json.JSONDecodeError, TypeError):
                    pass
            if capfirst_eligible:
                eligible = capfirst_eligible
                logger.info(
                    "Pre-inv gate: capability-first filter kept %d/%d candidates with retrieval ops",
                    len(eligible),
                    before_capfirst,
                )
            else:
                logger.warning(
                    "Pre-inv gate: capability-first filter found 0/%d candidates with retrieval ops — falling back to full pool",
                    before_capfirst,
                )

        logger.info(
            "Pre-inv gate Stage A: %d candidates pass hard filters", len(eligible)
        )

        # ── Predictor filter: skip candidates with high predicted loss_ratio ──
        if config.investigation_predictor_enabled and eligible:
            eligible = self._apply_predictor_filter(config, nb, eligible)

        # ── Slope reprieve: rescue high-slope candidates that failed Stage A ──
        cycle_reprieve_count = 0
        if config.slope_reprieve_enabled:
            eligible_ids = {r["result_id"] for r in eligible}
            reprieve_candidates = self._get_slope_reprieve_candidates(
                config,
                nb,
                investigated_fps,
                eligible_ids,
            )
            for rc in reprieve_candidates:
                if cycle_reprieve_count >= config.slope_reprieve_max_per_cycle:
                    break
                loss_ratio = float(
                    rc.get("loss_ratio") or rc.get("screening_loss_ratio") or 1.0
                )
                slope = float(rc.get("screening_slope") or 0.0)
                consistent = bool(rc.get("screening_slope_consistent"))
                reprieve_eligible = (
                    float(loss_ratio) < config.slope_reprieve_loss_floor
                    and float(slope) >= config.slope_reprieve_threshold
                    and (not config.slope_reprieve_consistent_required or consistent)
                )
                if reprieve_eligible:
                    rc["_screening_reprieve"] = True
                    rc["_reprieve_reason"] = (
                        f"slope={slope:.4f}"
                        f"_consistent={consistent}"
                        f"_loss={float(loss_ratio):.4f}"
                    )
                    cycle_reprieve_count += 1
                    eligible.append(rc)
                    logger.info(
                        "screening_reprieve_granted result_id=%s loss_ratio=%.4f "
                        "slope=%.4f consistent=%s cycle_reprieve_count=%d",
                        rc["result_id"][:8],
                        float(loss_ratio),
                        float(slope),
                        consistent,
                        cycle_reprieve_count,
                    )
                else:
                    logger.info(
                        "screening_reprieve_denied result_id=%s loss_ratio=%.4f "
                        "slope=%s consistent=%s reason=%s",
                        rc["result_id"][:8],
                        float(loss_ratio),
                        slope,
                        consistent,
                        "above_floor"
                        if float(loss_ratio) >= config.slope_reprieve_loss_floor
                        else "slope_insufficient",
                    )

        if not eligible:
            return []

        # ── Stage B: Composite score + rank ──
        ref_lr = self._get_reference_baseline_lr(nb)

        # Predictor-driven capability multiplier.  The GBM v2 capability heads
        # (P(induction_v2 ≥ 0.30), P(binding_v2 ≥ 0.30)) are the most
        # discriminating signal we have for "will this graph actually learn
        # induction/binding?".  Pre-2026-05-02 this gate ranked purely on
        # screening loss/stability, so the search kept investigating
        # architectures that pass S1 by reducing loss but never form heads
        # (mixture_of_recursions_block: 66.7% S1, 0.000 induction).  We
        # blend capability into the rank: multiplier = 0.5 + cap_score
        # (range 0.5×–1.5×, centred at 1.0 when the head has no signal),
        # so the predictor steers selection without overriding strong
        # screening evidence.
        cap_ensemble = None
        cap_op_stats = None
        cap_extract_features = None
        cap_enrich_features = None
        try:
            from ..intelligence.predictor import load_runtime_ensemble
            from ...synthesis.graph_features import (
                extract_graph_features_bundle as _extract_features_bundle,
                enrich_with_op_stats as _enrich_with_op_stats,
                load_op_stats as _load_op_stats,
            )

            cap_ensemble = load_runtime_ensemble()
            if cap_ensemble is not None and not cap_ensemble.is_fitted():
                cap_ensemble = None
            if (
                cap_ensemble is not None
                and hasattr(cap_ensemble, "has_capability_head")
                and not cap_ensemble.has_capability_head()
            ):
                # Ensemble fitted but no v2 capability head — degrade silently
                # to no-op (multiplier ≡ 1.0 via predict_capability_score==0.5).
                pass
            if cap_ensemble is not None:
                cap_op_stats = _load_op_stats(str(nb.db_path))
                cap_extract_features = _extract_features_bundle
                cap_enrich_features = _enrich_with_op_stats
        except Exception as exc:  # noqa: BLE001 - capability blend is best-effort
            logger.debug("Capability blend unavailable: %s", exc)
            cap_ensemble = None

        cap_boosted = 0
        cap_penalised = 0
        for row in eligible:
            base = LabNotebook.compute_pre_investigation_score(row, best_ref_lr=ref_lr)
            # Judgment boost: up to +15% for high-confidence candidates
            j = row.get("judgment_score")
            if j is not None and isinstance(j, (int, float)) and j > 0.5:
                base *= 1.0 + 0.15 * min(1.0, (j - 0.5) * 2.0)
            # Capability-score multiplier (predictor v2 heads).
            cap_score: float = 0.5  # no-signal default
            if cap_ensemble is not None and cap_extract_features is not None:
                try:
                    graph_json_str = row.get("graph_json") or ""
                    if graph_json_str:
                        graph_dict = json.loads(graph_json_str)
                        feats, ops = cap_extract_features(graph_dict)
                        if feats:
                            for op in ops:
                                if op:
                                    feats[f"op_{op}"] = feats.get(f"op_{op}", 0.0) + 1.0
                            cap_enrich_features(feats, ops, preloaded=cap_op_stats)
                            cap_score = float(
                                cap_ensemble.predict_capability_score(
                                    graph_features=feats
                                )
                            )
                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                    AttributeError,
                ) as cap_exc:
                    logger.debug(
                        "Capability score skipped for %s: %s",
                        str(row.get("result_id") or "")[:10],
                        cap_exc,
                    )
            cap_multiplier = 0.5 + max(0.0, min(1.0, cap_score))
            base *= cap_multiplier
            if cap_multiplier > 1.05:
                cap_boosted += 1
            elif cap_multiplier < 0.95:
                cap_penalised += 1
            row["_pre_inv_score"] = base
            row["_pre_inv_capability_score"] = cap_score
            # Apply reprieve score multiplier
            multiplier = (
                config.slope_reprieve_score_multiplier
                if row.get("_screening_reprieve")
                else 1.0
            )
            row["_pre_inv_effective_score"] = base * multiplier

        if cap_ensemble is not None and (cap_boosted or cap_penalised):
            logger.info(
                "Pre-inv gate capability blend: %d boosted (cap≥0.55) / "
                "%d penalised (cap≤0.45) / %d neutral across %d candidates",
                cap_boosted,
                cap_penalised,
                len(eligible) - cap_boosted - cap_penalised,
                len(eligible),
            )

        eligible.sort(key=lambda r: r.get("_pre_inv_effective_score", 0), reverse=True)
        top_n = eligible[: config.pre_inv_top_n]

        # Persist scores to leaderboard
        for row in eligible:
            try:
                nb.conn.execute(
                    "UPDATE leaderboard SET pre_inv_score = ? WHERE result_id = ?",
                    (row["_pre_inv_score"], row["result_id"]),
                )
            except sqlite3.OperationalError as exc:
                _fail_loud(
                    "continuous_investigation",
                    f"failed to persist pre-inv score for {row['result_id'][:8]}",
                    exc,
                )
        try:
            nb.conn.commit()
        except sqlite3.OperationalError as exc:
            _fail_loud(
                "continuous_investigation",
                "failed to commit pre-inv scores",
                exc,
            )

        logger.info(
            "Pre-inv gate Stage B: top %d scored [%s]",
            len(top_n),
            ", ".join(
                f"{r['result_id'][:8]}={r.get('_pre_inv_effective_score', 0):.1f}"
                f"{'(R)' if r.get('_screening_reprieve') else ''}"
                for r in top_n
            ),
        )

        # ── Stage C: Optional probe + reprieve eval ──
        if config.pre_inv_probe_enabled:
            probed = []
            for row in top_n:
                probe_lr = self._pre_inv_probe(config, nb, row["result_id"])
                if probe_lr is not None and probe_lr > config.pre_inv_probe_max_lr:
                    logger.info(
                        "Pre-inv probe rejected %s (lr=%.3f > %.3f)",
                        row["result_id"][:8],
                        probe_lr,
                        config.pre_inv_probe_max_lr,
                    )
                    continue
                probed.append(row)
            top_n = probed

        # Reprieve eval: 150-step extended screening for reprieve candidates
        if config.slope_reprieve_enabled:
            reprieve_passed = []
            for row in top_n:
                if not row.get("_screening_reprieve"):
                    reprieve_passed.append(row)
                    continue
                reprieve_lr = self._run_reprieve_eval(config, nb, row)
                if reprieve_lr is None or reprieve_lr >= 0.40:
                    logger.info(
                        "reprieve_eval_failed result_id=%s reprieve_loss_ratio=%s "
                        "original_loss_ratio=%s",
                        row["result_id"][:8],
                        f"{reprieve_lr:.4f}" if reprieve_lr is not None else "None",
                        row.get("loss_ratio"),
                    )
                    continue
                logger.info(
                    "reprieve_eval_passed result_id=%s reprieve_loss_ratio=%.4f "
                    "original_loss_ratio=%s",
                    row["result_id"][:8],
                    reprieve_lr,
                    row.get("loss_ratio"),
                )
                row["_reprieve_eval_loss_ratio"] = reprieve_lr
                reprieve_passed.append(row)
            top_n = reprieve_passed

        result_ids = [r["result_id"] for r in top_n if r.get("result_id")]

        # ── Stage D: Recipe re-roll for screened_out frontier models ──
        # Models that failed investigation (robustness < 0.5) but have
        # frontier-competitive real-token quality deserve reinvestigation
        # with fresh training programs before being permanently buried.
        reinvest_ids = self._get_reinvestigation_candidates(nb, exclude=set(result_ids))
        if reinvest_ids:
            logger.info(
                "Pre-inv gate Stage D: %d screened_out frontier models queued for recipe re-roll",
                len(reinvest_ids),
            )
            result_ids.extend(reinvest_ids)

        return result_ids

    _MAX_REINVESTIGATION_ATTEMPTS = 2

    def _get_reinvestigation_candidates(
        self,
        nb: LabNotebook,
        exclude: set,
        limit: int = 3,
    ) -> List[str]:
        """Find screened_out models with WikiText quality above the investigation tier.

        These are architectures that failed robustness (typically 1/3 training
        programs passed) but demonstrably generalise on real tokens.  They get
        reinvestigated with fresh training programs — same architecture, new
        recipe — before any architectural mutation is considered.

        Capped at ``_MAX_REINVESTIGATION_ATTEMPTS`` per model to prevent
        infinite re-roll loops.
        """
        max_attempts = self._MAX_REINVESTIGATION_ATTEMPTS
        try:
            rows = nb.conn.execute(
                """
                SELECT l.result_id, l.wikitext_score, l.investigation_robustness,
                       COALESCE(l.reinvestigation_count, 0) AS reinvest_count
                FROM leaderboard l
                WHERE l.tier = 'screened_out'
                  AND l.wikitext_score IS NOT NULL
                  AND l.wikitext_score > (
                      SELECT COALESCE(MAX(l2.wikitext_score), 0)
                      FROM leaderboard l2
                      WHERE l2.tier = 'investigation'
                  )
                  AND COALESCE(l.investigation_robustness, 0) < 0.5
                  AND COALESCE(l.reinvestigation_count, 0) < ?
                ORDER BY l.wikitext_score DESC
                LIMIT ?
            """,
                (max_attempts, limit + len(exclude)),
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.debug("Reinvestigation query failed: %s", e)
            return []

        candidates = [
            r["result_id"]
            for r in rows
            if r["result_id"] and r["result_id"] not in exclude
        ][:limit]

        # Increment reinvestigation count for selected candidates
        for rid in candidates:
            try:
                nb.conn.execute(
                    "UPDATE leaderboard SET reinvestigation_count = COALESCE(reinvestigation_count, 0) + 1 "
                    "WHERE result_id = ?",
                    (rid,),
                )
            except sqlite3.OperationalError as exc:
                _fail_loud(
                    "continuous_investigation",
                    f"failed to bump reinvestigation count for {rid[:8]}",
                    exc,
                )
            logger.info(
                "  Recipe re-roll candidate: %s (wikitext_score above investigation tier)",
                rid[:8],
            )
        if candidates:
            try:
                nb.conn.commit()
            except sqlite3.OperationalError as exc:
                _fail_loud(
                    "continuous_investigation",
                    "failed to commit reinvestigation counts",
                    exc,
                )

        return candidates

    def _get_slope_reprieve_candidates(
        self,
        config: RunConfig,
        nb: LabNotebook,
        investigated_fps: set,
        already_eligible_ids: set,
    ) -> list:
        """Query for screening candidates with high loss_ratio but good slope.

        Returns rows that passed health checks but have loss_ratio >= 0.40
        and screening_slope data available. Caller filters by threshold/consistency.
        """
        try:
            rows = nb.conn.execute(
                """SELECT pr.*, l.entry_id, l.tier, l.composite_score,
                          l.screening_loss_ratio, l.screening_novelty,
                          l.pre_inv_score, l.is_reference, l.reference_name
                   FROM program_results pr
                   JOIN leaderboard l ON l.result_id = pr.result_id
                   WHERE l.tier = 'screening'
                     AND COALESCE(l.is_reference, 0) = 0
                     AND pr.stage1_passed = 1
                     AND COALESCE(pr.has_nan_grad, 0) = 0
                     AND COALESCE(pr.has_nan_output, 0) = 0
                     AND COALESCE(pr.has_inf_output, 0) = 0
                     AND COALESCE(pr.has_zero_grad, 0) = 0
                     AND COALESCE(pr.graph_has_gradient_path, 1) = 1
                     AND pr.loss_ratio >= 0.40
                     AND pr.screening_slope IS NOT NULL
                   ORDER BY pr.screening_slope DESC
                   LIMIT ?""",
                (config.slope_reprieve_max_per_cycle * 3,),
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.debug("Slope reprieve query failed: %s", e)
            return []

        candidates = []
        for r in rows:
            row = dict(r)
            rid = row.get("result_id")
            fp = row.get("graph_fingerprint")
            if not rid or rid in already_eligible_ids:
                continue
            if investigated_fps and fp in investigated_fps:
                continue
            candidates.append(row)
        return candidates

    def _run_reprieve_eval(
        self,
        config: RunConfig,
        nb: LabNotebook,
        row: dict,
    ) -> Optional[float]:
        """Run extended screening eval for a reprieve candidate.

        Returns loss_ratio from the extended eval, or None on failure.
        """
        from ..shared_utils import resolve_device

        result_id = row["result_id"]
        try:
            details = nb.get_program_details([result_id])
            if not details or not details[0]:
                return None
            source = details[0]
            graph_json = source.get("graph_json")
            if not graph_json:
                return None

            reprieve_config = config.copy()
            reprieve_config.stage1_steps = config.slope_reprieve_eval_steps
            reprieve_config.stage1_batch_size = config.stage1_batch_size
            reprieve_config.n_programs = 1

            dev = resolve_device(config.device)

            from research.synthesis.compiler import compile_model

            model = compile_model(graph_json, reprieve_config, device=dev)
            if model is None:
                return None

            result = self._micro_train(
                model, reprieve_config, dev, graph_json=graph_json
            )
            lr = result.get("loss_ratio") if result else None
            return float(lr) if lr is not None else None
        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            logger.warning("Reprieve eval failed for %s: %s", result_id[:8], e)
            return None

    def _reference_margin_ceiling(
        self, config: RunConfig, nb: LabNotebook
    ) -> Optional[float]:
        """Convert the reference margin knob into a concrete Stage-A LR ceiling."""
        best_ref_lr = self._get_reference_baseline_lr(nb)
        if best_ref_lr is None:
            return None
        margin = max(0.1, float(config.pre_inv_reference_margin or 1.0))
        return float(best_ref_lr) * margin

    def _inline_investigate_candidate_training(
        self,
        config: RunConfig,
        inv_config: RunConfig,
        source_result_id: str,
        source: dict,
        exp_id: str,
        prog_idx: int,
        result_ids: list,
        ckpt,
        dev,
        results: dict,
    ) -> tuple:
        """Train all programs for one investigation candidate.

        Returns (tp_results, best_inv_model, training_programs, tp_sched).
        """
        graph_json_str = source.get("graph_json")
        arch_spec_json_str = source.get("arch_spec_json")
        model_source = source.get("model_source") or "graph_synthesis"

        # Generate training programs (queue-level scheduling telemetry)
        training_programs, tp_sched = synthesize_training_program_batch(
            n_programs=config.n_training_programs,
            n_steps=config.investigation_steps,
            max_seq_len=config.max_seq_len,
            seed_offset=prog_idx * 1000,
        )
        results.setdefault("training_program_scheduling", []).append(
            {
                "result_id": source_result_id,
                **tp_sched,
            }
        )

        # Test each (model x training_program) pair
        tp_results = []
        _best_inv_model = None
        _best_inv_model_lr = float("inf")
        for tp_i, tp in enumerate(training_programs):
            if self._stop_event.is_set():
                break

            # Reconstruct model fresh for each training program
            try:
                model = self._build_model_from_source(
                    model_source,
                    arch_spec_json_str,
                    graph_json_str,
                    config,
                    seq_len_override=config.max_seq_len,
                )
                if model is None:
                    raise RuntimeError(f"No model built for {source_result_id[:8]}")
            except (RuntimeError, ValueError, TypeError) as e:
                _fail_loud(
                    "continuous_investigation",
                    f"model reconstruction failed for {source_result_id[:8]} "
                    f"training program {tp_i + 1}/{len(training_programs)}",
                    e,
                )

            self._emit_event(
                "investigation_progress",
                {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": len(result_ids),
                    "source_result_id": source_result_id,
                    "training_program": tp_i + 1,
                    "total_programs": len(training_programs),
                    "status": f"training with {tp.name}",
                },
            )

            resume_state = ckpt.load_phase(exp_id, "investigation", prog_idx, tp_i)
            base_ctx = {"exp_id": exp_id, "phase": "investigation"}
            train_seed = self._stable_seed(
                exp_id, source_result_id, tp_i, "investigation"
            )
            self._live_training_context = {
                **base_ctx,
                "source_result_id": source_result_id,
                "candidate_index": prog_idx + 1,
                "total_candidates": len(result_ids),
                "training_program_index": tp_i + 1,
                "total_training_programs": len(training_programs),
                "training_program_label": tp.name,
                "training_seed": train_seed,
                "run_kind": "investigation",
                "checkpoint_manager": ckpt,
                "checkpoint_phase": "investigation",
                "checkpoint_candidate_idx": prog_idx,
                "checkpoint_seed_idx": tp_i,
                "checkpoint_interval_steps": int(
                    getattr(config, "phase_checkpoint_step_interval", 0) or 0
                ),
                "checkpoint_resume_state": (
                    resume_state
                    if resume_state and int(resume_state.get("step", 0) or 0) > 0
                    else None
                ),
            }
            try:
                tp_result = self._train_with_program(
                    model,
                    tp,
                    inv_config,
                    dev,
                    seed=train_seed,
                )
            finally:
                self._live_training_context = base_ctx
            tp_results.append(
                {
                    "training_program": tp.name,
                    "passed": tp_result.get("passed", False),
                    "loss_ratio": tp_result.get("loss_ratio"),
                    "final_loss": tp_result.get("final_loss"),
                }
            )

            # Retain the best-performing model for post-investigation
            # fingerprint completion (needs converged representations).
            _this_lr = tp_result.get("loss_ratio")
            if _this_lr is not None and (
                _best_inv_model is None or _this_lr < _best_inv_model_lr
            ):
                if _best_inv_model is not None:
                    del _best_inv_model
                _best_inv_model = model
                _best_inv_model_lr = _this_lr
            else:
                del model
            clear_gpu_memory()

        # Skip candidates where no training program could reconstruct the model
        if not tp_results:
            raise RuntimeError(
                f"Continuous investigation aborted for {source_result_id[:8]}: "
                f"model failed to reconstruct for all {len(training_programs)} "
                "training programs"
            )

        return tp_results, _best_inv_model, training_programs, tp_sched

    def _inline_investigate_fingerprint_completion(
        self,
        config: RunConfig,
        nb: LabNotebook,
        source: dict,
        source_result_id: str,
        best_inv_model,
        dev,
    ) -> tuple:
        """Run post-investigation fingerprint completion for one candidate.

        Returns (fingerprint_attempted, fingerprint_completed, investigation_passed_override).
        investigation_passed_override is True if fingerprint failure should downgrade
        the investigation_passed flag to False, False otherwise.
        """
        _fingerprint_completed = False
        _fingerprint_attempted = False
        _fp_dict = source.get("_behavioral_fingerprint")
        if best_inv_model is None or _fp_dict is None:
            return _fingerprint_attempted, _fingerprint_completed, False

        _fingerprint_attempted = True
        from ...eval.fingerprint import (
            BehavioralFingerprint,
        )
        from ...eval.fingerprint_runtime import (
            complete_fingerprint_post_investigation,
        )

        _fp = BehavioralFingerprint(
            **{
                k: v
                for k, v in _fp_dict.items()
                if k
                in {f.name for f in BehavioralFingerprint.__dataclass_fields__.values()}
            }
        )
        if _fp.fingerprint_completed_post_investigation:
            _fingerprint_completed = True
        else:
            # Attempt fingerprint completion with one retry
            for _attempt in range(2):
                try:
                    _fp = complete_fingerprint_post_investigation(
                        _fp,
                        best_inv_model,
                        seq_len=min(64, config.max_seq_len),
                        model_dim=config.model_dim,
                        vocab_size=config.vocab_size,
                        device=str(dev),
                    )
                    if _fp.fingerprint_completed_post_investigation:
                        _fingerprint_completed = True
                        _fp_dict_updated = _fp.to_dict()
                        source["_behavioral_fingerprint"] = _fp_dict_updated
                        source["novelty_confidence"] = (
                            0.9
                            if _fp.quality == "full"
                            else 0.4 + (_fp.analyses_succeeded * 0.1)
                            if _fp.quality == "partial"
                            else 0.3
                        )
                        source.update(
                            nb._behavioral_fingerprint_program_fields(
                                _fp_dict_updated,
                                novelty_confidence=source["novelty_confidence"],
                            )
                        )
                        nb.sync_behavioral_fingerprint_result(
                            result_id=source_result_id,
                            fp_payload=_fp_dict_updated,
                            novelty_confidence=source["novelty_confidence"],
                        )
                        logger.info(
                            "post_investigation_fingerprint_completed: "
                            "result_id=%s novelty_score=%.4f "
                            "novelty_valid=%s cka_source=%s attempt=%d",
                            source_result_id[:12],
                            _fp.novelty_score,
                            _fp.novelty_valid_for_promotion,
                            _fp.cka_source,
                            _attempt + 1,
                        )
                        break
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.error(
                        "post_investigation_fingerprint_failed: "
                        "result_id=%s attempt=%d error=%s",
                        source_result_id[:12],
                        _attempt + 1,
                        str(e),
                    )

        _should_downgrade = False
        if not _fingerprint_completed:
            _should_downgrade = True
            logger.warning(
                "investigation_fingerprint_incomplete: "
                "result_id=%s — downgrading investigation_passed to False",
                source_result_id[:12],
            )

        return _fingerprint_attempted, _fingerprint_completed, _should_downgrade

    def _record_inline_investigation_candidate(
        self,
        config: RunConfig,
        nb: LabNotebook,
        source: dict,
        source_result_id: str,
        exp_id: str,
        prog_idx: int,
        results: dict,
        tp_results: list,
        training_programs: list,
        tp_sched: dict,
        n_passed: int,
        robustness: float,
        best_tp: Optional[dict],
        best_lr: Optional[float],
        screening_lr,
        lr_multiplier,
        brittle_risk: bool,
        investigation_passed: bool,
        fingerprint_attempted: bool,
        fingerprint_completed: bool,
        dev,
        ckpt,
    ) -> None:
        """Record investigation results, submit benchmarks, save checkpoint."""
        graph_json_str = source.get("graph_json")
        arch_spec_json_str = source.get("arch_spec_json")
        model_source = source.get("model_source") or "graph_synthesis"

        _fp_incomplete = fingerprint_attempted and not fingerprint_completed
        investigation_entry = {
            "result_id": source_result_id,
            "robustness": robustness,
            "best_loss_ratio": best_lr,
            "screening_loss_ratio": screening_lr,
            "baseline_loss_ratio": source.get("baseline_loss_ratio"),
            "novelty_confidence": source.get("novelty_confidence"),
            "loss_ratio_multiplier": lr_multiplier,
            "brittle_risk": brittle_risk,
            "investigation_passed": investigation_passed,
            "fingerprint_incomplete": _fp_incomplete,
            "n_programs_passed": n_passed,
            "n_programs_tested": len(tp_results),
            "best_training_program": best_tp.get("training_program")
            if best_tp
            else None,
            "training_program_scheduling_avg_ms": tp_sched.get("scheduling_avg_ms"),
            "training_program_scheduling_max_ms": tp_sched.get("scheduling_max_ms"),
        }
        results["investigation_results"].append(investigation_entry)

        if best_lr and (
            results["best_loss_ratio"] is None or best_lr < results["best_loss_ratio"]
        ):
            results["best_loss_ratio"] = best_lr
        source_novelty = source.get("novelty_score")
        if source_novelty is not None and (
            results["best_novelty_score"] is None
            or source_novelty > results["best_novelty_score"]
        ):
            results["best_novelty_score"] = source_novelty

        # Update leaderboard
        best_tp_json = None
        if best_tp and best_tp.get("training_program"):
            for tp in training_programs:
                if tp.name == best_tp["training_program"]:
                    best_tp_json = json.dumps(json_safe(tp.to_dict()))
                    break

        # Submit benchmark evals to background thread so the
        # investigation loop can proceed to the next candidate.
        self._submit_investigation_eval(
            nb=nb,
            config=config,
            exp_id=exp_id,
            source=source,
            source_result_id=source_result_id,
            model_source=model_source,
            graph_json_str=graph_json_str,
            arch_spec_json_str=arch_spec_json_str,
            n_passed=n_passed,
            n_programs_tested=len(training_programs),
            best_lr=best_lr,
            best_tp_json=best_tp_json,
            robustness=robustness,
            investigation_passed=investigation_passed,
            fingerprint_incomplete=_fp_incomplete,
            dev=dev,
        )

        try:
            ckpt.save_phase(
                experiment_id=exp_id,
                phase="investigation",
                candidate_idx=prog_idx + 1,
                seed_idx=0,
                model_state_dict={},
                optimizer_state_dict={},
                step=0,
                metrics={"completed_candidate": prog_idx},
            )
            ckpt.save_phase(
                experiment_id=exp_id,
                phase="investigation",
                candidate_idx=-1,
                seed_idx=0,
                model_state_dict={},
                optimizer_state_dict={},
                step=0,
                metrics={"candidate_idx": prog_idx + 1},
            )
        except (OSError, RuntimeError) as e:
            logger.warning(
                "Continuous investigation checkpoint save failed for candidate %d: %s",
                prog_idx + 1,
                e,
            )

    def _submit_investigation_eval(
        self,
        *,
        nb: LabNotebook,
        config: RunConfig,
        exp_id: str,
        source: dict,
        source_result_id: str,
        model_source: str,
        graph_json_str,
        arch_spec_json_str,
        n_passed: int,
        n_programs_tested: int,
        best_lr,
        best_tp_json,
        robustness: float,
        investigation_passed: bool,
        fingerprint_incomplete: bool,
        dev,
    ) -> None:
        """Dispatch to benchmark eval (s1-passed) or v2-probe-only (s1-failed) and register the future."""
        if n_passed > 0:
            future = _submit_benchmark_eval(
                nb=nb,
                exp_id=exp_id,
                source_result_id=source_result_id,
                source=source,
                model_source=model_source,
                graph_json_str=graph_json_str,
                arch_spec_json_str=arch_spec_json_str,
                n_passed=n_passed,
                n_programs_tested=n_programs_tested,
                best_lr=best_lr,
                best_tp_json=best_tp_json,
                robustness=robustness,
                investigation_passed=investigation_passed,
                config=config,
                dev=dev,
                cached_json_load=self._cached_json_load,
                fingerprint_incomplete=fingerprint_incomplete,
                stop_event=self._stop_event,
            )
            kind = "benchmark"
        else:
            future = _submit_v2_probe_eval(
                nb=nb,
                exp_id=exp_id,
                source_result_id=source_result_id,
                source=source,
                model_source=model_source,
                graph_json_str=graph_json_str,
                arch_spec_json_str=arch_spec_json_str,
                n_passed=n_passed,
                n_programs_tested=n_programs_tested,
                best_lr=best_lr,
                best_tp_json=best_tp_json,
                robustness=robustness,
                investigation_passed=investigation_passed,
                config=config,
                dev=dev,
                cached_json_load=self._cached_json_load,
                stop_event=self._stop_event,
                fingerprint_incomplete=fingerprint_incomplete,
            )
            kind = "v2-probe"
        self._register_investigation_eval_future(
            exp_id=exp_id,
            future=future,
            kind=kind,
            source_result_id=source_result_id,
        )

    def _inline_investigate_one_candidate(
        self,
        config: RunConfig,
        inv_config: RunConfig,
        nb: LabNotebook,
        source_result_id: str,
        source: dict,
        exp_id: str,
        prog_idx: int,
        result_ids: list,
        ckpt,
        dev,
        results: dict,
    ) -> None:
        """Run training, fingerprint, and recording for one investigation candidate."""
        # Phase 1: Train all programs for this candidate
        tp_results, _best_inv_model, training_programs, tp_sched = (
            self._inline_investigate_candidate_training(
                config=config,
                inv_config=inv_config,
                source_result_id=source_result_id,
                source=source,
                exp_id=exp_id,
                prog_idx=prog_idx,
                result_ids=result_ids,
                ckpt=ckpt,
                dev=dev,
                results=results,
            )
        )

        # Compute robustness
        n_passed = sum(1 for r in tp_results if r.get("passed"))
        robustness = n_passed / max(len(tp_results), 1)
        best_tp = min(
            (r for r in tp_results if r.get("loss_ratio") is not None),
            key=lambda r: r["loss_ratio"],
            default=None,
        )
        best_lr = best_tp["loss_ratio"] if best_tp else None
        screening_lr = source.get("loss_ratio")
        lr_multiplier = self._investigation_loss_multiplier(screening_lr, best_lr)
        brittle_risk = lr_multiplier is not None and lr_multiplier > float(
            config.investigation_max_loss_ratio_multiplier
        )

        if n_passed > 0:
            results["stage1_passed"] += 1
        results["stage0_passed"] += 1
        results["stage05_passed"] += 1

        # Gate: pass investigation if loss quality is good enough.
        investigation_passed_early = (best_lr or 1.0) < 0.5 and (
            not brittle_risk or (best_lr is not None and best_lr < 0.3)
        )

        # Phase 2: Post-investigation fingerprint completion
        fp_attempted, fp_completed, fp_downgrade = (
            self._inline_investigate_fingerprint_completion(
                config=config,
                nb=nb,
                source=source,
                source_result_id=source_result_id,
                best_inv_model=_best_inv_model,
                dev=dev,
            )
        )
        if fp_downgrade:
            investigation_passed_early = False

        if _best_inv_model is not None:
            del _best_inv_model
            _best_inv_model = None
            clear_gpu_memory()

        # Brittle risk override: if the investigation LR is good on
        # its own merits (< 0.3), don't let the screening->investigation
        investigation_passed = investigation_passed_early

        # Phase 3: Record results, submit benchmarks, save checkpoint
        self._record_inline_investigation_candidate(
            config=config,
            nb=nb,
            source=source,
            source_result_id=source_result_id,
            exp_id=exp_id,
            prog_idx=prog_idx,
            results=results,
            tp_results=tp_results,
            training_programs=training_programs,
            tp_sched=tp_sched,
            n_passed=n_passed,
            robustness=robustness,
            best_tp=best_tp,
            best_lr=best_lr,
            screening_lr=screening_lr,
            lr_multiplier=lr_multiplier,
            brittle_risk=brittle_risk,
            investigation_passed=investigation_passed,
            fingerprint_attempted=fp_attempted,
            fingerprint_completed=fp_completed,
            dev=dev,
            ckpt=ckpt,
        )

    def _inline_investigation_loop(
        self,
        config: RunConfig,
        nb: LabNotebook,
        result_ids: list,
        inv_map: dict,
        exp_id: str,
        ckpt,
    ) -> dict:
        """Run the candidate loop for inline investigation.

        Returns the aggregated results dict.
        """
        resume_from_candidate = 0
        ckpt_state = ckpt.load_phase(exp_id, "investigation", -1, 0)
        if ckpt_state:
            resume_from_candidate = CheckpointManager.phase_resume_candidate_idx(
                ckpt_state
            )
            logger.info(
                "Resuming continuous investigation from candidate %d",
                resume_from_candidate,
            )

        results = {
            "total": len(result_ids),
            "stage0_passed": 0,
            "stage05_passed": 0,
            "stage1_passed": 0,
            "novel_count": 0,
            "best_loss_ratio": None,
            "best_novelty_score": None,
            "survivors": [],
            "investigation_results": [],
        }

        dev = resolve_device(config.device)
        str(dev)

        inv_config = config.copy()
        inv_config.stage1_steps = config.investigation_steps
        inv_config.stage1_batch_size = config.investigation_batch_size
        # Scale early stopping for longer investigation runs.
        step_ratio = config.investigation_steps / max(config.stage1_steps, 1)
        inv_config.early_stop_patience = int(config.early_stop_patience * step_ratio)
        inv_config.early_stop_min_steps = int(config.early_stop_min_steps * step_ratio)

        # Fetch all sources at once to avoid N+1 queries
        _build_source_map(nb, result_ids)

        for prog_idx, source_result_id in enumerate(result_ids):
            if prog_idx < resume_from_candidate:
                continue
            if self._stop_event.is_set():
                break

            # Cost check mid-investigation
            if (
                config.max_cost_dollars > 0
                and self.aria.total_cost >= config.max_cost_dollars
            ):
                logger.info("Cost limit reached during investigation")
                break

            self._update_progress(
                current_program=prog_idx + 1,
                status="investigating",
                aria_message=(
                    f"Investigating {prog_idx + 1}/{len(result_ids)}: "
                    f"{source_result_id[:8]}... "
                    f"({config.n_training_programs} training programs)"
                ),
            )

            self._emit_event(
                "investigation_progress",
                {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                },
            )

            # Fetch source program
            source = inv_map.get(source_result_id)
            if source is None:
                continue

            try:
                self._inline_investigate_one_candidate(
                    config=config,
                    inv_config=inv_config,
                    nb=nb,
                    source_result_id=source_result_id,
                    source=source,
                    exp_id=exp_id,
                    prog_idx=prog_idx,
                    result_ids=result_ids,
                    ckpt=ckpt,
                    dev=dev,
                    results=results,
                )
            except Exception as exc:
                logger.warning(
                    "Continuous investigation candidate %s failed; skipping: %s",
                    source_result_id[:8],
                    exc,
                )
                results.setdefault("candidate_failures", []).append(
                    {
                        "result_id": source_result_id,
                        "candidate_index": prog_idx,
                        "error": str(exc),
                    }
                )
                self._emit_event(
                    "investigation_progress",
                    {
                        "experiment_id": exp_id,
                        "current": prog_idx + 1,
                        "total": len(result_ids),
                        "source_result_id": source_result_id,
                        "status": "candidate_failed",
                        "error": str(exc),
                    },
                )
                clear_gpu_memory()
                continue

        if not results["investigation_results"] and results.get("candidate_failures"):
            first_failure = results["candidate_failures"][0]
            raise RuntimeError(
                "continuous_investigation: all "
                f"{len(results['candidate_failures'])}/{len(result_ids)} candidates "
                f"failed; first error for {first_failure['result_id'][:8]}: "
                f"{first_failure['error']}"
            )

        return results

    def _run_inline_investigation(
        self,
        config: RunConfig,
        nb: LabNotebook,
        leaderboard: list,
        n_experiments: int,
        limit_str: str,
        mode_reasoning: str,
    ):
        """Execute investigation phase inline (not threaded) for continuous mode."""
        # Use pre-investigation gate for candidate selection
        result_ids = self._pre_investigation_gate(config, nb, leaderboard)
        if not result_ids:
            self._run_continuous_synthesis(
                config, nb, n_experiments, limit_str, mode_reasoning
            )
            return

        # Build context for hypothesis formulation
        inv_map = _build_source_map(nb, result_ids)
        inv_context = build_investigation_context(list(inv_map.values()), leaderboard)
        hypothesis = self.aria.formulate_investigation_hypothesis(context=inv_context)
        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="investigation",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="llm_context",
                llm_used=True,
                fallback_used=False,
                used_context=True,
            ),
            created_by="inline_investigation",
        )

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="investigating",
                total_programs=len(result_ids),
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=(
                    f"[{limit_str}|investigation] Studying {len(result_ids)} candidates"
                ),
            )

        self._emit_event(
            "investigation_started",
            {
                "experiment_id": exp_id,
                "n_candidates": len(result_ids),
            },
        )
        ckpt = CheckpointManager(config.checkpoint_dir)

        self._live_training_context = {"exp_id": exp_id, "phase": "investigation"}
        try:
            results = self._inline_investigation_loop(
                config,
                nb,
                result_ids,
                inv_map,
                exp_id,
                ckpt,
            )
            self._finalize_inline_investigation(
                config=config,
                nb=nb,
                exp_id=exp_id,
                hypothesis=hypothesis,
                n_experiments=n_experiments,
                results=results,
            )

        except Exception as e:
            logger.warning(f"Inline investigation failed: {e}")
            self._publish_terminal_event(
                producer="runner.continuous_investigation",
                event_type="experiment_failed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "error": str(e),
                    "results": None,
                    "mode": "continuous_investigation",
                },
            )
            self._fail_experiment_compat(
                nb=nb,
                experiment_id=exp_id,
                error=str(e),
            )
            self._emit_event(
                "investigation_failed",
                {
                    "experiment_id": exp_id,
                    "error": str(e),
                },
            )
        finally:
            self._live_training_context = None

    def _finalize_inline_investigation(
        self,
        *,
        config: RunConfig,
        nb: LabNotebook,
        exp_id: str,
        hypothesis: str,
        n_experiments: int,
        results: dict,
    ) -> None:
        """Run post-loop finalization: drain background evals, score, summarize, escalate.

        Extracted from ``_run_inline_investigation`` solely to keep the
        outer method under the 150-line structural limit.  All side
        effects and state mutations happen here exactly as before — same
        events emitted, same notebook writes, same auto-escalation.
        """
        self._emit_event(
            "investigation_training_complete",
            {
                "experiment_id": exp_id,
                "results": results,
            },
        )
        self._update_progress(
            status="finalizing",
            aria_message=(
                "Investigation training complete; finalizing benchmark/probe writes."
            ),
        )
        eval_status = self._wait_for_investigation_eval_futures(exp_id)
        if eval_status:
            results["background_eval_status"] = eval_status

        # Complete experiment with LLM analysis
        results["perf_report"] = self._build_experiment_perf_report(results)
        results["perf_budget_gate"] = evaluate_perf_budget_gate(results["perf_report"])
        context = self._build_rich_context_for_experiment(
            results, config, hypothesis, nb
        )
        summary = self.aria.experiment_summary(results, context=context)
        llm_analysis = self.aria.analyze_results(results, context=context)
        insights = self._analyze_results(results, exp_id, nb, context=context)

        self._publish_terminal_event(
            producer="runner.continuous_investigation",
            event_type="experiment_completed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "results": results,
                "aria_summary": summary,
                "aria_mood": self.aria.state.mood,
                "insights": insights,
                "llm_analysis": llm_analysis,
                "mode": "continuous_investigation",
            },
        )
        self._complete_experiment_compat(
            nb=nb,
            experiment_id=exp_id,
            results=results,
            aria_summary=summary,
            insights=insights,
            llm_analysis=llm_analysis,
        )

        nb.flush_writes()
        # Auto-escalate to validation if strong candidates found
        self._auto_escalate(results, config, nb, phase="investigation")

        # Knowledge extraction after investigation
        self._maybe_extract_knowledge(config, nb, n_experiments)

        self._emit_event(
            "investigation_completed",
            {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            },
        )
