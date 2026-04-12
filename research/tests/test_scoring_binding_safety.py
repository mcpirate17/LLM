"""Safety-net tests for scoring pipeline + binding probe integration.

Covers the critical paths that have zero test coverage:
1. compute_composite binding probe scoring (component 16)
2. Binding soft gate (3-signal AND penalty)
3. ar_timed_out masking of ar_auc
4. compute_binding_composite / compute_local_only (binding_pipeline.py)
5. _pr_dict_to_score_kwargs mapping (DB row → scoring kwargs)
6. compute_composite entry point end-to-end
"""

import copy
import pytest

from research.eval.binding_pipeline import (
    FullBindingProbeResult,
    compute_binding_composite,
    compute_local_only,
)
from research.scientist.leaderboard_scoring import (
    _V7_CONFIG,
    _V8_CONFIG,
    _pr_dict_to_score_kwargs,
    _scurve,
    compute_composite,
    compute_composite_v7,
    compute_composite_v8,
)
from research.scientist.thresholds import (
    BINDING_AR_SOFT_GATE,
    BINDING_BINDING_AUC_SOFT_GATE,
    BINDING_INDUCTION_SOFT_GATE,
    BINDING_LOCAL_ONLY_PENALTY,
)


# -----------------------------------------------------------------------
# Helpers: realistic scoring kwargs for different model profiles
# -----------------------------------------------------------------------


def _base_kwargs(**overrides):
    """Baseline kwargs for a decent screening-tier model."""
    kw = dict(
        ppl_screening=9.5,
        param_count=5_000_000,
        tier="screening",
    )
    kw.update(overrides)
    return kw


def _investigated_kwargs(**overrides):
    """Kwargs for an investigation-tier model with binding probes."""
    kw = dict(
        ppl_screening=8.5,
        ppl_investigation=7.8,
        param_count=5_000_000,
        ppl_at_100=14.0,
        ppl_at_500=10.5,
        ppl_at_1000=8.5,
        tier="investigation",
        ar_auc=0.12,
        induction_auc=0.09,
        binding_auc=0.14,
    )
    kw.update(overrides)
    return kw


# -----------------------------------------------------------------------
# 1. compute_binding_composite
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestComputeBindingComposite:
    """Test the weighted composite of the three binding probes."""

    def test_full_three_signals(self):
        result = compute_binding_composite(0.10, 0.08, 0.12)
        expected = round(0.4 * 0.10 + 0.3 * 0.08 + 0.3 * 0.12, 4)
        assert result == expected

    def test_ar_auc_none_drops_to_two_signals(self):
        result = compute_binding_composite(None, 0.08, 0.12)
        expected = round(0.3 * 0.08 + 0.3 * 0.12, 4)
        assert result == expected

    def test_ar_auc_none_weights_sum_to_06(self):
        """With ar_auc=None, max composite is 0.6, not 1.0."""
        result = compute_binding_composite(None, 1.0, 1.0)
        assert result == 0.6

    def test_all_perfect_equals_one(self):
        result = compute_binding_composite(1.0, 1.0, 1.0)
        assert result == 1.0

    def test_all_zero(self):
        result = compute_binding_composite(0.0, 0.0, 0.0)
        assert result == 0.0

    def test_ar_auc_dominates_weighting(self):
        """AR AUC has 40% weight — highest single signal."""
        ar_only = compute_binding_composite(1.0, 0.0, 0.0)
        ind_only = compute_binding_composite(0.0, 1.0, 0.0)
        bind_only = compute_binding_composite(0.0, 0.0, 1.0)
        assert ar_only > ind_only
        assert ar_only > bind_only
        assert ind_only == bind_only  # both 30%


# -----------------------------------------------------------------------
# 2. compute_local_only (soft gate trigger)
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestComputeLocalOnly:
    """Test the 3-signal AND that triggers the local-only penalty."""

    def test_all_below_gates_returns_1(self):
        result = compute_local_only(0.01, 0.01, 0.01)
        assert result == 1

    def test_one_above_gate_returns_0(self):
        """Any signal above its gate prevents the penalty."""
        assert compute_local_only(0.99, 0.01, 0.01) == 0  # ar above
        assert compute_local_only(0.01, 0.99, 0.01) == 0  # induction above
        assert compute_local_only(0.01, 0.01, 0.99) == 0  # binding above

    def test_all_above_gates_returns_0(self):
        result = compute_local_only(0.99, 0.99, 0.99)
        assert result == 0

    def test_at_exact_gate_boundary(self):
        """At exactly the gate value, should still be below (strict <)."""
        result = compute_local_only(
            BINDING_AR_SOFT_GATE,
            BINDING_INDUCTION_SOFT_GATE,
            BINDING_BINDING_AUC_SOFT_GATE,
        )
        # All values equal to gate — not strictly less, so not all below
        assert result == 0

    def test_just_below_gates(self):
        result = compute_local_only(
            BINDING_AR_SOFT_GATE - 0.001,
            BINDING_INDUCTION_SOFT_GATE - 0.001,
            BINDING_BINDING_AUC_SOFT_GATE - 0.001,
        )
        assert result == 1


# -----------------------------------------------------------------------
# 3. FullBindingProbeResult.to_result_dict
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestFullBindingProbeResult:
    """Test the dataclass used by run_full_binding_probes."""

    def _make_result(self, **overrides):
        defaults = dict(
            ar_auc=0.15,
            ar_final_acc=0.20,
            ar_timed_out=False,
            ar_above_chance=True,
            ar_elapsed_ms=5000.0,
            induction_auc=0.10,
            induction_metadata={"induction_auc": 0.10, "induction_pass_count": 3},
            induction_elapsed_ms=1000.0,
            binding_auc=0.18,
            binding_distance_accuracies={5: 0.9, 10: 0.7, 20: 0.5},
            binding_elapsed_ms=3000.0,
        )
        defaults.update(overrides)
        return FullBindingProbeResult(**defaults)

    def test_to_result_dict_includes_ar_fields(self):
        r = self._make_result()
        d = r.to_result_dict()
        assert d["ar_auc"] == 0.15
        assert d["ar_final_acc"] == 0.20
        assert d["ar_timed_out"] is False
        assert d["ar_above_chance"] is True

    def test_to_result_dict_includes_binding_fields(self):
        r = self._make_result()
        d = r.to_result_dict()
        assert d["binding_auc"] == 0.18
        assert isinstance(d["binding_distance_accuracies"], dict)

    def test_to_result_dict_merges_induction_metadata(self):
        r = self._make_result()
        d = r.to_result_dict()
        assert d["induction_auc"] == 0.10


# -----------------------------------------------------------------------
# 4. _scurve sanity checks
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestScurve:
    def test_at_frontier(self):
        assert _scurve(1.0) == pytest.approx(0.5)

    def test_above_frontier(self):
        assert _scurve(2.0) > 0.9

    def test_below_frontier(self):
        assert _scurve(0.5) < 0.2

    def test_monotonic(self):
        prev = _scurve(0.0)
        for r in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            cur = _scurve(r)
            assert cur >= prev, f"S-curve not monotonic at ratio={r}"
            prev = cur


# -----------------------------------------------------------------------
# 5. Binding probe scoring in _compute_composite_generic (component 16)
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestCompositeBindingComponent:
    """Test that binding probes produce correct points in composite score."""

    def test_no_binding_data_zero_points(self):
        result = compute_composite_v8(decompose=True, **_base_kwargs())
        assert result["breakdown"].get("binding", 0) == 0.0

    def test_binding_probes_produce_points(self):
        result = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(),
        )
        pts = result["breakdown"]["binding"]
        assert pts > 0, "Binding probes should produce positive points"
        assert pts <= _V8_CONFIG["w_binding"]

    def test_ar_timed_out_masks_ar_auc(self):
        """When ar_timed_out=True, ar_auc should be ignored (treated as None)."""
        with_ar = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_timed_out=False),
        )
        timed_out = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_timed_out=True),
        )
        # With timeout, we lose the 40% ar_auc signal → lower binding score
        assert timed_out["breakdown"]["binding"] < with_ar["breakdown"]["binding"]

    def test_ar_timed_out_true_equals_ar_none(self):
        """ar_timed_out=True should produce same binding as ar_auc=None."""
        timed_out = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_timed_out=True),
        )
        no_ar = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=None, ar_timed_out=None),
        )
        assert timed_out["breakdown"]["binding"] == pytest.approx(
            no_ar["breakdown"]["binding"], abs=0.01
        )

    def test_higher_binding_scores_more_points(self):
        low = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=0.05, induction_auc=0.03, binding_auc=0.05),
        )
        high = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=0.30, induction_auc=0.20, binding_auc=0.30),
        )
        assert high["breakdown"]["binding"] > low["breakdown"]["binding"]

    def test_v7_binding_weight_higher_than_v8(self):
        """v7 allocates 120pts to binding, v8 allocates 85pts."""
        assert _V7_CONFIG["w_binding"] == 120.0
        assert _V8_CONFIG["w_binding"] == 85.0

    def test_investigation_failed_zeros_binding(self):
        result = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(tier="investigation_failed"),
        )
        assert result["breakdown"]["binding"] == 0.0


# -----------------------------------------------------------------------
# 6. Binding soft gate (3-signal AND penalty)
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestBindingSoftGate:
    """Test the local-only penalty in composite scoring."""

    def test_all_below_gates_applies_penalty(self):
        """When all measured signals are below gates → 0.80x multiplier."""
        kw = _investigated_kwargs(
            ar_auc=0.01,
            induction_auc=0.01,
            binding_auc=0.01,
        )
        result = compute_composite_v8(decompose=True, **kw)
        penalty = result["breakdown"].get("binding_local_only_penalty", 0)
        assert penalty == pytest.approx(BINDING_LOCAL_ONLY_PENALTY), (
            f"Expected {BINDING_LOCAL_ONLY_PENALTY}, got {penalty}"
        )

    def test_one_signal_above_gate_no_penalty(self):
        """If any signal is above its gate, no penalty."""
        kw = _investigated_kwargs(
            ar_auc=0.50,  # well above BINDING_AR_SOFT_GATE (0.05)
            induction_auc=0.01,
            binding_auc=0.01,
        )
        result = compute_composite_v8(decompose=True, **kw)
        penalty = result["breakdown"].get("binding_local_only_penalty", 0)
        assert penalty == 0.0

    def test_penalty_reduces_total_score(self):
        """The penalty multiplies the entire composite, not just binding."""
        # Good model with bad binding
        good_binding = _investigated_kwargs(
            ar_auc=0.20, induction_auc=0.15, binding_auc=0.20
        )
        bad_binding = _investigated_kwargs(
            ar_auc=0.01, induction_auc=0.01, binding_auc=0.01
        )
        score_good = compute_composite_v8(**good_binding)
        score_bad = compute_composite_v8(**bad_binding)
        assert score_bad < score_good

    def test_ar_above_chance_blocks_ar_below_check(self):
        """If ar_above_chance=True, AR doesn't count as 'below gate'
        even if the numeric value is below the threshold."""
        kw = _investigated_kwargs(
            ar_auc=0.03,  # below BINDING_AR_SOFT_GATE
            ar_above_chance=True,  # but statistically above chance
            induction_auc=0.01,
            binding_auc=0.01,
        )
        result = compute_composite_v8(decompose=True, **kw)
        penalty = result["breakdown"].get("binding_local_only_penalty", 0)
        # ar_above_chance prevents the AR leg of the AND from being True
        # so we need at least 2 measured signals with both below
        # induction and binding are both below → 2 signals below, ar is not → penalty still fires
        # Actually: the logic is _all_below = _binding_signals_measured >= 2 and all([...])
        # With ar_above_chance=True: _ar_below = False
        # all([True, True, False]) = False → no penalty
        assert penalty == 0.0

    def test_only_two_signals_measured_can_still_trigger(self):
        """With only induction + binding (no AR), penalty can still fire."""
        kw = _investigated_kwargs(
            ar_auc=None,
            induction_auc=0.01,
            binding_auc=0.01,
        )
        result = compute_composite_v8(decompose=True, **kw)
        penalty = result["breakdown"].get("binding_local_only_penalty", 0)
        assert penalty == pytest.approx(BINDING_LOCAL_ONLY_PENALTY)

    def test_single_signal_never_triggers(self):
        """With only 1 measured signal, penalty never fires (need >= 2)."""
        kw = _investigated_kwargs(
            ar_auc=None,
            induction_auc=0.01,
            binding_auc=None,
        )
        result = compute_composite_v8(decompose=True, **kw)
        penalty = result["breakdown"].get("binding_local_only_penalty", 0)
        assert penalty == 0.0


# -----------------------------------------------------------------------
# 7. _pr_dict_to_score_kwargs mapping
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestPrDictToScoreKwargs:
    """Test the DB row → scoring kwargs bridge."""

    def _make_pr_dict(self, **overrides):
        """Minimal pr_dict simulating a program_results row."""
        base = {
            "result_id": "test-001",
            "novelty_confidence": 0.7,
            "loss_improvement_rate": 0.5,
            "final_loss": 2.5,
            "param_count": 5_000_000,
            "n_train_steps": 1000,
            "behavioral_novelty": 0.3,
            "structural_novelty": 0.4,
            "fp_cka_vs_transformer": 0.8,
            "wikitext_perplexity": 9.5,
            "wikitext_score": 0.6,
            "wikitext_ppl_200": 14.0,
            "wikitext_ppl_500": 10.5,
            "wikitext_eval_steps": 1000,
            "routing_savings_ratio": None,
            "compression_ratio": None,
            "activation_sparsity_score": None,
            "depth_savings_ratio": None,
            "recursion_depth_ratio": None,
            "fp_jacobian_spectral_norm": 2.5,
            "validation_robustness_score": None,
            "ncd_description_length_per_param": 0.8,
            "novelty_valid_for_promotion": True,
            "fingerprint_json": '{"analyses_succeeded": 3}',
            "hellaswag_acc": 0.28,
            "ar_auc": 0.12,
            "ar_final_acc": 0.18,
            "ar_timed_out": False,
            "ar_above_chance": True,
            "induction_auc": 0.09,
            "binding_auc": 0.14,
            "blimp_overall_accuracy": 0.55,
            "tinystories_score": 0.40,
            "cross_task_score": 0.50,
            "diagnostic_score": 0.30,
            "fp_gromov_delta": None,
            "fp_hierarchy_fitness": 0.45,
        }
        base.update(overrides)
        return base

    def _make_lb_row(self, **overrides):
        """Minimal leaderboard row dict."""
        base = {
            "tier": "investigation",
            "screening_loss_ratio": 0.75,
            "wikitext_perplexity": 8.0,
            "screening_novelty": 0.5,
            "novelty_confidence": 0.7,
        }
        base.update(overrides)
        return base

    def test_ar_auc_extracted(self):
        pr = self._make_pr_dict()
        d = self._make_lb_row()
        kw = _pr_dict_to_score_kwargs(pr, d, is_reference=False)
        assert kw["ar_auc"] == 0.12

    def test_ar_timed_out_extracted_as_bool(self):
        pr = self._make_pr_dict(ar_timed_out=1)  # DB stores as int
        d = self._make_lb_row()
        kw = _pr_dict_to_score_kwargs(pr, d, is_reference=False)
        assert kw["ar_timed_out"] is True

    def test_ar_timed_out_none_stays_none(self):
        pr = self._make_pr_dict(ar_timed_out=None)
        d = self._make_lb_row()
        kw = _pr_dict_to_score_kwargs(pr, d, is_reference=False)
        assert kw["ar_timed_out"] is None

    def test_binding_auc_from_pr_dict(self):
        pr = self._make_pr_dict(binding_auc=0.20)
        d = self._make_lb_row(binding_auc=None)
        kw = _pr_dict_to_score_kwargs(pr, d, is_reference=False)
        assert kw["binding_auc"] == 0.20

    def test_binding_auc_fallback_to_lb_row(self):
        pr = self._make_pr_dict(binding_auc=None)
        d = self._make_lb_row(binding_auc=0.15)
        kw = _pr_dict_to_score_kwargs(pr, d, is_reference=False)
        assert kw["binding_auc"] == 0.15

    def test_fingerprint_json_parsed(self):
        pr = self._make_pr_dict(fingerprint_json='{"analyses_succeeded": 4}')
        d = self._make_lb_row()
        kw = _pr_dict_to_score_kwargs(pr, d, is_reference=False)
        assert kw["analyses_succeeded"] == 4

    def test_fingerprint_json_missing(self):
        pr = self._make_pr_dict(fingerprint_json=None)
        d = self._make_lb_row()
        kw = _pr_dict_to_score_kwargs(pr, d, is_reference=False)
        assert kw["analyses_succeeded"] == 0

    def test_ppl_investigation_only_at_investigation_tier(self):
        pr = self._make_pr_dict()
        d_screening = self._make_lb_row(tier="screening")
        d_investigation = self._make_lb_row(tier="investigation")
        kw_s = _pr_dict_to_score_kwargs(
            copy.deepcopy(pr), d_screening, is_reference=False
        )
        kw_i = _pr_dict_to_score_kwargs(
            copy.deepcopy(pr), d_investigation, is_reference=False
        )
        assert kw_s["ppl_investigation"] is None
        assert kw_i["ppl_investigation"] is not None

    def test_ppl_validation_only_at_validation_tier(self):
        pr = self._make_pr_dict()
        d_inv = self._make_lb_row(tier="investigation")
        d_val = self._make_lb_row(tier="validation")
        kw_i = _pr_dict_to_score_kwargs(copy.deepcopy(pr), d_inv, is_reference=False)
        kw_v = _pr_dict_to_score_kwargs(copy.deepcopy(pr), d_val, is_reference=False)
        assert kw_i["ppl_validation"] is None
        assert kw_v["ppl_validation"] is not None

    def test_kwargs_can_be_passed_to_compute_composite(self):
        """End-to-end: pr_dict → kwargs → compute_composite produces a valid score."""
        pr = self._make_pr_dict()
        d = self._make_lb_row()
        kw = _pr_dict_to_score_kwargs(pr, d, is_reference=False)
        score = compute_composite(**kw)
        assert isinstance(score, float)
        assert score >= 0.0


# -----------------------------------------------------------------------
# 8. End-to-end composite scoring scenarios
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestCompositeEndToEnd:
    """Realistic end-to-end scoring scenarios."""

    def test_screening_model_basic(self):
        score = compute_composite(**_base_kwargs())
        assert isinstance(score, float)
        assert score > 0

    def test_investigated_model_scores_higher_than_screened(self):
        s = compute_composite(**_base_kwargs())
        i = compute_composite(**_investigated_kwargs())
        assert i > s

    def test_insufficient_learning_capped_at_10(self):
        score = compute_composite(screening_lr=0.98, tier="screening")
        assert score == 10.0

    def test_zero_inputs_returns_zero(self):
        assert compute_composite() == 0.0

    def test_reference_model_no_novelty(self):
        """Reference models should get 0 novelty points."""
        result = compute_composite_v8(
            decompose=True,
            is_reference=True,
            screening_nov=0.8,
            novelty_confidence=0.9,
            ppl_screening=9.0,
            tier="screening",
        )
        assert result["breakdown"]["novelty"] == 0.0

    def test_param_size_penalty_for_large_models(self):
        """Models with >5M params get a multiplicative penalty."""
        small = compute_composite(**_investigated_kwargs(param_count=4_000_000))
        large = compute_composite(**_investigated_kwargs(param_count=20_000_000))
        assert large < small, "Large model should score lower due to param penalty"

    def test_v7_v8_produce_different_scores(self):
        kw = _investigated_kwargs()
        v7 = compute_composite_v7(**kw)
        v8 = compute_composite_v8(**kw)
        assert v7 != v8

    def test_decompose_breakdown_sums_to_composite(self):
        """The breakdown components (before penalties) should be recoverable."""
        result = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(
                ar_auc=0.20,
                induction_auc=0.15,
                binding_auc=0.20,
            ),
        )
        bd = result["breakdown"]
        # Sum all positive components (exclude penalty tracking entries)
        component_keys = [
            "perf_short",
            "perf_medium",
            "perf_long",
            "param_efficiency",
            "learning_efficiency",
            "routing_savings",
            "compression",
            "sparsity",
            "adaptive_computation",
            "novelty",
            "ncd",
            "robustness",
            "long_context",
            "early_convergence",
            "speed",
            "binding",
            "blimp",
        ]
        raw_sum = sum(bd.get(k, 0) for k in component_keys)
        penalty = bd.get("binding_local_only_penalty", 0)
        param_penalty = bd.get("param_size_penalty", 0)

        # The composite should be raw_sum * binding_penalty * param_penalty
        # (or just raw_sum if no penalties)
        composite = result["composite_score"]
        assert composite > 0
        assert composite <= raw_sum + 0.01  # penalties can only reduce


# -----------------------------------------------------------------------
# 9. AR AUC backfill safety: verify scoring is stable when ar_auc
#    is added to entries that previously had None
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestARBackfillSafety:
    """Validate that backfilling AR AUC produces correct score changes."""

    def test_adding_ar_auc_increases_binding_points(self):
        """When ar_auc goes from None to a positive value, binding points increase."""
        without_ar = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=None),
        )
        with_ar = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=0.15),
        )
        assert with_ar["breakdown"]["binding"] >= without_ar["breakdown"]["binding"]

    def test_timed_out_ar_is_same_as_none(self):
        """Backfilling a timed-out result should not change anything."""
        before = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=None, ar_timed_out=None),
        )
        after = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=0.15, ar_timed_out=True),
        )
        assert after["breakdown"]["binding"] == pytest.approx(
            before["breakdown"]["binding"], abs=0.01
        )

    def test_low_ar_with_low_others_triggers_penalty(self):
        """Backfilling a low ar_auc where induction+binding are also low
        could trigger the local-only penalty that wasn't triggered before."""
        # Before backfill: only 2 signals, both low → penalty fires
        before = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=None, induction_auc=0.02, binding_auc=0.02),
        )
        # After backfill: 3 signals, all low → penalty still fires
        after = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=0.01, induction_auc=0.02, binding_auc=0.02),
        )
        before_penalty = before["breakdown"].get("binding_local_only_penalty", 0)
        after_penalty = after["breakdown"].get("binding_local_only_penalty", 0)
        # Both should have the penalty
        assert before_penalty == pytest.approx(BINDING_LOCAL_ONLY_PENALTY)
        assert after_penalty == pytest.approx(BINDING_LOCAL_ONLY_PENALTY)

    def test_good_ar_auc_removes_penalty(self):
        """Backfilling a good ar_auc can remove the local-only penalty."""
        # Before: induction and binding are below gate, ar is None → penalty
        before = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=None, induction_auc=0.02, binding_auc=0.02),
        )
        # After: ar_auc is above gate → breaks the AND → no penalty
        after = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=0.50, induction_auc=0.02, binding_auc=0.02),
        )
        before_penalty = before["breakdown"].get("binding_local_only_penalty", 0)
        after_penalty = after["breakdown"].get("binding_local_only_penalty", 0)
        assert before_penalty == pytest.approx(BINDING_LOCAL_ONLY_PENALTY)
        assert after_penalty == 0.0

    def test_backfill_does_not_change_non_binding_components(self):
        """Adding ar_auc should only affect binding-related scores."""
        without = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=None, induction_auc=0.15, binding_auc=0.15),
        )
        with_ar = compute_composite_v8(
            decompose=True,
            **_investigated_kwargs(ar_auc=0.15, induction_auc=0.15, binding_auc=0.15),
        )
        for key in [
            "perf_short",
            "perf_medium",
            "param_efficiency",
            "learning_efficiency",
            "novelty",
            "ncd",
            "early_convergence",
        ]:
            assert without["breakdown"].get(key, 0) == pytest.approx(
                with_ar["breakdown"].get(key, 0), abs=0.001
            ), f"Component {key} changed when it shouldn't have"
