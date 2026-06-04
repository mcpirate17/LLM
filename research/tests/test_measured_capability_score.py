"""Unit tests for the measured capability RANK score (Phase 1.1).

Fast: exercises the pure composition `capability_score_from_descriptors`. The descriptor probe
itself (torch model build, ~0.4s/graph) is covered by measured_descriptors' own integration path.
"""

import pytest

from research.tools.measured_descriptors import (
    _CAPABILITY_WEIGHTS,
    _LIP_STABLE,
    capability_score_from_descriptors,
)

pytestmark = pytest.mark.unit

_MLP = dict(  # no backward routing, no gating, no data-dependence ⇒ floor
    long_range_reach=0.0,
    content_match_gating=0.0,
    content_dependence=0.0,
    causality_violation=0.0,
    measured_lipschitz=1.0,
)
_ATTN = dict(  # routes back, content-gated copy, data-dependent, causal, stable
    long_range_reach=0.6,
    content_match_gating=0.2,
    content_dependence=0.5,
    causality_violation=0.0,
    measured_lipschitz=1.0,
)


def test_attention_class_outranks_mlp_class():
    assert capability_score_from_descriptors(_ATTN) > capability_score_from_descriptors(
        _MLP
    )


def test_long_range_reach_and_gating_increase_score():
    base = capability_score_from_descriptors(_MLP)
    more_reach = capability_score_from_descriptors({**_MLP, "long_range_reach": 0.5})
    more_gate = capability_score_from_descriptors({**_MLP, "content_match_gating": 0.5})
    assert more_reach > base and more_gate > base


def test_causality_violation_not_penalized_by_default():
    # nas_funnel_ood_eval: causality_violation ROC 0.49 (noise at random init) ⇒ weight 0.
    base = capability_score_from_descriptors(_ATTN)
    assert (
        capability_score_from_descriptors({**_ATTN, "causality_violation": 0.9}) == base
    )


def test_lipschitz_gain_not_weighted_by_default():
    # the instability/lipschitz term proved positive-correlated, not a penalty ⇒ default weight 0.
    base = capability_score_from_descriptors(_ATTN)
    high = capability_score_from_descriptors(
        {**_ATTN, "measured_lipschitz": _LIP_STABLE + 8.0}
    )
    assert high == base


def test_missing_descriptors_default_to_zero_not_crash():
    # empty dict ⇒ all terms 0 ⇒ score 0.0 (no KeyError)
    assert capability_score_from_descriptors({}) == 0.0


def test_custom_weights_override_defaults():
    only_reach = capability_score_from_descriptors(
        _ATTN, weights={"long_range_reach": 1.0}
    )
    assert only_reach == pytest.approx(_ATTN["long_range_reach"])
    assert set(_CAPABILITY_WEIGHTS) >= {"long_range_reach", "content_match_gating"}
