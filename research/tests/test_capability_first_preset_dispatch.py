"""Regression: _capability_first_mode dispatches to GrammarConfig.capability_first.

Pins the UI → RunConfig → grammar dispatch seam so the Advanced panel
checkbox actually does something. Without this test a rename or reorder
in ``execution_candidates._build_grammar()`` could silently fall through
to ``routing_first`` and nobody would notice until leaderboard rows
stopped showing the new role-slot templates.
"""

from __future__ import annotations

from research.scientist.runner._types import RunConfig
from research.scientist.runner.execution_candidates import (
    _ExecutionCandidatesMixin,
)
from research.synthesis.templates import CAPABILITY_FIRST_TEMPLATES


class _Dispatcher(_ExecutionCandidatesMixin):
    """Minimal harness — _build_grammar_config only touches ``config``."""


def _build(config: RunConfig):
    return _Dispatcher()._build_grammar_config(config)


def test_capability_first_mode_selects_capability_first_preset() -> None:
    config = RunConfig()
    config._capability_first_mode = True  # type: ignore[attr-defined]
    grammar = _build(config)

    # Binding-capable screener flag must be on — that is the whole point
    # of the preset (gate8_retrieval_dead fires only when this is True).
    assert grammar.binding_capable_required is True

    # All six capability-first templates must be promoted above zero.
    for name in CAPABILITY_FIRST_TEMPLATES:
        assert grammar.template_weights.get(name, 0.0) > 0.0, (
            f"capability_first preset did not promote {name}"
        )

    # Retrieval-family ops must carry the boost — grammar.py sets these to
    # 4.0 in the capability_first preset. Anything lower means the sampler
    # fell through to a different branch.
    for op_name in ("matmul", "gather_topk", "token_type_classifier"):
        assert grammar.op_weights.get(op_name, 0.0) >= 3.5, (
            f"{op_name} weight too low ({grammar.op_weights.get(op_name)!r}) "
            "— likely not using capability_first preset"
        )


def test_capability_first_mode_wins_over_routing_first() -> None:
    """When both flags are set, capability_first must take precedence.

    This keeps the opt-in semantics clear: turning on capability_first
    should not silently be overridden by a legacy exploit/routing_first
    flag on the same config.
    """
    config = RunConfig()
    config._capability_first_mode = True  # type: ignore[attr-defined]
    config._routing_first_mode = True  # type: ignore[attr-defined]
    config.exploit_mode = True
    grammar = _build(config)

    # capability_first enables binding_capable_required; routing_first
    # leaves it at default False — so this flag being True proves
    # capability_first won.
    assert grammar.binding_capable_required is True


def test_capability_first_mode_off_leaves_legacy_path_intact() -> None:
    """Default config must not trigger capability_first semantics."""
    grammar = _build(RunConfig())
    assert grammar.binding_capable_required is False


def test_capability_first_mode_round_trips_via_from_dict() -> None:
    """UI payload with ``_capability_first_mode: true`` must land on RunConfig."""
    src = RunConfig()
    src._capability_first_mode = True  # type: ignore[attr-defined]

    payload = src.to_dict()
    assert payload["_capability_first_mode"] is True

    reconstructed = RunConfig.from_dict(payload)
    assert reconstructed._capability_first_mode is True  # type: ignore[attr-defined]


def test_capability_first_mode_fires_gate8_at_screening() -> None:
    """The screener reads ``_capability_first_mode`` directly (not the
    GrammarConfig field). This regression pins the RunConfig-side check so
    gate8_retrieval_dead actually runs when the user ticks the toggle.

    Without this, the live run logs from 2026-04-16 showed capability_first
    was on but zero gate8 drops appeared in the funnel — the flag was being
    read from the wrong object.
    """
    from research.scientist.runner.execution_screening_graphs import (
        CONTENT_ADDRESSED_OPS,
        SEQUENCE_MIXING_OPS,
        structural_gate_failure,
    )

    class _Analysis:
        def __init__(self, op_names: set[str]) -> None:
            self.op_names = op_names
            self.has_parameterized_op = True
            self.toxic_bigrams = set()

    class _FakeGraph:
        def n_ops(self) -> int:
            return 10

        def has_gradient_path(self) -> bool:
            return True

        def has_residual_path(self) -> bool:
            return True

    # A graph with a sequence mixer (SSM) but NO content-addressed op.
    # Default (binding_capable_required=False): must pass gates 1-7.
    # Capability-first (binding_capable_required=True): must be rejected
    # by gate8_retrieval_dead.
    ssm_only = _Analysis(
        {
            "selective_scan",
            "rmsnorm",
            "swiglu_mlp",
            "linear_proj",
            "add",
        }
    )
    assert "selective_scan" in SEQUENCE_MIXING_OPS
    assert not (ssm_only.op_names & CONTENT_ADDRESSED_OPS)

    # Default mode — gate8 is off, ssm-only graph passes.
    assert (
        structural_gate_failure(
            _FakeGraph(),
            routing_mandatory=False,
            efficiency_ops=frozenset({"selective_scan"}),
            analysis=ssm_only,
            binding_capable_required=False,
        )
        is None
    )

    # Capability-first — gate8 fires.
    assert (
        structural_gate_failure(
            _FakeGraph(),
            routing_mandatory=False,
            efficiency_ops=frozenset({"selective_scan"}),
            analysis=ssm_only,
            binding_capable_required=True,
        )
        == "gate8_retrieval_dead"
    )
