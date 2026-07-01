from __future__ import annotations

import torch
from torch import nn

from research.tools.dynamic_math_sweep import (
    DescriptorBundle,
    PARENT_VARIANT,
    SweepRecord,
    VariantDescriptor,
    build_variant_operator,
    default_variant_catalog,
    finalize_sweep_decisions,
    measure_operator_descriptors,
    run_dynamic_math_sweep,
    selected_summary,
    target_profile,
)


class IdentityOperator(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class CausalShiftMix(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
        return x + 0.25 * prev


def _bundle(
    *,
    reach: float,
    dep: float,
    gate: float,
    rank: float = 3.0,
    causality: float = 0.0,
    self_dom: float = 0.3,
    spectral: float = 1.0,
    energy: float = 1.0,
) -> DescriptorBundle:
    return DescriptorBundle(
        physics={
            "perm_equivariance": 0.5,
            "shift_equivariance": 0.5,
            "scale_homogeneity": 0.9,
            "energy_gain": energy,
            "spectral_radius": spectral,
        },
        measured={
            "long_range_reach": reach,
            "content_dependence": dep,
            "content_match_gating": gate,
            "causality_violation": causality,
            "measured_lipschitz": 1.0,
            "effective_rank": rank,
            "nonlinearity": 0.1,
            "self_dominance": self_dom,
        },
    )


def test_default_catalog_covers_required_variant_families() -> None:
    catalog = default_variant_catalog()

    assert catalog[0] is PARENT_VARIANT
    families = {variant.family for variant in catalog}
    assert {
        "algebraic",
        "spectral_trig",
        "calculus_dynamical",
        "kernel",
        "graph_diffusion",
    } <= families


def test_variant_wrapper_builds_finite_distinct_output() -> None:
    variant = next(
        v for v in default_variant_catalog() if v.variant_id == "dynamical_causal_integral"
    )
    op = build_variant_operator(IdentityOperator(), variant)
    x = torch.randn(2, 8, 6)

    y = op(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert not torch.allclose(y, x)


def test_unsupported_transform_fails_closed() -> None:
    variant = VariantDescriptor(
        variant_id="future_lambda",
        family="lambda_functional",
        transform="lambda_combinator_future",
    )

    records = run_dynamic_math_sweep(
        IdentityOperator(),
        candidate_id="c",
        candidate_name="toy",
        dim=6,
        run_id="r",
        catalog=(PARENT_VARIANT, variant),
        measure_fn=lambda _variant, _op: _bundle(reach=0.1, dep=0.1, gate=0.1),
    )

    assert records[1].failure_reason == "variant_build_failed"
    assert "unsupported math variant transform" in str(records[1].error)


def test_selection_picks_improving_noncollapsed_variant() -> None:
    profile = target_profile("binding")
    parent = SweepRecord(
        run_id="r",
        candidate_id="c",
        candidate_name="toy",
        variant=PARENT_VARIANT,
        parent_variant_id="parent",
        build_passed=True,
        validate_passed=True,
        compile_passed=True,
        physics_descriptors=dict(_bundle(reach=0.05, dep=0.05, gate=0.0).physics),
        measured_descriptors=dict(_bundle(reach=0.05, dep=0.05, gate=0.0).measured),
    )
    good = SweepRecord(
        run_id="r",
        candidate_id="c",
        candidate_name="toy",
        variant=VariantDescriptor("good", "algebraic", "reciprocal_cauchy_read"),
        parent_variant_id="parent",
        build_passed=True,
        validate_passed=True,
        compile_passed=True,
        physics_descriptors=dict(_bundle(reach=0.2, dep=0.4, gate=0.3).physics),
        measured_descriptors=dict(_bundle(reach=0.2, dep=0.4, gate=0.3).measured),
    )
    future_leak = SweepRecord(
        run_id="r",
        candidate_id="c",
        candidate_name="toy",
        variant=VariantDescriptor("leaky", "spectral", "dct_token_rotation"),
        parent_variant_id="parent",
        build_passed=True,
        validate_passed=True,
        compile_passed=True,
        physics_descriptors=dict(_bundle(reach=0.9, dep=0.9, gate=0.9).physics),
        measured_descriptors=dict(
            _bundle(reach=0.9, dep=0.9, gate=0.9, causality=0.5).measured
        ),
    )

    selected = finalize_sweep_decisions([parent, good, future_leak], profile=profile)

    assert selected is good
    assert good.decision == "selected"
    assert future_leak.failure_reason == "causality_violation"
    assert parent.decision == "parent"


def test_run_dynamic_math_sweep_with_fake_measure_logs_selection() -> None:
    catalog = (
        PARENT_VARIANT,
        VariantDescriptor(
            "weak",
            "kernel",
            "positive_cosine_kernel_read",
            axes={"op_math_family": "kernel_methods"},
        ),
        VariantDescriptor(
            "strong",
            "algebraic",
            "reciprocal_cauchy_read",
            axes={"op_physics_address_family": "reciprocal_cauchy"},
        ),
    )

    def fake_measure(variant: VariantDescriptor, _op: nn.Module) -> DescriptorBundle:
        if variant.variant_id == "strong":
            return _bundle(reach=0.4, dep=0.45, gate=0.25)
        if variant.variant_id == "weak":
            return _bundle(reach=0.051, dep=0.051, gate=0.0)
        return _bundle(reach=0.05, dep=0.05, gate=0.0)

    records = run_dynamic_math_sweep(
        IdentityOperator(),
        candidate_id="candidate-1",
        candidate_name="toy_candidate",
        dim=8,
        run_id="run-1",
        catalog=catalog,
        measure_fn=fake_measure,
    )
    summary = selected_summary(records)

    assert [record.variant_id for record in records] == ["parent", "weak", "strong"]
    assert records[-1].decision == "selected"
    assert records[1].failure_reason == "no_target_improvement"
    assert summary["math_variant_selected"] is True
    assert summary["math_variant_family"] == "algebraic"
    assert summary["math_variant_delta_content_match_gating"] > 0.0


def test_real_cpu_descriptor_runner_returns_finite_descriptors() -> None:
    bundle = measure_operator_descriptors(
        CausalShiftMix(),
        dim=8,
        physics_batch=1,
        physics_seq_len=8,
        physics_n_seeds=1,
        measured_batch=2,
        measured_gap=3,
        measured_n_seeds=1,
    )

    combined = bundle.combined()
    assert {"spectral_radius", "long_range_reach", "effective_rank"} <= set(combined)
    assert all(torch.isfinite(torch.tensor(value)) for value in combined.values())
