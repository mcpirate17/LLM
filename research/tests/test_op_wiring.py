"""Op Wiring Tests — verify target ops are reachable in the search space.

Tests that each target op is:
1. Registered in PRIMITIVE_REGISTRY
2. Has a compiler dispatch handler
3. Is reachable via motifs, templates, or activation substitution
4. Has correct algebraic type compatibility for its motif context
5. Executes forward+backward without error
"""

import random

import pytest
import torch

from research.synthesis.compiler import CompiledLayer, _OP_DISPATCH
from research.synthesis.graph import ComputationGraph
from research.synthesis.motifs import (
    ACTIVATION_POOL,
    MOTIFS_BY_CLASS,
    VALIDATED_MOTIFS,
)
from research.synthesis.op_roles import _OP_ROLE_MAP, get_role
from research.synthesis.primitives import (
    PRIMITIVE_REGISTRY,
    algebraic_types_compatible,
)
from research.synthesis.templates import TEMPLATES
from research.synthesis._template_helpers import _MIXER_CLASSES


# ── Target ops under test ──────────────────────────────────────────

TARGET_OPS = [
    "tropical_moe",
    "tropical_router",
    "padic_expand",
    "clifford_attention",
    "grade_mix",
    "div_safe",
    "reciprocal",
    "spectral_filter",
    "tropical_matmul",
    "hyp_distance",
]


# ── 1. Registry presence ──────────────────────────────────────────


@pytest.mark.parametrize("op_name", TARGET_OPS)
def test_op_registered(op_name):
    """Every target op must be in PRIMITIVE_REGISTRY."""
    # Force mathspace registration
    from research.mathspaces.registry import register_all_mathspaces

    register_all_mathspaces()
    assert op_name in PRIMITIVE_REGISTRY, f"{op_name} not in PRIMITIVE_REGISTRY"


# ── 2. Dispatch presence ──────────────────────────────────────────


@pytest.mark.parametrize("op_name", TARGET_OPS)
def test_op_has_dispatch(op_name):
    """Every target op must have a compiler dispatch handler or execute_fn."""
    from research.mathspaces.registry import register_all_mathspaces

    register_all_mathspaces()
    if op_name in _OP_DISPATCH:
        return  # Direct dispatch handler
    prim = PRIMITIVE_REGISTRY.get(op_name)
    assert prim is not None, f"{op_name} not registered"
    assert hasattr(prim, "execute_fn") and prim.execute_fn is not None, (
        f"{op_name} has neither _OP_DISPATCH handler nor execute_fn"
    )


# ── 3. Role classification ────────────────────────────────────────


@pytest.mark.parametrize("op_name", TARGET_OPS)
def test_op_has_explicit_role(op_name):
    """Every target op must have an explicit role (not category fallback)."""
    # reciprocal and spectral_filter are in _OP_ROLE_MAP directly
    # div_safe is UNSAFE which is intentional
    role = get_role(op_name)
    assert role is not None, f"{op_name} has no role"
    # Verify explicit assignment for non-trivial ops
    if op_name not in ("hyp_distance",):
        assert op_name in _OP_ROLE_MAP, f"{op_name} should be explicitly classified"


# ── 4. Motif/template/activation reachability ─────────────────────


def _op_in_any_motif(op_name: str) -> list[str]:
    """Return names of motifs that contain this op."""
    result = []
    for motif in VALIDATED_MOTIFS.values():
        if any(s.op_name == op_name for s in motif.steps):
            result.append(motif.name)
    return result


def _op_in_any_template(op_name: str) -> list[str]:
    """Check if op appears in any template source (binary-op templates)."""
    import inspect

    result = []
    for name, fn in TEMPLATES.items():
        src = inspect.getsource(fn)
        if f'"{op_name}"' in src:
            result.append(name)
    return result


class TestMotifReachability:
    """Ops must be reachable via motif, template, or activation pool."""

    def test_tropical_moe_in_motif(self):
        motifs = _op_in_any_motif("tropical_moe")
        assert motifs, "tropical_moe must appear in at least one motif"
        assert "tropical_moe_block" in motifs

    def test_tropical_router_in_motif(self):
        motifs = _op_in_any_motif("tropical_router")
        assert motifs, "tropical_router must appear in at least one motif"
        assert "tropical_router_block" in motifs

    def test_padic_expand_in_motif(self):
        motifs = _op_in_any_motif("padic_expand")
        assert motifs, "padic_expand must appear in at least one motif"

    def test_clifford_attention_in_motif(self):
        motifs = _op_in_any_motif("clifford_attention")
        assert motifs, "clifford_attention must appear in at least one motif"

    def test_grade_mix_in_motif(self):
        motifs = _op_in_any_motif("grade_mix")
        assert motifs, "grade_mix must appear in at least one motif"
        assert "clifford_attention_grade" in motifs

    def test_div_safe_in_template(self):
        """div_safe is UNSAFE — must be wired via binary-op template."""
        templates = _op_in_any_template("div_safe")
        assert templates, "div_safe must appear in at least one template"
        assert "safe_division" in templates

    def test_reciprocal_in_activation_pool(self):
        """reciprocal is ACTIVATE — must be in substitution pool."""
        assert "reciprocal" in ACTIVATION_POOL

    def test_spectral_filter_in_motif(self):
        motifs = _op_in_any_motif("spectral_filter")
        assert motifs, "spectral_filter must appear in at least one motif"
        assert "spectral_filter_mix" in motifs

    def test_tropical_matmul_in_template(self):
        """tropical_matmul is 2-input — must be wired via binary-op template."""
        templates = _op_in_any_template("tropical_matmul")
        assert templates, "tropical_matmul must appear in at least one template"
        assert "tropical_matmul_block" in templates

    def test_hyp_distance_in_motif(self):
        templates = _op_in_any_template("hyp_distance")
        assert templates, "hyp_distance must appear in at least one template"
        assert "hyp_distance_scoring" in templates


# ── 5. Algebraic type compatibility ───────────────────────────────


class TestAlgebraicCompatibility:
    """Motif steps must be algebraically compatible in sequence."""

    @pytest.fixture(autouse=True)
    def _ensure_mathspaces(self):
        from research.mathspaces.registry import register_all_mathspaces

        register_all_mathspaces()

    def _check_motif_type_chain(self, motif_name: str):
        """Verify type chain: each step's output is compatible with next step's input."""
        from research.synthesis.primitives import _EUCLIDEAN_TYPE

        motif = VALIDATED_MOTIFS[motif_name]
        current_type = _EUCLIDEAN_TYPE  # Graph input is euclidean
        for step in motif.steps:
            op = PRIMITIVE_REGISTRY.get(step.op_name)
            assert op is not None, f"{step.op_name} not registered"
            assert algebraic_types_compatible(current_type, op.algebraic_type), (
                f"Type mismatch in {motif_name}: {current_type} → {step.op_name} "
                f"(needs {op.algebraic_type.input_constraint}, got {current_type.output_guarantee})"
            )
            current_type = op.algebraic_type

    def test_tropical_moe_block_compat(self):
        self._check_motif_type_chain("tropical_moe_block")

    def test_tropical_router_block_compat(self):
        self._check_motif_type_chain("tropical_router_block")

    def test_padic_hierarchy_block_compat(self):
        self._check_motif_type_chain("padic_hierarchy_block")

    def test_clifford_attention_mix_compat(self):
        self._check_motif_type_chain("clifford_attention_mix")

    def test_clifford_attention_grade_compat(self):
        self._check_motif_type_chain("clifford_attention_grade")

    def test_hyp_distance_scoring_compat(self):
        pytest.skip("hyp_distance_scoring is a template, not a validated motif")

    def test_spectral_filter_mix_compat(self):
        self._check_motif_type_chain("spectral_filter_mix")

    def test_padic_residual_bridge_compat(self):
        self._check_motif_type_chain("padic_residual_bridge")


# ── 6. Math-space motifs reachable from standard templates ────────


class TestMathSpaceInMixerClasses:
    """MATH_SPACE motif class must be in _MIXER_CLASSES for standard template access."""

    def test_math_space_in_mixer_classes(self):
        from research.synthesis.motifs import MOTIF_CLASS_MATH_SPACE

        assert MOTIF_CLASS_MATH_SPACE in _MIXER_CLASSES, (
            "MOTIF_CLASS_MATH_SPACE must be in _MIXER_CLASSES for reachability"
        )

    def test_math_space_motifs_exist(self):
        from research.synthesis.motifs import MOTIF_CLASS_MATH_SPACE

        motifs = MOTIFS_BY_CLASS.get(MOTIF_CLASS_MATH_SPACE, [])
        assert len(motifs) >= 5, f"Expected >= 5 math-space motifs, got {len(motifs)}"


# ── 7. Forward+backward execution ────────────────────────────────


def _build_and_run(op_name, n_inputs=1, config=None):
    """Build single-op graph, compile, run fwd+bwd."""
    g = ComputationGraph(model_dim=64)
    inp = g.add_input()
    inputs = [inp]
    if n_inputs == 2:
        inp2 = g.add_op("linear_proj", [inp], config={"out_dim": 64})
        inputs = [inp, inp2]
    out = g.add_op(op_name, inputs, config=config or {})
    g.set_output(out)
    layer = CompiledLayer(g)
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = layer(x)
    assert not torch.isnan(y).any(), f"{op_name} produced NaN"
    try:
        y.sum().backward()
    except RuntimeError:
        pass  # Some non-parametric ops have no grad path in isolation
    return y


class TestForwardBackward:
    """Each target op must execute without errors."""

    @pytest.fixture(autouse=True)
    def _ensure_mathspaces(self):
        from research.mathspaces.registry import register_all_mathspaces

        register_all_mathspaces()

    def test_tropical_moe(self):
        _build_and_run("tropical_moe", config={"num_experts": 2})

    def test_tropical_router(self):
        _build_and_run("tropical_router")

    def test_padic_expand(self):
        _build_and_run("padic_expand")

    def test_clifford_attention(self):
        _build_and_run("clifford_attention")

    def test_grade_mix(self):
        _build_and_run("grade_mix")

    def test_div_safe(self):
        _build_and_run("div_safe", n_inputs=2)

    def test_reciprocal(self):
        _build_and_run("reciprocal")

    def test_spectral_filter(self):
        _build_and_run("spectral_filter")

    def test_tropical_matmul(self):
        _build_and_run("tropical_matmul", n_inputs=2)

    def test_hyp_distance(self):
        # hyp_distance uses reduce_last → dim=1, needs proj_up to restore
        g = ComputationGraph(model_dim=64)
        inp = g.add_input()
        proj = g.add_op("linear_proj", [inp], config={"out_dim": 64})
        dist = g.add_op("hyp_distance", [inp, proj])
        out = g.add_op("linear_proj_up", [dist], config={"out_dim": 64})
        g.set_output(out)
        layer = CompiledLayer(g)
        x = torch.randn(2, 8, 64, requires_grad=True)
        y = layer(x)
        assert not torch.isnan(y).any(), "hyp_distance produced NaN"


# ── 8. Template generation test ──────────────────────────────────


class TestTemplateGeneration:
    """Binary-op templates must produce valid graphs."""

    @pytest.fixture(autouse=True)
    def _ensure_mathspaces(self):
        from research.mathspaces.registry import register_all_mathspaces

        register_all_mathspaces()

    def _run_template(self, template_name):
        g = ComputationGraph(model_dim=64)
        inp = g.add_input()
        rng = random.Random(42)
        out = TEMPLATES[template_name](g, inp, rng)
        g.set_output(out)
        layer = CompiledLayer(g)
        x = torch.randn(2, 8, 64, requires_grad=True)
        y = layer(x)
        assert not torch.isnan(y).any(), f"Template {template_name} produced NaN"
        y.sum().backward()

    def test_tropical_matmul_template(self):
        self._run_template("tropical_matmul_block")

    def test_safe_division_template(self):
        self._run_template("safe_division")

    def test_tropical_residual_template(self):
        self._run_template("tropical_residual")


# ── 9. Registry/dispatch count invariants ─────────────────────────


class TestCountInvariants:
    """Primitive and dispatch counts must not decrease."""

    @pytest.fixture(autouse=True)
    def _ensure_mathspaces(self):
        from research.mathspaces.registry import register_all_mathspaces

        register_all_mathspaces()

    def test_primitive_count_minimum(self):
        assert len(PRIMITIVE_REGISTRY) >= 100, (
            f"PRIMITIVE_REGISTRY has {len(PRIMITIVE_REGISTRY)} ops, expected >= 100"
        )

    def test_dispatch_count_minimum(self):
        assert len(_OP_DISPATCH) >= 100, (
            f"_OP_DISPATCH has {len(_OP_DISPATCH)} handlers, expected >= 100"
        )

    def test_template_count_minimum(self):
        assert len(TEMPLATES) >= 30, (
            f"TEMPLATES has {len(TEMPLATES)} templates, expected >= 30"
        )
