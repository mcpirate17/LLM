"""Solo validator — single-pass scorecard for a generated component.

Runs four checks against a proposed nn.Module:
1. **Smoke**: compile + forward + backward + finite output + finite grad.
2. **Category metric**: ``mix_speed`` for lane modules only; other
   categories are recorded as skipped here (their behavior needs the
   in-context probe tier, not a solo pass).
3. **Property cross-check**: assert the runtime behavior matches the
   spec's declared math axes (e.g. tropical-declared modules must exhibit
   max-plus winner-take-all activation concentration).
4. **Persistence**: append the scorecard as a single JSONL line to
   ``component_fab/catalog/proposals.jsonl``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from component_fab.math_knobs import math_knobs_from_axes
from ..harness.standard_block import make_lane_test_block
from ..metrics.mix_speed import measure_mix_speed
from ..proposer.spec_generator import CATEGORY_LANE, ProposalSpec
from ..state.ledger import JsonlWriter


_REPO = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = _REPO / "component_fab" / "catalog" / "proposals.jsonl"


@dataclass(frozen=True, slots=True)
class SoloScorecard:
    proposal_id: str
    name: str
    category: str
    synthesis_kind: str
    math_axes: dict[str, Any]
    smoke: dict[str, Any]
    metrics: dict[str, Any]
    property_cross_check: dict[str, Any]
    promoted: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def smoke_test(module: nn.Module, *, dim: int, seq_len: int) -> dict[str, Any]:
    """Forward + backward on random input; shape + finiteness checks.

    Public: this is THE smoke check; ``viz.introspect`` adapts its keys for
    the UI rather than re-running its own forward/backward."""
    smoke: dict[str, Any] = {
        "compile_passed": False,
        "forward_passed": False,
        "backward_passed": False,
    }
    try:
        x = torch.randn(2, seq_len, dim, requires_grad=True)
        smoke["compile_passed"] = True
        y = module(x)
        if y.shape != x.shape:
            smoke["error"] = f"shape mismatch: {tuple(y.shape)} != {tuple(x.shape)}"
            return smoke
        smoke["forward_passed"] = True
        smoke["output_finite"] = bool(torch.isfinite(y).all().item())
        smoke["output_has_nan"] = bool(torch.isnan(y).any().item())
        smoke["output_has_inf"] = bool(torch.isinf(y).any().item())
        loss = y.pow(2).mean()
        loss.backward()
        smoke["backward_passed"] = True
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        smoke["param_grad_finite"] = all(
            g is None or torch.isfinite(g).all().item() for g in grads
        )
        smoke["input_grad_finite"] = bool(
            x.grad is None or torch.isfinite(x.grad).all().item()
        )
    except Exception as exc:
        smoke["error"] = f"{type(exc).__name__}: {exc}"
    return smoke


# Back-compat alias for pre-rename callers.
_smoke_test = smoke_test


def _category_metric(
    spec: ProposalSpec, module: nn.Module, *, dim: int, seq_len: int
) -> dict[str, Any]:
    if spec.category != CATEGORY_LANE:
        return {
            "skipped": True,
            "reason": f"category {spec.category} needs in-context probe",
        }
    block = make_lane_test_block(module, dim).eval()
    card = measure_mix_speed(block, seq_len=seq_len, feature_dim=dim, n_trials=2)
    return {
        "mix_half_life": card.mix_half_life,
        "peak_response_at_offset": card.peak_response_at_offset,
        "peak_response_magnitude": card.peak_response_magnitude,
        "mixes_globally": card.mixes_globally,
        "is_pure_local": card.is_pure_local,
    }


def _has_adapter_wrap(module: nn.Module) -> bool:
    """True if any submodule's class name contains 'Adapter' — the spec
    is wrapping the base mixer in a math-knob adapter, so the OUTER
    output reflects the adapter, not the inner algebra. Skip
    algebra/state/sparsity cross-checks in that case; the
    knob-specific check below remains the relevant signal.
    """
    return any("Adapter" in child.__class__.__name__ for child in module.modules())


def _property_cross_check(
    spec: ProposalSpec, module: nn.Module, *, dim: int, seq_len: int
) -> dict[str, Any]:
    declared = spec.math_axes
    findings: dict[str, Any] = {}
    x = torch.randn(2, seq_len, dim)
    module.eval()
    with torch.no_grad():
        y = module(x)

    adapter_wrapped = _has_adapter_wrap(module)
    findings["has_adapter_wrap"] = adapter_wrapped

    # Spectral rank measurement (to detect collapse or capacity limitation)
    flat_y = y.reshape(-1, dim)
    try:
        sv = torch.linalg.svdvals(flat_y)
        rank = (sv > sv.max() * 1e-4).sum().item()
        findings["activation_rank"] = int(rank)
    except Exception as exc:
        findings["activation_rank"] = 0
        findings["activation_rank_error"] = str(exc)

    declared_sparsity = str(declared.get("op_activation_sparsity_pattern") or "")
    if declared_sparsity == "top_k" and not adapter_wrapped:
        sparsity = float((y.abs() < 1e-6).float().mean().item())
        findings["measured_sparsity_ratio"] = sparsity
        findings["sparsity_declared_top_k"] = True
        findings["sparsity_consistent"] = sparsity > 0.1

    declared_state = int(declared.get("op_dynamical_has_state") or 0)
    if declared_state and not adapter_wrapped:
        x2 = x.clone()
        x2[:, 0] = x2[:, 0] + torch.randn_like(x2[:, 0])
        with torch.no_grad():
            y2 = module(x2)
        late_diff = float((y[:, -1] - y2[:, -1]).abs().mean().item())
        findings["measured_state_propagation_late_diff"] = late_diff
        findings["state_consistent"] = late_diff > 1e-4

    declared_algebra = str(declared.get("op_algebraic_space") or "")
    if declared_algebra == "tropical" and not adapter_wrapped:
        per_token_max_ratio = float(
            y.abs().amax(dim=-1).mean().item() / (y.abs().mean().item() + 1e-12)
        )
        findings["tropical_max_to_mean_ratio"] = per_token_max_ratio
        # Random-init tropical attention measures ~1.5; a smooth softmax-ish op
        # would be near 1.0. Threshold 1.3 separates them with margin.
        findings["tropical_consistent"] = per_token_max_ratio > 1.3
    _math_knob_cross_check(findings, declared, module, x, y, dim=dim)
    return findings


def _module_has_class_fragment(module: nn.Module, fragment: str) -> bool:
    return any(fragment in child.__class__.__name__ for child in module.modules())


def _module_attr_values(module: nn.Module, attr: str) -> list[Any]:
    return [getattr(child, attr) for child in module.modules() if hasattr(child, attr)]


def _future_drift(module: nn.Module, x: torch.Tensor) -> float:
    split = max(1, x.shape[1] // 2)
    x_future = x.clone()
    x_future[:, split:] = x_future[:, split:] + torch.randn_like(x_future[:, split:])
    module.eval()
    with torch.no_grad():
        y = module(x)
        y_future = module(x_future)
    return float((y[:, :split] - y_future[:, :split]).abs().max().item())


@dataclass(frozen=True, slots=True)
class _RangeKnobCheck:
    """Declarative spec for a math-knob whose consistency is a class-name
    fragment match plus a measured integer attribute landing in a range.

    ``upper``: ``None`` (lower-bound only, > 0), ``"strict"`` (< dim) or
    ``"inclusive"`` (<= dim).
    """

    knob: str
    attr: str
    measured_key: str
    consistent_key: str
    class_fragment: str
    upper: str | None


_RANGE_KNOB_CHECKS: tuple[_RangeKnobCheck, ...] = (
    _RangeKnobCheck(
        "linear_algebra_low_rank",
        "rank",
        "low_rank_measured_rank",
        "low_rank_consistent",
        "LowRank",
        "strict",
    ),
    _RangeKnobCheck(
        "sparse_matrix_banded",
        "bandwidth",
        "sparse_banded_measured_bandwidth",
        "sparse_banded_consistent",
        "SparseBanded",
        "inclusive",
    ),
    _RangeKnobCheck(
        "kernel_random_features",
        "n_features",
        "kernel_random_features_count",
        "kernel_random_features_consistent",
        "RandomFeatureKernel",
        None,
    ),
    _RangeKnobCheck(
        "multiscale_wavelet",
        "n_scales",
        "multiscale_wavelet_scales",
        "multiscale_wavelet_consistent",
        "MultiscaleWavelet",
        None,
    ),
)


def _measured_min_int(module: nn.Module, attr: str) -> int:
    values = [int(v) for v in _module_attr_values(module, attr)]
    return min(values) if values else 0


def _apply_range_knob_check(
    check: _RangeKnobCheck, findings: dict[str, Any], module: nn.Module, *, dim: int
) -> None:
    measured = _measured_min_int(module, check.attr)
    findings[check.measured_key] = measured
    if check.upper == "strict":
        in_range = 0 < measured < dim
    elif check.upper == "inclusive":
        in_range = 0 < measured <= dim
    else:
        in_range = measured > 0
    findings[check.consistent_key] = (
        _module_has_class_fragment(module, check.class_fragment) and in_range
    )


def _math_knob_cross_check(
    findings: dict[str, Any],
    declared: dict[str, Any],
    module: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int,
) -> None:
    del y
    knobs = math_knobs_from_axes(declared)
    if not knobs:
        return
    findings["declared_math_knobs"] = list(knobs)

    if "calculus_finite_difference" in knobs:
        drift = _future_drift(module, x)
        findings["calculus_future_drift"] = drift
        findings["calculus_consistent"] = (
            _module_has_class_fragment(module, "Calculus")
            or _module_has_class_fragment(module, "FiniteDifference")
        ) and drift < 1e-5

    for check in _RANGE_KNOB_CHECKS:
        if check.knob in knobs:
            _apply_range_knob_check(check, findings, module, dim=dim)

    if "graph_laplacian_diffusion" in knobs:
        step_count = _measured_min_int(module, "diffusion_steps")
        drift = _future_drift(module, x)
        findings["graph_diffusion_steps"] = step_count
        findings["graph_diffusion_future_drift"] = drift
        findings["graph_diffusion_consistent"] = (
            _module_has_class_fragment(module, "GraphDiffusion")
            and step_count > 0
            and drift < 1e-5
        )

    if "lambda_functional_blend" in knobs:
        gate_values = [
            float(torch.sigmoid(child.gate_logit).mean().item())
            for child in module.modules()
            if hasattr(child, "gate_logit")
        ]
        gate_mean = min(gate_values) if gate_values else 0.0
        findings["lambda_functional_gate_mean"] = gate_mean
        findings["lambda_functional_consistent"] = (
            _module_has_class_fragment(module, "LambdaFunctional")
            and 0.0 <= gate_mean < 0.1
        )


def _is_promotable(smoke: dict[str, Any], cross: dict[str, Any]) -> bool:
    required_smoke = (
        smoke.get("forward_passed"),
        smoke.get("backward_passed"),
        smoke.get("output_finite"),
        smoke.get("param_grad_finite"),
    )
    if not all(required_smoke):
        return False
    for key, value in cross.items():
        if key.endswith("_consistent") and value is False:
            return False
    return True


def validate_solo(
    spec: ProposalSpec, module: nn.Module, *, dim: int = 32, seq_len: int = 32
) -> SoloScorecard:
    smoke = smoke_test(module, dim=dim, seq_len=seq_len)
    if smoke.get("forward_passed"):
        metrics = _category_metric(spec, module, dim=dim, seq_len=seq_len)
        cross = _property_cross_check(spec, module, dim=dim, seq_len=seq_len)
    else:
        metrics = {"skipped": True, "reason": "forward failed"}
        cross = {"skipped": True, "reason": "forward failed"}
    promoted = _is_promotable(smoke, cross)
    return SoloScorecard(
        proposal_id=spec.proposal_id,
        name=spec.name,
        category=spec.category,
        synthesis_kind=spec.synthesis_kind,
        math_axes=dict(spec.math_axes),
        smoke=smoke,
        metrics=metrics,
        property_cross_check=cross,
        promoted=promoted,
    )


def append_scorecard(
    scorecard: SoloScorecard, path: Path | str = DEFAULT_CATALOG
) -> Path:
    out = Path(path)
    _get_scorecard_writer(out).write(asdict(scorecard))
    return out


# Per-path cache of JsonlWriter instances. ``run_autonomous._grade_spec`` calls
# ``append_scorecard`` once per spec per cycle (hundreds of writes/cycle); the
# per-write open+close was the dominant cost on the cycle write phase.
# Call ``close_scorecard_writers()`` to flush + release every cached handle
# (e.g. before reading the file from another reader, or at process exit).
_SCOREBOARD_WRITERS: dict[Path, JsonlWriter] = {}


def _get_scorecard_writer(path: Path) -> JsonlWriter:
    writer = _SCOREBOARD_WRITERS.get(path)
    if writer is None:
        writer = JsonlWriter(path)
        _SCOREBOARD_WRITERS[path] = writer
    return writer


def close_scorecard_writers(path: Path | str | None = None) -> None:
    """Flush and release the cached writer(s).

    With no argument, closes every cached writer. With a path, closes only
    that path. Safe to call multiple times; subsequent ``append_scorecard``
    calls will lazily open a fresh writer.
    """
    if path is None:
        for writer in _SCOREBOARD_WRITERS.values():
            writer.close()
        _SCOREBOARD_WRITERS.clear()
        return
    out = Path(path)
    writer = _SCOREBOARD_WRITERS.pop(out, None)
    if writer is not None:
        writer.close()
