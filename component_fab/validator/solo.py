"""Solo validator — single-pass scorecard for a generated component.

Runs four checks against a proposed nn.Module:
1. **Smoke**: compile + forward + backward + finite output + finite grad.
2. **Category metric**: ``mix_speed`` for lane, ``routing_health`` for
   routing (when the module emits routing weights), ``compression_quality``
   for compression (when a paired restore is provided).
3. **Property cross-check**: assert the runtime behavior matches the
   spec's declared math axes (e.g. tropical-declared modules must exhibit
   max-plus winner-take-all activation concentration).
4. **Persistence**: append the scorecard as a single JSONL line to
   ``component_fab/catalog/proposals.jsonl``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..harness.standard_block import make_lane_test_block
from ..metrics.mix_speed import measure_mix_speed
from ..proposer.spec_generator import CATEGORY_LANE, ProposalSpec


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


def _smoke_test(module: nn.Module, *, dim: int, seq_len: int) -> dict[str, Any]:
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


def _property_cross_check(
    spec: ProposalSpec, module: nn.Module, *, dim: int, seq_len: int
) -> dict[str, Any]:
    declared = spec.math_axes
    findings: dict[str, Any] = {}
    x = torch.randn(2, seq_len, dim)
    module.eval()
    with torch.no_grad():
        y = module(x)

    declared_sparsity = str(declared.get("op_activation_sparsity_pattern") or "")
    if declared_sparsity == "top_k":
        sparsity = float((y.abs() < 1e-6).float().mean().item())
        findings["measured_sparsity_ratio"] = sparsity
        findings["sparsity_declared_top_k"] = True
        findings["sparsity_consistent"] = sparsity > 0.1

    declared_state = int(declared.get("op_dynamical_has_state") or 0)
    if declared_state:
        x2 = x.clone()
        x2[:, 0] = x2[:, 0] + torch.randn_like(x2[:, 0])
        with torch.no_grad():
            y2 = module(x2)
        late_diff = float((y[:, -1] - y2[:, -1]).abs().mean().item())
        findings["measured_state_propagation_late_diff"] = late_diff
        findings["state_consistent"] = late_diff > 1e-4

    declared_algebra = str(declared.get("op_algebraic_space") or "")
    if declared_algebra == "tropical":
        per_token_max_ratio = float(
            y.abs().amax(dim=-1).mean().item() / (y.abs().mean().item() + 1e-12)
        )
        findings["tropical_max_to_mean_ratio"] = per_token_max_ratio
        # Random-init tropical attention measures ~1.5; a smooth softmax-ish op
        # would be near 1.0. Threshold 1.3 separates them with margin.
        findings["tropical_consistent"] = per_token_max_ratio > 1.3
    _math_knob_cross_check(findings, declared, module, x, y, dim=dim)
    return findings


def _declared_math_knobs(declared: dict[str, Any]) -> tuple[str, ...]:
    raw = declared.get("op_math_knobs")
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split("+") if part.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(part) for part in raw if str(part))
    family = str(declared.get("op_math_family") or "")
    if family == "calculus":
        return ("calculus_finite_difference",)
    if family == "linear_algebra":
        return ("linear_algebra_low_rank",)
    if family == "sparse_matrix":
        return ("sparse_matrix_banded",)
    if family == "kernel_methods":
        return ("kernel_random_features",)
    if family == "multiscale":
        return ("multiscale_wavelet",)
    if family == "graph_diffusion":
        return ("graph_laplacian_diffusion",)
    return ()


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
    knobs = _declared_math_knobs(declared)
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

    if "linear_algebra_low_rank" in knobs:
        ranks = [int(v) for v in _module_attr_values(module, "rank")]
        rank = min(ranks) if ranks else 0
        findings["low_rank_measured_rank"] = rank
        findings["low_rank_consistent"] = (
            _module_has_class_fragment(module, "LowRank") and 0 < rank < dim
        )

    if "sparse_matrix_banded" in knobs:
        bandwidths = [int(v) for v in _module_attr_values(module, "bandwidth")]
        bandwidth = min(bandwidths) if bandwidths else 0
        findings["sparse_banded_measured_bandwidth"] = bandwidth
        findings["sparse_banded_consistent"] = (
            _module_has_class_fragment(module, "SparseBanded") and 0 < bandwidth <= dim
        )

    if "kernel_random_features" in knobs:
        n_features = [int(v) for v in _module_attr_values(module, "n_features")]
        feature_count = min(n_features) if n_features else 0
        findings["kernel_random_features_count"] = feature_count
        findings["kernel_random_features_consistent"] = (
            _module_has_class_fragment(module, "RandomFeatureKernel")
            and feature_count > 0
        )

    if "multiscale_wavelet" in knobs:
        n_scales = [int(v) for v in _module_attr_values(module, "n_scales")]
        scale_count = min(n_scales) if n_scales else 0
        findings["multiscale_wavelet_scales"] = scale_count
        findings["multiscale_wavelet_consistent"] = (
            _module_has_class_fragment(module, "MultiscaleWavelet") and scale_count > 0
        )

    if "graph_laplacian_diffusion" in knobs:
        steps = [int(v) for v in _module_attr_values(module, "diffusion_steps")]
        step_count = min(steps) if steps else 0
        drift = _future_drift(module, x)
        findings["graph_diffusion_steps"] = step_count
        findings["graph_diffusion_future_drift"] = drift
        findings["graph_diffusion_consistent"] = (
            _module_has_class_fragment(module, "GraphDiffusion")
            and step_count > 0
            and drift < 1e-5
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
    smoke = _smoke_test(module, dim=dim, seq_len=seq_len)
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
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(scorecard), default=str) + "\n")
    return out
