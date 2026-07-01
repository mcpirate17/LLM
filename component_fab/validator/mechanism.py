"""Mechanistic probes for novel architectures (Run-2).

Confirms that a candidate uses its declared mechanism honestly, rather than
collapsing to a softmax-twin or an identity/constant path.

Battery includes:
- Routing health: entropy/balance/collapse of gated lanes.
- Relaxation: decreasing prediction error over training (surprise relaxation).
- Address entropy: slot utilization in content-addressed lanes.
- Ablation Δ: performance drop when the novel mechanism is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..metrics.routing_health import measure_routing_health
from ..proposer.algebraic_properties import measure_algebraic_properties


@dataclass(frozen=True, slots=True)
class MechanisticObservable:
    routing_entropy_mean: float = 0.0
    load_balance_cv: float = 0.0
    state_degeneracy: float = 0.0
    active_lane_fraction: float = 1.0
    relaxation_slope: float = 0.0
    address_entropy: float = 0.0
    ablation_delta: float = 0.0
    #: Measured convex-token-averaging signature in [0, 1]; high = the lane's
    #: end-to-end forward behaves like a softmax-shaped averager (a pathology to
    #: steer away from, per the non-QKV mission), low = genuinely novel geometry.
    softmax_twin_score: float = 0.0
    passed: bool = True
    notes: tuple[str, ...] = ()


def probe_observables(
    lane: nn.Module,
    *,
    dim: int,
    seq_len: int,
    batch_size: int = 4,
    n_train_steps: int = 40,
) -> MechanisticObservable:
    """Run the mechanistic observable battery on a single lane module."""
    notes = []
    trainable_params = [p for p in lane.parameters() if p.requires_grad]
    if n_train_steps > 0 and not trainable_params:
        notes.append("no_trainable_parameters")

    # 1. Routing Health (if applicable)
    routing = _check_routing(lane, dim=dim, seq_len=seq_len, batch_size=batch_size)

    # 2. Relaxation (if applicable)
    relaxation_slope = _check_relaxation(
        lane, dim=dim, seq_len=seq_len, n_steps=n_train_steps
    )

    # 3. Address Entropy (if applicable)
    addr_entropy = _check_address_entropy(
        lane, dim=dim, seq_len=seq_len, batch_size=batch_size
    )

    # 4. Softmax-twin signature: measure whether the lane's end-to-end forward
    #    behaves like a convex token-averager (the softmax structural tell). High
    #    is the pathology to demote downstream, not a target.
    twin_score = _check_softmax_twin(
        lane, dim=dim, seq_len=seq_len, batch_size=batch_size
    )

    # 5. Ablation Delta (if applicable)
    # TODO: Implement ablation delta probe

    return MechanisticObservable(
        routing_entropy_mean=routing.get("entropy", 0.0),
        load_balance_cv=routing.get("lb_cv", 0.0),
        state_degeneracy=routing.get("collapse", 0.0),
        active_lane_fraction=routing.get("active_frac", 1.0),
        relaxation_slope=relaxation_slope,
        address_entropy=addr_entropy,
        softmax_twin_score=twin_score,
        passed=bool(trainable_params or n_train_steps <= 0),
        notes=tuple(notes),
    )


def _check_softmax_twin(
    lane: nn.Module,
    *,
    dim: int,
    seq_len: int,
    batch_size: int,
) -> float:
    """Measured convex-token-averaging (softmax-twin) score for the lane forward.

    The lane obeys the same ``[B, L, D] -> [B, L, D]`` contract exercised by the
    relaxation probe, so it is directly usable as the operator ``f``. Sequence
    length is capped so this stays a cheap structural check.
    """
    probe_len = max(4, min(int(seq_len), 16))
    props = measure_algebraic_properties(
        lane, dim=dim, seq_len=probe_len, batch=batch_size, n_seeds=2
    )
    return props.softmax_twin_score


def _check_relaxation(
    lane: nn.Module,
    *,
    dim: int,
    seq_len: int,
    n_steps: int,
) -> float:
    """Measure the training loss slope on a short binding task.

    A negative slope means the mechanism is relaxing toward a stable
    state (learning to predict).
    """
    if n_steps <= 0:
        return 0.0

    params = [p for p in lane.parameters() if p.requires_grad]
    if not params:
        return 0.0

    optimizer = torch.optim.Adam(params, lr=3e-3)
    losses = []

    lane.train()
    for _ in range(n_steps):
        # [B, L, D] random continuous data and a 'identity' target
        # as a proxy for relaxation in the absence of a full LM.
        x = torch.randn(4, seq_len, dim)
        target = x.clone()

        optimizer.zero_grad()
        y = lane(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if len(losses) < 2:
        return 0.0

    # Simple linear regression slope
    x_axis = torch.arange(len(losses), dtype=torch.float32)
    y_axis = torch.tensor(losses, dtype=torch.float32)
    x_mean = x_axis.mean()
    y_mean = y_axis.mean()
    num = ((x_axis - x_mean) * (y_axis - y_mean)).sum()
    den = ((x_axis - x_mean) ** 2).sum()
    slope = (num / den).item() if den > 0 else 0.0
    return slope


def _check_address_entropy(
    lane: nn.Module,
    *,
    dim: int,
    seq_len: int,
    batch_size: int,
) -> float:
    """Measure the entropy of the address distribution in slotted lanes."""
    # Look for slot/address patterns
    if hasattr(lane, "n_slots") and hasattr(lane, "forward"):
        # Placeholder for future implementation using hooks
        pass
    return 0.0


def _check_routing(
    lane: nn.Module,
    *,
    dim: int,
    seq_len: int,
    batch_size: int,
) -> dict[str, float]:
    """Identify and probe routing layers within the lane."""
    # Look for known routing patterns
    if hasattr(lane, "write_route") and isinstance(
        lane.write_route, (nn.Linear, nn.ModuleList)
    ):
        # Common in SlotTableMemoryLane and MultiHeadSlotTableMemoryLane
        n_lanes = getattr(lane, "n_slots", 0) or getattr(lane, "n_heads", 1) * getattr(
            lane, "n_slots", 1
        )
        if n_lanes > 1:
            try:
                # We need a function that takes [B, L, D] and returns weights [B, L, n_lanes]
                def router_fn(x: torch.Tensor) -> torch.Tensor:
                    with torch.no_grad():
                        if hasattr(lane, "k"):
                            feat = torch.tanh(lane.k(x))
                        else:
                            feat = x

                        if isinstance(lane.write_route, nn.ModuleList):
                            # MultiHead grouped router
                            b, l, d = feat.shape
                            h = len(lane.write_route)
                            hd = d // h
                            feat_heads = feat.view(b, l, h, hd)
                            logits = torch.stack(
                                [
                                    r(feat_heads[:, :, i])
                                    for i, r in enumerate(lane.write_route)
                                ],
                                dim=2,
                            )
                        else:
                            logits = lane.write_route(feat)

                        return torch.softmax(logits.flatten(start_dim=2), dim=-1)

                health = measure_routing_health(
                    router_fn,
                    n_lanes=n_lanes,
                    seq_len=seq_len,
                    feature_dim=dim,
                    batch_size=batch_size,
                )
                return {
                    "entropy": health.routing_entropy_mean,
                    "lb_cv": health.load_balance_cv,
                    "collapse": health.mode_collapse_propensity,
                    "active_frac": health.active_lane_fraction,
                }
            except Exception:
                pass

    return {}
