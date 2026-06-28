"""Optimizer-as-program: the update rule as a searchable, parametric atom.

An optimizer is just a small differentiable program over ``(grad, momentum,
variance)`` — this is the space symbolic search explored to discover Lion. Here
the update is one continuous family with an identity-at-init default (AdamW), so a
generator/optimizer can dial it toward Lion, signSGD, plain momentum, or a blend
nobody has named:

    m = b1*m + (1-b1)*g                      # first moment (momentum)
    v = b2*v + (1-b2)*g^2                     # second moment
    adam_dir = m / (sqrt(v) + eps)           # AdamW direction (mix=0)
    lion_dir = sign(b3*m + (1-b3)*g)         # Lion direction  (mix=1)
    update   = lr * ((1-mix)*adam_dir + mix*lion_dir)

``mix`` slides AdamW <-> Lion; ``b1/b2/b3`` and ``log_lr`` are the remaining
knobs. The default spec is AdamW (``mix=0``). Candidates are graded the only way
that matters for an update rule — by the training trajectory they produce on a
fixed nano problem (fractional loss reduction), never by a name. A divergent
optimizer scores its (low) measured outcome; that IS the measurement, not an error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class UpdateSpec:
    """One update rule = a point in the AdamW<->Lion family.

    The default is AdamW; ``mix=1`` is Lion. All fields are continuous so the
    search can land between and beyond the named optimizers.
    """

    mix: float = 0.0  # 0 = AdamW direction, 1 = Lion (sign) direction
    beta1: float = 0.9  # first-moment decay
    beta2: float = 0.999  # second-moment decay
    beta3: float = 0.9  # Lion interpolation (momentum vs grad inside sign)
    log_lr: float = math.log(3e-3)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("mix", "beta1", "beta2", "beta3"):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]; got {val}")

    @property
    def lr(self) -> float:
        return math.exp(self.log_lr)

    @property
    def key(self) -> str:
        if self.mix == 0.0:
            family = "adamw"
        elif self.mix == 1.0:
            family = "lion"
        else:
            family = f"blend{self.mix:.2f}"
        return f"{family}@lr{self.lr:.1e}"


class ParametricOptimizer(torch.optim.Optimizer):
    """The update family above, as a drop-in ``torch.optim.Optimizer``."""

    def __init__(self, params, spec: UpdateSpec | None = None) -> None:
        spec = spec or UpdateSpec()
        super().__init__(params, dict(lr=spec.lr))
        self.spec = spec

    @torch.no_grad()
    def step(self) -> None:  # type: ignore[override]
        s = self.spec
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                m.mul_(s.beta1).add_(g, alpha=1.0 - s.beta1)
                v.mul_(s.beta2).addcmul_(g, g, value=1.0 - s.beta2)
                adam_dir = m / (v.sqrt() + s.eps)
                lion_dir = torch.sign(s.beta3 * m + (1.0 - s.beta3) * g)
                direction = (1.0 - s.mix) * adam_dir + s.mix * lion_dir
                p.add_(direction, alpha=-lr)


# --------------------------------------------------------------------------- #
# Grading harness — a harder fixed problem, scored by the WHOLE trajectory
# (Area-Under-Loss-Curve), not the final loss.
#
# Why AULC and not final-loss reduction: on an easy problem every non-divergent
# optimizer drives the final loss to ~0, so final-loss fractional reduction
# saturates near 1.0 and cannot RANK update rules (the failure noted in
# name_free_discovery_next_steps). AULC integrates the relative loss over every
# step, so a faster-converging rule scores higher even when all candidates
# eventually reach the same floor — convergence SPEED is exactly what separates
# AdamW / Lion / their blends. The target is also deepened so the landscape has
# real dynamic range over the horizon rather than collapsing in a few steps.
# --------------------------------------------------------------------------- #
def _problem(seed: int, device: str) -> tuple[nn.Module, Tensor, Tensor]:
    """A deterministic deep non-convex regression: 3-layer tanh target, fixed data.

    Deeper than a single hidden layer so it does not saturate within the horizon
    — the trajectory keeps moving, which is what AULC needs to discriminate.
    """
    # Reset the global RNG so data, target weights and start point are identical
    # across optimizer candidates (module init draws from the global generator).
    torch.manual_seed(seed)
    n, d_in, d_hidden = 256, 16, 48
    x = torch.randn(n, d_in, device=device)
    target = nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.Tanh(),
        nn.Linear(d_hidden, d_hidden),
        nn.Tanh(),
        nn.Linear(d_hidden, 1),
    ).to(device)
    for param in target.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        y = target(x)
    model = nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, 1),
    ).to(device)
    # Deterministic re-init so the starting point is identical across optimizers.
    torch.manual_seed(seed + 1)
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model, x, y


def grade_optimizer(
    spec: UpdateSpec, *, steps: int = 80, seed: int = 0, device: str = "cpu"
) -> float:
    """Area-Under-Loss-Curve score of the update rule on the nano problem.

    ``score = 1 - mean_t(loss_t / loss_0)`` over the optimization trajectory:
    1.0 = drove the loss to zero immediately, 0 = no progress, <0 = the loss
    spent the run above its starting point (slow or diverging). A non-finite loss
    at any step is reported as the measured low score (-1.0), never raised.
    Higher = converges faster AND lower — the property that ranks update rules.
    """
    model, x, y = _problem(seed, device)
    opt = ParametricOptimizer(model.parameters(), spec)
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        loss0 = float(loss_fn(model(x), y))
    if loss0 <= 0.0 or not math.isfinite(loss0):
        return -1.0
    rel_area = 0.0  # running sum of loss_t / loss_0 across the trajectory
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            lt = float(loss_fn(model(x), y))
        if not math.isfinite(lt):
            return -1.0
        rel_area += lt / loss0
    return 1.0 - rel_area / steps
