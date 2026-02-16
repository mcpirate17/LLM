"""
Optimizer Synthesis

Generate optimizer update rules from primitives instead of using only AdamW.
Each synthesized optimizer defines how to update parameters given gradients.

Examples:
- Spectral momentum: maintain FFT of gradient history
- Sign + frequency: combine sign updates with frequency-domain signals
- Tropical gradient: min-plus accumulation of gradient history
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class OptimizerComponent:
    """A component of the optimizer update rule."""
    name: str
    description: str = ""


@dataclass
class SynthesizedOptimizer:
    """A synthesized optimizer."""
    name: str
    components: List[str] = field(default_factory=list)
    lr: float = 3e-4
    weight_decay: float = 0.01
    description: str = ""
    seed: int = 0

    def create(self, params, **kwargs) -> torch.optim.Optimizer:
        """Create an optimizer instance."""
        lr = kwargs.get("lr", self.lr)
        wd = kwargs.get("weight_decay", self.weight_decay)

        if "spectral_momentum" in self.components:
            return SpectralMomentumOptimizer(params, lr=lr, weight_decay=wd)
        elif "sign_descent" in self.components:
            return SignDescentOptimizer(params, lr=lr, weight_decay=wd)
        elif "tropical_grad" in self.components:
            return TropicalGradientOptimizer(params, lr=lr, weight_decay=wd)
        elif "lion_variant" in self.components:
            return LionVariantOptimizer(params, lr=lr, weight_decay=wd)
        else:
            # Default: AdamW with modified betas
            betas = (0.9, 0.999)
            if "high_momentum" in self.components:
                betas = (0.95, 0.999)
            if "low_momentum" in self.components:
                betas = (0.8, 0.95)
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "components": self.components,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "description": self.description,
            "seed": self.seed,
        }


class SpectralMomentumOptimizer(torch.optim.Optimizer):
    """Optimizer that maintains FFT of gradient history.

    Instead of exponential moving average (like Adam), this maintains
    the gradient history in frequency domain. Low-frequency gradient
    components (persistent directions) get higher effective learning rate.
    """

    def __init__(self, params, lr=3e-4, weight_decay=0.01, history_len=16):
        defaults = dict(lr=lr, weight_decay=weight_decay, history_len=history_len)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            history_len = group["history_len"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["grad_history"] = torch.zeros(
                        history_len, *grad.shape, device=grad.device
                    )
                    state["idx"] = 0

                state["step"] += 1
                idx = state["idx"] % history_len
                state["grad_history"][idx] = grad
                state["idx"] = idx + 1

                # FFT of gradient history
                n_filled = min(state["step"], history_len)
                history = state["grad_history"][:n_filled]

                if n_filled >= 2:
                    # Spectral analysis along history dimension
                    freq = torch.fft.rfft(history, dim=0)
                    # Low frequency = persistent direction = amplify
                    n_freq = freq.shape[0]
                    weights = torch.linspace(2.0, 0.5, n_freq, device=grad.device)
                    for i in range(n_freq):
                        freq[i] *= weights[i]
                    # Reconstruct weighted gradient
                    weighted = torch.fft.irfft(freq, n=n_filled, dim=0)
                    update = weighted[-1]  # Most recent weighted
                else:
                    update = grad

                # Weight decay
                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                p.data.add_(update, alpha=-lr)

        return loss


class SignDescentOptimizer(torch.optim.Optimizer):
    """Sign descent with momentum — like Lion but from primitives.

    Update rule: sign(momentum * beta + grad * (1-beta))
    """

    def __init__(self, params, lr=1e-4, weight_decay=0.01, beta=0.9):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta=beta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta = group["beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(grad)

                m = state["momentum"]
                update = torch.sign(m * beta + grad * (1 - beta))

                # Update momentum
                state["momentum"] = m * beta + grad * (1 - beta)

                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-lr)

        return loss


class TropicalGradientOptimizer(torch.optim.Optimizer):
    """Tropical gradient optimizer: min-plus accumulation.

    Instead of averaging gradients (Euclidean), uses tropical
    algebra: tracks the minimum gradient direction, which
    corresponds to the "shortest path" to the optimum.
    """

    def __init__(self, params, lr=3e-4, weight_decay=0.01, decay=0.9):
        defaults = dict(lr=lr, weight_decay=weight_decay, decay=decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            decay = group["decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["tropical_acc"] = grad.clone()
                    state["v"] = torch.zeros_like(grad)

                # Tropical accumulation: element-wise min of |grad| history
                # but keeping sign information
                acc = state["tropical_acc"]
                v = state["v"]

                # Min of absolute values (tropical add)
                abs_acc = acc.abs()
                abs_grad = grad.abs()
                new_abs = torch.minimum(abs_acc * decay, abs_grad)
                new_sign = torch.where(abs_grad < abs_acc * decay,
                                      grad.sign(), acc.sign())
                state["tropical_acc"] = new_sign * new_abs

                # Second moment (for adaptive LR)
                v.mul_(0.999).addcmul_(grad, grad, value=0.001)
                state["v"] = v

                # Update
                denom = v.sqrt().clamp(min=1e-8)
                update = state["tropical_acc"] / denom

                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-lr)

        return loss


class LionVariantOptimizer(torch.optim.Optimizer):
    """Variant of Lion (sign-based) with learned interpolation."""

    def __init__(self, params, lr=1e-4, weight_decay=0.01,
                 beta1=0.9, beta2=0.99):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                       beta1=beta1, beta2=beta2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            b1 = group["beta1"]
            b2 = group["beta2"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["m"] = torch.zeros_like(grad)

                m = state["m"]

                # Update = sign(interpolation)
                update = torch.sign(m * b1 + grad * (1 - b1))

                # Weight decay
                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-lr)

                # Momentum update (different interpolation)
                state["m"] = m * b2 + grad * (1 - b2)

        return loss


# ── Synthesis ─────────────────────────────────────────────────────────

OPTIMIZER_RECIPES = [
    ("adamw_standard", ["adamw"], "Standard AdamW"),
    ("adamw_high_momentum", ["adamw", "high_momentum"], "AdamW with high momentum"),
    ("adamw_low_momentum", ["adamw", "low_momentum"], "AdamW with low momentum"),
    ("spectral_momentum", ["spectral_momentum"], "Spectral momentum optimizer"),
    ("sign_descent", ["sign_descent"], "Sign descent with momentum"),
    ("tropical_gradient", ["tropical_grad"], "Tropical gradient accumulation"),
    ("lion_variant", ["lion_variant"], "Lion-style sign-based optimizer"),
]


def synthesize_optimizer(seed: Optional[int] = None) -> SynthesizedOptimizer:
    """Generate a random optimizer."""
    rng = random.Random(seed)

    name, components, desc = rng.choice(OPTIMIZER_RECIPES)

    # Randomize hyperparameters
    lr = 10 ** rng.uniform(-4.5, -3.0)  # 3e-5 to 1e-3
    wd = 10 ** rng.uniform(-3, -1)      # 0.001 to 0.1

    return SynthesizedOptimizer(
        name=name,
        components=components,
        lr=lr,
        weight_decay=wd,
        description=desc,
        seed=seed or 0,
    )
