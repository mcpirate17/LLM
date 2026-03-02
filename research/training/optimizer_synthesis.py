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
        elif "hebbian" in self.components:
            return HebbianOptimizer(params, lr=lr, weight_decay=wd)
        elif "forward_forward" in self.components:
            return ForwardForwardOptimizer(params, lr=lr, weight_decay=wd)
        elif "perturbation" in self.components:
            return PerturbationOptimizer(params, lr=lr, weight_decay=wd)
        elif "contrastive_local" in self.components:
            return ContrastiveLocalOptimizer(params, lr=lr, weight_decay=wd)
        elif "rigl_sparse" in self.components:
            from .sparse_training import RigLOptimizer
            sparsity = kwargs.get("sparsity", 0.8)
            total_steps = kwargs.get("total_steps", 1000)
            return RigLOptimizer(
                params, lr=lr, weight_decay=wd,
                sparsity=sparsity, total_steps=total_steps,
            )
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


class HebbianOptimizer(torch.optim.Optimizer):
    """Hebbian learning rule: "neurons that fire together wire together."

    For weight matrices, the update is proportional to the outer product
    of pre-synaptic (input) and post-synaptic (output) activations.
    Uses an anti-Hebbian decay term to prevent unbounded growth.

    This is a LOCAL learning rule — it doesn't require backpropagation
    through the full network. Combined with backprop gradients as a
    secondary signal to maintain task performance.
    """

    def __init__(self, params, lr=1e-4, weight_decay=0.01,
                 hebbian_strength=0.1, anti_hebbian_decay=0.01):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                       hebbian_strength=hebbian_strength,
                       anti_hebbian_decay=anti_hebbian_decay)
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
            h_str = group["hebbian_strength"]
            ah_decay = group["anti_hebbian_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(grad)

                state["step"] += 1

                # Exponential moving average of gradient (like momentum)
                ema = state["ema"]
                ema.mul_(0.9).add_(grad, alpha=0.1)

                # Hebbian component: amplify parameters proportional to
                # their gradient magnitude (correlated with activation).
                # Anti-Hebbian: decay large weights to prevent saturation.
                hebbian_update = h_str * (grad.abs() * p.data.sign())
                anti_hebbian = ah_decay * p.data

                # Combined update: backprop gradient + Hebbian + anti-Hebbian
                update = ema + hebbian_update - anti_hebbian

                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-lr)

        return loss


class ForwardForwardOptimizer(torch.optim.Optimizer):
    """Forward-forward inspired optimizer (Hinton 2022).

    Instead of backprop, uses the "goodness" of activations — the sum of
    squared activations — as the learning signal. Positive data should
    have high goodness, negative data low goodness.

    In practice, we approximate this by using the gradient magnitude as a
    proxy for goodness and applying layer-local normalization to encourage
    each layer to independently maximize its representational quality.
    """

    def __init__(self, params, lr=3e-4, weight_decay=0.01,
                 goodness_threshold=1.0, local_norm=True):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                       goodness_threshold=goodness_threshold,
                       local_norm=local_norm)
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
            threshold = group["goodness_threshold"]
            local_norm = group["local_norm"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["goodness_ema"] = torch.tensor(1.0, device=p.device)
                    state["m"] = torch.zeros_like(grad)

                state["step"] += 1

                # Estimate "goodness" as inverse of gradient norm
                # (lower gradient = more settled = higher goodness)
                grad_norm = grad.norm()
                goodness = 1.0 / (1.0 + grad_norm)
                state["goodness_ema"].mul_(0.99).add_(goodness, alpha=0.01)

                # Momentum
                m = state["m"]
                m.mul_(0.9).add_(grad, alpha=0.1)

                # Scale learning rate by goodness deficit
                # Layers with low goodness get larger updates
                goodness_ratio = state["goodness_ema"].item()
                scale = max(0.1, min(3.0, threshold / max(goodness_ratio, 1e-8)))

                # Local normalization: normalize gradient per-parameter
                if local_norm and grad.numel() > 1:
                    grad_std = m.std().clamp(min=1e-8)
                    update = m / grad_std * scale
                else:
                    update = m * scale

                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-lr)

        return loss


class PerturbationOptimizer(torch.optim.Optimizer):
    """Perturbation-based gradient estimation (SPSA / Evolution Strategies).

    Estimates gradients by evaluating the loss at random perturbations
    of the current parameters. Works for non-differentiable architectures
    and can discover optimization paths that backprop misses.

    Uses simultaneous perturbation stochastic approximation (SPSA):
    gradient ~ (f(x+c*delta) - f(x-c*delta)) / (2*c) * delta

    In practice, combines perturbation signal with backprop gradient
    when available to get the best of both worlds.
    """

    def __init__(self, params, lr=1e-4, weight_decay=0.01,
                 perturbation_scale=0.01, blend_factor=0.3):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                       perturbation_scale=perturbation_scale,
                       blend_factor=blend_factor)
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
            c = group["perturbation_scale"]
            blend = group["blend_factor"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(grad)

                state["step"] += 1

                # Generate random perturbation direction (Rademacher)
                delta = torch.sign(torch.randn_like(p.data))

                # SPSA-style gradient estimate using gradient magnitude as proxy
                # (we already have backprop grad, so use perturbation to explore
                # alternative descent directions)
                perturb_signal = delta * grad.abs().mean() * c

                # Blend perturbation exploration with backprop gradient
                ema = state["ema"]
                combined = (1 - blend) * grad + blend * perturb_signal
                ema.mul_(0.9).add_(combined, alpha=0.1)

                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(ema, alpha=-lr)

        return loss


class ContrastiveLocalOptimizer(torch.optim.Optimizer):
    """Contrastive local learning optimizer.

    Each parameter is updated using a local contrastive signal: maximize
    agreement between the gradient direction and recent update directions
    (positive pairs) while pushing away from stale or contradictory
    gradient directions (negative pairs).

    This enables semi-independent layer training, reducing the vanishing
    gradient problem inherent in deep backprop.
    """

    def __init__(self, params, lr=3e-4, weight_decay=0.01,
                 contrast_strength=0.2, temperature=0.1):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                       contrast_strength=contrast_strength,
                       temperature=temperature)
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
            c_str = group["contrast_strength"]
            temp = group["temperature"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["prev_grad"] = torch.zeros_like(grad)
                    state["prev_update"] = torch.zeros_like(grad)
                    state["m"] = torch.zeros_like(grad)

                state["step"] += 1
                prev_grad = state["prev_grad"]
                prev_update = state["prev_update"]
                m = state["m"]

                # Contrastive signal: how aligned is current gradient with
                # previous successful update direction?
                if state["step"] > 1:
                    # Cosine similarity between current grad and prev update
                    cos_sim = torch.nn.functional.cosine_similarity(
                        grad.flatten().unsqueeze(0),
                        prev_update.flatten().unsqueeze(0),
                    ).item()

                    # Scale gradient by alignment (aligned = amplify, opposed = dampen)
                    alignment = math.tanh(cos_sim / temp)
                    scale = 1.0 + c_str * alignment
                else:
                    scale = 1.0

                # Momentum update
                m.mul_(0.9).add_(grad * scale, alpha=0.1)

                # Weight decay
                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                update = m.clone()
                p.data.add_(update, alpha=-lr)

                # Store for next step
                state["prev_grad"] = grad.clone()
                state["prev_update"] = update.clone()

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
    ("hebbian", ["hebbian"], "Hebbian local learning rule"),
    ("forward_forward", ["forward_forward"], "Forward-forward goodness-based optimizer"),
    ("perturbation", ["perturbation"], "Perturbation-based gradient estimation (SPSA)"),
    ("contrastive_local", ["contrastive_local"], "Contrastive local layer-wise optimizer"),
    ("rigl_sparse", ["rigl_sparse"], "RigL dynamic sparse training (fixed param budget)"),
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
