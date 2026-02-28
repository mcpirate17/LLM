"""Auto-generated Python fallback kernel for reciprocal."""

import torch

class ReciprocalFallback:
    """Fallback handler for reciprocal."""

    def __call__(self, module, x):
        """Execute reciprocal operation."""
        # Differentiable stable reciprocal: 1 / (x + epsilon*sgn(x))
        ones = torch.ones_like(x)
        sign = torch.where(x >= 0, ones, -ones)
        return 1.0 / (x + 1e-6 * sign.clamp(min=1.0))
