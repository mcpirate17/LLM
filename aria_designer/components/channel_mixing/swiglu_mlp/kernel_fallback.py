"""Auto-generated Python fallback kernel for swiglu_mlp."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwigluMlpFallback:
    """Fallback handler for swiglu_mlp."""

    def __call__(self, module, x):
        """Execute swiglu_mlp operation."""
        if not hasattr(module, 'gate_proj'):
            # Initialize projections if missing (lazy init for designer context)
            D = x.shape[-1]
            hidden = D * 3
            module.gate_proj = torch.nn.Linear(D, hidden, bias=False)
            module.up_proj = torch.nn.Linear(D, hidden, bias=False)
            module.down_proj = torch.nn.Linear(hidden, D, bias=False)

        gate = module.gate_proj(x)
        up = module.up_proj(x)
        activated = F.silu(gate) * up
        return module.down_proj(activated)
