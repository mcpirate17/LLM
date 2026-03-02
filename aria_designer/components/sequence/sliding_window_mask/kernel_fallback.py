"""Auto-generated Python fallback kernel for sliding_window_mask."""

import torch
import torch.nn.functional as F

class SlidingWindowMaskFallback:
    """Fallback handler for sliding_window_mask."""

    def __call__(self, module, x, window_size=32):
        """Execute sliding_window_mask operation."""
        B, S, D = x.shape
        W = int(window_size)
        
        # Python Fallback: O(S^2) masking
        W_safe = min(W, S)
        row_idx = torch.arange(S, device=x.device).unsqueeze(1)
        col_idx = torch.arange(S, device=x.device).unsqueeze(0)
        dist = (row_idx - col_idx)
        
        # Causal sliding window: col <= row AND dist < W
        mask = (dist >= 0) & (dist < W_safe)
        # Numerical decay
        decay = torch.exp(-dist.float().clamp(min=0) / max(W_safe / 4, 1.0))
        
        # Normalize per-position to maintain signal scale
        final_mask = (mask.float() * decay)
        final_mask = final_mask / final_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        return torch.bmm(final_mask.unsqueeze(0).expand(B, -1, -1), x)
