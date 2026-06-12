"""Python fallback kernel for softmax_attention."""

import torch.nn.functional as F

from aria_designer.components.base import make_causal_attention_handler

ComponentHandler = make_causal_attention_handler(
    lambda scores, config: F.softmax(scores, dim=-1), mask_value=float("-inf")
)
