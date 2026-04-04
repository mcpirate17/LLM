"""Python fallback kernel for graph_attention.

Identical causal self-attention logic to softmax_attention — delegates to avoid duplication.
(In fallback mode, edge features are not available.)
"""

from aria_designer.components.mixing.softmax_attention.kernel_fallback import (
    ComponentHandler,
)

__all__ = ["ComponentHandler"]
