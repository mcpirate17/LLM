"""
Novel Mathematical Spaces as Primitives

Non-Euclidean geometries and alternative algebraic structures
that can be used as building blocks in synthesized computation graphs.
"""

from . import clifford, compression, hyperbolic, padic, spiking, tropical
from .registry import register_all_mathspaces

__all__ = [
	"register_all_mathspaces",
	"hyperbolic",
	"tropical",
	"padic",
	"clifford",
	"compression",
	"spiking",
]

# Ensure mathspace ops are always registered when this package is imported.
# This handles the circular-import case where synthesis.primitives can't
# eagerly import us (because we're mid-import). Idempotent.
register_all_mathspaces()
