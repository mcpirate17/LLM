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
