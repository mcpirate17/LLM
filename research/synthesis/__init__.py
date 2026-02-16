"""
Program Synthesis Engine

Generates computation graphs from primitive tensor operations,
compiles them to live PyTorch modules, and validates them.
"""

from .primitives import PrimitiveOp, PRIMITIVE_REGISTRY, get_primitive, list_primitives
from .graph import OpNode, ComputationGraph
from .grammar import GrammarConfig, generate_layer_graph
from .compiler import compile_graph, CompiledLayer, SynthesizedModel
from .validator import validate_graph, ValidationResult
from .serializer import graph_to_json, graph_from_json
