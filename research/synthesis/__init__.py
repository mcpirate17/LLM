"""
Program Synthesis Engine

Generates computation graphs from primitive tensor operations,
compiles them to live PyTorch modules, and validates them.
"""

from .primitives import (
    PrimitiveOp as PrimitiveOp,
    PRIMITIVE_REGISTRY as PRIMITIVE_REGISTRY,
    get_primitive as get_primitive,
    list_primitives as list_primitives,
)
from .graph import OpNode as OpNode, ComputationGraph as ComputationGraph
from .grammar import (
    GrammarConfig as GrammarConfig,
    generate_layer_graph as generate_layer_graph,
)
from .compiler import (
    compile_graph as compile_graph,
    CompiledLayer as CompiledLayer,
    SynthesizedModel as SynthesizedModel,
)
from .validator import (
    validate_graph as validate_graph,
    ValidationResult as ValidationResult,
)
from .serializer import (
    graph_to_json as graph_to_json,
    graph_from_json as graph_from_json,
)
