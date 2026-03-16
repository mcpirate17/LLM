"""
Routing & Compression Schema Validator.
Manual implementation of routing_compression.v1.schema.json validation
to avoid external dependencies like jsonschema.
"""

from typing import Any, Dict, List, Optional

VALID_ROUTING_KINDS = {
    "uniform",
    "mod_topk",
    "early_exit",
    "adaptive_recursion",
    "token_merge",
    "cascade",
    "speculative",
    "mixture_of_paths",
    "multi_lane"
}

VALID_COMPRESSION_KINDS = {
    "dense_matrix",
    "low_rank",
    "shared_basis",
    "tied_proj",
    "grouped_linear",
    "block_sparse",
    "structured_sparse",
    "semi_structured_2_4",
    "bottleneck_proj",
    "quantized"
}

def validate_routing_compression(data: Any) -> None:
    """Validate data against routing_compression.v1 schema.
    Raises ValueError if invalid.
    """
    if not isinstance(data, dict):
        raise ValueError("Schema data must be an object")
    
    if "routing" not in data:
        raise ValueError("Missing required property: routing")
    if "compression" not in data:
        raise ValueError("Missing required property: compression")
        
    r = data["routing"]
    if not isinstance(r, dict) or "kind" not in r:
        raise ValueError("routing must be an object with 'kind'")
    if r["kind"] not in VALID_ROUTING_KINDS:
        raise ValueError(f"Invalid routing kind: {r['kind']}")
        
    c = data["compression"]
    if not isinstance(c, dict) or "kind" not in c:
        raise ValueError("compression must be an object with 'kind'")
    if c["kind"] not in VALID_COMPRESSION_KINDS:
        raise ValueError(f"Invalid compression kind: {c['kind']}")
