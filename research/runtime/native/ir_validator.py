"""Validate native runner IR documents against the v1 JSON schema."""

import json
import os

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "schemas", "native_ir.v1.json"
)

_schema_cache = None


def _load_schema():
    global _schema_cache
    if _schema_cache is None:
        with open(_SCHEMA_PATH, "r") as f:
            _schema_cache = json.load(f)
    return _schema_cache


def validate_ir(ir_doc: dict) -> list[str]:
    """Validate an IR document. Returns list of error messages (empty = valid)."""
    errors = []

    # Schema validation (if jsonschema available)
    if HAS_JSONSCHEMA:
        schema = _load_schema()
        validator = jsonschema.Draft202012Validator(schema)
        for err in validator.iter_errors(ir_doc):
            errors.append(f"schema: {err.message} at {list(err.absolute_path)}")
    else:
        # Manual validation fallback
        for field in (
            "schema_version",
            "model_dim",
            "nodes",
            "edges",
            "output_node_id",
        ):
            if field not in ir_doc:
                errors.append(f"missing required field: {field}")
        if ir_doc.get("schema_version") != "native_ir.v1":
            errors.append(f"unsupported schema_version: {ir_doc.get('schema_version')}")

    if errors:
        return errors

    # Structural validation beyond schema
    node_ids = set()
    for node in ir_doc.get("nodes", []):
        nid = node["id"]
        if nid in node_ids:
            errors.append(f"duplicate node id: {nid}")
        node_ids.add(nid)
        for inp_id in node.get("input_ids", []):
            if inp_id not in node_ids and inp_id >= nid:
                # Forward references - check after all nodes loaded
                pass

    # Validate all input_ids reference existing nodes
    for node in ir_doc.get("nodes", []):
        for inp_id in node.get("input_ids", []):
            if inp_id not in node_ids:
                errors.append(
                    f"node {node['id']} references non-existent input {inp_id}"
                )

    # Validate edges reference existing nodes
    for edge in ir_doc.get("edges", []):
        if edge["source"] not in node_ids:
            errors.append(f"edge source {edge['source']} not in nodes")
        if edge["target"] not in node_ids:
            errors.append(f"edge target {edge['target']} not in nodes")

    # Validate output_node_id exists
    if ir_doc.get("output_node_id") not in node_ids:
        errors.append(f"output_node_id {ir_doc.get('output_node_id')} not in nodes")

    # Cycle detection (Kahn's algorithm)
    adj = {nid: [] for nid in node_ids}
    in_degree = {nid: 0 for nid in node_ids}
    for edge in ir_doc.get("edges", []):
        src, tgt = edge["source"], edge["target"]
        if src in adj and tgt in adj:
            adj[src].append(tgt)
            in_degree[tgt] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        nid = queue.pop(0)
        visited += 1
        for neighbor in adj[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited < len(node_ids):
        errors.append("graph contains a cycle")

    return errors


def validate_ir_json(json_str: str) -> list[str]:
    """Validate an IR JSON string. Returns list of error messages."""
    try:
        doc = json.loads(json_str)
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"]
    return validate_ir(doc)
