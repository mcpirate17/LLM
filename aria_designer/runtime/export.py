import torch
import io
from .compiler import compile_workflow


def export_onnx(workflow_json, components_dir, opset_version=14):
    """Export workflow to ONNX bytes."""
    model = compile_workflow(workflow_json, components_dir)
    model.eval()

    node_map = {n["id"]: n for n in workflow_json["nodes"]}

    # Find source nodes (no incoming edges)
    targets = {e["target"] for e in workflow_json["edges"]}
    source_nodes = [nid for nid in node_map if nid not in targets]

    # Build dummy inputs — default shape [1, 16, 64]
    dummy_inputs = {}
    for nid in source_nodes:
        params = node_map[nid].get("params", {})
        dim = params.get("dim", 64)
        seq = params.get("seq_len", 16)
        dummy_inputs[nid] = torch.randn(1, seq, dim)

    buffer = io.BytesIO()
    try:
        torch.onnx.export(
            model,
            (dummy_inputs,),
            buffer,
            input_names=list(dummy_inputs.keys()),
            opset_version=opset_version,
            do_constant_folding=True,
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}") from e

    return buffer.getvalue()
