import torch
import io
import os
from .compiler import compile_workflow
from .dispatch import KernelDispatcher

def export_onnx(workflow_json, components_dir, opset_version=14):
    """
    Export workflow to ONNX bytes.
    """
    # 1. Compile to torch.nn.Module
    model = compile_workflow(workflow_json, components_dir)
    model.eval()
    
    # 2. Infer input shapes to generate dummy input
    # We need to run shape inference manually since WorkflowModule doesn't expose it directly yet
    # (it does validation but doesn't store the inferred input shapes for us easily, 
    #  though it stores topological order).
    
    dispatcher = KernelDispatcher()
    
    # Re-construct node_rules for shape inference
    # This duplicates some logic from compiler.py but is necessary without refactoring
    node_ids = [n['id'] for n in workflow_json['nodes']]
    node_map = {n['id']: n for n in workflow_json['nodes']}
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    # We need to map component types to ShapeRules. 
    # For now, we'll use a simplified mapping or rely on the manifest if we had access.
    # Since we don't have easy access to manifests here without Registry, 
    # let's assume we can get it from the model's registry.
    
    registry = model.component_registry
    
    # Find input nodes
    source_nodes = []
    edges = workflow_json['edges']
    targets = {e['target'] for e in edges}
    for nid in node_ids:
        if nid not in targets:
            source_nodes.append(nid)
            
    # Create dummy inputs
    dummy_inputs = {}
    # Default shape B=1, S=16, D=64 if unknown
    # Ideally we'd parse this from the 'input' component params if available
    
    for nid in source_nodes:
        # Check if it has a shape param
        comp = node_map[nid]
        # Heuristic: [1, 16, 64]
        dummy_inputs[nid] = torch.randn(1, 16, 64)
        
    # We need to pass args as a tuple or dict depending on how forward is implemented.
    # WorkflowModule.forward takes a dict `inputs`.
    # torch.onnx.export expects args to be passed to model(*args).
    # Since model() signature is `forward(inputs)`, we pass one arg: the dict.
    # BUT torch.onnx.export doesn't support Dict inputs well in older versions, 
    # or it traces it.
    
    # Let's try tracing with the dict.
    
    buffer = io.BytesIO()
    
    try:
        torch.onnx.export(
            model,
            (dummy_inputs,),
            buffer,
            input_names=list(dummy_inputs.keys()),
            # output_names we'd need to know
            opset_version=opset_version,
            do_constant_folding=True
        )
    except Exception as e:
        # Fallback: if dict fails, maybe wrapper?
        # For now, raise
        raise RuntimeError(f"ONNX export failed: {e}")
        
    return buffer.getvalue()
