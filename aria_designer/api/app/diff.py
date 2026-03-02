from typing import List, Dict, Any
from .models import PatchOpModel

def diff_graphs(graph_a: Dict[str, Any], graph_b: Dict[str, Any]) -> List[PatchOpModel]:
    """
    Compare two graphs and return operations to transform A into B.
    """
    ops = []
    
    nodes_a = {n["id"]: n for n in graph_a.get("nodes", [])}
    nodes_b = {n["id"]: n for n in graph_b.get("nodes", [])}
    
    # 1. Removed nodes
    for nid in nodes_a:
        if nid not in nodes_b:
            ops.append(PatchOpModel(op="remove_node", node_id=nid))
            
    # 2. Added nodes
    for nid, node in nodes_b.items():
        if nid not in nodes_a:
            ops.append(PatchOpModel(
                op="add_node", 
                payload={
                    "component_id": node["component_type"],
                    "params": node.get("params", {}),
                    "position": node.get("ui_meta", {}).get("position", {"x": 0, "y": 0})
                }
            ))
            
    # 3. Mutated params
    for nid, node_b in nodes_b.items():
        if nid in nodes_a:
            node_a = nodes_a[nid]
            if node_a.get("params") != node_b.get("params"):
                for k, v in node_b.get("params", {}).items():
                    if node_a.get("params", {}).get(k) != v:
                        ops.append(PatchOpModel(
                            op="mutate_param",
                            node_id=nid,
                            payload={"param_name": k, "value": v}
                        ))
                        
    # 4. Edges (simplified: just rewire if target changes)
    edges_a = {e["id"]: e for e in graph_a.get("edges", [])}
    edges_b = {e["id"]: e for e in graph_b.get("edges", [])}
    
    for eid, edge in edges_b.items():
        if eid not in edges_a:
            # New edge
            ops.append(PatchOpModel(
                op="rewire",
                edge_id=eid,
                payload={
                    "source": edge["source"],
                    "source_port": edge["source_port"],
                    "target": edge["target"],
                    "target_port": edge["target_port"]
                }
            ))
            
    return ops
