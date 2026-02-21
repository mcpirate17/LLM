import random
import json
from uuid import uuid4
from typing import Dict, Any, List
from .models import AriaPatchProposalModel, PatchOpModel
from .database import get_workflow, save_proposal, list_components

def refine_winner(workflow_id: str, num_variations: int = 3) -> List[str]:
    """
    Generate variations of a workflow using evolutionary mutations.
    Returns list of proposal IDs created.
    """
    workflow = get_workflow(workflow_id)
    if not workflow:
        raise ValueError("Workflow not found")
        
    graph = json.loads(workflow["graph_json"])
    
    proposal_ids = []
    attempts = 0
    max_attempts = max(num_variations * 8, 8)

    while len(proposal_ids) < num_variations and attempts < max_attempts:
        attempts += 1
        # Pick a mutation strategy
        mutation_type = random.choice(["mutate_param", "replace_activation", "add_layer"])
        ops = []
        rationale = ""
        
        if mutation_type == "mutate_param":
            # Find a param to mutate
            candidates = []
            for node in graph["nodes"]:
                if node.get("params"):
                    for k, v in node["params"].items():
                        if isinstance(v, (int, float)):
                            candidates.append((node["id"], k, v))
            
            if candidates:
                nid, param, val = random.choice(candidates)
                # Mutate by +/- 10-50%
                new_val = val * random.uniform(0.5, 1.5)
                if isinstance(val, int):
                    new_val = int(new_val)
                
                ops.append(PatchOpModel(
                    op="mutate_param",
                    node_id=nid,
                    payload={"param_name": param, "value": new_val}
                ))
                rationale = f"Evolution: Mutated {param} in {nid} from {val} to {new_val}"
                
        elif mutation_type == "replace_activation":
            # Find activation nodes
            candidates = [n for n in graph["nodes"] if n["component_type"] in ["math/relu", "math/gelu", "math/silu"]]
            if candidates:
                node = random.choice(candidates)
                current = node["component_type"]
                options = ["math/relu", "math/gelu", "math/silu"]
                options.remove(current)
                if options:
                    new_type = random.choice(options)
                    ops.append(PatchOpModel(
                        op="replace_node",
                        node_id=node["id"],
                        payload={"new_component_id": new_type}
                    ))
                    rationale = f"Evolution: Swapped {current} for {new_type} in {node['id']}"

        # If no ops generated (e.g. no params), fallback or skip
        if not ops:
            continue
            
        # Create proposal
        proposal_id = f"evo_{uuid4().hex[:10]}"
        patch = AriaPatchProposalModel(
            workflow_id=workflow_id,
            base_version=workflow["version"],
            rationale=rationale,
            ops=ops
        )
        
        save_proposal(
            proposal_id=proposal_id,
            workflow_id=workflow_id,
            patch_json=json.dumps(patch.model_dump()),
            rationale=rationale,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        proposal_ids.append(proposal_id)
        
    return proposal_ids

from datetime import datetime, timezone
