from .dispatch import KernelDispatcher

class ExecutionPlanner:
    def __init__(self, registry=None):
        self.dispatcher = KernelDispatcher()
        self.registry = registry

    def plan(self, workflow_json):
        """
        Produce an execution plan:
        - Topological order of nodes
        - Resolved implementation for each node
        - Memory schedule (buffer indices for each edge)
        """
        # 1. Validate and get topo order
        nodes_config = {n['id']: n for n in workflow_json['nodes']}
        node_ids = list(nodes_config.keys())
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        edges = workflow_json['edges']

        c_edges = []
        for e in edges:
            c_edges.append((node_to_idx[e['source']], node_to_idx[e['target']], 0, 0))

        res = self.dispatcher.validate_graph(node_ids, c_edges)
        if not res['valid']:
            raise ValueError(f"Invalid graph: {res['error']}")

        topo_order = res['topo_order']

        # 2. Kernel resolution
        implementations = []
        for node_idx in topo_order:
            node_id = node_ids[node_idx]
            config = nodes_config[node_id]
            comp_type = config['component_type']

            # Choice: for now, we just know which ones have C kernels
            # (relu, gelu, silu, add, mul, matmul, linear, rmsnorm)
            has_native = False
            cid = comp_type.split("/")[-1]
            if hasattr(self.dispatcher.lib, f"aria_{cid}_f32"):
                has_native = True

            implementations.append({
                "node_id": node_id,
                "type": comp_type,
                "implementation": "native" if has_native else "fallback"
            })

        # 3. Simple memory schedule (no optimization for now)
        # In a real one, we'd use a liveness analysis to reuse buffers.
        memory_schedule = []
        for e in edges:
            memory_schedule.append({
                "edge_id": e['id'],
                "source": e['source'],
                "target": e['target'],
                "buffer_id": e['id'] # Each edge gets its own buffer for now
            })

        return {
            "topo_order": [node_ids[idx] for idx in topo_order],
            "implementations": implementations,
            "memory_schedule": memory_schedule
        }
