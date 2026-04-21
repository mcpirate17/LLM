import logging
from collections import defaultdict

import torch.nn as nn
from .component_catalog import RuntimeComponentCatalog
from .dispatch import KernelDispatcher
from .port_dtypes import find_unsupported_edge_dtype_pairings

logger = logging.getLogger(__name__)


class WorkflowModule(nn.Module):
    def __init__(self, workflow_json, component_catalog):
        super().__init__()
        self.workflow_id = workflow_json["workflow_id"]
        self.name = workflow_json["name"]
        self.nodes_config = {n["id"]: n for n in workflow_json["nodes"]}
        self.edges = workflow_json["edges"]
        self.component_catalog = component_catalog

        # Precompute edge adjacency — O(E) once instead of O(N×E) per forward
        self._edges_by_target = defaultdict(list)
        self._source_set = set()
        for e in self.edges:
            self._edges_by_target[e["target"]].append(e)
            self._source_set.add(e["source"])
        self._sink_nodes = frozenset(self.nodes_config.keys()) - self._source_set

        dtype_issues = find_unsupported_edge_dtype_pairings(
            workflow_json,
            self.component_catalog.get_manifest,
        )
        if dtype_issues:
            first = dtype_issues[0]
            raise ValueError(first["message"])

        # Validate and get topological order
        dispatcher = KernelDispatcher()
        node_ids = list(self.nodes_config.keys())
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        c_edges = [
            (node_to_idx[e["source"]], node_to_idx[e["target"]], 0, 0)
            for e in self.edges
        ]

        res = dispatcher.validate_graph(node_ids, c_edges)
        if not res["valid"]:
            raise ValueError(f"Invalid graph: {res['error']}")

        self.topo_order = [node_ids[idx] for idx in res["topo_order"]]

        # Instantiate submodules
        self.submodules = nn.ModuleDict()
        self.node_handlers = {}
        missing_runtime = []
        for node_id, config in self.nodes_config.items():
            comp_type = config["component_type"]
            params = config["params"]

            handler_class = self.component_catalog.get_handler(comp_type)
            if not handler_class:
                missing_runtime.append(f"{node_id} ({comp_type})")
                continue

            handler = handler_class()
            self.node_handlers[node_id] = handler
            module = handler.build(params)
            if isinstance(module, nn.Module):
                self.submodules[node_id] = module

        if missing_runtime:
            missing_list = ", ".join(missing_runtime)
            raise ValueError(
                "Missing runtime kernel_fallback.py for component(s): "
                f"{missing_list}. Add component fallback kernels or remove these nodes."
            )

    def forward(self, inputs):
        """
        inputs: dict mapping node_id to input tensors for source nodes.
        Returns: dict mapping node_id to output tensors for sink nodes.
        """
        node_outputs = {}

        # Initial inputs for source nodes
        for nid, val in inputs.items():
            node_outputs[nid] = val

        edges_by_target = self._edges_by_target  # local ref avoids repeated attr lookup

        for node_id in self.topo_order:
            # If it's a source node and we already have its output, skip
            if node_id in node_outputs and node_id not in edges_by_target:
                continue

            # Gather inputs from edges — O(degree) not O(E)
            node_inputs = {}
            for e in edges_by_target.get(node_id, ()):
                src_val = node_outputs.get(e["source"])
                port_name = e["target_port"] if e["target_port"] else "x"
                node_inputs[port_name] = src_val

            # Execute node
            config = self.nodes_config[node_id]
            comp_type = config["component_type"]
            params = config["params"]

            if node_id in self.submodules:
                module = self.submodules[node_id]
                if len(node_inputs) > 0:
                    node_outputs[node_id] = self._invoke_node(
                        node_id, module, node_inputs, params
                    )
            else:
                out = self._invoke_handler(node_id, comp_type, node_inputs, params)
                if out is not None:
                    node_outputs[node_id] = out

        return {
            nid: node_outputs[nid] for nid in self._sink_nodes if nid in node_outputs
        }

    def _invoke_node(self, node_id, module, node_inputs, params):
        # Prefer handler.forward() if available — it knows the port names
        handler = self.node_handlers.get(node_id)
        if handler and hasattr(handler, "forward"):
            try:
                result = handler.forward(node_inputs, params)
                if isinstance(result, dict):
                    return result.get(
                        "y", result.get("out", next(iter(result.values())))
                    )
                return result
            except Exception:
                logger.debug(
                    "Native dispatch failed for node %s, using fallback",
                    node_id,
                    exc_info=True,
                )

        if len(node_inputs) == 1:
            key, val = next(iter(node_inputs.items()))
            attempts = (
                lambda: module(val),
                lambda: module(**{key: val}),
            )
        else:
            attempts = (
                lambda: module(**node_inputs),
                lambda: module(*tuple(node_inputs.values())),
            )

        for attempt in attempts:
            try:
                return attempt()
            except TypeError:
                continue

        # Last resort: use handler forward or raise
        out = self._invoke_handler(
            node_id, self.nodes_config[node_id]["component_type"], node_inputs, params
        )
        if out is not None:
            return out

        if len(node_inputs) == 1:
            key, val = next(iter(node_inputs.items()))
            return module(**{key: val})
        return module(**node_inputs)

    def _invoke_handler(self, node_id, comp_type, node_inputs, params):
        handler = self.node_handlers.get(node_id)
        if handler is None:
            handler_class = self.component_catalog.get_handler(comp_type)
            if handler_class:
                handler = handler_class()
                self.node_handlers[node_id] = handler

        if handler is None:
            return None

        out = handler.forward(node_inputs, params)
        if isinstance(out, dict):
            return out.get("y", list(out.values())[0])
        return out


def compile_workflow(workflow_json, components_dir):
    component_catalog = RuntimeComponentCatalog(components_dir)
    return WorkflowModule(workflow_json, component_catalog)
