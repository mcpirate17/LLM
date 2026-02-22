import torch
import torch.nn as nn
import importlib.util
import os
import yaml
from .dispatch import KernelDispatcher
from .port_dtypes import find_unsupported_edge_dtype_pairings

class WorkflowModule(nn.Module):
    def __init__(self, workflow_json, component_registry):
        super().__init__()
        self.workflow_id = workflow_json['workflow_id']
        self.name = workflow_json['name']
        self.nodes_config = {n['id']: n for n in workflow_json['nodes']}
        self.edges = workflow_json['edges']
        self.component_registry = component_registry

        dtype_issues = find_unsupported_edge_dtype_pairings(
            workflow_json,
            self.component_registry.get_manifest,
        )
        if dtype_issues:
            first = dtype_issues[0]
            raise ValueError(first["message"])

        # Validate and get topological order
        dispatcher = KernelDispatcher()
        node_ids = list(self.nodes_config.keys())
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        c_edges = []
        for e in self.edges:
            # We use indices for the C validator
            c_edges.append((node_to_idx[e['source']], node_to_idx[e['target']], 0, 0))

        res = dispatcher.validate_graph(node_ids, c_edges)
        if not res['valid']:
            raise ValueError(f"Invalid graph: {res['error']}")

        self.topo_order = [node_ids[idx] for idx in res['topo_order']]

        # Instantiate submodules
        self.submodules = nn.ModuleDict()
        self.node_handlers = {}
        for node_id, config in self.nodes_config.items():
            comp_type = config['component_type']
            params = config['params']

            handler_class = self.component_registry.get_handler(comp_type)
            if handler_class:
                handler = handler_class()
                self.node_handlers[node_id] = handler
                module = handler.build(params)
                if isinstance(module, nn.Module):
                    self.submodules[node_id] = module

    def forward(self, inputs):
        """
        inputs: dict mapping node_id to input tensors for source nodes.
        Returns: dict mapping node_id to output tensors for sink nodes.
        """
        node_outputs = {}

        # Initial inputs for source nodes
        for nid, val in inputs.items():
            node_outputs[nid] = val

        for node_id in self.topo_order:
            # If it's a source node and we already have its output, skip
            if node_id in node_outputs and not self._has_incoming_edges(node_id):
                continue

            # Gather inputs from edges
            node_inputs = {}
            for e in self.edges:
                if e['target'] == node_id:
                    src_val = node_outputs.get(e['source'])
                    # If multiple outputs from source, we'd need port handling
                    # For now, simplify.
                    port_name = e['target_port'] if e['target_port'] else 'x'
                    node_inputs[port_name] = src_val

            # Execute node
            config = self.nodes_config[node_id]
            comp_type = config['component_type']
            params = config['params']

            if node_id in self.submodules:
                module = self.submodules[node_id]
                if len(node_inputs) > 0:
                    node_outputs[node_id] = self._invoke_node(node_id, module, node_inputs, params)
            else:
                # Pure functional component fallback or placeholder
                out = self._invoke_handler(node_id, comp_type, node_inputs, params)
                if out is not None:
                    node_outputs[node_id] = out

        # Identify sink nodes (no outgoing edges)
        sinks = self._get_sink_nodes()
        return {nid: node_outputs[nid] for nid in sinks if nid in node_outputs}

    def _has_incoming_edges(self, node_id):
        for e in self.edges:
            if e['target'] == node_id:
                return True
        return False

    def _get_sink_nodes(self):
        targets = {e['target'] for e in self.edges}
        sources = {e['source'] for e in self.edges}
        all_nodes = set(self.nodes_config.keys())
        # Sinks are nodes that are NOT sources for any edge
        return all_nodes - sources

    def _invoke_node(self, node_id, module, node_inputs, params):
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

        out = self._invoke_handler(node_id, self.nodes_config[node_id]['component_type'], node_inputs, params)
        if out is not None:
            return out

        if len(node_inputs) == 1:
            key, val = next(iter(node_inputs.items()))
            return module(**{key: val})
        return module(**node_inputs)

    def _invoke_handler(self, node_id, comp_type, node_inputs, params):
        handler = self.node_handlers.get(node_id)
        if handler is None:
            handler_class = self.component_registry.get_handler(comp_type)
            if handler_class:
                handler = handler_class()
                self.node_handlers[node_id] = handler

        if handler is None:
            return None

        out = handler.forward(node_inputs, params)
        if isinstance(out, dict):
            return out.get('y', list(out.values())[0])
        return out

class ComponentRegistry:
    def __init__(self, components_dir):
        self.components_dir = components_dir
        self.handlers = {}
        self.manifests = {}

    def _resolve_component_dir(self, component_type):
        parts = component_type.split("/")
        if len(parts) == 2:
            cat, cid = parts
            component_dir = os.path.join(self.components_dir, cat, cid)
            if os.path.isdir(component_dir):
                return component_dir
            return None

        cid = component_type
        for cat in os.listdir(self.components_dir):
            category_dir = os.path.join(self.components_dir, cat)
            if not os.path.isdir(category_dir):
                continue
            component_dir = os.path.join(category_dir, cid)
            if os.path.isdir(component_dir):
                return component_dir
        return None

    def get_manifest(self, component_type):
        if component_type in self.manifests:
            return self.manifests[component_type]

        component_dir = self._resolve_component_dir(component_type)
        if component_dir is None:
            return None

        manifest_path = os.path.join(component_dir, "manifest.yaml")
        if not os.path.exists(manifest_path):
            return None

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        self.manifests[component_type] = manifest
        return manifest

    def get_handler(self, component_type):
        if component_type in self.handlers:
            return self.handlers[component_type]

        component_dir = self._resolve_component_dir(component_type)
        if component_dir is None:
            return None

        path = os.path.join(component_dir, "kernel_fallback.py")
        if os.path.exists(path):
            return self._load_handler(component_type, path)

        return None

    def _load_handler(self, component_type, path):
        cid = os.path.basename(os.path.dirname(path))
        spec = importlib.util.spec_from_file_location(f"handler_{cid}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.handlers[component_type] = module.ComponentHandler
        return module.ComponentHandler

def compile_workflow(workflow_json, components_dir):
    registry = ComponentRegistry(components_dir)
    return WorkflowModule(workflow_json, registry)
