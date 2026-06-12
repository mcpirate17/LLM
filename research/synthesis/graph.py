"""
Computation Graph DAG Representation

A computation graph is a DAG of OpNodes, where each node applies a primitive
operation to its inputs. The graph takes (B, S, D) tensors as input and
produces (B, S, D) tensors as output.

Shape tracking is built into the graph — every node knows its output shape
at construction time, so invalid graphs are rejected before compilation.
"""

from __future__ import annotations

import copy
import heapq
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import xxhash

logger = logging.getLogger(__name__)

import numpy as np
from .graph_ir_builder import (
    build_graph_ir,
    estimate_reachable_params,
)
from .native_analysis import analyze_ir
from .native_topology import compute_topological_order
from .primitives import (
    OP_NAME_ALIASES,
    PrimitiveOp,
    get_primitive,
    PRIMITIVE_REGISTRY,
)

_JSON_ATOMS = (str, int, float, bool, type(None))


def copy_jsonlike(obj: Any) -> Any:
    """Deep-copy JSON-like data (dict/list/tuple/atoms) without deepcopy overhead.

    graph.metadata snapshots dominated generate_layer_graph profiles (~51%
    via copy.deepcopy's memo/dispatch machinery). Metadata is JSON-shaped, so
    a type-dispatched walk is ~6x faster. Unknown types fall back to a real
    copy.deepcopy so rollback correctness is preserved for any future value.
    """
    cls = obj.__class__
    if cls is dict:
        return {k: copy_jsonlike(v) for k, v in obj.items()}
    if cls is list:
        return [copy_jsonlike(v) for v in obj]
    if cls in _JSON_ATOMS:
        return obj
    if cls is tuple:
        return tuple(copy_jsonlike(v) for v in obj)
    return copy.deepcopy(obj)


_UNCHANGED_UNARY_SHAPE_RULES = frozenset(
    {
        "identity",
        "outer",
        "transpose_seq_dim",
        "roll",
        "gather",
        "scatter",
        "cumulative",
        "softmax",
        "causal_mask",
        "scale",
        "bias",
        "sort",
        "unsort",
    }
)


@dataclass(slots=True)
class ShapeInfo:
    """Tracked shape through the computation graph.

    We use symbolic shapes: ("B", "S", "D") for standard layer I/O.
    Some ops change dimensions — e.g., split halves D, linear changes D.
    We track the actual numeric D through the graph.
    """

    batch: str = "B"  # always "B"
    seq: str = "S"  # "S" or "S//2+1" (for FFT)
    dim: int = 0  # concrete feature dimension

    @property
    def is_standard(self) -> bool:
        """Is this a standard (B, S, D) shape?"""
        return self.seq == "S"

    @property
    def is_freq_domain(self) -> bool:
        return self.seq == "S//2+1"

    def __eq__(self, other):
        if not isinstance(other, ShapeInfo):
            return False
        return (
            self.batch == other.batch
            and self.seq == other.seq
            and self.dim == other.dim
        )

    def __hash__(self):
        return hash((self.batch, self.seq, self.dim))

    def to_dict(self) -> dict:
        return {"batch": self.batch, "seq": self.seq, "dim": self.dim}

    @classmethod
    def from_dict(cls, d: dict) -> ShapeInfo:
        return cls(**d)


_SHAPE_INFO_CACHE: dict[tuple[str, int], ShapeInfo] = {}


def _shape_info(dim: int, seq: str = "S") -> ShapeInfo:
    """Return a shared immutable-by-convention shape for graph construction."""
    key = (seq, int(dim))
    shape = _SHAPE_INFO_CACHE.get(key)
    if shape is None:
        shape = ShapeInfo(seq=seq, dim=int(dim))
        _SHAPE_INFO_CACHE[key] = shape
    return shape


_CONFIG_REPR_CACHE_MAX = 4096
_CONFIG_REPR_CACHE: dict[tuple, str] = {}


def _format_config_items(items: tuple) -> str:
    return f"[{','.join(f'{key}={value}' for key, value in items)}]"


def _canonical_config_repr(config: Dict) -> str:
    if not config:
        return ""
    if len(config) == 1:
        key, value = next(iter(config.items()))
        return f"[{key}={value}]"
    try:
        cache_key = tuple(config.items())
        cached = _CONFIG_REPR_CACHE.get(cache_key)
        if cached is not None:
            return cached
    except TypeError:
        sorted_items = tuple(
            item
            for _, item in sorted(
                (f"{key}={value}", (key, value)) for key, value in config.items()
            )
        )
        return _format_config_items(sorted_items)

    sorted_items = tuple(
        item
        for _, item in sorted(
            (f"{key}={value}", (key, value)) for key, value in cache_key
        )
    )
    result = _format_config_items(sorted_items)
    if len(_CONFIG_REPR_CACHE) >= _CONFIG_REPR_CACHE_MAX:
        _CONFIG_REPR_CACHE.clear()
    _CONFIG_REPR_CACHE[cache_key] = result
    return result


@dataclass(slots=True)
class OpNode:
    """A single node in the computation graph."""

    id: int
    op_name: str
    input_ids: List[int]  # IDs of input nodes (empty for graph inputs)
    output_shape: ShapeInfo = field(default_factory=ShapeInfo)
    depth: int = 0
    # Config for parameterized ops
    config: Dict = field(default_factory=dict)
    # Metadata
    is_input: bool = False
    is_output: bool = False
    _config_repr: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._config_repr is None:
            self._config_repr = (
                _canonical_config_repr(self.config) if self.config else ""
            )

    @property
    def op(self) -> PrimitiveOp:
        return get_primitive(self.op_name)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "op_name": self.op_name,
            "input_ids": self.input_ids,
            "output_shape": self.output_shape.to_dict(),
            "depth": self.depth,
            "config": self.config,
            "is_input": self.is_input,
            "is_output": self.is_output,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OpNode:
        return cls(
            id=d["id"],
            op_name=d["op_name"],
            input_ids=list(d["input_ids"]),
            output_shape=ShapeInfo.from_dict(d["output_shape"]),
            depth=int(d.get("depth", 0)),
            config=dict(d.get("config", {})),
            is_input=d.get("is_input", False),
            is_output=d.get("is_output", False),
        )


@dataclass(slots=True)
class ComputationGraphIR:
    """Memory-contiguous representation of a computation graph.
    Designed for fast structural analysis and JIT-compiled execution.
    """

    model_dim: int
    op_codes: np.ndarray  # int32, shape (N,)
    # input_indices[i, j] is the index of the j-th input to node i
    # -1 indicates no input (for padded slots)
    input_indices: np.ndarray  # int32, shape (N, 2)
    output_node_idx: int
    configs: List[Dict]
    node_ids: Optional[np.ndarray] = None
    param_estimates: Optional[np.ndarray] = None
    node_dims: Optional[np.ndarray] = None
    node_seq_flags: Optional[np.ndarray] = None
    node_ids_are_contiguous: bool = False
    source_version: int = 0  # _ir_version of source ComputationGraph at construction
    analysis_cache: Dict[str, object] = field(default_factory=dict, repr=False)

    def is_stale(self, graph: "ComputationGraph") -> bool:
        """Check if this IR was built from an older version of the graph."""
        return self.source_version != graph._ir_version

    def n_nodes(self) -> int:
        return len(self.op_codes)

    def analyze_structure(self, include_reachable: bool = False):
        cache_key = "with_reachable" if include_reachable else "summary"
        cached = self.analysis_cache.get(cache_key)
        if cached is not None:
            return cached
        result = analyze_ir(self, include_reachable=include_reachable)
        self.analysis_cache[cache_key] = result
        return result

    def has_gradient_path(self) -> bool:
        """Check if there's a differentiable path from input to output.
        Uses sparse reverse traversal over input_indices.
        """
        return bool(self.analyze_structure().has_gradient_path)

    @staticmethod
    def batch_op_distribution(
        ir_list: List[ComputationGraphIR], n_opcodes: int
    ) -> np.ndarray:
        """Compute opcode counts for a batch of IRs.
        Returns array of shape (batch_size, n_opcodes).
        """
        batch_size = len(ir_list)
        counts = np.zeros((batch_size, n_opcodes), dtype=np.int32)
        for i, ir in enumerate(ir_list):
            op_codes = ir.op_codes
            non_input = op_codes[op_codes != 0]
            if len(non_input) > 0:
                counts[i] = np.bincount(non_input, minlength=n_opcodes)
        return counts

    def n_params_estimate(self) -> int:
        """Estimate total learnable parameters using the IR. Cached."""
        return int(self.analyze_structure().param_estimate)


class ComputationGraph:
    """A DAG of primitive operations representing a single layer.

    Input: (B, S, D) tensor
    Output: (B, S, D) tensor

    The graph tracks shapes through every node, ensuring validity.
    """

    def __init__(self, model_dim: int):
        self.model_dim = model_dim
        self.nodes: Dict[int, OpNode] = {}
        self._next_id = 0
        self._input_node_id: Optional[int] = None
        self._output_node_id: Optional[int] = None
        self.metadata: Dict = {}
        self._cache: Dict = {}  # lazily computed properties
        self._ir_version: int = 0  # incremented on every structural mutation

    def copy(self) -> "ComputationGraph":
        """Create a structural copy without generic deepcopy overhead."""
        clone = ComputationGraph(self.model_dim)
        clone.nodes = {
            node_id: replace(
                node,
                input_ids=list(node.input_ids),
                output_shape=replace(node.output_shape),
                config=dict(node.config),
            )
            for node_id, node in self.nodes.items()
        }
        clone._next_id = self._next_id
        clone._input_node_id = self._input_node_id
        clone._output_node_id = self._output_node_id
        clone.metadata = dict(self.metadata)
        clone._ir_version = self._ir_version
        return clone

    @property
    def input_node(self) -> Optional[OpNode]:
        if self._input_node_id is not None:
            return self.nodes.get(self._input_node_id)
        return None

    @property
    def output_node(self) -> Optional[OpNode]:
        if self._output_node_id is not None:
            return self.nodes.get(self._output_node_id)
        return None

    def add_input(self) -> int:
        """Add the input node. Returns node ID."""
        node_id = self._next_id
        self._next_id += 1
        shape = _shape_info(self.model_dim)
        node = OpNode(
            id=node_id,
            op_name="input",
            input_ids=[],
            output_shape=shape,
            depth=0,
            is_input=True,
        )
        self.nodes[node_id] = node
        self._input_node_id = node_id
        self._ir_version += 1
        if self._cache:
            self._cache.clear()
        return node_id

    def add_op(
        self, op_name: str, input_ids: List[int], config: Optional[Dict] = None
    ) -> int:
        """Add an operation node. Returns node ID.

        Raises ValueError if shapes don't compose.
        """
        alias = OP_NAME_ALIASES.get(op_name)
        canonical_name = op_name if alias is None else alias
        op = PRIMITIVE_REGISTRY.get(canonical_name)
        if op is None:
            if canonical_name != "input":
                raise ValueError(f"Unknown op: {op_name}")

        config_dict = config if config is not None else {}
        input_count = len(input_ids)
        if input_count == 1:
            input_node = self.nodes.get(input_ids[0])
            if input_node is None:
                raise ValueError(f"Input node {input_ids[0]} doesn't exist")
            output_shape = self._compute_shape_fast(
                op,
                input_node.output_shape,
                None,
                input_count,
                config_dict,
            )
        elif input_count == 2:
            left_node = self.nodes.get(input_ids[0])
            if left_node is None:
                raise ValueError(f"Input node {input_ids[0]} doesn't exist")
            right_node = self.nodes.get(input_ids[1])
            if right_node is None:
                raise ValueError(f"Input node {input_ids[1]} doesn't exist")
            output_shape = self._compute_shape_fast(
                op,
                left_node.output_shape,
                right_node.output_shape,
                input_count,
                config_dict,
            )
        else:
            input_shapes = []
            for iid in input_ids:
                input_node = self.nodes.get(iid)
                if input_node is None:
                    raise ValueError(f"Input node {iid} doesn't exist")
                input_shapes.append(input_node.output_shape)
            output_shape = self._compute_shape(op, input_shapes, config_dict)

        if input_count == 1:
            depth = input_node.depth + 1
        elif input_count == 2:
            left_depth = left_node.depth
            right_depth = right_node.depth
            depth = (left_depth if left_depth >= right_depth else right_depth) + 1
        else:
            depth = 1 + max((self.nodes[iid].depth for iid in input_ids), default=0)

        node_id = self._next_id
        self._next_id += 1
        node = OpNode(
            id=node_id,
            op_name=canonical_name,
            input_ids=input_ids,
            output_shape=output_shape,
            depth=depth,
            config=config_dict,
            _config_repr=_canonical_config_repr(config_dict) if config_dict else "",
        )
        self.nodes[node_id] = node
        self._ir_version += 1
        if self._cache:
            self._cache.clear()
        return node_id

    def _compute_shape_fast(
        self,
        op: PrimitiveOp,
        s0: ShapeInfo,
        s1: ShapeInfo | None,
        input_count: int,
        config: Dict,
    ) -> ShapeInfo:
        """Compute output shape for the common one- and two-input add_op paths."""
        rule = op.shape_rule

        if rule in _UNCHANGED_UNARY_SHAPE_RULES:
            return s0

        if rule == "binary_broadcast":
            if input_count != 2 or s1 is None:
                raise ValueError(f"Binary op requires 2 inputs, got {input_count}")
            if s0.dim != s1.dim and s0.dim != 1 and s1.dim != 1:
                raise ValueError(
                    f"Binary op {op.name}: incompatible dims {s0.dim} vs {s1.dim}"
                )
            if s0.seq != s1.seq:
                raise ValueError(
                    f"Binary op {op.name}: incompatible seq {s0.seq} vs {s1.seq}"
                )
            return s0 if s0.dim >= s1.dim else s1

        if rule == "linear":
            out_dim = config.get("out_dim", s0.dim)
            return s0 if out_dim == s0.dim else _shape_info(out_dim, s0.seq)

        if rule == "reduce_last":
            return _shape_info(1, s0.seq)

        if rule == "reduce_seq":
            return _shape_info(s0.dim, "1")

        if rule == "matmul":
            if input_count != 2 or s1 is None:
                raise ValueError("Matmul needs 2 inputs")
            return s0 if s0.dim == s1.dim else _shape_info(s1.dim, s0.seq)

        if rule == "split":
            if op.name == "split2":
                n = 2
            elif op.name == "split3":
                n = 3
            elif op.name == "split4":
                n = 4
            else:
                n = int(config.get("n_splits", 2))
            return _shape_info(s0.dim // n, s0.seq)

        if rule == "rfft":
            return _shape_info(s0.dim, "S//2+1")

        if rule == "irfft":
            return _shape_info(s0.dim, "S")

        if rule == "concat":
            if input_count == 2 and s1 is not None:
                return _shape_info(s0.dim + s1.dim, s0.seq)
            return _shape_info(s0.dim, s0.seq)

        raise ValueError(f"Unknown shape rule: {rule}")

    def set_output(self, node_id: int) -> None:
        """Mark a node as the graph output."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} doesn't exist")
        node = self.nodes[node_id]
        # Output must be (B, S, model_dim)
        if node.output_shape.dim != self.model_dim:
            raise ValueError(
                f"Output node has dim={node.output_shape.dim}, "
                f"but model_dim={self.model_dim}"
            )
        if not node.output_shape.is_standard:
            raise ValueError(
                f"Output node has seq={node.output_shape.seq}, must be 'S'"
            )
        node.is_output = True
        self._output_node_id = node_id
        self._ir_version += 1
        if self._cache:
            self._cache.clear()

    def _compute_shape(
        self, op: PrimitiveOp, input_shapes: List[ShapeInfo], config: Dict
    ) -> ShapeInfo:
        # guardrail: allow-god-function
        """Compute output shape given an op and input shapes."""
        rule = op.shape_rule

        if not input_shapes:
            raise ValueError(f"Op {op.name} requires inputs")

        s0 = input_shapes[0]

        if rule == "identity":
            return _shape_info(s0.dim, s0.seq)

        elif rule == "binary_broadcast":
            if len(input_shapes) != 2:
                raise ValueError(
                    f"Binary op requires 2 inputs, got {len(input_shapes)}"
                )
            s1 = input_shapes[1]
            # Dims must match or one must be 1
            if s0.dim != s1.dim and s0.dim != 1 and s1.dim != 1:
                raise ValueError(
                    f"Binary op {op.name}: incompatible dims {s0.dim} vs {s1.dim}"
                )
            if s0.seq != s1.seq:
                raise ValueError(
                    f"Binary op {op.name}: incompatible seq {s0.seq} vs {s1.seq}"
                )
            return _shape_info(max(s0.dim, s1.dim), s0.seq)

        elif rule == "reduce_last":
            return _shape_info(1, s0.seq)

        elif rule == "reduce_seq":
            return _shape_info(s0.dim, "1")

        elif rule == "matmul":
            if len(input_shapes) != 2:
                raise ValueError("Matmul needs 2 inputs")
            s1 = input_shapes[1]
            # Match compiler.py logic:
            # - If dims match, it's an attention-style [B,S,D] @ [B,S,D] -> [B,S,D]
            # - Otherwise, it's a projection [B,S,D] @ [B,D,K] -> [B,S,K]
            if s0.dim == s1.dim:
                return _shape_info(s0.dim, s0.seq)
            return _shape_info(s1.dim, s0.seq)

        elif rule == "outer":
            # Outer product of features — would be D*D which is too large
            # Instead we produce (B, S, D) by doing outer then project
            return _shape_info(s0.dim, s0.seq)

        elif rule == "transpose_seq_dim":
            return _shape_info(s0.dim, s0.seq)  # we handle internally

        elif rule == "concat":
            if not input_shapes:
                raise ValueError("Concat needs at least 1 input")
            # Sum dimensions across all inputs
            total_dim = sum(s.dim for s in input_shapes)
            return _shape_info(total_dim, s0.seq)

        elif rule == "split":
            # Determine split divisor from op name or config
            if op.name == "split2":
                n = 2
            elif op.name == "split3":
                n = 3
            elif op.name == "split4":
                n = 4
            else:
                n = int(config.get("n_splits", 2))

            if s0.dim % n != 0:
                # Fallback: if not perfectly divisible, last split gets remainder
                # but for simplicity in research, we often just want floor
                return _shape_info(s0.dim // n, s0.seq)
            return _shape_info(s0.dim // n, s0.seq)

        elif rule == "linear":
            out_dim = config.get("out_dim", s0.dim)
            return _shape_info(out_dim, s0.seq)

        elif rule == "roll":
            return _shape_info(s0.dim, s0.seq)

        elif rule == "gather":
            return _shape_info(s0.dim, s0.seq)

        elif rule == "scatter":
            return _shape_info(s0.dim, s0.seq)

        elif rule == "rfft":
            return _shape_info(s0.dim, "S//2+1")

        elif rule == "irfft":
            return _shape_info(s0.dim, "S")

        elif rule in (
            "cumulative",
            "softmax",
            "causal_mask",
            "scale",
            "bias",
            "sort",
            "unsort",
        ):
            return _shape_info(s0.dim, s0.seq)

        else:
            raise ValueError(f"Unknown shape rule: {rule}")

    def topological_order(self) -> List[int]:
        """Return node IDs in a canonical topological order (inputs first). Cached.

        Uses a stable sort (Kahn's algorithm) with tie-breakers to ensure that
        structurally identical graphs always produce the same order regardless
        of original node ID assignments.
        """
        if "topo" in self._cache:
            return self._cache["topo"]
        order = compute_topological_order(self)
        self._cache["topo"] = order
        return order

    def get_reachable_nodes(self) -> set:
        """Return set of node IDs reachable from the output via backward BFS.

        A node is "reachable" if it lies on any path from the output node
        back to the input node.  Nodes not in this set are dead branches.
        """
        if "reachable" in self._cache:
            return self._cache["reachable"]
        if self._output_node_id is None:
            self._cache["reachable"] = set()
            return set()
        visited: set[int] = set()
        stack = [self._output_node_id]
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            node = self.nodes.get(node_id)
            if node is None:
                continue
            visited.add(node_id)
            stack.extend(node.input_ids)
        self._cache["reachable"] = visited
        return visited

    def get_dead_nodes(self) -> set:
        """Return set of node IDs that are NOT reachable from the output."""
        reachable = self.get_reachable_nodes()
        if len(reachable) == len(self.nodes):
            return set()
        return set(self.nodes.keys()) - reachable

    def children_map(self) -> Dict[int, List[int]]:
        """Return parent→children adjacency. Cached until graph mutation."""
        cached = self._cache.get("children")
        if cached is not None:
            return cached
        children: Dict[int, List[int]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for parent_id in node.input_ids:
                if parent_id in children:
                    children[parent_id].append(nid)
        self._cache["children"] = children
        return children

    def prune_unreachable_nodes(self) -> int:
        """Remove dead-branch nodes not connected to the output.

        Returns the number of nodes removed.
        """
        dead = self.get_dead_nodes()
        if not dead:
            return 0
        for nid in dead:
            del self.nodes[nid]
        self._ir_version += 1
        self._cache.clear()
        if self._input_node_id in dead:
            self._input_node_id = None
        return len(dead)

    def depth(self) -> int:
        """Longest path from input to output. Cached."""
        if "depth" in self._cache:
            return self._cache["depth"]
        if not self.nodes:
            self._cache["depth"] = 0
            return 0
        output_node = self.output_node
        result = int(output_node.depth) if output_node is not None else 0
        self._cache["depth"] = result
        return result

    def n_ops(self) -> int:
        """Number of non-input nodes.

        Generation calls this inside budget checks while the graph is mutating,
        so a cache is invalidated almost every time. There is only one input
        node by construction; compute the count directly instead of scanning.
        """
        input_adjustment = (
            1
            if self._input_node_id is not None and self._input_node_id in self.nodes
            else 0
        )
        return len(self.nodes) - input_adjustment

    def n_params_estimate(self) -> int:
        """Estimate total learnable parameters. Cached."""
        if "n_params" in self._cache:
            return self._cache["n_params"]
        result = estimate_reachable_params(self, self.get_reachable_nodes())
        self._cache["n_params"] = result
        return result

    def has_gradient_path(self) -> bool:
        """Check if there's a differentiable path from input to output. Cached.
        Vectorized via IR lowering for high-throughput architecture filtering.
        """
        if "grad_path" in self._cache:
            return self._cache["grad_path"]
        result = bool(self._analysis_ir().analyze_structure().has_gradient_path)
        self._cache["grad_path"] = result
        return result

    def fingerprint(self) -> str:
        """Structural fingerprint (hash of the graph topology + ops). Cached.

        Canonical representation: abstracts away specific node IDs to ensure
        identical architectures always produce the same hash regardless of
        generation order or ID assignment.
        """
        if "fingerprint" in self._cache:
            return self._cache["fingerprint"]

        # Use a stable topological order (Kahn's or similar)
        # and replace node IDs with their rank in that order.
        order = self.topological_order()
        topology_inputs = self._cache.get("canonical_topology_inputs")
        if topology_inputs is not None:
            _, _, op_names, config_strs, node_inputs = topology_inputs
        else:
            op_names = config_strs = node_inputs = None

        if len(order) == self._next_id:
            rank_strs = [""] * self._next_id
            for rank, nid in enumerate(order):
                rank_strs[nid] = str(rank)
            id_to_rank = None
        else:
            id_to_rank = {nid: i for i, nid in enumerate(order)}
            rank_strs = None

        desc = []
        desc_append = desc.append
        for nid in order:
            if op_names is not None and nid < len(op_names):
                if rank_strs is not None:
                    ranks = ",".join(rank_strs[iid] for iid in node_inputs[nid])
                else:
                    ranks = ",".join(str(id_to_rank[iid]) for iid in node_inputs[nid])
                desc_append(f"{op_names[nid]}{config_strs[nid]}({ranks})")
            else:
                node = self.nodes[nid]
                if rank_strs is not None:
                    ranks = ",".join(rank_strs[iid] for iid in node.input_ids)
                else:
                    ranks = ",".join(str(id_to_rank[iid]) for iid in node.input_ids)
                desc_append(f"{node.op_name}{node._config_repr}({ranks})")

        # Include model_dim in fingerprint
        # Z13: Include routing/compression policy in fingerprint
        rc_str = ""
        rc = self.metadata.get("routing_compression")
        if rc:
            r_kind = rc.get("routing", {}).get("kind", "unknown")
            c_kind = rc.get("compression", {}).get("kind", "unknown")
            rc_str = f"|rc={r_kind}:{c_kind}"

        key = f"dim={self.model_dim}{rc_str}|" + "|".join(desc)
        result = xxhash.xxh64(key.encode()).hexdigest()
        self._cache["fingerprint"] = result
        return result

    def to_dict(self) -> dict:
        return {
            "model_dim": self.model_dim,
            "nodes": {str(k): v.to_dict() for k, v in self.nodes.items()},
            "input_node_id": self._input_node_id,
            "output_node_id": self._output_node_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ComputationGraph:
        g = cls(d["model_dim"])
        for k, v in d["nodes"].items():
            node = OpNode.from_dict(v)
            g.nodes[node.id] = node
        g._next_id = max(g.nodes.keys()) + 1 if g.nodes else 0
        g._input_node_id = d.get("input_node_id")
        g._output_node_id = d.get("output_node_id")
        g.metadata = dict(d.get("metadata", {}))
        g._refresh_node_depths()
        return g

    def _refresh_node_depths(self) -> None:
        depths: Dict[int, int] = {}
        for nid in compute_topological_order(self):
            node = self.nodes[nid]
            if node.is_input:
                node.depth = 0
            else:
                node.depth = 1 + max(
                    (depths.get(input_id, 0) for input_id in node.input_ids),
                    default=0,
                )
            depths[nid] = node.depth

    def lower_to_ir(self) -> ComputationGraphIR:
        """Lower the graph to its compact IR representation. Cached.

        Only reachable nodes (connected to the output) are included;
        dead branches are silently stripped during lowering.
        """
        if "ir" in self._cache:
            return self._cache["ir"]

        reachable = self.get_reachable_nodes()
        contiguous_all_nodes = len(self.nodes) == self._next_id
        all_nodes_reachable = len(reachable) == len(self.nodes)
        if not reachable:
            node_ids = []
            assume_contiguous_ids = False
        elif contiguous_all_nodes and all_nodes_reachable:
            node_ids = range(len(self.nodes))
            assume_contiguous_ids = True
        else:
            node_ids = self._topological_order_for_nodes(reachable)
            assume_contiguous_ids = False

        ir = build_graph_ir(
            self,
            node_ids=node_ids,
            ir_cls=ComputationGraphIR,
            assume_contiguous_ids=assume_contiguous_ids,
        )
        self._cache["ir"] = ir
        return ir

    def _topological_order_for_nodes(self, node_ids: set[int]) -> List[int]:
        """Return deterministic topological order for an induced node subset."""
        in_degree = {
            nid: sum(
                1 for input_id in self.nodes[nid].input_ids if input_id in node_ids
            )
            for nid in node_ids
        }
        children: Dict[int, List[int]] = {nid: [] for nid in node_ids}
        for nid in node_ids:
            for input_id in self.nodes[nid].input_ids:
                if input_id in node_ids:
                    children[input_id].append(nid)

        order: List[int] = []
        canonical_id_map: Dict[int, int] = {}
        ready: list[tuple[str, tuple[int, ...], str, int]] = []
        for nid, degree in in_degree.items():
            if degree == 0:
                node = self.nodes[nid]
                heapq.heappush(ready, (node.op_name, (), node._config_repr, nid))

        while ready:
            _, _, _, nid = heapq.heappop(ready)
            canonical_id_map[nid] = len(order)
            order.append(nid)
            for child_id in children[nid]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    child = self.nodes[child_id]
                    input_keys = tuple(
                        canonical_id_map[input_id]
                        for input_id in child.input_ids
                        if input_id in node_ids
                    )
                    heapq.heappush(
                        ready, (child.op_name, input_keys, child._config_repr, child_id)
                    )

        if len(order) < len(node_ids):
            return sorted(node_ids)
        return order

    def _analysis_ir(self) -> ComputationGraphIR:
        cached = self._cache.get("analysis_ir")
        if cached is not None:
            return cached

        contiguous_ids = len(self.nodes) == self._next_id
        ir = build_graph_ir(
            self,
            node_ids=range(len(self.nodes)) if contiguous_ids else self.nodes.keys(),
            ir_cls=ComputationGraphIR,
            assume_contiguous_ids=contiguous_ids,
        )
        self._cache["analysis_ir"] = ir
        return ir

    def describe(self) -> str:
        """Human-readable description of the graph."""
        lines = [
            f"ComputationGraph(dim={self.model_dim}, ops={self.n_ops()}, "
            f"depth={self.depth()}, params~{self.n_params_estimate()})"
        ]
        for nid in self.topological_order():
            node = self.nodes[nid]
            inputs = ", ".join(f"n{i}" for i in node.input_ids)
            shape = f"({node.output_shape.seq},{node.output_shape.dim})"
            prefix = ""
            if node.is_input:
                prefix = "[INPUT] "
            elif node.is_output:
                prefix = "[OUTPUT] "
            lines.append(f"  n{nid}: {prefix}{node.op_name}({inputs}) -> {shape}")
        return "\n".join(lines)

    def has_residual_path(self) -> bool:
        """Detect if there is a direct residual (add) path from input to output. Cached."""
        if "has_residual" in self._cache:
            return self._cache["has_residual"]

        input_ids = [nid for nid, node in self.nodes.items() if node.is_input]
        if not input_ids or self._output_node_id is None:
            return False

        main_input = input_ids[0]

        # Heuristic: is the input ID a direct input to ANY 'add' node?
        # Most residuals in this project use 'add' nodes for the skip connection.
        res = False
        for node in self.nodes.values():
            if node.op_name == "add" and main_input in node.input_ids:
                res = True
                break

        self._cache["has_residual"] = res
        return res
