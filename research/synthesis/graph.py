"""
Computation Graph DAG Representation

A computation graph is a DAG of OpNodes, where each node applies a primitive
operation to its inputs. The graph takes (B, S, D) tensors as input and
produces (B, S, D) tensors as output.

Shape tracking is built into the graph — every node knows its output shape
at construction time, so invalid graphs are rejected before compilation.
"""

from __future__ import annotations

import xxhash
import logging
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

import numpy as np
from .graph_ir_builder import build_graph_ir, resolve_reachable_node_ids
from .native_analysis import analyze_ir
from .native_topology import compute_topological_order
from .primitives import (
    PrimitiveOp,
    get_primitive,
    PRIMITIVE_REGISTRY,
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


@dataclass(slots=True)
class OpNode:
    """A single node in the computation graph."""

    id: int
    op_name: str
    input_ids: List[int]  # IDs of input nodes (empty for graph inputs)
    output_shape: ShapeInfo = field(default_factory=ShapeInfo)
    # Config for parameterized ops
    config: Dict = field(default_factory=dict)
    # Metadata
    is_input: bool = False
    is_output: bool = False

    @property
    def op(self) -> PrimitiveOp:
        return get_primitive(self.op_name)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "op_name": self.op_name,
            "input_ids": self.input_ids,
            "output_shape": self.output_shape.to_dict(),
            "config": self.config,
            "is_input": self.is_input,
            "is_output": self.is_output,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OpNode:
        return cls(
            id=d["id"],
            op_name=d["op_name"],
            input_ids=d["input_ids"],
            output_shape=ShapeInfo.from_dict(d["output_shape"]),
            config=d.get("config", {}),
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
    def batch_has_gradient_path(ir_list: List[ComputationGraphIR]) -> np.ndarray:
        """Check gradient path for a list of IRs."""
        return np.fromiter((ir.has_gradient_path() for ir in ir_list), dtype=bool)

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
        shape = ShapeInfo(dim=self.model_dim)
        node = OpNode(
            id=node_id,
            op_name="input",
            input_ids=[],
            output_shape=shape,
            is_input=True,
        )
        self.nodes[node_id] = node
        self._input_node_id = node_id
        self._ir_version += 1
        self._cache.clear()
        return node_id

    def add_op(
        self, op_name: str, input_ids: List[int], config: Optional[Dict] = None
    ) -> int:
        """Add an operation node. Returns node ID.

        Raises ValueError if shapes don't compose.
        """
        if op_name not in PRIMITIVE_REGISTRY and op_name != "input":
            raise ValueError(f"Unknown op: {op_name}")

        # Get input shapes
        input_shapes = []
        for iid in input_ids:
            if iid not in self.nodes:
                raise ValueError(f"Input node {iid} doesn't exist")
            input_shapes.append(self.nodes[iid].output_shape)

        # Compute output shape
        op = get_primitive(op_name)
        output_shape = self._compute_shape(op, input_shapes, config or {})

        node_id = self._next_id
        self._next_id += 1
        node = OpNode(
            id=node_id,
            op_name=op_name,
            input_ids=input_ids,
            output_shape=output_shape,
            config=config or {},
        )
        self.nodes[node_id] = node
        self._ir_version += 1
        self._cache.clear()
        return node_id

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
        self._cache.clear()

    def _compute_shape(
        self, op: PrimitiveOp, input_shapes: List[ShapeInfo], config: Dict
    ) -> ShapeInfo:
        """Compute output shape given an op and input shapes."""
        rule = op.shape_rule

        if not input_shapes:
            raise ValueError(f"Op {op.name} requires inputs")

        s0 = input_shapes[0]

        if rule == "identity":
            return ShapeInfo(dim=s0.dim, seq=s0.seq)

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
            return ShapeInfo(dim=max(s0.dim, s1.dim), seq=s0.seq)

        elif rule == "reduce_last":
            return ShapeInfo(dim=1, seq=s0.seq)

        elif rule == "reduce_seq":
            return ShapeInfo(dim=s0.dim, seq="1")

        elif rule == "matmul":
            if len(input_shapes) != 2:
                raise ValueError("Matmul needs 2 inputs")
            s1 = input_shapes[1]
            # Match compiler.py logic:
            # - If dims match, it's an attention-style [B,S,D] @ [B,S,D] -> [B,S,D]
            # - Otherwise, it's a projection [B,S,D] @ [B,D,K] -> [B,S,K]
            if s0.dim == s1.dim:
                return ShapeInfo(dim=s0.dim, seq=s0.seq)
            return ShapeInfo(dim=s1.dim, seq=s0.seq)

        elif rule == "outer":
            # Outer product of features — would be D*D which is too large
            # Instead we produce (B, S, D) by doing outer then project
            return ShapeInfo(dim=s0.dim, seq=s0.seq)

        elif rule == "transpose_seq_dim":
            return ShapeInfo(dim=s0.dim, seq=s0.seq)  # we handle internally

        elif rule == "concat":
            if not input_shapes:
                raise ValueError("Concat needs at least 1 input")
            # Sum dimensions across all inputs
            total_dim = sum(s.dim for s in input_shapes)
            return ShapeInfo(dim=total_dim, seq=s0.seq)

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
                return ShapeInfo(dim=s0.dim // n, seq=s0.seq)
            return ShapeInfo(dim=s0.dim // n, seq=s0.seq)

        elif rule == "linear":
            out_dim = config.get("out_dim", s0.dim)
            return ShapeInfo(dim=out_dim, seq=s0.seq)

        elif rule == "roll":
            return ShapeInfo(dim=s0.dim, seq=s0.seq)

        elif rule == "gather":
            return ShapeInfo(dim=s0.dim, seq=s0.seq)

        elif rule == "scatter":
            return ShapeInfo(dim=s0.dim, seq=s0.seq)

        elif rule == "rfft":
            return ShapeInfo(dim=s0.dim, seq="S//2+1")

        elif rule == "irfft":
            return ShapeInfo(dim=s0.dim, seq="S")

        elif rule in (
            "cumulative",
            "softmax",
            "causal_mask",
            "scale",
            "bias",
            "sort",
            "unsort",
        ):
            return ShapeInfo(dim=s0.dim, seq=s0.seq)

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
        ir = self._analysis_ir()
        analysis = ir.analyze_structure(include_reachable=True)
        mask = analysis.reachable_mask
        node_ids = (
            ir.node_ids
            if ir.node_ids is not None
            else np.arange(ir.n_nodes(), dtype=np.int32)
        )
        visited = {int(node_ids[idx]) for idx in np.flatnonzero(mask)}
        self._cache["reachable"] = visited
        return visited

    def get_dead_nodes(self) -> set:
        """Return set of node IDs that are NOT reachable from the output."""
        return set(self.nodes.keys()) - self.get_reachable_nodes()

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
        result = int(self.lower_to_ir().analyze_structure().depth)
        self._cache["depth"] = result
        return result

    def n_ops(self) -> int:
        """Number of non-input nodes. Cached."""
        if "n_ops" in self._cache:
            return self._cache["n_ops"]
        result = sum(1 for n in self.nodes.values() if not n.is_input)
        self._cache["n_ops"] = result
        return result

    def n_params_estimate(self) -> int:
        """Estimate total learnable parameters. Cached."""
        if "n_params" in self._cache:
            return self._cache["n_params"]
        ir = self.lower_to_ir()
        result = ir.n_params_estimate()
        self._cache["n_params"] = result
        return result

    def has_gradient_path(self) -> bool:
        """Check if there's a differentiable path from input to output. Cached.
        Vectorized via IR lowering for high-throughput architecture filtering.
        """
        if "grad_path" in self._cache:
            return self._cache["grad_path"]
        result = bool(self.lower_to_ir().analyze_structure().has_gradient_path)
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
        id_to_rank = {nid: i for i, nid in enumerate(order)}

        desc = []
        for nid in order:
            node = self.nodes[nid]
            ranks = tuple(str(id_to_rank[iid]) for iid in node.input_ids)
            if node.config:
                config_items = sorted(f"{k}={v}" for k, v in node.config.items())
                config_str = f"[{','.join(config_items)}]"
            else:
                config_str = ""
            desc.append(f"{node.op_name}{config_str}({','.join(ranks)})")

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
        g.metadata = d.get("metadata", {})
        return g

    def lower_to_ir(self) -> ComputationGraphIR:
        """Lower the graph to its compact IR representation. Cached.

        Only reachable nodes (connected to the output) are included;
        dead branches are silently stripped during lowering.
        """
        if "ir" in self._cache:
            return self._cache["ir"]

        ir = build_graph_ir(
            self,
            node_ids=resolve_reachable_node_ids(self),
            ir_cls=ComputationGraphIR,
        )
        self._cache["ir"] = ir
        return ir

    def _analysis_ir(self) -> ComputationGraphIR:
        cached = self._cache.get("analysis_ir")
        if cached is not None:
            return cached

        ir = build_graph_ir(
            self,
            node_ids=sorted(self.nodes.keys()),
            ir_cls=ComputationGraphIR,
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
