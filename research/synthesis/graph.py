"""
Computation Graph DAG Representation

A computation graph is a DAG of OpNodes, where each node applies a primitive
operation to its inputs. The graph takes (B, S, D) tensors as input and
produces (B, S, D) tensors as output.

Shape tracking is built into the graph — every node knows its output shape
at construction time, so invalid graphs are rejected before compilation.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from .primitives import PrimitiveOp, get_primitive, PRIMITIVE_REGISTRY


@dataclass
class ShapeInfo:
    """Tracked shape through the computation graph.

    We use symbolic shapes: ("B", "S", "D") for standard layer I/O.
    Some ops change dimensions — e.g., split halves D, linear changes D.
    We track the actual numeric D through the graph.
    """
    batch: str = "B"   # always "B"
    seq: str = "S"     # "S" or "S//2+1" (for FFT)
    dim: int = 0       # concrete feature dimension

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
        return self.batch == other.batch and self.seq == other.seq and self.dim == other.dim

    def __hash__(self):
        return hash((self.batch, self.seq, self.dim))

    def to_dict(self) -> dict:
        return {"batch": self.batch, "seq": self.seq, "dim": self.dim}

    @classmethod
    def from_dict(cls, d: dict) -> ShapeInfo:
        return cls(**d)


@dataclass
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
        return node_id

    def add_op(self, op_name: str, input_ids: List[int],
               config: Optional[Dict] = None) -> int:
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

    def _compute_shape(self, op: PrimitiveOp, input_shapes: List[ShapeInfo],
                       config: Dict) -> ShapeInfo:
        """Compute output shape given an op and input shapes."""
        rule = op.shape_rule

        if not input_shapes:
            raise ValueError(f"Op {op.name} requires inputs")

        s0 = input_shapes[0]

        if rule == "identity":
            return ShapeInfo(dim=s0.dim, seq=s0.seq)

        elif rule == "binary_broadcast":
            if len(input_shapes) != 2:
                raise ValueError(f"Binary op {op.name} needs 2 inputs")
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
            # (B, S, D) x (B, D, K) -> (B, S, K)
            # or (B, S, D) x (B, S, D) -> (B, S, D) for batched
            return ShapeInfo(dim=s1.dim, seq=s0.seq)

        elif rule == "outer":
            # Outer product of features — would be D*D which is too large
            # Instead we produce (B, S, D) by doing outer then project
            return ShapeInfo(dim=s0.dim, seq=s0.seq)

        elif rule == "transpose_seq_dim":
            return ShapeInfo(dim=s0.dim, seq=s0.seq)  # we handle internally

        elif rule == "split":
            n = 2 if op.name == "split2" else 3
            if s0.dim % n != 0:
                raise ValueError(f"Can't split dim={s0.dim} into {n} parts")
            return ShapeInfo(dim=s0.dim // n, seq=s0.seq)

        elif rule == "concat":
            if len(input_shapes) != 2:
                raise ValueError("Concat needs 2 inputs")
            s1 = input_shapes[1]
            if s0.seq != s1.seq:
                raise ValueError(f"Concat: seq mismatch {s0.seq} vs {s1.seq}")
            return ShapeInfo(dim=s0.dim + s1.dim, seq=s0.seq)

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

        elif rule in ("cumulative", "softmax", "causal_mask", "scale", "bias", "sort", "unsort"):
            return ShapeInfo(dim=s0.dim, seq=s0.seq)

        else:
            raise ValueError(f"Unknown shape rule: {rule}")

    def topological_order(self) -> List[int]:
        """Return node IDs in topological order (inputs first)."""
        visited = set()
        order = []

        def visit(node_id: int):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes[node_id]
            for inp_id in node.input_ids:
                visit(inp_id)
            order.append(node_id)

        if self._output_node_id is not None:
            visit(self._output_node_id)
        else:
            for nid in self.nodes:
                visit(nid)

        return order

    def depth(self) -> int:
        """Longest path from input to output."""
        if not self.nodes:
            return 0
        depths: Dict[int, int] = {}
        for nid in self.topological_order():
            node = self.nodes[nid]
            if not node.input_ids:
                depths[nid] = 0
            else:
                depths[nid] = max(depths.get(iid, 0) for iid in node.input_ids) + 1
        return max(depths.values()) if depths else 0

    def n_ops(self) -> int:
        """Number of non-input nodes."""
        return sum(1 for n in self.nodes.values() if not n.is_input)

    def n_params_estimate(self) -> int:
        """Estimate total learnable parameters."""
        total = 0
        D = self.model_dim
        for node in self.nodes.values():
            if node.is_input:
                continue
            op = get_primitive(node.op_name)
            if op.has_params:
                formula = op.param_formula.replace("D", str(D))
                try:
                    total += int(eval(formula))
                except Exception:
                    total += D * D  # fallback estimate
        return total

    def has_gradient_path(self) -> bool:
        """Check if there's a differentiable path from input to output."""
        if self._input_node_id is None or self._output_node_id is None:
            return False

        # BFS backwards from output
        reachable = set()
        queue = deque([self._output_node_id])
        while queue:
            nid = queue.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            node = self.nodes[nid]
            for inp_id in node.input_ids:
                queue.append(inp_id)

        return self._input_node_id in reachable

    def fingerprint(self) -> str:
        """Structural fingerprint (hash of the graph topology + ops)."""
        desc = []
        for nid in self.topological_order():
            node = self.nodes[nid]
            desc.append(f"{node.op_name}({','.join(map(str, node.input_ids))})")
        key = "|".join(desc)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

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

    def describe(self) -> str:
        """Human-readable description of the graph."""
        lines = [f"ComputationGraph(dim={self.model_dim}, ops={self.n_ops()}, "
                 f"depth={self.depth()}, params~{self.n_params_estimate()})"]
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
