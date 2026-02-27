"""Differentiable DAG search via continuous relaxation of graph topology.

Uses Gumbel-Softmax routing to maintain differentiability while searching
over discrete DAG connections, inspired by DARTS but adapted for
heterogeneous computational graphs.

Reference: ARIA_NEXT_GEN_ARCHITECTURE.md §3
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DifferentiableDAG(nn.Module):
    """Differentiable architecture search over a DAG of heterogeneous ops.

    Each node receives a Gumbel-Softmax-weighted sum of all preceding node
    outputs.  As temperature -> 0 the routing becomes a hard DAG.

    Reference: ARIA_NEXT_GEN_ARCHITECTURE.md §3
    """

    def __init__(self, num_nodes: int, dim: int, operations: nn.ModuleList):
        """
        Args:
            num_nodes: Total nodes including input (node 0) and output (last).
            dim: Feature dimension expected by all operations.
            operations: ModuleList of length num_nodes. ops[0] is identity/unused.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.ops = operations

        # Architecture params: alpha[i, j] = logit weight of edge j -> i
        self.alpha = nn.Parameter(torch.zeros(num_nodes, num_nodes))

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, ...).
            temperature: Gumbel-Softmax temperature. Lower = harder routing.

        Returns:
            Output of the final node.
        """
        node_outputs: List[torch.Tensor] = [x]

        for i in range(1, self.num_nodes):
            # Mask: only allow edges from earlier nodes (DAG constraint)
            logits = self.alpha[i, :i]  # (i,)

            # Gumbel-Softmax for differentiable discrete routing
            weights = F.gumbel_softmax(logits, tau=temperature, hard=False)

            # Weighted aggregation of predecessor outputs
            node_input = torch.zeros_like(node_outputs[0])
            for j, w in enumerate(weights):
                node_input = node_input + w * node_outputs[j]

            # Apply this node's operation
            node_output = self.ops[i](node_input)
            node_outputs.append(node_output)

        return node_outputs[-1]

    def discretize(self, keep_top_k: int = 2) -> List[tuple]:
        """Extract the discrete DAG by selecting top-k incoming edges per node.

        Returns:
            List of (target_node, source_node, weight) tuples.
        """
        edges = []
        with torch.no_grad():
            for i in range(1, self.num_nodes):
                logits = self.alpha[i, :i]
                probs = F.softmax(logits, dim=0)
                k = min(keep_top_k, len(probs))
                topk_vals, topk_idx = torch.topk(probs, k)
                for val, idx in zip(topk_vals, topk_idx):
                    edges.append((i, idx.item(), val.item()))
        return edges


class DARTSSearchLoop:
    """Bi-level optimisation loop for DifferentiableDAG search.

    Alternates between:
    1. Training network weights (ops) on train split.
    2. Updating architecture params (alpha) on val split.

    Reference: ARIA_NEXT_GEN_ARCHITECTURE.md §3
    """

    def __init__(
        self,
        dag: DifferentiableDAG,
        train_loader,
        val_loader,
        weight_lr: float = 1e-3,
        arch_lr: float = 3e-4,
        temperature_schedule: Optional[callable] = None,
    ):
        self.dag = dag
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Separate optimisers for weights vs architecture params
        weight_params = [p for n, p in dag.named_parameters() if n != 'alpha']
        self.weight_opt = torch.optim.Adam(weight_params, lr=weight_lr)
        self.arch_opt = torch.optim.Adam([dag.alpha], lr=arch_lr)

        self.temperature_schedule = temperature_schedule or (lambda step: max(0.1, 1.0 - step * 0.01))

    def step(self, epoch: int, loss_fn) -> dict:
        """Run one epoch of bi-level optimisation.

        Returns:
            Dict with train_loss, val_loss, temperature.
        """
        temp = self.temperature_schedule(epoch)

        # Phase 1: update weights on training data
        self.dag.train()
        train_losses = []
        for xb, yb in self.train_loader:
            self.weight_opt.zero_grad()
            out = self.dag(xb, temperature=temp)
            loss = loss_fn(out, yb)
            loss.backward()
            self.weight_opt.step()
            train_losses.append(loss.item())

        # Phase 2: update architecture params on validation data
        val_losses = []
        for xb, yb in self.val_loader:
            self.arch_opt.zero_grad()
            out = self.dag(xb, temperature=temp)
            loss = loss_fn(out, yb)
            loss.backward()
            self.arch_opt.step()
            val_losses.append(loss.item())

        return {
            'train_loss': sum(train_losses) / max(len(train_losses), 1),
            'val_loss': sum(val_losses) / max(len(val_losses), 1),
            'temperature': temp,
        }
