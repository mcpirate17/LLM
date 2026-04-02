"""
One-shot pruning utilities.

Provides lightweight offline pruning routines (Wanda-style / SparseGPT-style)
for evaluating quality-retention without retraining.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import measure_loss as _measure_loss_shared


@dataclass(slots=True)
class OneShotPruneResult:
    method: str
    target_sparsity: float
    actual_sparsity: float
    n_params_total: int
    n_params_pruned: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "method": self.method,
            "target_sparsity": self.target_sparsity,
            "actual_sparsity": self.actual_sparsity,
            "n_params_total": self.n_params_total,
            "n_params_pruned": self.n_params_pruned,
        }


def _iter_prunable_params(model: nn.Module) -> Iterable[torch.nn.Parameter]:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:
            continue
        if "embed" in name.lower():
            continue
        yield param


def _score_tensor(weight: torch.Tensor, method: str) -> torch.Tensor:
    w = weight.detach().abs()
    method_norm = str(method or "wanda").strip().lower()
    if method_norm == "sparsegpt":
        return w
    row_scale = torch.sqrt(
        torch.clamp((weight.detach() ** 2).mean(dim=-1, keepdim=True), min=1e-12)
    )
    return w * row_scale


def apply_one_shot_pruning(
    model: nn.Module,
    target_sparsity: float = 0.5,
    method: str = "wanda",
) -> OneShotPruneResult:
    """In-place one-shot pruning over prunable 2D+ parameters."""
    target = float(max(0.0, min(0.95, target_sparsity)))
    total_params = 0
    total_pruned = 0

    for param in _iter_prunable_params(model):
        numel = int(param.numel())
        total_params += numel
        if numel <= 1 or target <= 0.0:
            continue

        score = _score_tensor(param, method=method).reshape(-1)
        k_prune = int(round(numel * target))
        k_prune = max(0, min(k_prune, numel - 1))
        if k_prune <= 0:
            continue

        prune_idx = torch.topk(score, k=k_prune, largest=False).indices
        mask = torch.ones(numel, device=param.device, dtype=param.dtype)
        mask[prune_idx] = 0.0
        mask = mask.reshape_as(param)

        with torch.no_grad():
            param.mul_(mask)

        total_pruned += k_prune

    actual = (total_pruned / total_params) if total_params > 0 else 0.0
    return OneShotPruneResult(
        method=str(method or "wanda").strip().lower() or "wanda",
        target_sparsity=target,
        actual_sparsity=float(actual),
        n_params_total=total_params,
        n_params_pruned=total_pruned,
    )


def estimate_lm_ce_loss(
    model: nn.Module,
    input_batches: List[torch.Tensor],
    device: torch.device,
) -> Optional[float]:
    """Estimate average CE loss on provided token batches without training."""
    return _measure_loss_shared(model, input_batches, device)


def run_dense_vs_structured_sparse_ablation(
    model_dim: int = 128,
    vocab_size: int = 2048,
    seq_len: int = 48,
    batch_size: int = 4,
    steps: int = 16,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare dense vs structured sparse linear ops on speed and training loss."""
    from ..synthesis.graph import ComputationGraph
    from ..synthesis.compiler import compile_model

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(13)

    variants = [
        ("dense", "linear_proj", {"out_dim": model_dim}),
        ("nm_2_4", "nm_sparse_linear", {"out_dim": model_dim, "n": 2, "m": 4}),
        (
            "block_16",
            "block_sparse_linear",
            {"out_dim": model_dim, "block_size": 16, "block_density": 0.25},
        ),
    ]

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
    targets = input_ids.roll(shifts=-1, dims=1)
    rows: List[Dict[str, Any]] = []

    for label, op_name, config in variants:
        graph = ComputationGraph(model_dim)
        i0 = graph.add_input()
        node = graph.add_op(op_name, [i0], config=config)
        graph.set_output(node)

        try:
            model = compile_model(
                [graph], vocab_size=vocab_size, max_seq_len=seq_len
            ).to(dev)
        except Exception as e:
            rows.append(
                {"label": label, "op_name": op_name, "error": str(e), "passed": False}
            )
            continue

        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses: List[float] = []
        t0 = time.perf_counter()
        for _ in range(max(1, int(steps))):
            opt.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                targets[:, :-1].reshape(-1),
            )
            if torch.isnan(loss) or torch.isinf(loss):
                break
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rows.append(
            {
                "label": label,
                "op_name": op_name,
                "passed": len(losses) > 0,
                "steps": len(losses),
                "final_loss": float(losses[-1]) if losses else None,
                "avg_step_ms": float(elapsed_ms / max(len(losses), 1)),
                "total_time_ms": float(elapsed_ms),
            }
        )

    dense = next(
        (r for r in rows if r.get("label") == "dense" and r.get("passed")), None
    )
    if dense:
        dense_loss = float(dense.get("final_loss") or 1.0)
        dense_step_ms = float(dense.get("avg_step_ms") or 1.0)
        for row in rows:
            if not row.get("passed") or row.get("label") == "dense":
                continue
            row["loss_ratio_vs_dense"] = float(
                row.get("final_loss") or dense_loss
            ) / max(dense_loss, 1e-8)
            row["speedup_vs_dense"] = dense_step_ms / max(
                float(row.get("avg_step_ms") or dense_step_ms), 1e-8
            )

    return {
        "device": str(dev),
        "steps": int(steps),
        "rows": rows,
        "dense_row": dense,
    }
