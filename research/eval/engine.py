"""
Unified Training Engine

Core execution logic for model training and evaluation.
Consolidates loops from evaluator.py and eval/utils.py to ensure
consistent measurement of loss, gradients, and efficiency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class TrainingResult:
    final_loss: float = float("inf")
    initial_loss: float = 0.0
    loss_ratio: float = 1.0
    steps_completed: int = 0
    throughput_tok_s: float = 0.0
    peak_memory_mb: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    passed: bool = False
    error: Optional[str] = None
    loss_curve: List[float] = field(default_factory=list)
    grad_norm_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

def run_micro_train(
    model: nn.Module,
    batches: List[torch.Tensor],
    vocab_size: int,
    n_steps: int = 500,
    lr: float = 3e-4,
    clip_grad: float = 1.0,
    device: torch.device = torch.device("cpu"),
    early_exit_threshold: Optional[float] = None,
    collect_telemetry: bool = True
) -> TrainingResult:
    """Execute a micro-training run with telemetry collection."""
    res = TrainingResult()
    if not batches:
        res.error = "No training batches provided"
        return res

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=(device.type == "cuda"))
    
    total_tokens = 0
    start_time = time.time()
    
    try:
        for step in range(n_steps):
            batch = batches[step % len(batches)]
            total_tokens += batch.numel()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with timing
            t0 = time.time()
            logits = model(batch)
            res.forward_time_ms = (res.forward_time_ms * step + (time.time() - t0) * 1000) / (step + 1)
            
            # Loss calculation (standardized)
            sl = logits[:, :-1].contiguous()
            if sl.shape[-1] > vocab_size:
                sl = sl[..., :vocab_size]
            
            targets = batch[:, 1:].reshape(-1)
            loss = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), targets)
            
            if step == 0:
                res.initial_loss = loss.item()
            
            if not torch.isfinite(loss):
                res.error = f"Non-finite loss at step {step}"
                break
                
            # Backward pass with timing
            t0 = time.time()
            loss.backward()
            res.backward_time_ms = (res.backward_time_ms * step + (time.time() - t0) * 1000) / (step + 1)
            
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
            optimizer.step()
            res.final_loss = loss.item()
            res.steps_completed = step + 1
            
            # Early exit if requested
            if early_exit_threshold and res.final_loss < early_exit_threshold:
                break
                
        elapsed = time.time() - start_time
        res.throughput_tok_s = total_tokens / max(elapsed, 1e-6)
        if device.type == "cuda":
            res.peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            
        res.loss_ratio = res.final_loss / max(res.initial_loss, 1e-6)
        res.passed = res.steps_completed >= (n_steps // 2) # Heuristic pass
        
    except Exception as e:
        res.error = str(e)
        res.passed = False
        
    return res
