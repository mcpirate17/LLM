"""
Synthetic Diagnostic Tasks

Four targeted synthetic tasks that test specific architectural capabilities:
  1. Copy — information routing across positions
  2. Induction Heads — in-context pattern matching
  3. Periodic — periodicity detection
  4. Selective Copy — gated/selective information routing

Each task trains a fresh deepcopy of the model from random init on
deterministic data for a small number of steps, then measures accuracy
on positions where the correct answer is known.
"""

from __future__ import annotations

import copy
import gc
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIAG_SEP_TOKEN = 255        # separator token
DIAG_MARK_TOKEN = 254       # marker for selective copy
DIAG_STEPS = 100            # training steps per task
DIAG_BATCH_SIZE = 8
DIAG_SEQ_LEN = 64
DIAG_LR = 1e-3
DIAG_EVAL_BATCHES = 4       # batches for eval pass

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticTaskResult:
    task_name: str
    accuracy: float = 0.0
    loss: float = float("inf")
    steps_trained: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DiagnosticSuiteResult:
    tasks: List[DiagnosticTaskResult] = field(default_factory=list)
    diagnostic_score: float = 0.0
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "tasks": [t.to_dict() for t in self.tasks],
            "diagnostic_score": self.diagnostic_score,
            "total_time_ms": self.total_time_ms,
        }


# ---------------------------------------------------------------------------
# Data generators
#
# Each returns (input_ids, critical_mask, critical_targets):
#   input_ids:       (B, S)     — the training sequence
#   critical_mask:   (B, S-1)   — bool mask over next-token prediction positions
#   critical_targets:(B, S-1)   — same as input_ids[:, 1:] (shifted targets)
# ---------------------------------------------------------------------------

def generate_copy_task(
    batch_size: int = DIAG_BATCH_SIZE,
    seq_len: int = DIAG_SEQ_LEN,
    device: str = "cpu",
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy task: [t1 t2 ... tk SEP t1 t2 ... tk]

    Critical positions: after SEP, where model reproduces source tokens.
    """
    dev = torch.device(device)
    half = (seq_len - 1) // 2  # tokens before SEP
    ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=dev)

    source = torch.randint(
        0, DIAG_MARK_TOKEN, (batch_size, half),
        device=dev, generator=rng,
    )
    ids[:, :half] = source
    ids[:, half] = DIAG_SEP_TOKEN
    copy_len = min(half, seq_len - half - 1)
    ids[:, half + 1 : half + 1 + copy_len] = source[:, :copy_len]

    targets = ids[:, 1:]  # (B, S-1)
    mask = torch.zeros(batch_size, seq_len - 1, dtype=torch.bool, device=device)
    # Critical: positions in the copy region (after SEP)
    # In target-space, position i predicts ids[:, i+1].
    # Copy region in ids starts at half+1, so target positions half..half+copy_len-1
    mask[:, half : half + copy_len] = True

    return ids, mask, targets


def generate_induction_task(
    batch_size: int = DIAG_BATCH_SIZE,
    seq_len: int = DIAG_SEQ_LEN,
    device: str = "cpu",
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Induction heads: sequences with repeated bigrams [... A B ... A _]

    Scatter ~8 bigrams; second occurrence of A should predict B.
    Critical positions: after each repeated first-token.
    """
    dev = torch.device(device)
    n_bigrams = 8
    ids = torch.randint(
        0, DIAG_MARK_TOKEN, (batch_size, seq_len),
        device=dev, generator=rng,
    )
    mask = torch.zeros(batch_size, seq_len - 1, dtype=torch.bool, device=dev)

    # Vectorized: compute positions once, generate all tokens at once
    pos_first = torch.arange(2, 2 + n_bigrams * 2, 2, device=dev)
    pos_second = torch.arange(seq_len // 2, seq_len // 2 + n_bigrams * 2, 2, device=dev)
    valid = (pos_first + 1 < seq_len) & (pos_second + 1 < seq_len)
    n_valid = valid.sum().item()

    if n_valid > 0:
        pos_first = pos_first[valid]
        pos_second = pos_second[valid]
        # Generate all bigram tokens at once: (batch_size, n_valid)
        a_toks = torch.randint(0, DIAG_MARK_TOKEN, (batch_size, n_valid), device=dev, generator=rng)
        b_toks = torch.randint(0, DIAG_MARK_TOKEN, (batch_size, n_valid), device=dev, generator=rng)
        # Scatter into ids using advanced indexing
        batch_idx = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, n_valid)
        ids[batch_idx, pos_first.unsqueeze(0).expand(batch_size, -1)] = a_toks
        ids[batch_idx, (pos_first + 1).unsqueeze(0).expand(batch_size, -1)] = b_toks
        ids[batch_idx, pos_second.unsqueeze(0).expand(batch_size, -1)] = a_toks
        ids[batch_idx, (pos_second + 1).unsqueeze(0).expand(batch_size, -1)] = b_toks
        mask[:, pos_second] = True

    targets = ids[:, 1:]
    return ids, mask, targets


def generate_periodic_task(
    batch_size: int = DIAG_BATCH_SIZE,
    seq_len: int = DIAG_SEQ_LEN,
    device: str = "cpu",
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Periodic: repeating pattern [A B C D A B C D A B ...]

    Period randomly chosen from {3, 4, 5, 6} per sample.
    Critical positions: every position in 2nd+ repetition.
    """
    dev = torch.device(device)
    max_period = 6
    ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=dev)
    mask = torch.zeros(batch_size, seq_len - 1, dtype=torch.bool, device=dev)

    # Generate all periods and patterns at once
    periods = torch.randint(3, 7, (batch_size,), device=dev, generator=rng)
    patterns = torch.randint(0, DIAG_MARK_TOKEN, (batch_size, max_period), device=dev, generator=rng)
    pos_idx = torch.arange(seq_len, device=dev).unsqueeze(0)  # (1, S)

    for b in range(batch_size):
        p = periods[b].item()
        ids[b] = patterns[b, pos_idx[0] % p]
        mask[b, p - 1:] = True

    targets = ids[:, 1:]
    return ids, mask, targets


def generate_selective_copy_task(
    batch_size: int = DIAG_BATCH_SIZE,
    seq_len: int = DIAG_SEQ_LEN,
    device: str = "cpu",
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Selective copy: [MARK t1 x x MARK t2 x x SEP t1 t2]

    Source has some tokens preceded by MARK_TOKEN; after SEP, only
    marked tokens appear in order.
    Critical positions: after SEP.
    """
    dev = torch.device(device)
    n_marks = 4  # number of marked tokens
    source_len = n_marks * 4  # source region length (mark + token + 2 fillers each)
    # Need: source_len + 1 (SEP) + n_marks (copy) <= seq_len
    if source_len + 1 + n_marks > seq_len:
        n_marks = max(2, (seq_len - 1) // 5)
        source_len = n_marks * 4

    ids = torch.randint(
        0, DIAG_MARK_TOKEN, (batch_size, seq_len),
        device=dev, generator=rng,
    )
    mask = torch.zeros(batch_size, seq_len - 1, dtype=torch.bool, device=dev)

    # Vectorized: mark positions are at 0, 4, 8, ... and values at 1, 5, 9, ...
    mark_positions = torch.arange(0, n_marks * 4, 4, device=dev)
    val_positions = mark_positions + 1
    valid = val_positions < source_len
    mark_positions = mark_positions[valid]
    val_positions = val_positions[valid]
    n_valid = mark_positions.shape[0]

    # Generate all marked values at once
    marked_vals = torch.randint(0, DIAG_MARK_TOKEN, (batch_size, n_valid), device=dev, generator=rng)

    # Set mark tokens and values for all batches
    ids[:, mark_positions] = DIAG_MARK_TOKEN
    batch_idx = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, n_valid)
    ids[batch_idx, val_positions.unsqueeze(0).expand(batch_size, -1)] = marked_vals

    # Set SEP and copy region
    sep_pos = source_len
    ids[:, sep_pos] = DIAG_SEP_TOKEN
    copy_positions = sep_pos + 1 + torch.arange(n_valid, device=dev)
    copy_valid = copy_positions < seq_len
    if copy_valid.any():
        cp = copy_positions[copy_valid]
        ids[batch_idx[:, :cp.shape[0]], cp.unsqueeze(0).expand(batch_size, -1)] = marked_vals[:, :cp.shape[0]]

    # Critical: target positions for the copy region
    target_positions = sep_pos + torch.arange(n_valid, device=dev)
    target_valid = target_positions < seq_len - 1
    if target_valid.any():
        mask[:, target_positions[target_valid]] = True

    targets = ids[:, 1:]
    return ids, mask, targets


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

DIAGNOSTIC_TASKS = {
    "copy": generate_copy_task,
    "induction": generate_induction_task,
    "periodic": generate_periodic_task,
    "selective_copy": generate_selective_copy_task,
}

# ---------------------------------------------------------------------------
# Training + eval loop
# ---------------------------------------------------------------------------

def _train_and_eval_task(
    model: nn.Module,
    task_fn,
    task_name: str,
    device: str = "cpu",
    n_steps: int = DIAG_STEPS,
    seed: int = 42,
) -> DiagnosticTaskResult:
    """Train a fresh copy of model on one diagnostic task, then eval."""
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    result = DiagnosticTaskResult(task_name=task_name)

    try:
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        # We assume model is already a fresh copy if we want to avoid deepcopy issues
        # Or we move it to device here
        task_model = model.to(dev)
        task_model.train()

        optimizer = torch.optim.AdamW(
            task_model.parameters(), lr=DIAG_LR, weight_decay=0.01,
        )

        # Determine vocab_size from model's embedding layer
        vocab_size = None
        for m in task_model.modules():
            if isinstance(m, nn.Embedding):
                vocab_size = m.num_embeddings
                break
        if vocab_size is None:
            result.error = "no_embedding_found"
            return result

        # Training
        rng = torch.Generator(device=dev)
        rng.manual_seed(seed)
        for step in range(n_steps):
            input_ids, _, _ = task_fn(
                batch_size=DIAG_BATCH_SIZE, seq_len=DIAG_SEQ_LEN,
                device=str(dev), rng=rng,
            )
            with torch.amp.autocast(
                device_type=dev.type, dtype=torch.bfloat16,
                enabled=(dev.type == "cuda"),
            ):
                logits = task_model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab_size),
                    input_ids[:, 1:].reshape(-1),
                )

            if torch.isnan(loss) or torch.isinf(loss):
                result.error = "nan_or_inf_loss"
                result.steps_trained = step
                return result

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(task_model.parameters(), 1.0)
            optimizer.step()

        result.steps_trained = n_steps

        # Evaluation: measure accuracy on critical positions
        task_model.eval()
        eval_rng = torch.Generator(device=dev)
        eval_rng.manual_seed(seed + 10000)
        total_correct = 0
        total_critical = 0
        total_loss = 0.0

        with torch.no_grad():
            for _ in range(DIAG_EVAL_BATCHES):
                input_ids, crit_mask, crit_targets = task_fn(
                    batch_size=DIAG_BATCH_SIZE, seq_len=DIAG_SEQ_LEN,
                    device=str(dev), rng=eval_rng,
                )
                with torch.amp.autocast(
                    device_type=dev.type, dtype=torch.bfloat16,
                    enabled=(dev.type == "cuda"),
                ):
                    logits = task_model(input_ids)

                # Next-token predictions: logits[:, :-1]
                preds = logits[:, :-1].argmax(dim=-1)  # (B, S-1)
                n_crit = crit_mask.sum().item()
                if n_crit > 0:
                    correct = ((preds == crit_targets) & crit_mask).sum().item()
                    total_correct += correct
                    total_critical += n_crit

                    # Loss on critical positions only
                    flat_logits = logits[:, :-1].reshape(-1, vocab_size)
                    flat_targets = crit_targets.reshape(-1)
                    flat_mask = crit_mask.reshape(-1)
                    if flat_mask.any():
                        crit_loss = F.cross_entropy(
                            flat_logits[flat_mask],
                            flat_targets[flat_mask],
                        )
                        total_loss += crit_loss.item()

        if total_critical > 0:
            result.accuracy = total_correct / total_critical
        if DIAG_EVAL_BATCHES > 0:
            result.loss = total_loss / DIAG_EVAL_BATCHES

    except Exception as e:
        result.error = str(e)[:200]
    finally:
        # Cleanup
        if "task_model" in dir():
            del task_model
        if "optimizer" in dir():
            del optimizer
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return result


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_diagnostic_suite(
    model_or_graph: Any,
    device: str = "cuda",
    n_steps: int = DIAG_STEPS,
    seed: int = 42,
) -> DiagnosticSuiteResult:
    """Run all diagnostic tasks on a model or graph and return aggregated results."""
    from ..synthesis.compiler import compile_graph
    from ..synthesis.serializer import ComputationGraph

    t0 = time.time()
    suite = DiagnosticSuiteResult()

    for task_name, task_fn in DIAGNOSTIC_TASKS.items():
        # Compile a fresh model per task to ensure clean weights and avoid deepcopy issues
        if isinstance(model_or_graph, ComputationGraph):
            model = compile_graph(model_or_graph)
        else:
            model = copy.deepcopy(model_or_graph)

        task_result = _train_and_eval_task(
            model, task_fn, task_name,
            device=device, n_steps=n_steps, seed=seed,
        )
        suite.tasks.append(task_result)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Compute mean accuracy across non-errored tasks
    accs = [t.accuracy for t in suite.tasks if t.error is None]
    suite.diagnostic_score = sum(accs) / len(accs) if accs else 0.0
    suite.total_time_ms = (time.time() - t0) * 1000

    return suite
