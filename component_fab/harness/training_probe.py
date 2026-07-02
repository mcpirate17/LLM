"""One token-task training/eval core for the harness probes.

Absorbs the near-identical train/eval scaffolds that used to live in
``harder_binding_tasks`` (single + checkpointed runs), ``binding_validity``
and ``adversarial_retention``, plus the continuous-lane twin used by the
nano probes (``nano_bind_probe``, ``nano_induction_probe``,
``range_binding_probe``, ``capability_probes``).

Failure policy (the ONE policy, from ``binding_tests_v2``): a failed probe
is logged loudly (``logger.warning`` with traceback) and scored 0 /
``converged=False`` — never silently swallowed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TypeVar

import torch
from torch import nn

from .tiny_lm import TinyLM, TinyLMConfig

logger = logging.getLogger(__name__)

# (input_ids[B, L], query_positions[B, Q], target_ids[B, Q])
TokenBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
TokenBatchFn = Callable[[torch.Generator], TokenBatch]


def _tensor_finite_summary(name: str, tensor: torch.Tensor) -> str:
    """Compact diagnostics for non-finite tensors without dumping payloads."""
    with torch.no_grad():
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        n_total = int(detached.numel())
        n_finite = int(finite.sum().item()) if n_total else 0
        pieces = [
            f"{name}: shape={tuple(detached.shape)}",
            f"dtype={detached.dtype}",
            f"finite={n_finite}/{n_total}",
        ]
        if n_finite:
            finite_values = detached[finite]
            pieces.append(f"min={float(finite_values.min().item()):.6g}")
            pieces.append(f"max={float(finite_values.max().item()):.6g}")
        return ", ".join(pieces)


def _require_finite_tensor(name: str, tensor: torch.Tensor, *, step: int) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(
            f"non-finite {name} at step {step}; {_tensor_finite_summary(name, tensor)}"
        )


def _materialize_parameters(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    params = list(parameters)
    if not params:
        raise ValueError("parameters must contain at least one trainable parameter")
    return params


def _require_finite_gradients(
    parameters: Sequence[nn.Parameter], *, step: int
) -> None:
    bad: list[str] = []
    for idx, param in enumerate(parameters):
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad.append(_tensor_finite_summary(f"grad[{idx}]", param.grad))
            if len(bad) >= 3:
                break
    if bad:
        joined = "; ".join(bad)
        raise FloatingPointError(f"non-finite gradient at step {step}; {joined}")


def gather_logits_at(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``logits[B, L, V]`` + ``positions[B, Q]`` -> ``[B, Q, V]``."""
    b, q = positions.shape
    v = logits.shape[-1]
    return logits.gather(1, positions.unsqueeze(-1).expand(b, q, v))


def seeded_generator(seed: int) -> torch.Generator:
    """Seed the global torch RNG (model init) and return a data generator."""
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def stable_probe_init(model: nn.Module, *, n_blocks: int) -> None:
    """GPT-2-style init with depth-scaled residual-output std."""
    init_std = 0.02
    scaled_init_std = init_std / math.sqrt(max(1, 2 * n_blocks))
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            is_residual_out = (
                name.endswith(".fc2")
                or name.endswith(".out")
                or name.endswith(".out_proj")
            )
            std = scaled_init_std if is_residual_out else init_std
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def build_tiny_lm(
    lane_factory: Callable[[int], nn.Module],
    *,
    vocab_size: int,
    dim: int,
    n_blocks: int,
    max_seq_len: int,
    use_position_embedding: bool = True,
    use_rope: bool = True,
    use_ffn: bool = False,
    ffn_mult: int = 4,
    embedding_kind: str = "dense",
    ecc_code_length: int = 8,
    ecc_field_size: int = 257,
    hash_n_buckets: int | None = None,
    jl_rank: int | None = None,
    jl_seed: int = 5,
    stable_init: bool = False,
    device: str = "cpu",
) -> TinyLM:
    """The one TinyLM builder shared by every token-task probe."""
    cfg = TinyLMConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        use_position_embedding=use_position_embedding,
        use_rope=use_rope,
        max_seq_len=max_seq_len,
        use_ffn=use_ffn,
        ffn_mult=ffn_mult,
        embedding_kind=embedding_kind,
        ecc_code_length=ecc_code_length,
        ecc_field_size=ecc_field_size,
        hash_n_buckets=hash_n_buckets,
        jl_rank=jl_rank,
        jl_seed=jl_seed,
    )
    model = TinyLM(lane_factory, cfg)
    if stable_init:
        stable_probe_init(model, n_blocks=n_blocks)
    return model.to(device)


@dataclass(frozen=True, slots=True)
class CheckpointMetrics:
    step: int
    train_loss: float
    train_accuracy: float
    eval_accuracy: float


@dataclass(frozen=True, slots=True)
class TokenTaskTrace:
    initial_loss: float
    final_loss: float
    final_train_accuracy: float
    converged: bool
    checkpoints: tuple[CheckpointMetrics, ...]

    def checkpoint_at(self, step: int) -> CheckpointMetrics | None:
        for row in self.checkpoints:
            if row.step == step:
                return row
        return None


def evaluate_token_accuracy(
    model: nn.Module,
    batch_fn: TokenBatchFn,
    *,
    generator: torch.Generator,
    n_eval_batches: int = 8,
    device: str = "cpu",
) -> float:
    """Argmax accuracy at the query positions over ``n_eval_batches``."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_eval_batches):
            ids, positions, targets = batch_fn(generator)
            ids = ids.to(device)
            positions = positions.to(device)
            targets = targets.to(device)
            predictions = gather_logits_at(model(ids), positions).argmax(dim=-1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
    return correct / max(1, total)


def train_token_task(
    model: nn.Module,
    train_batch_fn: TokenBatchFn,
    *,
    rng: torch.Generator,
    eval_at_steps: Sequence[int],
    eval_batch_fn: TokenBatchFn,
    eval_seed: int | None,
    n_eval_batches: int = 8,
    learning_rate: float = 3e-3,
    device: str = "cpu",
    probe: str,
    max_grad_norm: float | None = 1.0,
) -> TokenTaskTrace:
    """Train to ``max(eval_at_steps)``; record metrics at every checkpoint.

    ``eval_seed=None`` continues the *training* generator for eval (the
    legacy single-shot ``run_one_task`` semantics); an int re-seeds a fresh
    generator at every checkpoint so the eval set is identical across
    checkpoints and across models.
    """
    steps = sorted({int(s) for s in eval_at_steps if int(s) > 0})
    if not steps:
        raise ValueError("eval_at_steps must contain at least one positive step")
    checkpoints_at = set(steps)

    params = _materialize_parameters(model.parameters())
    optim = torch.optim.Adam(params, lr=learning_rate)
    initial_loss = float("nan")
    final_train_acc = 0.0
    # Track the running loss as a detached tensor; .item() per step is a
    # GPU->CPU sync that serializes the training loop. Sync once at the end.
    last_loss: torch.Tensor | None = None
    rows: list[CheckpointMetrics] = []
    converged = True
    try:
        model.train()
        for step in range(1, steps[-1] + 1):
            ids, positions, targets = train_batch_fn(rng)
            ids = ids.to(device)
            positions = positions.to(device)
            targets = targets.to(device)
            query_logits = gather_logits_at(model(ids), positions)
            _require_finite_tensor("query_logits", query_logits, step=step)
            loss = nn.functional.cross_entropy(
                query_logits.reshape(-1, query_logits.shape[-1]),
                targets.reshape(-1),
            )
            _require_finite_tensor("loss", loss, step=step)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            _require_finite_gradients(params, step=step)
            if max_grad_norm is not None:
                grad_norm = nn.utils.clip_grad_norm_(params, max_grad_norm)
                _require_finite_tensor("grad_norm", grad_norm, step=step)
            optim.step()
            if step == 1:
                initial_loss = float(loss.item())
            last_loss = loss.detach()
            if step not in checkpoints_at:
                continue
            final_train_acc = float(
                (query_logits.argmax(dim=-1) == targets).float().mean().item()
            )
            eval_generator = (
                rng if eval_seed is None else torch.Generator().manual_seed(eval_seed)
            )
            eval_accuracy = evaluate_token_accuracy(
                model,
                eval_batch_fn,
                generator=eval_generator,
                n_eval_batches=n_eval_batches,
                device=device,
            )
            model.train()
            rows.append(
                CheckpointMetrics(
                    step=step,
                    train_loss=float(loss.item()),
                    train_accuracy=final_train_acc,
                    eval_accuracy=eval_accuracy,
                )
            )
    except Exception:  # noqa: BLE001 - one bad lane must not abort the cohort
        logger.warning("token-task probe %r failed; scoring 0.0", probe, exc_info=True)
        converged = False
    final_loss = float(last_loss.item()) if last_loss is not None else float("nan")
    return TokenTaskTrace(
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_train_accuracy=final_train_acc,
        converged=converged,
        checkpoints=tuple(rows),
    )


# ---------------- Continuous-lane twin ----------------

TargetT = TypeVar("TargetT")


@dataclass(frozen=True, slots=True)
class LaneHeadTrace:
    checkpoint_values: tuple[float, ...]
    final_loss: float

    @property
    def final_checkpoint_value(self) -> float:
        return self.checkpoint_values[-1] if self.checkpoint_values else 0.0


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        return float((logits.argmax(dim=-1) == labels).float().mean().item())


def train_lane_head(
    forward: Callable[[torch.Tensor], torch.Tensor],
    parameters: Iterable[nn.Parameter],
    sample_batch: Callable[[], tuple[torch.Tensor, TargetT]],
    loss_fn: Callable[[torch.Tensor, TargetT], torch.Tensor],
    *,
    n_train_steps: int,
    learning_rate: float = 3e-3,
    checkpoint_at_steps: tuple[int, ...] = (),
    checkpoint_metric: Callable[[torch.Tensor, TargetT], float] | None = None,
    max_grad_norm: float | None = 1.0,
) -> LaneHeadTrace:
    """Adam loop over ``forward(sample) -> loss`` with optional checkpoints.

    Raises ``FloatingPointError`` on non-finite predictions, loss, gradient, or
    gradient norm. The exception includes compact tensor diagnostics so the
    caller can identify whether divergence started in the lane, head, loss, or
    optimizer step.
    """
    params = _materialize_parameters(parameters)
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    metric = checkpoint_metric or classification_accuracy
    checkpoint_values: list[float] = []
    last_loss: torch.Tensor | None = None
    for step in range(1, int(n_train_steps) + 1):
        x, target = sample_batch()
        predictions = forward(x)
        _require_finite_tensor("predictions", predictions, step=step)
        loss = loss_fn(predictions, target)
        _require_finite_tensor("loss", loss, step=step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _require_finite_gradients(params, step=step)
        if max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(params, max_grad_norm)
            _require_finite_tensor("grad_norm", grad_norm, step=step)
        optimizer.step()
        last_loss = loss.detach()
        if step in checkpoint_at_steps:
            checkpoint_values.append(metric(predictions, target))
    final_loss = float(last_loss.item()) if last_loss is not None else float("nan")
    return LaneHeadTrace(
        checkpoint_values=tuple(checkpoint_values),
        final_loss=final_loss,
    )
