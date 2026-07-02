"""MiniMax-M3-align M3X-C1 promotion wiring for compact embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from torch import nn

from .harder_binding_tasks import (
    HardBindingResult,
    HardBindingTask,
    default_hard_binding_tasks,
    run_one_task,
)
from .tiny_lm import CausalConv1dLane

M3X_C1_EMBEDDING_KINDS: tuple[str, ...] = (
    "dense",
    "ecc_codeword",
    "modulo_hash",
    "jl_low_rank",
)


@dataclass(frozen=True, slots=True)
class M3XC1PromotionRow:
    embedding_kind: str
    task_name: str
    mixer_label: str
    n_params: int
    embedding_params: int
    train_loss_initial: float
    train_loss_final: float
    train_accuracy_final: float
    eval_accuracy: float
    chance_accuracy: float
    converged: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compact_embedding_param_count(
    embedding_kind: str,
    *,
    vocab_size: int,
    dim: int,
    ecc_code_length: int,
    ecc_field_size: int,
    hash_n_buckets: int | None = None,
    jl_rank: int | None = None,
) -> int:
    if embedding_kind == "dense":
        return int(vocab_size * dim)
    if embedding_kind == "ecc_codeword":
        if dim % ecc_code_length != 0:
            raise ValueError("dim must be divisible by ecc_code_length")
        return int(ecc_code_length * ecc_field_size * (dim // ecc_code_length))
    if embedding_kind == "modulo_hash":
        return int((hash_n_buckets or ecc_field_size) * dim)
    if embedding_kind == "jl_low_rank":
        return int((jl_rank or ecc_field_size) * dim)
    raise ValueError(f"unknown M3X-C1 embedding kind: {embedding_kind!r}")


def _select_task(task_name: str, *, seed: int) -> HardBindingTask:
    for task in default_hard_binding_tasks(seed=seed):
        if task.name == task_name:
            return task
    raise ValueError(f"unknown hard-binding task: {task_name!r}")


def _promotion_row(
    embedding_kind: str,
    result: HardBindingResult,
    *,
    vocab_size: int,
    dim: int,
    ecc_code_length: int,
    ecc_field_size: int,
    hash_n_buckets: int | None,
    jl_rank: int | None,
) -> M3XC1PromotionRow:
    return M3XC1PromotionRow(
        embedding_kind=embedding_kind,
        task_name=result.task_name,
        mixer_label=result.mixer_label,
        n_params=result.n_params,
        embedding_params=compact_embedding_param_count(
            embedding_kind,
            vocab_size=vocab_size,
            dim=dim,
            ecc_code_length=ecc_code_length,
            ecc_field_size=ecc_field_size,
            hash_n_buckets=hash_n_buckets,
            jl_rank=jl_rank,
        ),
        train_loss_initial=result.train_loss_initial,
        train_loss_final=result.train_loss_final,
        train_accuracy_final=result.train_accuracy_final,
        eval_accuracy=result.eval_accuracy,
        chance_accuracy=result.chance_accuracy,
        converged=result.converged,
    )


def run_m3x_c1_embedding_promotion_probe(
    *,
    lane_factory: Callable[[int], nn.Module] = CausalConv1dLane,
    mixer_label: str = "causal_conv",
    task_name: str = "multi_query_kv_recall",
    embedding_kinds: tuple[str, ...] = M3X_C1_EMBEDDING_KINDS,
    dim: int = 64,
    n_blocks: int = 1,
    n_train_steps: int = 20,
    batch_size: int = 8,
    learning_rate: float = 3e-3,
    ecc_code_length: int = 8,
    ecc_field_size: int = 17,
    hash_n_buckets: int | None = None,
    jl_rank: int | None = None,
    jl_seed: int = 5,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[M3XC1PromotionRow, ...]:
    """Run the same harder-binding probe across dense/ECC/hash/JL embeddings."""
    task = _select_task(task_name, seed=seed)
    rows: list[M3XC1PromotionRow] = []
    for embedding_kind in embedding_kinds:
        result = run_one_task(
            lane_factory,
            task,
            mixer_label=f"{mixer_label}:{embedding_kind}",
            dim=dim,
            n_blocks=n_blocks,
            n_train_steps=n_train_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            embedding_kind=embedding_kind,
            ecc_code_length=ecc_code_length,
            ecc_field_size=ecc_field_size,
            hash_n_buckets=hash_n_buckets,
            jl_rank=jl_rank,
            jl_seed=jl_seed,
            seed=seed,
            device=device,
        )
        rows.append(
            _promotion_row(
                embedding_kind,
                result,
                vocab_size=task.vocab_size,
                dim=dim,
                ecc_code_length=ecc_code_length,
                ecc_field_size=ecc_field_size,
                hash_n_buckets=hash_n_buckets,
                jl_rank=jl_rank,
            )
        )
    return tuple(rows)
