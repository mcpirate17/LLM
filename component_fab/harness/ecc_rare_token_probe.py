"""MiniMax-M3-align M3X-C1 rare-token controls for ECC embeddings.

This probe trains compact shared embedding/output-head models on a few-shot
rare-token identity task. ECC, modulo-hash, and JL-low-rank controls receive the
same trainable parameter budget, so the score isolates whether structured
codeword sharing beats unstructured compact representations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from component_fab.generator.ecc_codeword_embedding import (
    ECCCodewordEmbedding,
    ECCCodewordOutputHead,
    JLLowRankEmbedding,
    MaterializedWeightOutputHead,
    ModuloHashEmbedding,
)


@dataclass(frozen=True, slots=True)
class RareTokenProbeConfig:
    vocab_size: int = 128
    dim: int = 64
    code_length: int = 8
    field_size: int = 17
    rare_bucket: int = 3
    rare_group_size: int = 8
    batch_size: int = 64
    rare_per_batch: int = 2
    train_steps: int = 10
    learning_rate: float = 5e-2
    seed: int = 123
    data_seed: int = 9
    jl_seed: int = 5
    max_grad_norm: float = 1.0

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.dim % self.code_length != 0:
            raise ValueError("dim must be divisible by code_length")
        if self.field_size <= 2:
            raise ValueError("field_size must be greater than 2")
        if not 0 <= self.rare_bucket < self.field_size:
            raise ValueError("rare_bucket must be in [0, field_size)")
        if self.rare_group_size <= 1:
            raise ValueError("rare_group_size must be greater than 1")
        if self.batch_size <= self.rare_per_batch:
            raise ValueError("batch_size must exceed rare_per_batch")
        if self.rare_per_batch <= 0:
            raise ValueError("rare_per_batch must be positive")
        if self.train_steps <= 0:
            raise ValueError("train_steps must be positive")


@dataclass(frozen=True, slots=True)
class RareTokenProbeResult:
    name: str
    trainable_params: int
    final_loss: float
    rare_accuracy: float
    rare_margin: float


class _SharedEmbeddingModel(nn.Module):
    def __init__(self, embedding: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.embedding = embedding
        self.head = head

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(ids))


def rare_token_ids(config: RareTokenProbeConfig) -> torch.Tensor:
    config.validate()
    ids = torch.arange(config.rare_bucket, config.vocab_size, config.field_size)
    if ids.numel() < config.rare_group_size:
        raise ValueError(
            "rare_group_size exceeds available tokens for rare_bucket "
            f"{config.rare_bucket}: requested {config.rare_group_size}, "
            f"available {ids.numel()}"
        )
    return ids[: config.rare_group_size]


def _common_token_ids(config: RareTokenProbeConfig) -> torch.Tensor:
    all_ids = torch.arange(config.vocab_size, dtype=torch.long)
    return all_ids[all_ids.remainder(config.field_size) != config.rare_bucket]


def _make_model(name: str, config: RareTokenProbeConfig) -> _SharedEmbeddingModel:
    if name == "ecc_codeword":
        embedding = ECCCodewordEmbedding(
            config.vocab_size,
            config.dim,
            code_length=config.code_length,
            field_size=config.field_size,
        )
        return _SharedEmbeddingModel(embedding, ECCCodewordOutputHead(embedding))
    if name == "modulo_hash":
        embedding = ModuloHashEmbedding(
            config.vocab_size, config.dim, n_buckets=config.field_size
        )
        return _SharedEmbeddingModel(embedding, MaterializedWeightOutputHead(embedding))
    if name == "jl_low_rank":
        embedding = JLLowRankEmbedding(
            config.vocab_size,
            config.dim,
            rank=config.field_size,
            seed=config.jl_seed,
        )
        return _SharedEmbeddingModel(embedding, MaterializedWeightOutputHead(embedding))
    raise ValueError(f"unknown rare-token control: {name!r}")


def _sample_batch(
    *,
    rare_ids: torch.Tensor,
    common_ids: torch.Tensor,
    config: RareTokenProbeConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    common_count = config.batch_size - config.rare_per_batch
    common = common_ids[
        torch.randint(common_ids.numel(), (common_count,), generator=generator)
    ]
    rare = rare_ids[
        torch.randint(rare_ids.numel(), (config.rare_per_batch,), generator=generator)
    ]
    ids = torch.cat((common, rare), dim=0)
    return ids[torch.randperm(ids.numel(), generator=generator)]


def _evaluate_rare(
    model: _SharedEmbeddingModel, rare_ids: torch.Tensor
) -> tuple[float, float]:
    with torch.no_grad():
        logits = model(rare_ids)
        predictions = logits.argmax(dim=-1)
        rare_accuracy = float((predictions == rare_ids).float().mean().item())
        target_logits = logits.gather(1, rare_ids.unsqueeze(1)).squeeze(1)
        target_mask = torch.zeros_like(logits, dtype=torch.bool)
        target_mask.scatter_(1, rare_ids.unsqueeze(1), True)
        non_target_logits = logits.masked_fill(target_mask, -torch.inf)
        margins = target_logits - non_target_logits.max(dim=-1).values
        rare_margin = float(margins.mean().item())
    return rare_accuracy, rare_margin


def run_rare_token_embedding_probe(
    config: RareTokenProbeConfig | None = None,
) -> tuple[RareTokenProbeResult, ...]:
    """Train ECC, hash, and JL controls on the same few-shot rare-token task."""
    cfg = config or RareTokenProbeConfig()
    cfg.validate()
    rare_ids = rare_token_ids(cfg)
    common_ids = _common_token_ids(cfg)
    results: list[RareTokenProbeResult] = []
    for name in ("ecc_codeword", "modulo_hash", "jl_low_rank"):
        torch.manual_seed(cfg.seed)
        model = _make_model(name, cfg)
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate)
        generator = torch.Generator().manual_seed(cfg.data_seed)
        final_loss = torch.tensor(float("nan"))
        for _ in range(cfg.train_steps):
            ids = _sample_batch(
                rare_ids=rare_ids,
                common_ids=common_ids,
                config=cfg,
                generator=generator,
            )
            logits = model(ids)
            loss = F.cross_entropy(logits, ids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm)
            optimizer.step()
            final_loss = loss.detach()
        rare_accuracy, rare_margin = _evaluate_rare(model, rare_ids)
        results.append(
            RareTokenProbeResult(
                name=name,
                trainable_params=sum(p.numel() for p in parameters),
                final_loss=float(final_loss.item()),
                rare_accuracy=rare_accuracy,
                rare_margin=rare_margin,
            )
        )
    return tuple(results)
