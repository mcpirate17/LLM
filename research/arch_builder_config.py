from __future__ import annotations

from dataclasses import dataclass

from .defaults import MODEL_DIM, VOCAB_SIZE


@dataclass
class BuildConfig:
    """Concrete hyperparameters for building a model from an ArchSpec."""

    dim: int = MODEL_DIM
    n_heads: int = 8
    n_kv_heads: int = 4
    n_layers: int = 6
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = 512
    mlp_ratio: float = 3.0
    dropout: float = 0.0
    moe_num_experts: int = 4
    moe_topk: int = 2
    compression_factor: int = 4
