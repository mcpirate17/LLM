"""MiniMax-M3-align M3X-C1: compact ECC-codeword token embeddings.

Tokens are represented by deterministic polynomial codewords over a small prime
field. Each code symbol indexes a small learned table, and the selected symbol
vectors are concatenated to form the token embedding. The paired output head
reuses the same symbol tables, so TinyLM can keep input/output weight sharing
without storing a dense ``vocab_size * dim`` matrix.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    limit = int(math.sqrt(value))
    for candidate in range(3, limit + 1, 2):
        if value % candidate == 0:
            return False
    return True


def _num_polynomial_coefficients(vocab_size: int, field_size: int) -> int:
    coefficients = 1
    capacity = field_size
    while capacity < vocab_size:
        coefficients += 1
        capacity *= field_size
    return coefficients


def make_polynomial_codewords(
    vocab_size: int,
    *,
    code_length: int,
    field_size: int,
) -> torch.Tensor:
    """Return ``[vocab_size, code_length]`` Reed-Solomon-style integer codes."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if code_length <= 0:
        raise ValueError("code_length must be positive")
    if field_size <= 2 or not _is_prime(field_size):
        raise ValueError("field_size must be an odd prime")
    if code_length >= field_size:
        raise ValueError("code_length must be smaller than field_size")

    n_coefficients = _num_polynomial_coefficients(vocab_size, field_size)
    if n_coefficients > code_length:
        raise ValueError(
            "vocab_size is too large for an injective code with the requested "
            "code_length and field_size"
        )

    ids = torch.arange(vocab_size, dtype=torch.long)
    remaining = ids.clone()
    coeffs = torch.empty(vocab_size, n_coefficients, dtype=torch.long)
    for degree in range(n_coefficients):
        coeffs[:, degree] = remaining.remainder(field_size)
        remaining = torch.div(remaining, field_size, rounding_mode="floor")

    points = torch.arange(1, code_length + 1, dtype=torch.long)
    powers = torch.empty(n_coefficients, code_length, dtype=torch.long)
    powers[0].fill_(1)
    for degree in range(1, n_coefficients):
        powers[degree] = powers[degree - 1].mul(points).remainder(field_size)
    return coeffs.matmul(powers).remainder(field_size)


class ECCCodewordEmbedding(nn.Module):
    """Compact embedding table backed by error-correcting token codewords."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        code_length: int = 8,
        field_size: int = 257,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if dim % code_length != 0:
            raise ValueError("dim must be divisible by code_length")
        self.vocab_size = vocab_size
        self.dim = dim
        self.code_length = code_length
        self.field_size = field_size
        self.symbol_dim = dim // code_length
        self.n_coefficients = _num_polynomial_coefficients(vocab_size, field_size)
        self.init_std = init_std

        codewords = make_polynomial_codewords(
            vocab_size, code_length=code_length, field_size=field_size
        )
        self.register_buffer("codewords", codewords, persistent=True)
        self.symbol_tables = nn.Parameter(
            torch.empty(code_length, field_size, self.symbol_dim)
        )
        self.reset_parameters()

    @property
    def minimum_distance_lower_bound(self) -> int:
        return max(1, self.code_length - self.n_coefficients + 1)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.symbol_tables, mean=0.0, std=self.init_std)

    def compact_parameter_count(self) -> int:
        return int(self.symbol_tables.numel())

    def dense_parameter_count(self) -> int:
        return int(self.vocab_size * self.dim)

    def compression_ratio(self) -> float:
        return self.dense_parameter_count() / max(1, self.compact_parameter_count())

    def symbols_for(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("ids must be an integer tensor")
        return self.codewords[ids.to(dtype=torch.long)]

    def _embed_symbols(self, symbols: torch.Tensor) -> torch.Tensor:
        flat = symbols.reshape(-1, self.code_length)
        positions = torch.arange(self.code_length, device=flat.device).unsqueeze(0)
        gathered = self.symbol_tables[positions, flat]
        return gathered.reshape(*symbols.shape[:-1], self.dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self._embed_symbols(self.symbols_for(ids))

    def materialize_weight(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        weight = self._embed_symbols(self.codewords)
        if dtype is not None and weight.dtype != dtype:
            weight = weight.to(dtype=dtype)
        return weight

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, dim={self.dim}, "
            f"code_length={self.code_length}, field_size={self.field_size}, "
            f"compact_params={self.compact_parameter_count()}"
        )


class ModuloHashEmbedding(nn.Module):
    """Equal-budget collision control for compact embedding experiments."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        n_buckets: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if n_buckets <= 0:
            raise ValueError("n_buckets must be positive")
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_buckets = n_buckets
        self.init_std = init_std
        self.weight = nn.Parameter(torch.empty(n_buckets, dim))
        self.register_buffer(
            "bucket_ids",
            torch.arange(vocab_size, dtype=torch.long).remainder(n_buckets),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self.init_std)

    def compact_parameter_count(self) -> int:
        return int(self.weight.numel())

    def dense_parameter_count(self) -> int:
        return int(self.vocab_size * self.dim)

    def compression_ratio(self) -> float:
        return self.dense_parameter_count() / max(1, self.compact_parameter_count())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weight[self.bucket_ids[ids.to(dtype=torch.long)]]

    def materialize_weight(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        weight = self.weight[self.bucket_ids]
        if dtype is not None and weight.dtype != dtype:
            weight = weight.to(dtype=dtype)
        return weight

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, dim={self.dim}, "
            f"n_buckets={self.n_buckets}, "
            f"compact_params={self.compact_parameter_count()}"
        )


class JLLowRankEmbedding(nn.Module):
    """Equal-budget Johnson-Lindenstrauss low-rank embedding control."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        rank: int,
        seed: int = 5,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.vocab_size = vocab_size
        self.dim = dim
        self.rank = rank
        self.seed = seed
        self.init_std = init_std
        generator = torch.Generator().manual_seed(seed)
        codes = torch.randn(vocab_size, rank, generator=generator)
        self.register_buffer("codes", F.normalize(codes, dim=-1), persistent=True)
        self.basis = nn.Parameter(torch.empty(rank, dim))
        self.reset_parameters(generator=generator)

    def reset_parameters(self, *, generator: torch.Generator | None = None) -> None:
        nn.init.normal_(
            self.basis, mean=0.0, std=self.init_std, generator=generator
        )

    def compact_parameter_count(self) -> int:
        return int(self.basis.numel())

    def dense_parameter_count(self) -> int:
        return int(self.vocab_size * self.dim)

    def compression_ratio(self) -> float:
        return self.dense_parameter_count() / max(1, self.compact_parameter_count())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.codes[ids.to(dtype=torch.long)] @ self.basis

    def materialize_weight(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        weight = self.codes @ self.basis
        if dtype is not None and weight.dtype != dtype:
            weight = weight.to(dtype=dtype)
        return weight

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, dim={self.dim}, rank={self.rank}, "
            f"compact_params={self.compact_parameter_count()}"
        )


class MaterializedWeightOutputHead(nn.Module):
    """Shared logit head for compact embeddings without dense head parameters."""

    def __init__(self, embedding: nn.Module) -> None:
        super().__init__()
        self.vocab_size = embedding.vocab_size
        self.dim = embedding.dim
        # Keep a strong unregistered reference; TinyLM already owns the module.
        self.__dict__["_embedding"] = embedding

    @property
    def embedding(self) -> nn.Module:
        return self.__dict__["_embedding"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.embedding.materialize_weight(dtype=x.dtype)
        return F.linear(x, weight)

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, dim={self.dim}, shared=materialized"


class ECCCodewordOutputHead(MaterializedWeightOutputHead):
    """Logit head that shares an ``ECCCodewordEmbedding`` without dense weights."""

    def __init__(self, embedding: ECCCodewordEmbedding) -> None:
        super().__init__(embedding)

    @property
    def embedding(self) -> ECCCodewordEmbedding:
        return self.__dict__["_embedding"]

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, dim={self.dim}, shared=ecc_codeword"
