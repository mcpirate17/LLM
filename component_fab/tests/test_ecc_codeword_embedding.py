"""MiniMax-M3-align M3X-C1 tests for ECC codeword embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from component_fab.generator.ecc_codeword_embedding import (
    ECCCodewordEmbedding,
    ECCCodewordOutputHead,
)
from component_fab.harness.tiny_lm import TinyLM, TinyLMConfig, count_trainable_params


def test_polynomial_codewords_have_error_correcting_distance() -> None:
    embedding = ECCCodewordEmbedding(
        vocab_size=64,
        dim=32,
        code_length=8,
        field_size=17,
    )

    codes = embedding.codewords
    distances = codes.unsqueeze(1).ne(codes.unsqueeze(0)).sum(dim=-1)
    off_diagonal = ~torch.eye(codes.shape[0], dtype=torch.bool)

    assert embedding.minimum_distance_lower_bound == 7
    assert int(distances[off_diagonal].min().item()) >= 7


def test_ecc_codewords_separate_mod_hash_collisions() -> None:
    embedding = ECCCodewordEmbedding(
        vocab_size=64,
        dim=32,
        code_length=8,
        field_size=17,
    )
    ids = torch.tensor([3, 20])

    modulo_hash_codes = ids.remainder(17).view(-1, 1).expand(-1, 8)
    ecc_codes = embedding.codewords[ids]

    assert torch.equal(modulo_hash_codes[0], modulo_hash_codes[1])
    assert not torch.equal(ecc_codes[0], ecc_codes[1])
    assert int(ecc_codes[0].ne(ecc_codes[1]).sum().item()) >= 7


def test_embedding_uses_compact_symbol_tables_and_receives_gradients() -> None:
    torch.manual_seed(0)
    embedding = ECCCodewordEmbedding(
        vocab_size=128,
        dim=64,
        code_length=8,
        field_size=17,
    )
    ids = torch.tensor([[0, 1, 17, 34], [3, 20, 64, 127]])

    out = embedding(ids)
    loss = out.square().mean()
    loss.backward()

    symbols = embedding.symbols_for(ids)
    active = torch.zeros(
        embedding.code_length, embedding.field_size, dtype=torch.bool
    )
    for position in range(embedding.code_length):
        active[position, symbols[..., position].reshape(-1)] = True
    grad_energy = embedding.symbol_tables.grad.abs().sum(dim=-1)

    assert out.shape == (2, 4, 64)
    assert embedding.compact_parameter_count() == 8 * 17 * 8
    assert embedding.compact_parameter_count() < embedding.dense_parameter_count()
    assert torch.count_nonzero(grad_energy[active]).item() > 0


def test_output_head_shares_ecc_tables_without_dense_parameters() -> None:
    torch.manual_seed(1)
    embedding = ECCCodewordEmbedding(
        vocab_size=32,
        dim=16,
        code_length=4,
        field_size=17,
    )
    head = ECCCodewordOutputHead(embedding)
    x = torch.randn(2, 3, 16)

    logits = head(x)
    manual = F.linear(x, embedding.materialize_weight(dtype=x.dtype))
    logits.square().mean().backward()

    assert list(head.parameters()) == []
    assert torch.allclose(logits, manual)
    assert embedding.symbol_tables.grad is not None
    assert float(embedding.symbol_tables.grad.abs().sum().item()) > 0.0


def test_tinylm_ecc_embedding_is_opt_in_compact_and_trainable() -> None:
    torch.manual_seed(2)
    dense_cfg = TinyLMConfig(vocab_size=128, dim=32, n_blocks=1)
    ecc_cfg = TinyLMConfig(
        vocab_size=128,
        dim=32,
        n_blocks=1,
        embedding_kind="ecc_codeword",
        ecc_code_length=8,
        ecc_field_size=17,
    )
    dense_model = TinyLM(lambda _dim: nn.Identity(), dense_cfg)
    ecc_model = TinyLM(lambda _dim: nn.Identity(), ecc_cfg)
    ids = torch.randint(0, 128, (2, 6))

    logits = ecc_model(ids)
    logits.mean().backward()

    assert dense_model.lm_head.weight is dense_model.embed.weight
    assert isinstance(ecc_model.embed, ECCCodewordEmbedding)
    assert isinstance(ecc_model.lm_head, ECCCodewordOutputHead)
    assert ecc_model.lm_head.embedding is ecc_model.embed
    assert logits.shape == (2, 6, 128)
    assert ecc_model.embed.compact_parameter_count() < dense_model.embed.weight.numel()
    assert count_trainable_params(ecc_model) < count_trainable_params(dense_model)
    assert ecc_model.embed.symbol_tables.grad is not None
