"""Unit tests for RoPE module + integration with TinyLM."""

from __future__ import annotations


import pytest
import torch

from component_fab.generator.primitive_templates import (
    SparsemaxAttention,
    TropicalAttention,
)
from component_fab.harness.rope import RotaryEmbedding, apply_rope
from component_fab.harness.tiny_lm import (
    SoftmaxCausalAttention,
    TinyLM,
    TinyLMConfig,
)


# ---------- RoPE module ----------


def test_rope_cos_sin_shapes() -> None:
    rope = RotaryEmbedding(dim=64, max_seq_len=128)
    cos, sin = rope(seq_len=32, device=torch.device("cpu"), dtype=torch.float32)
    assert cos.shape == (32, 32)  # (seq_len, dim/2)
    assert sin.shape == (32, 32)


def test_rope_odd_dim_rejected() -> None:
    with pytest.raises(ValueError, match="even dim"):
        RotaryEmbedding(dim=63)


def test_rope_seq_len_overflow_rejected() -> None:
    rope = RotaryEmbedding(dim=32, max_seq_len=64)
    with pytest.raises(ValueError, match="exceeds cached"):
        rope(seq_len=65, device=torch.device("cpu"), dtype=torch.float32)


def test_apply_rope_preserves_norm() -> None:
    """Rotation should preserve L2 norm per (pos, halved-pair)."""
    torch.manual_seed(0)
    rope = RotaryEmbedding(dim=16, max_seq_len=8)
    x = torch.randn(2, 8, 16)
    cos, sin = rope(8, device=x.device, dtype=x.dtype)
    y = apply_rope(x, cos, sin)
    # ||(x1, x2)|| == ||(x1*cos - x2*sin, x1*sin + x2*cos)||
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_apply_rope_position_zero_is_identity() -> None:
    """At position 0, cos=1, sin=0 → rotation is identity."""
    rope = RotaryEmbedding(dim=16, max_seq_len=8)
    x = torch.randn(1, 1, 16)
    cos, sin = rope(1, device=x.device, dtype=x.dtype)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x, y, atol=1e-6)


def test_apply_rope_position_dependent() -> None:
    """Same token at different positions should rotate differently."""
    rope = RotaryEmbedding(dim=16, max_seq_len=8)
    token = torch.ones(1, 1, 16)
    cos4, sin4 = rope(4, device=token.device, dtype=token.dtype)
    rotated = torch.stack(
        [
            apply_rope(token, cos4[i : i + 1], sin4[i : i + 1]).squeeze()
            for i in range(4)
        ]
    )
    # Position 0 = unchanged; positions 1..3 differ from each other.
    for i in range(1, 4):
        assert not torch.allclose(rotated[0], rotated[i], atol=1e-3)
    for i in range(1, 4):
        for j in range(i + 1, 4):
            assert not torch.allclose(rotated[i], rotated[j], atol=1e-3)


# ---------- Attention classes accept use_rope ----------


@pytest.mark.parametrize(
    "cls", [SoftmaxCausalAttention, TropicalAttention, SparsemaxAttention]
)
def test_attention_use_rope_forward(cls) -> None:
    attn = cls(dim=32, use_rope=True, max_seq_len=128).eval()
    x = torch.randn(2, 100, 32)  # seq_len > 64 to ensure RoPE actually engaged
    with torch.no_grad():
        y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(
    "cls", [SoftmaxCausalAttention, TropicalAttention, SparsemaxAttention]
)
def test_attention_default_no_rope(cls) -> None:
    """Default constructor (no kwargs) must keep the old behavior — rope is None."""
    attn = cls(dim=32)
    assert getattr(attn, "rope", "MISSING") is None, (
        f"{cls.__name__} should default to use_rope=False; got rope={attn.rope!r}"
    )


# ---------- TinyLM end-to-end ----------


def test_tinylm_new_default_accepts_long_seq() -> None:
    """The whole point of Tier 2: TinyLM with defaults should accept seq_len > 256."""
    cfg = TinyLMConfig(vocab_size=100, dim=32, n_blocks=2, max_seq_len=512)
    model = TinyLM(SoftmaxCausalAttention, cfg).eval()
    # attach RoPE post-hoc (mirrors _attach_rope_to_attention)
    for m in model.modules():
        if isinstance(m, SoftmaxCausalAttention):
            m.rope = RotaryEmbedding(m.dim, max_seq_len=cfg.max_seq_len)
    ids = torch.randint(0, 100, (1, 400))
    with torch.no_grad():
        out = model(ids)
    assert out.shape == (1, 400, 100)
    assert torch.isfinite(out).all()


def test_tinylm_legacy_config_still_works() -> None:
    """Old checkpoint config (use_position_embedding=True, max_seq_len=256) loads."""
    cfg = TinyLMConfig(
        vocab_size=100,
        dim=32,
        n_blocks=2,
        use_position_embedding=True,
        use_rope=False,
        max_seq_len=256,
    )
    model = TinyLM(SoftmaxCausalAttention, cfg).eval()
    assert model.pos_embed is not None
    assert model.pos_embed.weight.shape == (256, 32)
    ids = torch.randint(0, 100, (1, 256))
    with torch.no_grad():
        out = model(ids)
    assert out.shape == (1, 256, 100)
