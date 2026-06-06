"""Tiny lane-pluggable LM for fab-level binding tasks.

A minimal ``Embedding -> [lane, norm]*n_blocks -> LMHead`` wrapper so we
can train any fab lane primitive on token-id sequences with a real
cross-entropy LM head. Used by ``harder_binding_tasks`` for harder
discrete/symbolic binding probes than ``nano_bind`` (continuous vector,
single key, single slot).

Same shape as ``research/tools/small_ar_story_calibration.py:TinyCausalAttentionLM``
but lane-agnostic: any ``nn.Module`` mapping ``[B, L, D] -> [B, L, D]`` can be
the mixer. The mixer is **the only thing that differs** between a fab
candidate and the baselines — embedding, depth, head, optimizer, steps
are identical, so fair comparison is straightforward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from ..generator.primitive_templates._core import get_causal_mask
from .rope import RotaryEmbedding, apply_rope


@dataclass(frozen=True, slots=True)
class TinyLMConfig:
    vocab_size: int
    dim: int = 64
    n_blocks: int = 2
    # RoPE is the new default for positional info — applied inside attention
    # lanes to Q and K, extrapolates past the trained seq_len cleanly.
    # ``use_position_embedding=True`` exists only for loading legacy
    # checkpoints (pre-RoPE), which had a learned abs pos embed capped at
    # ``max_seq_len``; setting both flags True is supported but redundant.
    use_position_embedding: bool = False
    use_rope: bool = True
    max_seq_len: int = 1024
    # Pre-norm Transformer block (mixer + FFN). Disable for the legacy
    # mixer-only block used by the discrete-binding probes.
    use_ffn: bool = False
    ffn_mult: int = 4
    ffn_kind: str = "swiglu"


class _MLP(nn.Module):
    """SwiGLU channel mixer: ``W3( SiLU(W1 x) * (W2 x) )`` (table-stakes; the
    de-facto default gated FFN in modern LLMs). The hidden width is scaled by
    2/3 so the 3-matrix SwiGLU has ~the same param count as the prior 2-matrix
    ``Linear→GELU→Linear`` FFN at the same ``mult`` (standard practice)."""

    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        self.fc1 = nn.Linear(dim, hidden)  # gate (SiLU)
        self.fc2 = nn.Linear(dim, hidden)  # value
        self.fc3 = nn.Linear(hidden, dim)  # project back

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(torch.nn.functional.silu(self.fc1(x)) * self.fc2(x))


class _GELUMLP(nn.Module):
    """Legacy two-matrix FFN used by pre-SwiGLU checkpoints."""

    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        hidden = dim * mult
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _LaneBlock(nn.Module):
    """One pre-norm block. With FFN this is the standard Transformer
    pattern: ``x -> norm -> mixer -> +x -> norm -> FFN -> +x``.

    Without FFN it's the mixer-only pattern used by the discrete-binding
    probes (no language modeling). Same skeleton, FFN behind a flag.
    """

    def __init__(
        self,
        lane: nn.Module,
        dim: int,
        *,
        use_ffn: bool,
        ffn_mult: int,
        ffn_kind: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.lane = lane
        self.norm2: nn.LayerNorm | None
        self.mlp: nn.Module | None
        if use_ffn:
            self.norm2 = nn.LayerNorm(dim)
            if ffn_kind == "swiglu":
                self.mlp = _MLP(dim, mult=ffn_mult)
            elif ffn_kind == "gelu":
                self.mlp = _GELUMLP(dim, mult=ffn_mult)
            else:
                raise ValueError(f"unknown TinyLM FFN kind: {ffn_kind!r}")
        else:
            self.norm2 = None
            self.mlp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.lane(self.norm1(x))
        if self.mlp is not None and self.norm2 is not None:
            x = x + self.mlp(self.norm2(x))
        return x


class TinyLM(nn.Module):
    """``Embedding -> LaneBlock * n_blocks -> LayerNorm -> LMHead``.

    ``lane_factory`` produces a fresh ``nn.Module`` for each block (so
    each block's parameters are independent). The factory takes ``dim``
    and returns a position-mixing module operating on ``[B, L, D]``.
    """

    def __init__(
        self,
        lane_factory: Callable[[int], nn.Module],
        config: TinyLMConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embed: nn.Embedding | None
        if config.use_position_embedding:
            self.pos_embed = nn.Embedding(config.max_seq_len, config.dim)
        else:
            self.pos_embed = None
        self.blocks = nn.ModuleList(
            [
                _LaneBlock(
                    lane_factory(config.dim),
                    config.dim,
                    use_ffn=config.use_ffn,
                    ffn_mult=config.ffn_mult,
                    ffn_kind=config.ffn_kind,
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Tie embedding and head weights to keep param count low.
        self.lm_head.weight = self.embed.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, l = ids.shape
        h = self.embed(ids)
        if self.pos_embed is not None:
            positions = torch.arange(l, device=ids.device).unsqueeze(0).expand(b, l)
            h = h + self.pos_embed(positions)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.lm_head(h)


# ---------- Standard-mixer baselines ----------


class SoftmaxCausalAttention(nn.Module):
    """Single-head causal softmax attention — the obvious baseline for binding.

    Pass ``use_rope=True`` to apply RoPE to Q and K (replaces absolute pos
    embedding; lets the model accept seq_len up to ``max_seq_len``). Default
    is False for backward compatibility with the abs-pos-embed-on-input
    architecture.
    """

    def __init__(
        self, dim: int, *, use_rope: bool = False, max_seq_len: int = 1024
    ) -> None:
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = float(dim) ** -0.5
        self.rope = RotaryEmbedding(dim, max_seq_len=max_seq_len) if use_rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.rope is not None:
            cos, sin = self.rope(l, device=x.device, dtype=x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        affinity = torch.einsum("bid,bjd->bij", q, k) * self.scale
        mask = get_causal_mask(l, x.device, x.dtype)
        attn = torch.softmax(affinity + mask, dim=-1)
        return torch.einsum("bij,bjd->bid", attn, v)


class CausalConv1dLane(nn.Module):
    """Depthwise causal Conv1d — local-only mixing baseline.

    Strong on short-range patterns, weak on long-gap binding by construction.
    Useful negative control: if a fab lane only beats this on short tasks,
    it's local-only.
    """

    def __init__(self, dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1  # left pad for causality
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, D, L]
        h = x.transpose(1, 2)
        h = torch.nn.functional.pad(h, (self.pad, 0))
        h = self.conv(h).transpose(1, 2)
        return self.proj(h)


class MultiHeadCausalAttention(nn.Module):
    """GPT-2 style multi-head causal softmax attention (pure mixer).

    The canonical frontier baseline. TinyLM wraps it with pre-norm, residual and
    an FFN, so this is just the attention mixer (no FFN/residual here).
    """

    def __init__(self, dim: int, *, n_heads: int | None = None) -> None:
        super().__init__()
        if n_heads is None:
            n_heads = max(1, dim // 16)
            while n_heads > 1 and dim % n_heads != 0:
                n_heads -= 1
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.scale = float(self.head_dim) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        affinity = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask = get_causal_mask(l, x.device, x.dtype)
        attn = torch.softmax(affinity + mask, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(b, l, self.dim)
        return self.proj(out)


def _compiled_reference_mixer_factory(arch: str) -> Callable[[int], nn.Module]:
    """Build a residual/norm-free reference mixer lane from the NAS compiler.

    Reuses the validated op implementations (``selective_scan``, ``gated_delta``)
    rather than re-deriving Mamba/Mamba2 by hand. Returns a factory taking ``dim``
    and yielding ``compile_model([mixer_graph]).layers[0]`` — a pure ``[B,L,D]``
    mixer (TinyLM adds the norm/residual/FFN around it).
    """

    def factory(dim: int) -> nn.Module:
        from research.synthesis.compiler import compile_model
        from research.synthesis.graph import ComputationGraph

        g = ComputationGraph(dim)
        inp = g.add_input()
        if arch == "mamba":
            conv = g.add_op("conv1d_seq", [inp])
            act = g.add_op("silu", [conv])
            ssm = g.add_op("selective_scan", [act])
            out = g.add_op("gated_linear", [ssm], {"out_dim": dim})
        elif arch == "mamba2":  # Gated DeltaNet / Mamba2-family
            gd = g.add_op("gated_delta", [inp])
            out = g.add_op("linear_proj", [gd], {"out_dim": dim})
        else:
            raise ValueError(f"no compiled mixer for arch {arch!r}")
        g.set_output(out)
        return compile_model([g], use_ir=False).layers[0]

    return factory


def lane_factory_for_baseline(name: str) -> Callable[[int], nn.Module]:
    """Resolve a baseline name to a lane factory."""
    if name == "softmax_attention":
        return SoftmaxCausalAttention
    if name == "causal_conv":
        return CausalConv1dLane
    if name == "gpt2":
        return MultiHeadCausalAttention
    if name in ("mamba", "mamba2"):
        return _compiled_reference_mixer_factory(name)
    raise ValueError(f"unknown baseline: {name}")


def striped_lane_factory(
    candidate_factory: Callable[[int], nn.Module],
    *,
    attn_every: int = 2,
    attn_name: str = "softmax_attention",
) -> Callable[[int], nn.Module]:
    """Wrap a candidate mixer into a striped hybrid: attention every ``attn_every``
    blocks, the candidate everywhere else.

    Why: the hybrid-architecture literature (Poli et al. 2024 striped stacks +8.1%
    on MAD; Lahoti et al. 2025) finds that a token-mixer's value is its contribution
    *inside* an attention stack, and that standalone quality does NOT predict hybrid
    quality. So a fair "is this component worth keeping" test interleaves it with
    full attention rather than running it solo. ``TinyLM`` calls the lane factory
    once per block in order, so a position counter assigns block 0 (and every
    ``attn_every``-th block) to attention and the rest to the candidate — e.g.
    ``attn_every=2`` → ``[attn, cand, attn, cand, …]`` (1:1); ``attn_every=4`` →
    ``[attn, cand, cand, cand, …]`` (1 full : 3 candidate, the Lahoti 3:1 regime).

    The returned factory is single-use per ``TinyLM`` (it carries block-position
    state); build a fresh one for each model.
    """
    attn_factory = lane_factory_for_baseline(attn_name)
    position = {"i": -1}

    def factory(dim: int) -> nn.Module:
        position["i"] += 1
        if position["i"] % max(1, attn_every) == 0:
            return attn_factory(dim)
        return candidate_factory(dim)

    return factory


DEFAULT_BASELINE_NAMES: tuple[str, ...] = ("softmax_attention", "causal_conv")
# Frontier preset: opt-in via cohort --baselines. Compares candidates against
# real GPT-2 / Mamba / Mamba2 mixers, not just nano softmax/conv.
FRONTIER_BASELINE_NAMES: tuple[str, ...] = (
    "softmax_attention",
    "gpt2",
    "mamba",
    "mamba2",
)


# ---------- Param-count utility ----------


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def param_match_warning(
    candidate_lm: TinyLM, baseline_lms: dict[str, TinyLM], tolerance: float = 0.05
) -> list[str]:
    """Sanity check that candidate and baselines have comparable param counts.

    Returns a list of warning strings (empty when within tolerance). Fair
    comparison requires same scale — if a candidate has 10× the params of a
    baseline, win/loss says nothing about the mixer.
    """
    out: list[str] = []
    cand_count = count_trainable_params(candidate_lm)
    for name, lm in baseline_lms.items():
        b_count = count_trainable_params(lm)
        if b_count == 0:
            continue
        ratio = cand_count / b_count
        if abs(math.log(ratio)) > math.log(1.0 + tolerance):
            out.append(
                f"param count drift: candidate={cand_count} {name}={b_count} "
                f"ratio={ratio:.3f}"
            )
    return out
