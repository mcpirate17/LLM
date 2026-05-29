"""Multi-model multi-scale BLiMP scaling study with checkpointed v2 evals.

Trains a TinyLM(lane) on wikitext-103 BPE to a configurable step count,
saving checkpoints at intermediate step targets. At each checkpoint,
evaluates: (a) full BLiMP (67 subtasks), (b) selective_copy + variable_delay
v2 tests at quality settings.

Scales: 30M, 60M, 120M params via (dim, n_blocks) sizing.
Steps: 10K, 20K, 40K — single training to 40K with checkpoints at all 3.

Models supported (selected by --lane-name):
  - softmax_ffn          (GPT2-style baseline)
  - block_gated_parallel (fab day-3 BLiMP champion)
  - simplified_mamba     (selective SSM)
  - recursive_depth_router (runs.db top-BLiMP template)
  - hetero_moe_block     (runs.db top-BLiMP template)
  - tropical_attention   (fab winner-take-all)

Early-stop rule: if --early-stop-blimp passed and the 10K-checkpoint BLiMP
falls below it, the rest of the training is skipped (saves 75% of compute on a
clearly-bad architecture).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from component_fab.generator.code_generator import generate_module
from component_fab.generator.block_templates import (
    GatedParallelBlock,
    HeteroMoEBlock,
    RecursiveDepthRouterBlock,
    ThreeLaneAdaptive,
)
from component_fab.generator.primitive_templates import (
    LinearStateSpaceLane,
    MultiscaleWaveletLane,
    PhaseLockAttention,
    ReciprocalRankAttention,
    SparsemaxAttention,
    TropicalAttention,
)
from component_fab.harness.binding_tests_v2 import (
    test_dyck2_v3,
    test_npi_synthetic_v2,
    test_selective_copy,
    test_variable_delay_repeat,
)
from component_fab.harness.tiny_lm import (
    TinyLM,
    TinyLMConfig,
    count_trainable_params,
    lane_factory_for_baseline,
)
from research.defaults import VOCAB_SIZE
from research.eval.blimp_eval import evaluate_blimp
from research.eval.utils import tokenize_file
from research.eval.wikitext_eval import _download_wikitext


# Param sizing presets matching the user's 30M/60M/120M targets at vocab=100K.
PARAM_SIZING: dict[str, dict[str, int]] = {
    "30M": {"dim": 256, "n_blocks": 6},
    "60M": {"dim": 448, "n_blocks": 8},
    "120M": {"dim": 640, "n_blocks": 12},
}

_REPO = Path(__file__).resolve().parents[2]
_SAVED_WINNERS_PATH = _REPO / "component_fab" / "catalog" / "saved_winners.json"
_DEFAULT_PROPOSAL_ID = "improve_tropical_gate_block_gated_parallel_84f0ccd08a"


class _SimplifiedMambaLane(nn.Module):
    """Selective state-space lane (simplified Mamba/S6).

    Each token computes its OWN dt (time step), A (state-transition), B (input
    projection) — i.e., selectivity is content-dependent. Unlike
    ``LinearStateSpaceLane`` which has fixed-per-channel transition matrices,
    this lane lets each token gate state updates based on its own embedding.
    Critically: the dt-gating is the load-bearing part of why Mamba beats
    linear SSMs on BLiMP-like tasks.

    Implementation note: this is a simplified scalar-state version (not the
    full Mamba complex state). Sufficient for binding/induction comparisons
    but not a faithful Mamba reproduction.
    """

    def __init__(self, dim: int, state_dim: int | None = None) -> None:
        super().__init__()
        state_dim = state_dim or dim
        self.dim = dim
        self.state_dim = state_dim
        # Content-dependent dt projection — the selectivity mechanism.
        self.dt_proj = nn.Linear(dim, state_dim)
        # Content-dependent B (input → state) projection.
        self.B_proj = nn.Linear(dim, state_dim)
        # Static A (per-state-dim decay rate, like Mamba's per-channel a).
        self.a_log = nn.Parameter(torch.zeros(state_dim))
        # Output projection.
        self.C_proj = nn.Linear(dim, state_dim)
        self.out_proj = nn.Linear(state_dim, dim, bias=False)
        # Gate (Mamba's SiLU gate over the SSM output).
        self.gate_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-token dt (positive via softplus).
        dt = torch.nn.functional.softplus(self.dt_proj(x))  # [B, L, state]
        # Per-token B (input gate).
        B = self.B_proj(x)  # [B, L, state]
        # Per-token C (output projection).
        C = self.C_proj(x)  # [B, L, state]
        # Discrete-time recurrence with content-dependent transitions.
        a = -torch.exp(self.a_log)  # negative for stability (real Mamba uses neg eig)
        # Parallel associative (Kogge-Stone) scan over h[t] = exp(dt*a) * h[t-1] + dt*B.
        # log-domain: log_a_bar[t] = dt[t] * a; b[t] = dt[t] * B[t].
        # Existing scan expects [..., L] with last dim = sequence, so transpose state/L axes.
        from research.synthesis.compiler_ops_sequence import _parallel_associative_scan

        log_a_bar = (dt * a).transpose(-1, -2).contiguous()  # [B, state, L]
        b_bar = (dt * B).transpose(-1, -2).contiguous()  # [B, state, L]
        h_t = _parallel_associative_scan(log_a_bar, b_bar)  # [B, state, L]
        h = h_t.transpose(-1, -2)  # [B, L, state]
        y_scalar = (C * h).sum(dim=-1, keepdim=True)  # [B, L, 1]
        y_seq = y_scalar.expand(-1, -1, self.dim)  # [B, L, dim]
        ssm_out = self.out_proj(y_seq)
        gate = torch.nn.functional.silu(self.gate_proj(x))
        return ssm_out * gate


class AdjacentTokenMergeLane(nn.Module):
    """Binding-specialist lane built on the ``adjacent_token_merge`` primitive.

    The primitive (``compiler_ops_routing._op_adjacent_token_merge``) is a
    parameter-free, strictly-causal token compressor: it merges even-stride
    tokens into their predecessor and then restores the original seq_len via a
    nearest-kept mapping (``(B,S,D) -> (B,S,D)``). On its own it carries no
    learnable capacity, so we surround it with learnable in/out projections and
    a sigmoid content gate to make it a trainable mixer.

    As a bare mixer it relies on the outer ``_LaneBlock`` for the
    norm/residual/FFN structure (the merge is information-destructive and
    ``REQUIRES_RESIDUAL_BYPASS`` — the outer ``x + lane(norm1(x))`` supplies it).

    The merge op has data-dependent control flow (``searchsorted``/dynamic
    ``arange``) and mutates a ``routing_telemetry`` dict on ``self`` every call,
    both of which poison the dynamo recompile cache, so it is isolated via
    ``@torch._dynamo.disable`` (see [[feedback_dynamo_disable_synthesis_forwards]]).
    It is also forced to fp32 to avoid the bf16 ``scatter_add`` dtype mismatch
    under AMP.
    """

    def __init__(self, dim: int, keep_frac: float = 0.5) -> None:
        super().__init__()
        self.dim = dim
        self.keep_frac = float(keep_frac)
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    @torch._dynamo.disable
    def _merge(self, h: torch.Tensor) -> torch.Tensor:
        from research.synthesis.compiler_ops_routing import _op_adjacent_token_merge

        seq_len = h.shape[1]
        n_keep = max(1, int(round(seq_len * self.keep_frac)))
        with torch.autocast(device_type=h.device.type, enabled=False):
            return _op_adjacent_token_merge(self, [h.float()], {"n_keep": n_keep})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = self._merge(self.in_proj(x)).to(x.dtype)
        gate = torch.sigmoid(self.gate(x))
        return self.out_proj(merged) * gate


# 2026-05-28: nano-scale BLiMP winners (seq=512 sweet-spot study) — lane name →
# (graphs-table fingerprint, description). Single source of truth shared by the
# lane factory below and `tools/apply_mixer_fingerprint_pretrain.py`.
WINNER_LANE_FINGERPRINTS: dict[str, tuple[str, str]] = {
    "pq_rope_winner": (
        "9be78a43c07948c4",  # pragma: allowlist secret
        "pq_embedding_moe_block_rope (BLiMP 0.5543 @ 10k seq512)",
    ),
    "semiring_winner": (
        "a2f747a20982907a",  # pragma: allowlist secret
        "learnable_semiring_attention_block (BLiMP 0.5651 @ 10k seq512)",
    ),
}


def _build_lane_factory(
    name: str, top_k_frac: float = 0.25
) -> Callable[[int], nn.Module]:
    """Resolve a lane-name to a lane factory for scaling tests."""
    # guardrail: allow-god-function
    if name in ("softmax_ffn", "softmax_attention"):
        return lane_factory_for_baseline("softmax_attention")
    if name == "tropical_attention":
        return TropicalAttention
    if name == "sparsemax_attention":
        return SparsemaxAttention
    if name == "simplified_mamba":
        return _SimplifiedMambaLane
    if name == "multiscale_wavelet":
        return MultiscaleWaveletLane
    if name == "linear_ssm":
        return LinearStateSpaceLane
    if name == "causal_conv":
        return lane_factory_for_baseline("causal_conv")
    if name == "adjacent_token_merge_lane":
        return AdjacentTokenMergeLane

    # Novel mixer lanes ported from synthesis ops (AR-gate 1.0, top nano BLiMP),
    # now with RoPE-capable QKV base for scaling tests.
    if name == "reciprocal_rank_attention":
        return lambda d: ReciprocalRankAttention(d, use_rope=True)
    if name == "phase_lock_attention":
        return lambda d: PhaseLockAttention(d, use_rope=True)

    if name == "reciprocal_phase_two_lane":
        # Both induction-strong novel lanes only (no tropical/sparsemax), to test
        # whether a 2-lane preserves the standout nano_induction_nearest that the
        # 3-lane dilutes (0.44 single -> 0.29 in reciprocal_phase_tropical).
        def factory(dim: int) -> nn.Module:
            return GatedParallelBlock(
                lambda d: ReciprocalRankAttention(d, use_rope=True),
                lambda d: PhaseLockAttention(d, use_rope=True),
                dim,
            )

        return factory

    if name == "reciprocal_phase_tropical_three_lane":

        def factory(dim: int) -> nn.Module:
            return ThreeLaneAdaptive(
                lambda d: ReciprocalRankAttention(d, use_rope=True),
                lambda d: PhaseLockAttention(d, use_rope=True),
                lambda d: TropicalAttention(d),
                dim,
            )

        return factory

    if name == "reciprocal_sparsemax_wavelet_three_lane":

        def factory(dim: int) -> nn.Module:
            return ThreeLaneAdaptive(
                lambda d: ReciprocalRankAttention(d, use_rope=True),
                lambda d: SparsemaxAttention(d),
                lambda d: MultiscaleWaveletLane(d),
                dim,
            )

        return factory

    if name == "tropical_sparsemax_wavelet_three_lane":

        def factory(dim: int) -> nn.Module:
            return ThreeLaneAdaptive(
                lambda d: TropicalAttention(d),
                lambda d: SparsemaxAttention(d),
                lambda d: MultiscaleWaveletLane(d),
                dim,
            )

        return factory

    if name == "tropical_sparsemax_two_lane":
        # 2026-05-19: 3-lane sublane ablation showed wavelet alone broken
        # on every structural probe; tropical+sparsemax recover the full
        # hybrid capability. This composite drops wavelet to test whether
        # the 2-lane is simpler-and-better.
        def factory(dim: int) -> nn.Module:
            def lane_a(d):
                return TropicalAttention(d)

            def lane_b(d):
                return SparsemaxAttention(d)

            return GatedParallelBlock(lane_a, lane_b, dim)

        return factory

    if name == "top_ar_block":
        # 2026-05-19: faithful reproduction of fp 7fb0412ec57a1213 (the
        # leaderboard-best AR-curriculum scorer at 0.9046, ~13M params, 1000
        # wikitext steps). Dual mixer with conv1d_seq + swiglu between the
        # two attentions, 3-way residual to original input.
        from component_fab.harness.top_ar_block import (
            TopArchBlock,
            LocalWindowAttention,
        )

        def factory(dim: int) -> nn.Module:
            def mixer_a(d):
                return TropicalAttention(d)

            def mixer_b(d):
                return LocalWindowAttention(d, window_size=16)

            return TopArchBlock(dim, mixer_a, mixer_b)

        return factory

    if name == "top_ar_block_with_two_lane":
        # 2026-05-19: same scaffold as top_ar_block but mixer_a (the
        # tropical_attention slot) is replaced with our 2-lane composite
        # (GatedParallelBlock of tropical + sparsemax). Tests whether the
        # 2-lane is a productive substitution inside the AR-friendly scaffold.
        from component_fab.harness.top_ar_block import (
            TopArchBlock,
            LocalWindowAttention,
        )

        def factory(dim: int) -> nn.Module:
            def two_lane(d):
                def la(dd):
                    return TropicalAttention(dd)

                def lb(dd):
                    return SparsemaxAttention(dd)

                return GatedParallelBlock(la, lb, d)

            def mixer_b(d):
                return LocalWindowAttention(d, window_size=16)

            return TopArchBlock(dim, two_lane, mixer_b)

        return factory

    if name == "block_gated_parallel":

        def factory(dim: int) -> nn.Module:
            def anchor(d):
                return TropicalAttention(d)

            def wavelet(d):
                return MultiscaleWaveletLane(d)

            return GatedParallelBlock(anchor, wavelet, dim)

        return factory

    if name == "recursive_depth_router":

        def factory(dim: int) -> nn.Module:
            def anchor(d):
                return TropicalAttention(d)

            return RecursiveDepthRouterBlock(anchor, dim, max_depth=4)

        return factory

    if name == "hetero_moe_block":

        def factory(dim: int) -> nn.Module:
            def anchor(d):
                return TropicalAttention(d)

            return HeteroMoEBlock(
                anchor,
                (LinearStateSpaceLane, MultiscaleWaveletLane, TropicalAttention),
                dim,
            )

        return factory

    if name.startswith("ensemble_top_ar_") or name == "ensemble_top_ar_plus_three_lane":
        # 2026-05-19: parallel-sum ensemble lanes built from the top AR-curriculum
        # graphs in runs.db. Lazy-import the factory builder to avoid a
        # module-import-time dependency on sqlite/synthesis paths. Names:
        #   ensemble_top_ar_Nway (N in 1..4) — equal-weight parallel-sum of top-N
        #   ensemble_top_ar_plus_three_lane — top-4 graphs sum + ThreeLaneAsBlock
        # All ensemble lanes have internal norm/FFN/residual via the underlying
        # graphs — pair with mixer_fingerprint via `--no-ffn` if the TinyLM
        # outer FFN must be skipped; see [[feedback_rope_or_pe_required]] for
        # the screening recipe.
        from research.tools.ensemble_screening import (
            _load_top_graphs,
            _make_ensemble_lane_factory,
            _make_ensemble_plus_three_lane_factory,
        )

        if name == "ensemble_top_ar_plus_three_lane":
            specs = _load_top_graphs(4)
            return _make_ensemble_plus_three_lane_factory(specs)
        suffix = name[len("ensemble_top_ar_") :]
        if not suffix.endswith("way"):
            raise ValueError(f"expected '_Nway' suffix in {name}")
        n = int(suffix[: -len("way")])
        specs = _load_top_graphs(n)
        return _make_ensemble_lane_factory(specs)

    if name == "local_ssm_diff":
        # 2026-05-21 Phase-1 cross-bias mining: the only rows with both
        # binding_intermediate > 0.7, ar_curriculum > 0.4, and induction=1.0
        # were local_attn_ssm_hybrid variants with local_window_attn +
        # conv1d_seq + selective_scan + diff_attention.
        from research.tools.ensemble_screening import (
            _load_graphs_by_fingerprint,
            _make_ensemble_lane_factory,
        )

        specs = _load_graphs_by_fingerprint(
            (
                (
                    "bb0b8d5856da1f29",  # pragma: allowlist secret
                    "local_window + conv + selective_scan + diff_attention",
                    0.7975,
                ),
                (
                    "5c5013c79d1f0a51",  # pragma: allowlist secret
                    "local_window + conv + selective_scan + diff_attention alt",
                    0.4069,
                ),
            )
        )
        return _make_ensemble_lane_factory(specs)

    if name == "routed_compress":
        # 2026-05-21 Phase-1 cross-bias mining: cluster 2 — graphs that clear
        # binding_intermediate > 0.7 AND induction_intermediate > 0.5 AND
        # ar_curriculum > 0.3, sharing the `latent_compress + difficulty_routed
        # + routed_bottleneck` templates. Ops: softmax_attention +
        # token_type_classifier + entropy_score + rope_rotate + spectral_filter
        # (+ latent_attention_compressor for one variant).
        from research.tools.ensemble_screening import (
            ROUTED_COMPRESS_FPS,
            _load_graphs_by_fingerprint,
            _make_ensemble_lane_factory,
        )

        return _make_ensemble_lane_factory(
            _load_graphs_by_fingerprint(ROUTED_COMPRESS_FPS)
        )

    if name == "local_ssm_diff_rope":
        # 2026-05-21: controlled ablation of `local_ssm_diff` with `rope_rotate`
        # nodes injected before every `local_window_attn` / `diff_attention`
        # node in the cluster-1 graphs. The function-based `_op_local_window_attn`
        # is invisible to `_attach_rope_to_attention` (which walks nn.Module
        # subclasses), so the bare `local_ssm_diff` lane has no Q/K positional
        # signal. Caveat: the 2026-05-21 RoPE-coverage audit of past
        # mixer_fingerprint runs found failing AR spread evenly across all
        # RoPE cohorts — this variant is a controlled probe, not a fix.
        from research.tools.ensemble_screening import (
            CROSS_BIAS_FPS,
            _inject_rope_before_ops,
            _load_graphs_by_fingerprint,
            _make_ensemble_lane_factory,
        )

        specs = _load_graphs_by_fingerprint(CROSS_BIAS_FPS)
        roped = [
            (fp, desc, db_auc, _inject_rope_before_ops(g))
            for fp, desc, db_auc, g in specs
        ]
        return _make_ensemble_lane_factory(roped)

    if name in ("pq_rope_winner", "semiring_winner"):
        # 2026-05-28: nano-scale BLiMP winners (seq=512 sweet-spot study). Each
        # is a single synthesized block reconstructed from its graph_json in the
        # `graphs` dedup table (these were screened-only, never promoted to a
        # program_results row). Wrapped as a 1-branch ensemble = the block
        # itself (its own norm/residual built in → pair with mixer_fingerprint
        # `--no-ffn`). Powers the from-scratch Chinchilla pretrain.
        from research.tools.ensemble_screening import (
            _load_graphs_from_graphs_table,
            _make_ensemble_lane_factory,
        )

        fp, desc = WINNER_LANE_FINGERPRINTS[name]
        specs = _load_graphs_from_graphs_table(((fp, desc, 0.0),))
        return _make_ensemble_lane_factory(specs)

    raise ValueError(f"unknown lane name: {name}")


def _saved_winner_factory(
    proposal_id_or_name: str, top_k_frac: float = 0.25
) -> tuple[str, Callable[[int], nn.Module], dict]:
    """Resolve a pinned fab winner to the exact code-generator path."""
    data = json.loads(_SAVED_WINNERS_PATH.read_text(encoding="utf-8"))
    candidates = [
        row
        for row in data.get("winners", [])
        if row.get("proposal_id") == proposal_id_or_name
        or row.get("name") == proposal_id_or_name
        or str(row.get("proposal_id") or "").startswith(proposal_id_or_name)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"saved winner '{proposal_id_or_name}' resolved to {len(candidates)} rows"
        )
    row = candidates[0]
    axes = dict(row["math_axes"])
    label = str(row.get("proposal_id") or row.get("name") or proposal_id_or_name)

    def factory(dim: int) -> nn.Module:
        return generate_module(axes, dim=dim, top_k_frac=top_k_frac)

    return label, factory, axes


def _append_jsonl(path: Path | None, row: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str, sort_keys=True) + "\n")


def _rng_state_payload() -> dict:
    payload: dict = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng_state(payload: dict | None) -> None:
    if not isinstance(payload, dict):
        return
    torch_state = payload.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = payload.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _save_training_checkpoint(
    *,
    checkpoint_dir: Path | None,
    label: str,
    step: int,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    metadata: dict,
) -> str | None:
    if checkpoint_dir is None:
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
    path = checkpoint_dir / f"{safe_label}_step_{step:06d}.pt"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "rng_state": _rng_state_payload(),
            "metadata": metadata,
        },
        tmp_path,
    )
    tmp_path.replace(path)
    return str(path)


def _load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    device: str,
) -> dict:
    payload = torch.load(path, map_location=device)  # nosec B614 - locally-produced checkpoint, not network-sourced
    model.load_state_dict(payload["model_state_dict"])
    optim.load_state_dict(payload["optimizer_state_dict"])
    _restore_rng_state(payload.get("rng_state"))
    return payload


def _ensure_wikitext_paths(
    variant: str, max_chars_train: int, max_chars_val: int
) -> tuple[Path, Path]:
    """Force-refresh stale tiny WikiText cache files before long runs."""
    cache_dir = Path.home() / ".cache" / "aria" / "wikitext" / variant
    train_path = cache_dir / "train.txt"
    val_path = cache_dir / "validation.txt"
    requested = ((train_path, max_chars_train), (val_path, max_chars_val))
    for path, max_chars in requested:
        if max_chars <= 0 or not path.exists():
            continue
        # Existing cache files predate this runner and may be tiny 2MB slices.
        min_expected = int(max_chars * 0.8)
        if path.stat().st_size < min_expected:
            path.unlink()
    return _download_wikitext(variant, max_chars_train, max_chars_val)


def _load_wikitext_tokens(
    *,
    variant: str,
    vocab_size: int,
    max_chars_train: int,
    max_chars_val: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    train_path, val_path = _ensure_wikitext_paths(
        variant, max_chars_train, max_chars_val
    )
    train_tokens = torch.as_tensor(
        tokenize_file(train_path, vocab_size), dtype=torch.long
    ).contiguous()
    val_tokens = torch.as_tensor(
        tokenize_file(val_path, vocab_size), dtype=torch.long
    ).contiguous()
    return train_tokens, val_tokens, int(train_tokens.numel()), int(val_tokens.numel())


class _RandomWindowBatcher:
    """Sample fresh random token windows without materializing a huge batch cache."""

    def __init__(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        device: str,
        seed: int,
    ) -> None:
        if int(tokens.numel()) < seq_len + 2:
            raise ValueError("not enough tokens for requested sequence length")
        self.device = torch.device(device)
        self.tokens = tokens.to(self.device).contiguous()
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        # Sample starts on the same device as tokens — keeps the gather GPU-side
        # and removes the per-step H2D transfer that dominated old data wait time.
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self._offsets_cache: dict[int, torch.Tensor] = {
            self.seq_len: torch.arange(
                self.seq_len, dtype=torch.long, device=self.device
            ).unsqueeze(0)
        }
        self.max_start = int(self.tokens.numel()) - self.seq_len - 1

    def _offsets_for(self, seq_len: int) -> torch.Tensor:
        seq_len = int(seq_len)
        offsets = self._offsets_cache.get(seq_len)
        if offsets is None:
            offsets = torch.arange(
                seq_len, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            self._offsets_cache[seq_len] = offsets
        return offsets

    def next(self, seq_len: int | None = None) -> torch.Tensor:
        seq_len = self.seq_len if seq_len is None else int(seq_len)
        if seq_len <= 0 or seq_len > self.seq_len:
            raise ValueError(f"seq_len must be in [1, {self.seq_len}], got {seq_len}")
        max_start = int(self.tokens.numel()) - seq_len - 1
        if max_start <= 0:
            raise ValueError("not enough tokens for requested sequence length")
        starts = torch.randint(
            0,
            max_start,
            (self.batch_size, 1),
            generator=self.generator,
            device=self.device,
        )
        return self.tokens[starts + self._offsets_for(seq_len)]

    def fixed_batches(self, n_batches: int) -> list[torch.Tensor]:
        return [self.next() for _ in range(int(n_batches))]


def _gpt2_style_init(model: nn.Module, n_blocks: int) -> None:
    """GPT-2 style initialization. Fixes the dim=256 explosion observed
    at TinyLM's default ``nn.Embedding`` init = Normal(0, 1), which at
    larger dim produces per-vector norms (sqrt(dim) * 1) too large for
    the tied lm_head, causing loss=170 at step 1 (vs expected ~11.5 =
    ln(vocab)).

    Reference: Radford et al. 2019 (GPT-2). All linear/embedding layers
    Normal(0, 0.02). Residual-stream output projections scaled
    1/sqrt(2 * n_blocks) (Karpathy nano-gpt convention).
    """
    init_std = 0.02
    scaled_init_std = init_std / (2 * n_blocks) ** 0.5
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Residual-stream output projections get scaled init.
            is_resid_proj = (
                name.endswith(".fc2")  # MLP output projection
                or name.endswith(".out")  # common 'out' naming in lanes
                or name.endswith(".out_proj")  # mamba-style
            )
            std = scaled_init_std if is_resid_proj else init_std
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def _build_tinylm(
    lane_factory: Callable[[int], nn.Module],
    *,
    dim: int,
    n_blocks: int,
    vocab_size: int = VOCAB_SIZE,
    max_seq_len: int = 1024,
    use_ffn: bool = True,
    use_rope: bool = True,
    use_position_embedding: bool = False,
) -> TinyLM:
    """Build a TinyLM with RoPE-enabled attention by default.

    Defaults flipped on 2026-05-18 after the seq_len=256 cap was identified
    as the cause of CUDA gather OOB in the hard probes. RoPE lets attention
    lanes accept seq_len up to ``max_seq_len`` without retraining a larger
    absolute embedding. ``use_position_embedding=True`` is retained for
    loading legacy pre-RoPE checkpoints only.
    """
    cfg = TinyLMConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        use_position_embedding=use_position_embedding,
        use_rope=use_rope,
        max_seq_len=max_seq_len,
        use_ffn=use_ffn,
        ffn_mult=4,
    )
    model = TinyLM(lane_factory, cfg)
    if use_rope:
        _attach_rope_to_attention(model, max_seq_len=max_seq_len)
    _gpt2_style_init(model, n_blocks)
    return model


def _attach_rope_to_attention(model: nn.Module, *, max_seq_len: int) -> None:
    """Walk ``model`` and attach a ``RotaryEmbedding`` to every attention lane.

    Composite lanes (ThreeLaneAdaptive, GatedParallelBlock, etc.) construct
    inner ``TropicalAttention`` / ``SparsemaxAttention`` / ``SoftmaxCausalAttention``
    via hardcoded factories that don't know about RoPE — we retrofit them
    post-construction instead of threading ``use_rope`` through every factory.
    """
    from component_fab.harness.rope import RotaryEmbedding
    from component_fab.harness.tiny_lm import SoftmaxCausalAttention

    attention_types = (SoftmaxCausalAttention, TropicalAttention, SparsemaxAttention)
    for module in model.modules():
        if (
            isinstance(module, attention_types)
            and getattr(module, "rope", None) is None
        ):
            module.rope = RotaryEmbedding(module.dim, max_seq_len=max_seq_len)


def _causal_lm_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
    )


def _eval_ppl(model: TinyLM, batches: list[torch.Tensor]) -> float:
    if not batches:
        return float("nan")
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in batches:
            logits = model(batch)
            total_loss += float(_causal_lm_loss(logits, batch).item())
            n += 1
    return float(torch.exp(torch.tensor(total_loss / max(1, n))).item())


def _eval_v2_at_dim(
    model_lane_factory: Callable[[int], nn.Module], dim: int
) -> dict[str, float]:
    """Quality-settings v2 binding-test suite at the same dim as the trained
    model. Runs the four discriminative tests: selective_copy,
    variable_delay, dyck_2_v3 (deep-prefix+noise), npi_synthetic_v2
    (long-distance licensor). Each does its own small training; measures
    architectural capability independent of the wikitext-trained weights.
    """
    sc = test_selective_copy(
        model_lane_factory, dim=dim, n_blocks=2, n_train_steps=300, seed=0
    )
    vd_dict, vd_mean = test_variable_delay_repeat(
        model_lane_factory, dim=dim, n_blocks=2, n_train_steps=300, seed=0
    )
    dyck = test_dyck2_v3(
        model_lane_factory, dim=dim, n_blocks=2, n_train_steps=300, seed=0
    )
    npi = test_npi_synthetic_v2(
        model_lane_factory, dim=dim, n_blocks=2, n_train_steps=300, seed=0
    )
    return {
        "v2_sc": sc,
        "v2_vd_mean": vd_mean,
        "v2_vd_per_delay": vd_dict,
        "v2_dyck_v3": dyck,
        "v2_npi_v2": npi,
    }


def _make_lr_schedule(
    base_lr: float, warmup_steps: int, n_train_steps_total: int, final_lr_frac: float
) -> Callable[[int], float]:
    def _lr_at(s: int) -> float:
        if s < warmup_steps:
            return base_lr * (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(1, n_train_steps_total - warmup_steps)
        return base_lr * (
            final_lr_frac
            + (1 - final_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
        )

    return _lr_at


def _eval_checkpoint(
    model,
    val_batches,
    factory,
    dim: int,
    *,
    target: int,
    step: int,
    last_loss_val: float,
    last_grad_norm: float,
    cur_lr: float,
    best_ppl: float | None,
    best_ppl_step: int | None,
    started: float,
    blimp_n_per_subtask: int,
    seq_len: int,
    device: str,
) -> tuple[dict, float, int]:
    ckpt_t0 = time.monotonic()
    post_ppl = _eval_ppl(model, val_batches)
    if best_ppl is None or post_ppl < best_ppl:
        best_ppl = post_ppl
        best_ppl_step = step
    blimp = evaluate_blimp(
        model,
        vocab_size=VOCAB_SIZE,
        device=device,
        n_per_subtask=blimp_n_per_subtask,
        max_seq_len=seq_len,
    )
    v2 = _eval_v2_at_dim(factory, dim)
    ck = {
        "checkpoint_step": target,
        "actual_step": step,
        "train_loss": last_loss_val,
        "grad_norm": last_grad_norm,
        "lr": cur_lr,
        "post_train_ppl": post_ppl,
        "best_ppl": best_ppl,
        "best_ppl_step": best_ppl_step,
        "blimp_overall": float(blimp.overall_accuracy or 0),
        "blimp_status": str(blimp.status or ""),
        "blimp_by_subtask": dict(blimp.subtask_accuracies or {}),
        "v2_sc": v2["v2_sc"],
        "v2_vd_mean": v2["v2_vd_mean"],
        "v2_vd_per_delay": v2["v2_vd_per_delay"],
        "v2_dyck_v3": v2["v2_dyck_v3"],
        "v2_npi_v2": v2["v2_npi_v2"],
        "eval_elapsed_s": round(time.monotonic() - ckpt_t0, 1),
        "wall_clock_s": round(time.monotonic() - started, 1),
    }
    return ck, best_ppl, best_ppl_step


def _train_step(
    model,
    optim,
    batch,
    grad_clip_max_norm: float,
) -> tuple[float, float, str | None]:
    logits = model(batch)
    loss = _causal_lm_loss(logits, batch)
    if not torch.isfinite(loss):
        return 0.0, 0.0, "nonfinite_loss"
    optim.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
    if torch.is_tensor(grad_norm) and not torch.isfinite(grad_norm):
        return 0.0, 0.0, "nonfinite_grad"
    optim.step()
    last_loss_val = float(loss.item())
    last_grad_norm = float(
        grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
    )
    return last_loss_val, last_grad_norm, None


def run_scaling_cell(
    lane_name: str | None,
    size: str,
    *,
    proposal_id: str | None = None,
    n_train_steps_total: int,
    checkpoint_steps: tuple[int, ...],
    batch_size: int = 8,
    seq_len: int = 256,
    n_eval_batches: int = 32,
    max_chars_train: int = 200_000_000,
    max_chars_val: int = 2_000_000,
    blimp_n_per_subtask: int = 25,
    learning_rate: float = 1e-4,
    early_stop_blimp: float | None = None,
    ppl_stop_factor: float = 2.0,
    max_epoch_equivalent: float = 50.0,
    grad_spike_threshold: float = 10.0,
    grad_spike_patience: int = 5,
    checkpoint_log: Path | None = None,
    device: str = "cuda",
    quiet: bool = False,
) -> dict:
    """One model × one size × full training to n_train_steps_total with
    checkpoints. Returns the checkpoint results."""
    # guardrail: allow-god-function
    if size not in PARAM_SIZING:
        raise ValueError(f"unknown size: {size}")
    dim = PARAM_SIZING[size]["dim"]
    n_blocks = PARAM_SIZING[size]["n_blocks"]
    axes: dict | None = None
    if proposal_id:
        run_label, factory, axes = _saved_winner_factory(proposal_id)
        lane_label = run_label
    elif lane_name:
        factory = _build_lane_factory(lane_name)
        lane_label = lane_name
    else:
        raise ValueError("either lane_name or proposal_id is required")

    if not quiet:
        print(
            f"\n=== {lane_label} @ {size} (dim={dim}, n_blocks={n_blocks}) ===",
            flush=True,
        )

    model = _build_tinylm(factory, dim=dim, n_blocks=n_blocks)
    model = model.to(device)
    n_params = count_trainable_params(model)
    if not quiet:
        print(f"  params: {n_params / 1e6:.1f}M", flush=True)

    train_tokens, val_tokens, n_train_tokens, n_val_tokens = _load_wikitext_tokens(
        variant="wikitext-103-raw-v1",
        vocab_size=VOCAB_SIZE,
        max_chars_train=max_chars_train,
        max_chars_val=max_chars_val,
    )
    token_visits = int(n_train_steps_total) * int(batch_size) * int(seq_len)
    epoch_equivalent = token_visits / max(1, n_train_tokens)
    if epoch_equivalent > max_epoch_equivalent:
        return {
            "status": "refused_excessive_data_reuse",
            "lane": lane_label,
            "proposal_id": proposal_id,
            "size": size,
            "train_tokens": n_train_tokens,
            "planned_token_visits": token_visits,
            "epoch_equivalent": epoch_equivalent,
            "max_epoch_equivalent": max_epoch_equivalent,
        }

    train_batcher = _RandomWindowBatcher(
        train_tokens, batch_size=batch_size, seq_len=seq_len, device=device, seed=42
    )
    val_batcher = _RandomWindowBatcher(
        val_tokens, batch_size=batch_size, seq_len=seq_len, device=device, seed=123
    )
    val_batches = val_batcher.fixed_batches(n_eval_batches)

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    checkpoints: list[dict] = []
    started = time.monotonic()

    pre_ppl = _eval_ppl(model, val_batches)
    metadata = {
        "event": "start",
        "status": "running",
        "lane": lane_label,
        "proposal_id": proposal_id,
        "math_axes": axes,
        "size": size,
        "dim": dim,
        "n_blocks": n_blocks,
        "n_params": n_params,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "n_train_steps_total": n_train_steps_total,
        "checkpoint_steps": list(sorted(checkpoint_steps)),
        "train_tokens": n_train_tokens,
        "val_tokens": n_val_tokens,
        "planned_token_visits": token_visits,
        "epoch_equivalent": round(epoch_equivalent, 4),
        "pre_ppl": pre_ppl,
        "ppl_stop_factor": ppl_stop_factor,
    }
    _append_jsonl(checkpoint_log, metadata)
    if not quiet:
        print(
            f"  pre-train PPL: {pre_ppl:.1f}; train_tokens={n_train_tokens:,}; "
            f"planned_token_visits={token_visits:,}; epochs≈{epoch_equivalent:.2f}",
            flush=True,
        )

    step = 0
    next_ckpt_idx = 0
    sorted_ckpts = sorted(checkpoint_steps)
    _lr_at = _make_lr_schedule(
        base_lr=learning_rate,
        warmup_steps=1000,
        n_train_steps_total=n_train_steps_total,
        final_lr_frac=0.1,
    )
    grad_clip_max_norm = 1.0  # Standard transformer gradient clipping.

    model.train()
    loss_history: list[tuple[int, float, float]] = []  # (step, loss, grad_norm)
    log_every = 500  # Print per-step diagnostics every N steps for early monitoring.
    best_ppl: float | None = None
    best_ppl_step: int | None = None
    last_loss_val = float("nan")
    last_grad_norm = float("nan")
    grad_spikes = 0
    stop_reason: str | None = None
    while step < n_train_steps_total:
        cur_lr = _lr_at(step)
        for pg in optim.param_groups:
            pg["lr"] = cur_lr
        batch = train_batcher.next()
        last_loss_val, last_grad_norm, step_err = _train_step(
            model, optim, batch, grad_clip_max_norm
        )
        if step_err is not None:
            stop_reason = f"{step_err}_step_{step + 1}"
            break
        step += 1
        if last_grad_norm > grad_spike_threshold:
            grad_spikes += 1
        else:
            grad_spikes = 0
        if grad_spikes >= grad_spike_patience:
            stop_reason = f"grad_spike>{grad_spike_threshold}_for_{grad_spikes}_steps"
            break
        # Diagnostic logging — captures the early loss trajectory + grad
        # health so we can SEE if the architecture is exploding before
        # the first checkpoint at step 10K.
        if not quiet and (step <= 50 or step % log_every == 0):
            loss_history.append((step, last_loss_val, last_grad_norm))
            print(
                f"  step {step:>6d} lr={cur_lr:.2e} loss={last_loss_val:>7.3f} "
                f"grad_norm={last_grad_norm:>7.3f} ppl≈{math.exp(min(last_loss_val, 30)):.1f}",
                flush=True,
            )
        # Hit a checkpoint?
        if next_ckpt_idx < len(sorted_ckpts) and step >= sorted_ckpts[next_ckpt_idx]:
            target = sorted_ckpts[next_ckpt_idx]
            ck, best_ppl, best_ppl_step = _eval_checkpoint(
                model,
                val_batches,
                factory,
                dim,
                target=target,
                step=step,
                last_loss_val=last_loss_val,
                last_grad_norm=last_grad_norm,
                cur_lr=cur_lr,
                best_ppl=best_ppl,
                best_ppl_step=best_ppl_step,
                started=started,
                blimp_n_per_subtask=blimp_n_per_subtask,
                seq_len=seq_len,
                device=device,
            )
            post_ppl = ck["post_train_ppl"]
            if best_ppl is not None and post_ppl > best_ppl * ppl_stop_factor:
                ck["early_stop_reason"] = (
                    f"val_ppl {post_ppl:.1f} > {ppl_stop_factor:.2f}x "
                    f"best {best_ppl:.1f} at step {best_ppl_step}"
                )
                stop_reason = ck["early_stop_reason"]
            checkpoints.append(ck)
            _append_jsonl(checkpoint_log, {"event": "checkpoint", **ck})
            if not quiet:
                print(
                    f"  ckpt @ step {step:>6d}: ppl={post_ppl:>7.1f} "
                    f"BLiMP={ck['blimp_overall']:.4f} v2_sc={ck['v2_sc']:.3f} "
                    f"v2_vd={ck['v2_vd_mean']:.3f} dyck={ck['v2_dyck_v3']:.3f} "
                    f"npi={ck['v2_npi_v2']:.3f} eval_t={ck['eval_elapsed_s']:.0f}s "
                    f"total_t={ck['wall_clock_s']:.0f}s",
                    flush=True,
                )
            if stop_reason is not None:
                if not quiet:
                    print(f"  EARLY-STOP: {stop_reason}", flush=True)
                break
            # Early-stop rule.
            if early_stop_blimp is not None and ck["blimp_overall"] < early_stop_blimp:
                if not quiet:
                    print(
                        f"  EARLY-STOP: BLiMP {ck['blimp_overall']:.4f} < "
                        f"{early_stop_blimp:.4f} at step {step}; halting this cell.",
                        flush=True,
                    )
                return {
                    "status": "early_stopped",
                    "lane": lane_label,
                    "proposal_id": proposal_id,
                    "size": size,
                    "dim": dim,
                    "n_blocks": n_blocks,
                    "n_params": n_params,
                    "pre_ppl": pre_ppl,
                    "checkpoints": checkpoints,
                    "early_stop_at_step": step,
                    "early_stop_blimp_threshold": early_stop_blimp,
                }
            next_ckpt_idx += 1
            model.train()
        if next_ckpt_idx >= len(sorted_ckpts):
            break

    if stop_reason is not None:
        result = {
            "status": "early_stopped",
            "stop_reason": stop_reason,
            "lane": lane_label,
            "proposal_id": proposal_id,
            "math_axes": axes,
            "size": size,
            "dim": dim,
            "n_blocks": n_blocks,
            "n_params": n_params,
            "pre_ppl": pre_ppl,
            "train_tokens": n_train_tokens,
            "val_tokens": n_val_tokens,
            "planned_token_visits": token_visits,
            "epoch_equivalent": epoch_equivalent,
            "checkpoints": checkpoints,
            "elapsed_total_s": round(time.monotonic() - started, 1),
        }
        _append_jsonl(checkpoint_log, {"event": "stop", **result})
        return result

    return {
        "status": "completed",
        "lane": lane_label,
        "proposal_id": proposal_id,
        "math_axes": axes,
        "size": size,
        "dim": dim,
        "n_blocks": n_blocks,
        "n_params": n_params,
        "pre_ppl": pre_ppl,
        "train_tokens": n_train_tokens,
        "val_tokens": n_val_tokens,
        "planned_token_visits": token_visits,
        "epoch_equivalent": epoch_equivalent,
        "checkpoints": checkpoints,
        "elapsed_total_s": round(time.monotonic() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-name", default=None, type=str)
    parser.add_argument(
        "--proposal-id",
        default=None,
        type=str,
        help=(
            "Saved winner proposal_id/name/prefix. Defaults to the pinned "
            "block_gated_parallel BLiMP champion when --lane-name is omitted."
        ),
    )
    parser.add_argument("--size", required=True, choices=list(PARAM_SIZING.keys()))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-log", default=None, type=Path)
    parser.add_argument("--checkpoint-steps", default="10000,20000,40000", type=str)
    parser.add_argument("--n-train-steps", default=40000, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--seq-len", default=256, type=int)
    parser.add_argument("--n-eval-batches", default=32, type=int)
    parser.add_argument("--max-chars-train", default=200_000_000, type=int)
    parser.add_argument("--max-chars-val", default=2_000_000, type=int)
    parser.add_argument("--blimp-n-per-subtask", default=25, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument(
        "--early-stop-blimp",
        type=float,
        default=None,
        help="If 10K-checkpoint BLiMP falls below this, halt the cell.",
    )
    parser.add_argument("--ppl-stop-factor", default=2.0, type=float)
    parser.add_argument("--max-epoch-equivalent", default=50.0, type=float)
    parser.add_argument("--grad-spike-threshold", default=10.0, type=float)
    parser.add_argument("--grad-spike-patience", default=5, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    proposal_id = args.proposal_id
    if args.lane_name is None and proposal_id is None:
        proposal_id = _DEFAULT_PROPOSAL_ID
    checkpoint_log = args.checkpoint_log or args.output.with_suffix(".jsonl")
    ckpts = tuple(int(x) for x in args.checkpoint_steps.split(",") if x.strip())
    result = run_scaling_cell(
        args.lane_name,
        args.size,
        proposal_id=proposal_id,
        n_train_steps_total=args.n_train_steps,
        checkpoint_steps=ckpts,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_eval_batches=args.n_eval_batches,
        max_chars_train=args.max_chars_train,
        max_chars_val=args.max_chars_val,
        blimp_n_per_subtask=args.blimp_n_per_subtask,
        learning_rate=args.learning_rate,
        early_stop_blimp=args.early_stop_blimp,
        ppl_stop_factor=args.ppl_stop_factor,
        max_epoch_equivalent=args.max_epoch_equivalent,
        grad_spike_threshold=args.grad_spike_threshold,
        grad_spike_patience=args.grad_spike_patience,
        checkpoint_log=checkpoint_log,
        device=args.device,
        quiet=args.quiet,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if not args.quiet:
        print(f"\nwrote {args.output}", flush=True)
        print(f"checkpoint log: {checkpoint_log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
