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
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from component_fab.generator.code_generator import generate_module
from component_fab.generator.primitive_templates import (
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
)
from research.defaults import VOCAB_SIZE
from research.eval.blimp_eval import evaluate_blimp
from research.eval.utils import tokenize_file
from research.eval.wikitext_eval import _download_wikitext
from research.tools._scaling_lanes import (
    AdjacentTokenMergeLane,  # noqa: F401 – re-exported for test_adjacent_token_merge_causality
    WINNER_LANE_FINGERPRINTS,  # noqa: F401 – re-exported for apply_mixer_fingerprint_pretrain
    _build_lane_factory,
)


# Param sizing presets matching the user's 30M/60M/120M targets at vocab=100K.
PARAM_SIZING: dict[str, dict[str, int]] = {
    "30M": {"dim": 256, "n_blocks": 6},
    "60M": {"dim": 448, "n_blocks": 8},
    "120M": {"dim": 640, "n_blocks": 12},
}

_REPO = Path(__file__).resolve().parents[2]
_SAVED_WINNERS_PATH = _REPO / "component_fab" / "catalog" / "saved_winners.json"
_DEFAULT_PROPOSAL_ID = "improve_tropical_gate_block_gated_parallel_84f0ccd08a"


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
    ffn_kind: str = "swiglu",
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
        ffn_kind=ffn_kind,
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


def _resolve_lane(
    lane_name: str | None,
    proposal_id: str | None,
) -> tuple[str, object, dict | None]:
    """Resolve (lane_label, factory, axes) from lane_name or proposal_id."""
    axes: dict | None = None
    if proposal_id:
        run_label, factory, axes = _saved_winner_factory(proposal_id)
        return run_label, factory, axes
    if lane_name:
        factory = _build_lane_factory(lane_name)
        return lane_name, factory, axes
    raise ValueError("either lane_name or proposal_id is required")


def _check_epoch_guard(
    lane_label: str,
    proposal_id: str | None,
    size: str,
    n_train_tokens: int,
    n_train_steps_total: int,
    batch_size: int,
    seq_len: int,
    max_epoch_equivalent: float,
) -> dict | None:
    """Return a refusal dict if planned token visits exceed the epoch cap, else None."""
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
    return None


def _log_start_metadata(
    checkpoint_log: Path | None,
    lane_label: str,
    proposal_id: str | None,
    axes: dict | None,
    size: str,
    dim: int,
    n_blocks: int,
    n_params: int,
    batch_size: int,
    seq_len: int,
    n_train_steps_total: int,
    checkpoint_steps: tuple[int, ...],
    n_train_tokens: int,
    n_val_tokens: int,
    token_visits: int,
    epoch_equivalent: float,
    pre_ppl: float,
    ppl_stop_factor: float,
    quiet: bool,
) -> None:
    """Write the 'start' event to the checkpoint log and print the pre-ppl banner."""
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


def _process_checkpoint_hit(
    ctx: "_LoopCtx",
    next_ckpt_idx: int,
    step: int,
    last_loss_val: float,
    last_grad_norm: float,
    cur_lr: float,
    best_ppl: float | None,
    best_ppl_step: int | None,
) -> tuple[dict, float | None, int | None, str | None]:
    """Evaluate one checkpoint; return (ck, best_ppl, best_ppl_step, stop_reason)."""
    target = ctx.sorted_ckpts[next_ckpt_idx]
    ck, best_ppl, best_ppl_step = _eval_checkpoint(
        ctx.model,
        ctx.val_batches,
        ctx.factory,
        ctx.dim,
        target=target,
        step=step,
        last_loss_val=last_loss_val,
        last_grad_norm=last_grad_norm,
        cur_lr=cur_lr,
        best_ppl=best_ppl,
        best_ppl_step=best_ppl_step,
        started=ctx.started,
        blimp_n_per_subtask=ctx.blimp_n_per_subtask,
        seq_len=ctx.seq_len,
        device=ctx.device,
    )
    post_ppl = ck["post_train_ppl"]
    stop_reason: str | None = None
    if best_ppl is not None and post_ppl > best_ppl * ctx.ppl_stop_factor:
        ck["early_stop_reason"] = (
            f"val_ppl {post_ppl:.1f} > {ctx.ppl_stop_factor:.2f}x "
            f"best {best_ppl:.1f} at step {best_ppl_step}"
        )
        stop_reason = ck["early_stop_reason"]
    _append_jsonl(ctx.checkpoint_log, {"event": "checkpoint", **ck})
    if not ctx.quiet:
        print(
            f"  ckpt @ step {step:>6d}: ppl={post_ppl:>7.1f} "
            f"BLiMP={ck['blimp_overall']:.4f} v2_sc={ck['v2_sc']:.3f} "
            f"v2_vd={ck['v2_vd_mean']:.3f} dyck={ck['v2_dyck_v3']:.3f} "
            f"npi={ck['v2_npi_v2']:.3f} eval_t={ck['eval_elapsed_s']:.0f}s "
            f"total_t={ck['wall_clock_s']:.0f}s",
            flush=True,
        )
    return ck, best_ppl, best_ppl_step, stop_reason


@dataclasses.dataclass
class _LoopCtx:
    """Bundles all inputs for _run_training_loop and its helpers."""

    model: TinyLM
    optim: torch.optim.Optimizer
    train_batcher: "_RandomWindowBatcher"
    val_batches: list[torch.Tensor]
    factory: object
    dim: int
    n_train_steps_total: int
    sorted_ckpts: list[int]
    learning_rate: float
    lane_label: str
    proposal_id: str | None
    axes: dict | None
    size: str
    n_blocks: int
    n_params: int
    pre_ppl: float
    n_train_tokens: int
    n_val_tokens: int
    token_visits: int
    epoch_equivalent: float
    started: float
    blimp_n_per_subtask: int
    seq_len: int
    device: str
    ppl_stop_factor: float
    early_stop_blimp: float | None
    grad_spike_threshold: float
    grad_spike_patience: int
    checkpoint_log: Path | None
    quiet: bool


def _build_common_result(ctx: _LoopCtx, checkpoints: list[dict]) -> dict:
    """Assemble fields shared by every terminal result dict."""
    return {
        "lane": ctx.lane_label,
        "proposal_id": ctx.proposal_id,
        "math_axes": ctx.axes,
        "size": ctx.size,
        "dim": ctx.dim,
        "n_blocks": ctx.n_blocks,
        "n_params": ctx.n_params,
        "pre_ppl": ctx.pre_ppl,
        "train_tokens": ctx.n_train_tokens,
        "val_tokens": ctx.n_val_tokens,
        "planned_token_visits": ctx.token_visits,
        "epoch_equivalent": ctx.epoch_equivalent,
        "checkpoints": checkpoints,
        "elapsed_total_s": round(time.monotonic() - ctx.started, 1),
    }


def _run_loop_while(
    ctx: _LoopCtx,
    lr_fn: Callable[[int], float],
    grad_clip_max_norm: float,
) -> tuple[list[dict], str | None, int]:
    """Execute the training while-loop.

    Returns (checkpoints, stop_reason, final_step).  stop_reason is None on
    normal completion, ``"__blimp_early_stop__"`` when the BLiMP threshold
    fires, or a descriptive string for grad-spike / non-finite errors.
    """
    loss_history: list[tuple[int, float, float]] = []  # (step, loss, grad_norm)
    log_every = 500  # Print per-step diagnostics every N steps for early monitoring.
    checkpoints: list[dict] = []
    best_ppl: float | None = None
    best_ppl_step: int | None = None
    last_loss_val = float("nan")
    last_grad_norm = float("nan")
    grad_spikes = 0
    stop_reason: str | None = None
    step = 0
    next_ckpt_idx = 0
    while step < ctx.n_train_steps_total:
        cur_lr = lr_fn(step)
        for pg in ctx.optim.param_groups:
            pg["lr"] = cur_lr
        batch = ctx.train_batcher.next()
        last_loss_val, last_grad_norm, step_err = _train_step(
            ctx.model, ctx.optim, batch, grad_clip_max_norm
        )
        if step_err is not None:
            stop_reason = f"{step_err}_step_{step + 1}"
            break
        step += 1
        if last_grad_norm > ctx.grad_spike_threshold:
            grad_spikes += 1
        else:
            grad_spikes = 0
        if grad_spikes >= ctx.grad_spike_patience:
            stop_reason = (
                f"grad_spike>{ctx.grad_spike_threshold}_for_{grad_spikes}_steps"
            )
            break
        # Diagnostic logging -- captures the early loss trajectory + grad
        # health so we can SEE if the architecture is exploding before
        # the first checkpoint at step 10K.
        if not ctx.quiet and (step <= 50 or step % log_every == 0):
            loss_history.append((step, last_loss_val, last_grad_norm))
            print(
                f"  step {step:>6d} lr={cur_lr:.2e} loss={last_loss_val:>7.3f} "
                f"grad_norm={last_grad_norm:>7.3f} ppl≈{math.exp(min(last_loss_val, 30)):.1f}",
                flush=True,
            )
        # Hit a checkpoint?
        if (
            next_ckpt_idx < len(ctx.sorted_ckpts)
            and step >= ctx.sorted_ckpts[next_ckpt_idx]
        ):
            ck, best_ppl, best_ppl_step, stop_reason = _process_checkpoint_hit(
                ctx,
                next_ckpt_idx,
                step,
                last_loss_val,
                last_grad_norm,
                cur_lr,
                best_ppl,
                best_ppl_step,
            )
            checkpoints.append(ck)
            if stop_reason is not None:
                if not ctx.quiet:
                    print(f"  EARLY-STOP: {stop_reason}", flush=True)
                break
            # Early-stop rule.
            if (
                ctx.early_stop_blimp is not None
                and ck["blimp_overall"] < ctx.early_stop_blimp
            ):
                if not ctx.quiet:
                    print(
                        f"  EARLY-STOP: BLiMP {ck['blimp_overall']:.4f} < "
                        f"{ctx.early_stop_blimp:.4f} at step {step}; halting this cell.",
                        flush=True,
                    )
                stop_reason = "__blimp_early_stop__"
                break
            next_ckpt_idx += 1
            ctx.model.train()
        if next_ckpt_idx >= len(ctx.sorted_ckpts):
            break
    return checkpoints, stop_reason, step


def _run_training_loop(ctx: _LoopCtx) -> dict:
    """Run the main training loop and return the final result dict."""
    lr_fn = _make_lr_schedule(
        base_lr=ctx.learning_rate,
        warmup_steps=1000,
        n_train_steps_total=ctx.n_train_steps_total,
        final_lr_frac=0.1,
    )
    grad_clip_max_norm = 1.0  # Standard transformer gradient clipping.
    ctx.model.train()
    checkpoints, stop_reason, step = _run_loop_while(ctx, lr_fn, grad_clip_max_norm)
    if stop_reason == "__blimp_early_stop__":
        return {
            "status": "early_stopped",
            "lane": ctx.lane_label,
            "proposal_id": ctx.proposal_id,
            "size": ctx.size,
            "dim": ctx.dim,
            "n_blocks": ctx.n_blocks,
            "n_params": ctx.n_params,
            "pre_ppl": ctx.pre_ppl,
            "checkpoints": checkpoints,
            "early_stop_at_step": step,
            "early_stop_blimp_threshold": ctx.early_stop_blimp,
        }
    common = _build_common_result(ctx, checkpoints)
    if stop_reason is not None:
        result = {"status": "early_stopped", "stop_reason": stop_reason, **common}
        _append_jsonl(ctx.checkpoint_log, {"event": "stop", **result})
        return result
    return {"status": "completed", **common}


def _init_and_build_ctx(
    model: TinyLM,
    optim: torch.optim.Optimizer,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    factory: object,
    n_train_tokens: int,
    n_val_tokens: int,
    dim: int,
    n_blocks: int,
    n_params: int,
    lane_label: str,
    proposal_id: str | None,
    axes: dict | None,
    size: str,
    n_train_steps_total: int,
    checkpoint_steps: tuple[int, ...],
    batch_size: int,
    seq_len: int,
    n_eval_batches: int,
    learning_rate: float,
    blimp_n_per_subtask: int,
    device: str,
    ppl_stop_factor: float,
    early_stop_blimp: float | None,
    grad_spike_threshold: float,
    grad_spike_patience: int,
    token_visits: int,
    epoch_equivalent: float,
    checkpoint_log: Path | None,
    quiet: bool,
) -> tuple[_LoopCtx, list[torch.Tensor], float]:
    """Build batchers, log start metadata, eval pre-PPL, return (ctx, val_batches, pre_ppl)."""
    train_batcher = _RandomWindowBatcher(
        train_tokens, batch_size=batch_size, seq_len=seq_len, device=device, seed=42
    )
    val_batches = _RandomWindowBatcher(
        val_tokens, batch_size=batch_size, seq_len=seq_len, device=device, seed=123
    ).fixed_batches(n_eval_batches)
    started = time.monotonic()
    pre_ppl = _eval_ppl(model, val_batches)
    _log_start_metadata(
        checkpoint_log,
        lane_label,
        proposal_id,
        axes,
        size,
        dim,
        n_blocks,
        n_params,
        batch_size,
        seq_len,
        n_train_steps_total,
        checkpoint_steps,
        n_train_tokens,
        n_val_tokens,
        token_visits,
        epoch_equivalent,
        pre_ppl,
        ppl_stop_factor,
        quiet,
    )
    ctx = _LoopCtx(
        model=model,
        optim=optim,
        train_batcher=train_batcher,
        val_batches=val_batches,
        factory=factory,
        dim=dim,
        n_train_steps_total=n_train_steps_total,
        sorted_ckpts=sorted(checkpoint_steps),
        learning_rate=learning_rate,
        lane_label=lane_label,
        proposal_id=proposal_id,
        axes=axes,
        size=size,
        n_blocks=n_blocks,
        n_params=n_params,
        pre_ppl=pre_ppl,
        n_train_tokens=n_train_tokens,
        n_val_tokens=n_val_tokens,
        token_visits=token_visits,
        epoch_equivalent=epoch_equivalent,
        started=started,
        blimp_n_per_subtask=blimp_n_per_subtask,
        seq_len=seq_len,
        device=device,
        ppl_stop_factor=ppl_stop_factor,
        early_stop_blimp=early_stop_blimp,
        grad_spike_threshold=grad_spike_threshold,
        grad_spike_patience=grad_spike_patience,
        checkpoint_log=checkpoint_log,
        quiet=quiet,
    )
    return ctx, val_batches, pre_ppl


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
    """One model x one size x full training to n_train_steps_total with
    checkpoints. Returns the checkpoint results."""
    if size not in PARAM_SIZING:
        raise ValueError(f"unknown size: {size}")
    dim = PARAM_SIZING[size]["dim"]
    n_blocks = PARAM_SIZING[size]["n_blocks"]
    lane_label, factory, axes = _resolve_lane(lane_name, proposal_id)
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
    guard = _check_epoch_guard(
        lane_label,
        proposal_id,
        size,
        n_train_tokens,
        n_train_steps_total,
        batch_size,
        seq_len,
        max_epoch_equivalent,
    )
    if guard is not None:
        return guard
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ctx, _vb, _pppl = _init_and_build_ctx(
        model,
        optim,
        train_tokens,
        val_tokens,
        factory,
        n_train_tokens,
        n_val_tokens,
        dim,
        n_blocks,
        n_params,
        lane_label,
        proposal_id,
        axes,
        size,
        n_train_steps_total,
        checkpoint_steps,
        batch_size,
        seq_len,
        n_eval_batches,
        learning_rate,
        blimp_n_per_subtask,
        device,
        ppl_stop_factor,
        early_stop_blimp,
        grad_spike_threshold,
        grad_spike_patience,
        token_visits,
        epoch_equivalent,
        checkpoint_log,
        quiet,
    )
    return _run_training_loop(ctx)


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
