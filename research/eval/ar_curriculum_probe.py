"""AR Curriculum probe — staged associative recall with breaking-point detection.

Trains the model cumulatively across 6 disjoint-token stages with progressively
harder vocabulary and in-context distractor pressure. Evaluates each stage at
final-train so the per-stage learning curve falls out as a side-effect.

Headline metrics:

* ``ar_curriculum_auc_pair_final`` — mean held_pair_acc across the 6 final-stage
  evals. Bounded [0, 1]. Higher = more capability.
* ``ar_curriculum_s0_retention`` — held_pair_acc at S0 (the easiest stage)
  divided by 1.0 (the matched-compute frozen-S0 ceiling). Lower = more
  catastrophic forgetting from cumulative training.
* ``ar_curriculum_max_passing_stage`` — last stage where held_pair_acc beat
  4× chance. -1 = no stage cleared.

Stages are token-disjoint so cumulative training cannot trivially generalize
between stages — held_pair gains at S_n+1 reflect real learning past S_n.

This probe is the production successor to ``ar_intermediate_probe`` for tiers
that need rich rank discrimination. ``ar_validation`` becomes optional /
post-champion confirmation as of 2026-05-09.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .associative_recall import _get_special_tokens
from ._probe_runtime import disable_native_probe_dispatch
from .utils import clip_grad_norm, make_adamw, model_vocab_size

AR_CURRICULUM_METRIC_VERSION = "ar_curriculum_v1"

# Stage curriculum (S0–S5). Token ranges are disjoint per stage; vocab requirement
# is sum(n_keys + n_values) + vocab_lo + 2 special tokens = 1576.
STAGE_CONFIGS: tuple[dict[str, int], ...] = (
    {
        "n_keys": 8,
        "n_values": 4,
        "pairs_per_example": 1,
        "n_train_pairs": 3,
        "n_held_pairs": 1,
    },
    {
        "n_keys": 16,
        "n_values": 6,
        "pairs_per_example": 2,
        "n_train_pairs": 6,
        "n_held_pairs": 2,
    },
    {
        "n_keys": 32,
        "n_values": 8,
        "pairs_per_example": 2,
        "n_train_pairs": 8,
        "n_held_pairs": 4,
    },
    {
        "n_keys": 64,
        "n_values": 12,
        "pairs_per_example": 2,
        "n_train_pairs": 16,
        "n_held_pairs": 8,
    },
    {
        "n_keys": 128,
        "n_values": 16,
        "pairs_per_example": 3,
        "n_train_pairs": 32,
        "n_held_pairs": 16,
    },
    {
        "n_keys": 256,
        "n_values": 24,
        "pairs_per_example": 4,
        "n_train_pairs": 64,
        "n_held_pairs": 24,
    },
)
VOCAB_LO = 1000

DEFAULT_STEPS_PER_STAGE = 1000
DEFAULT_BATCH_SIZE = 16
DEFAULT_EVAL_BATCHES = 32
DEFAULT_LR = 1e-3
DEFAULT_TIMEOUT_S = 300.0


@dataclass(frozen=True, slots=True)
class ARCurriculumConfig:
    """Cumulative-curriculum config. Default: 6 stages × 1000 steps."""

    seed: int = 0
    steps_per_stage: int = DEFAULT_STEPS_PER_STAGE
    batch_size: int = DEFAULT_BATCH_SIZE
    eval_batches: int = DEFAULT_EVAL_BATCHES
    lr: float = DEFAULT_LR
    timeout_s: float = DEFAULT_TIMEOUT_S
    copy_model: bool = True
    mode: str = "cumulative"  # "cumulative" or "frozen_s0" (matched-compute control)
    pass_threshold_x_chance: float = 4.0


@dataclass(frozen=True, slots=True)
class _StageSpec:
    stage_idx: int
    n_key_tokens: int
    n_value_tokens: int
    pairs_per_example: int
    n_train_pairs: int
    n_held_pairs: int
    train_keys: torch.Tensor
    train_values: torch.Tensor
    held_keys: torch.Tensor
    held_values: torch.Tensor
    value_lo: int
    value_hi: int


@dataclass(slots=True)
class ARCurriculumResult:
    metric_version: str = AR_CURRICULUM_METRIC_VERSION
    auc_pair_final: float = 0.0
    auc_class_final: float = 0.0
    s0_held_pair_acc: float = 0.0
    s0_retention: float = 0.0
    max_passing_stage: int = -1
    per_stage_held_pair_acc: list[float] = field(default_factory=list)
    per_stage_held_class_acc: list[float] = field(default_factory=list)
    per_stage_lift_pair: list[float] = field(default_factory=list)
    per_stage_z_score_pair: list[float] = field(default_factory=list)
    per_stage_chance_pair: list[float] = field(default_factory=list)
    learning_curve: list[dict[str, Any]] = field(default_factory=list)
    steps_trained: int = 0
    n_eval_examples: int = 0
    mode: str = "cumulative"
    elapsed_ms: float = 0.0
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ar_curriculum_metric_version": self.metric_version,
            "ar_curriculum_auc_pair_final": self.auc_pair_final,
            "ar_curriculum_auc_class_final": self.auc_class_final,
            "ar_curriculum_s0_held_pair_acc": self.s0_held_pair_acc,
            "ar_curriculum_s0_retention": self.s0_retention,
            "ar_curriculum_max_passing_stage": self.max_passing_stage,
            "ar_curriculum_per_stage_held_pair_acc": json.dumps(
                self.per_stage_held_pair_acc
            ),
            "ar_curriculum_per_stage_held_class_acc": json.dumps(
                self.per_stage_held_class_acc
            ),
            "ar_curriculum_per_stage_lift_pair": json.dumps(self.per_stage_lift_pair),
            "ar_curriculum_per_stage_z_score_pair": json.dumps(
                self.per_stage_z_score_pair
            ),
            "ar_curriculum_per_stage_chance_pair": json.dumps(
                self.per_stage_chance_pair
            ),
            "ar_curriculum_learning_curve_json": json.dumps(
                self.learning_curve, sort_keys=True
            ),
            "ar_curriculum_steps_trained": self.steps_trained,
            "ar_curriculum_n_eval_examples": self.n_eval_examples,
            "ar_curriculum_mode": self.mode,
            "ar_curriculum_elapsed_ms": self.elapsed_ms,
            "ar_curriculum_status": self.status,
            "ar_curriculum_error": self.error,
        }


def _build_stages(seed: int, device: torch.device) -> list[_StageSpec]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    cursor = VOCAB_LO
    out: list[_StageSpec] = []
    for i, cfg in enumerate(STAGE_CONFIGS):
        key_lo = cursor
        cursor += cfg["n_keys"]
        value_lo = cursor
        cursor += cfg["n_values"]
        total = cfg["n_train_pairs"] + cfg["n_held_pairs"]
        if cfg["n_keys"] < total * 2:
            raise ValueError(
                f"stage {i}: n_keys={cfg['n_keys']} too small for {total} pairs"
            )
        key_perm = torch.randperm(cfg["n_keys"], generator=gen)[: total * 2] + key_lo
        key_pairs = key_perm.reshape(total, 2)
        value_offsets = torch.arange(total, dtype=torch.long) % cfg["n_values"]
        value_offsets = value_offsets[torch.randperm(total, generator=gen)]
        values = value_lo + value_offsets
        n_train = cfg["n_train_pairs"]
        out.append(
            _StageSpec(
                stage_idx=i,
                n_key_tokens=cfg["n_keys"],
                n_value_tokens=cfg["n_values"],
                pairs_per_example=cfg["pairs_per_example"],
                n_train_pairs=cfg["n_train_pairs"],
                n_held_pairs=cfg["n_held_pairs"],
                train_keys=key_pairs[:n_train].contiguous().to(device),
                train_values=values[:n_train].contiguous().to(device),
                held_keys=key_pairs[n_train:].contiguous().to(device),
                held_values=values[n_train:].contiguous().to(device),
                value_lo=value_lo,
                value_hi=value_lo + cfg["n_values"],
            )
        )
    return out


def _make_batch(
    stage: _StageSpec,
    *,
    split: str,
    batch_size: int,
    sep_token: int,
    ans_token: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = stage.train_keys if split == "train" else stage.held_keys
    n_pairs = int(stage.pairs_per_example)
    value_span = int(stage.value_hi - stage.value_lo)
    if value_span < n_pairs:
        raise ValueError(
            f"stage {stage.stage_idx}: value span {value_span} < pairs {n_pairs}"
        )
    q_idx = torch.randint(
        0, keys.shape[0], (batch_size,), device=device, generator=generator
    )
    d_idx = torch.randint(
        0,
        stage.train_keys.shape[0],
        (batch_size, n_pairs - 1),
        device=device,
        generator=generator,
    )
    ex_keys = torch.empty((batch_size, n_pairs, 2), dtype=torch.long, device=device)
    ex_values = torch.empty((batch_size, n_pairs), dtype=torch.long, device=device)
    ex_keys[:, 0, :] = keys.index_select(0, q_idx)
    flat_d = d_idx.reshape(-1)
    ex_keys[:, 1:, :] = stage.train_keys.index_select(0, flat_d).reshape(
        batch_size, n_pairs - 1, 2
    )
    scores = torch.rand((batch_size, value_span), device=device, generator=generator)
    order = torch.argsort(scores, dim=1)[:, :n_pairs]
    ex_values[:, :] = order + int(stage.value_lo)
    shuffle = torch.argsort(
        torch.rand((batch_size, n_pairs), device=device, generator=generator), dim=1
    )
    ex_keys = ex_keys.gather(1, shuffle.unsqueeze(-1).expand(-1, -1, 2))
    ex_values = ex_values.gather(1, shuffle)
    seq_len = 3 * n_pairs + 4
    ids = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    pair_pos = torch.arange(n_pairs, device=device)
    ids[:, pair_pos * 3] = ex_keys[:, :, 0]
    ids[:, pair_pos * 3 + 1] = ex_keys[:, :, 1]
    ids[:, pair_pos * 3 + 2] = ex_values
    sep_pos = 3 * n_pairs
    ids[:, sep_pos] = int(sep_token)
    ids[:, sep_pos + 1] = keys[q_idx, 0]
    ids[:, sep_pos + 2] = keys[q_idx, 1]
    ids[:, sep_pos + 3] = int(ans_token)
    query_pos = (shuffle == 0).to(torch.long).argmax(dim=1)
    targets = ex_values.gather(1, query_pos.unsqueeze(1)).squeeze(1)
    return ids, targets


def _train_one_batch(
    model: nn.Module,
    ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    opt: torch.optim.Optimizer,
    stage: _StageSpec,
    ans_pos: int,
) -> float | None:
    opt.zero_grad(set_to_none=True)
    logits = model(ids)
    pred = logits[:, ans_pos, stage.value_lo : stage.value_hi].float()
    loss = F.cross_entropy(pred, targets - int(stage.value_lo))
    if not torch.isfinite(loss):
        return None
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach().item())


@torch.no_grad()
def _evaluate_stage(
    model: nn.Module,
    stage: _StageSpec,
    *,
    sep_token: int,
    ans_token: int,
    device: torch.device,
    seed: int,
    eval_batches: int,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    n_pairs = int(stage.pairs_per_example)
    ans_pos = 3 * n_pairs + 3
    pair_correct = class_correct = total = 0
    n_classes = 4
    for _ in range(eval_batches):
        ids, targets = _make_batch(
            stage,
            split="held",
            batch_size=batch_size,
            sep_token=sep_token,
            ans_token=ans_token,
            device=device,
            generator=gen,
        )
        logits = model(ids)
        pred = logits[:, ans_pos, stage.value_lo : stage.value_hi].argmax(dim=-1)
        pred = pred + int(stage.value_lo)
        pair_correct += int((pred == targets).sum().item())
        pred_cls = (pred - int(stage.value_lo)).remainder(n_classes)
        targ_cls = (targets - int(stage.value_lo)).remainder(n_classes)
        class_correct += int((pred_cls == targ_cls).sum().item())
        total += int(targets.shape[0])
    return pair_correct / max(total, 1), class_correct / max(total, 1)


def _err_result(t0: float, status: str, msg: str) -> ARCurriculumResult:
    return ARCurriculumResult(
        status=status,
        error=msg,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


def required_vocab_size() -> int:
    """Smallest model vocab that satisfies the curriculum's token span."""
    span = sum(c["n_keys"] + c["n_values"] for c in STAGE_CONFIGS)
    return VOCAB_LO + span + 2  # +2 for SEP/ANS special tokens


def ar_curriculum_probe(
    model: nn.Module,
    *,
    cfg: ARCurriculumConfig | None = None,
    device: str = "cuda",
) -> ARCurriculumResult:
    """Run the AR curriculum probe.

    ``cfg.copy_model=True`` preserves the caller's model by training a deepcopy.
    ``cfg.mode='cumulative'`` is the production setting; ``'frozen_s0'`` runs a
    matched-compute control (all steps on stage 0 only) and is used for
    calibration anchoring, not per-arch screening.
    """
    cfg = cfg or ARCurriculumConfig()
    if cfg.mode not in ("cumulative", "frozen_s0"):
        return _err_result(time.perf_counter(), "error", f"unknown_mode:{cfg.mode}")

    t0 = time.perf_counter()
    dev = torch.device(device)
    try:
        probe_model = copy.deepcopy(model).to(dev) if cfg.copy_model else model.to(dev)
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "copy_failed", str(exc))

    try:
        sep_token, ans_token = _get_special_tokens(probe_model)
        required = max(required_vocab_size(), int(sep_token) + 1, int(ans_token) + 1)
        model_vocab = model_vocab_size(probe_model)
        if model_vocab is not None and int(model_vocab) < required:
            return _err_result(
                t0, "error", f"model_vocab_too_small:{model_vocab}<required:{required}"
            )
        stages = _build_stages(int(cfg.seed), dev)
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "error", f"setup_failed:{exc}")

    n_stages = len(stages)
    n_eval_examples = int(cfg.eval_batches) * int(cfg.batch_size)
    deadline = t0 + float(cfg.timeout_s)
    train_gen = torch.Generator(device=dev)
    train_gen.manual_seed(int(cfg.seed))
    learning_curve: list[dict[str, Any]] = []
    steps_done = 0
    status = "ok"
    error: str | None = None

    try:
        with disable_native_probe_dispatch(probe_model, device=str(dev)):
            opt = make_adamw(
                probe_model.parameters(),
                lr=float(cfg.lr),
                fused_if_available=(dev.type == "cuda"),
            )
            if cfg.mode == "cumulative":
                for stage_idx, stage in enumerate(stages):
                    if time.perf_counter() > deadline:
                        status = "timeout"
                        break
                    err = _train_stage(
                        probe_model,
                        stage,
                        n_steps=int(cfg.steps_per_stage),
                        batch_size=int(cfg.batch_size),
                        sep_token=sep_token,
                        ans_token=ans_token,
                        device=dev,
                        generator=train_gen,
                        opt=opt,
                        deadline=deadline,
                        stage_label=f"S{stage_idx}",
                        learning_curve=learning_curve,
                    )
                    if err is not None:
                        return _err_result(t0, "non_finite_loss", err)
                    steps_done += int(cfg.steps_per_stage)
            else:  # frozen_s0
                err = _train_stage(
                    probe_model,
                    stages[0],
                    n_steps=int(cfg.steps_per_stage) * n_stages,
                    batch_size=int(cfg.batch_size),
                    sep_token=sep_token,
                    ans_token=ans_token,
                    device=dev,
                    generator=train_gen,
                    opt=opt,
                    deadline=deadline,
                    stage_label="S0_frozen",
                    learning_curve=learning_curve,
                )
                if err is not None:
                    return _err_result(t0, "non_finite_loss", err)
                steps_done = int(cfg.steps_per_stage) * n_stages

            per_stage_pair: list[float] = []
            per_stage_class: list[float] = []
            for stage in stages:
                pa, ca = _evaluate_stage(
                    probe_model,
                    stage,
                    sep_token=sep_token,
                    ans_token=ans_token,
                    device=dev,
                    seed=int(cfg.seed) + 1000,
                    eval_batches=int(cfg.eval_batches),
                    batch_size=int(cfg.batch_size),
                )
                per_stage_pair.append(round(pa, 4))
                per_stage_class.append(round(ca, 4))
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "error", f"train_or_eval:{exc}")
    finally:
        del probe_model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    chance_per_stage = [round(1.0 / s.n_value_tokens, 4) for s in stages]
    lift_per_stage = [
        round((acc - c) / (1.0 - c) if (1.0 - c) > 0 else 0.0, 4)
        for acc, c in zip(per_stage_pair, chance_per_stage)
    ]
    z_per_stage = []
    for acc, c in zip(per_stage_pair, chance_per_stage):
        se = (c * (1 - c) / max(n_eval_examples, 1)) ** 0.5
        z_per_stage.append(round((acc - c) / se if se > 0 else 0.0, 2))
    auc_pair = round(sum(per_stage_pair) / max(len(per_stage_pair), 1), 4)
    auc_class = round(sum(per_stage_class) / max(len(per_stage_class), 1), 4)
    s0_acc = float(per_stage_pair[0]) if per_stage_pair else 0.0
    last_passing = -1
    for i, (acc, c) in enumerate(zip(per_stage_pair, chance_per_stage)):
        if acc >= c * float(cfg.pass_threshold_x_chance):
            last_passing = i

    return ARCurriculumResult(
        auc_pair_final=auc_pair,
        auc_class_final=auc_class,
        s0_held_pair_acc=round(s0_acc, 4),
        s0_retention=round(s0_acc, 4),  # vs 1.0 ceiling from frozen_s0 control
        max_passing_stage=last_passing,
        per_stage_held_pair_acc=per_stage_pair,
        per_stage_held_class_acc=per_stage_class,
        per_stage_lift_pair=lift_per_stage,
        per_stage_z_score_pair=z_per_stage,
        per_stage_chance_pair=chance_per_stage,
        learning_curve=learning_curve,
        steps_trained=steps_done,
        n_eval_examples=n_eval_examples,
        mode=cfg.mode,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        status=status,
        error=error,
    )


def _train_stage(
    model: nn.Module,
    stage: _StageSpec,
    *,
    n_steps: int,
    batch_size: int,
    sep_token: int,
    ans_token: int,
    device: torch.device,
    generator: torch.Generator,
    opt: torch.optim.Optimizer,
    deadline: float,
    stage_label: str,
    learning_curve: list[dict[str, Any]],
) -> str | None:
    ans_pos = 3 * int(stage.pairs_per_example) + 3
    model.train()
    for step in range(n_steps):
        if time.perf_counter() > deadline:
            return None
        ids, targets = _make_batch(
            stage,
            split="train",
            batch_size=batch_size,
            sep_token=sep_token,
            ans_token=ans_token,
            device=device,
            generator=generator,
        )
        loss = _train_one_batch(
            model, ids, targets, opt=opt, stage=stage, ans_pos=ans_pos
        )
        if loss is None:
            return f"non_finite_loss at {stage_label} step={step}"
        if step % 100 == 0 or step == n_steps - 1:
            learning_curve.append(
                {"stage": stage_label, "step": step, "loss": round(loss, 4)}
            )
    return None


__all__ = [
    "AR_CURRICULUM_METRIC_VERSION",
    "STAGE_CONFIGS",
    "VOCAB_LO",
    "ARCurriculumConfig",
    "ARCurriculumResult",
    "ar_curriculum_probe",
    "required_vocab_size",
]
