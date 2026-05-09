"""Medium associative-recall probe for post-Nano, pre-Small screening.

This probe deliberately sits between ``ar_gate`` and
``ar_validation``:

* harder than AR Gate: integer-token episodic values, multi-pair contexts,
  disjoint held key pairs, and value-class diagnostics;
* cheaper than AR Validation/easy25: fewer pairs, shorter sequences, lower default
  train/eval budget, and CPU-compatible tiny configs for unit tests.

Format:

    k1a k1b v1 ... kNa kNb vN [SEP] kQa kQb [ANS] -> vQ

With ``episodic_values=True`` the value assigned to a key is sampled per
example, so a model cannot solve the task by memorizing the static key table.
Held-pair evaluation queries key tokens that never appeared as query keys
during training, while held-class accuracy records whether the predicted value
lands in the correct coarse value class.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch
import torch.nn as nn

from ._kv_pair import (
    KVPairTable,
    evaluate_kv_split,
    kv_table_to_device,
    make_kv_pair_batch,
    train_kv_one_batch,
)
from ._probe_runtime import disable_native_probe_dispatch
from .associative_recall import _get_special_tokens
from .utils import chance_lift, clip01, make_adamw, model_vocab_size

ARIntermediatePairTable: TypeAlias = KVPairTable
make_ar_intermediate_batch = make_kv_pair_batch
_table_to_device = kv_table_to_device
_evaluate_split = evaluate_kv_split
_train_one_batch = train_kv_one_batch

AR_INTERMEDIATE_METRIC_VERSION = "ar_intermediate_probe_v4_lift_auc_eval256"

DEFAULT_VOCAB_LO = 512
DEFAULT_KEY_TOKENS = 256
DEFAULT_VALUE_TOKENS = 48
DEFAULT_VALUE_CLASSES = 12
DEFAULT_TRAIN_PAIRS = 96
DEFAULT_HELD_PAIRS = 32
DEFAULT_PAIRS_PER_EXAMPLE = 5
DEFAULT_TRAIN_STEPS = 1_500
DEFAULT_EVAL_EVERY = 150
DEFAULT_BATCH_SIZE = 16
DEFAULT_EVAL_EXAMPLES = 256
DEFAULT_LR = 1e-3
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_THRESHOLD = 0.12


@dataclass(frozen=True, slots=True)
class ARIntermediateConfig:
    seed: int = 0
    vocab_lo: int = DEFAULT_VOCAB_LO
    n_key_tokens: int = DEFAULT_KEY_TOKENS
    n_value_tokens: int = DEFAULT_VALUE_TOKENS
    n_value_classes: int = DEFAULT_VALUE_CLASSES
    n_train_pairs: int = DEFAULT_TRAIN_PAIRS
    n_held_pairs: int = DEFAULT_HELD_PAIRS
    pairs_per_example: int = DEFAULT_PAIRS_PER_EXAMPLE
    train_steps: int = DEFAULT_TRAIN_STEPS
    eval_every: int = DEFAULT_EVAL_EVERY
    batch_size: int = DEFAULT_BATCH_SIZE
    n_eval: int = DEFAULT_EVAL_EXAMPLES
    lr: float = DEFAULT_LR
    timeout_s: float = DEFAULT_TIMEOUT_S
    threshold: float = DEFAULT_THRESHOLD
    episodic_values: bool = True
    copy_model: bool = True


@dataclass(slots=True)
class ARIntermediateResult:
    metric_version: str = AR_INTERMEDIATE_METRIC_VERSION
    train_pair_acc: float = 0.0
    held_pair_acc: float = 0.0
    held_class_acc: float = 0.0
    pair_chance_acc: float = 0.0
    class_chance_acc: float = 0.0
    held_pair_lift: float = 0.0
    held_class_lift: float = 0.0
    early_held_pair_acc: float = 0.0
    final_held_pair_acc: float = 0.0
    best_held_pair_acc: float = 0.0
    improvement: float = 0.0
    slope_per_100_steps: float = 0.0
    auc: float = 0.0
    auc_lift: float = 0.0
    learning_curve: list[dict[str, float | int]] = field(default_factory=list)
    steps_to_threshold: int | None = None
    score: float = 0.0
    steps_trained: int = 0
    status: str = "ok"
    elapsed_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ar_intermediate_metric_version": self.metric_version,
            "ar_intermediate_train_pair_acc": self.train_pair_acc,
            "ar_intermediate_held_pair_acc": self.held_pair_acc,
            "ar_intermediate_held_class_acc": self.held_class_acc,
            "ar_intermediate_pair_chance_acc": self.pair_chance_acc,
            "ar_intermediate_class_chance_acc": self.class_chance_acc,
            "ar_intermediate_held_pair_lift": self.held_pair_lift,
            "ar_intermediate_held_class_lift": self.held_class_lift,
            "ar_intermediate_early_held_pair_acc": self.early_held_pair_acc,
            "ar_intermediate_final_held_pair_acc": self.final_held_pair_acc,
            "ar_intermediate_best_held_pair_acc": self.best_held_pair_acc,
            "ar_intermediate_improvement": self.improvement,
            "ar_intermediate_slope_per_100_steps": self.slope_per_100_steps,
            "ar_intermediate_auc": self.auc,
            "ar_intermediate_auc_lift": self.auc_lift,
            "ar_intermediate_learning_curve_json": json.dumps(
                self.learning_curve,
                sort_keys=True,
            ),
            "ar_intermediate_steps_to_threshold": self.steps_to_threshold,
            "ar_intermediate_diagnostic_score": self.score,
            "ar_intermediate_steps_trained": self.steps_trained,
            "ar_intermediate_status": self.status,
            "ar_intermediate_elapsed_ms": self.elapsed_ms,
            "ar_intermediate_error": self.error,
        }


def build_ar_intermediate_pair_table(
    cfg: ARIntermediateConfig,
) -> ARIntermediatePairTable:
    total_pairs = int(cfg.n_train_pairs) + int(cfg.n_held_pairs)
    if total_pairs <= 0:
        raise ValueError("at least one train or held pair is required")
    if int(cfg.n_key_tokens) < total_pairs * 2:
        raise ValueError("n_key_tokens must provide two unique tokens per pair")
    if int(cfg.n_value_tokens) <= 0:
        raise ValueError("n_value_tokens must be positive")
    if int(cfg.n_value_classes) <= 0 or int(cfg.n_value_classes) > int(
        cfg.n_value_tokens
    ):
        raise ValueError("n_value_classes must be in [1, n_value_tokens]")
    if int(cfg.pairs_per_example) < 2:
        raise ValueError("pairs_per_example must be at least 2")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(cfg.seed))
    key_tokens = torch.randperm(int(cfg.n_key_tokens), generator=gen)[: total_pairs * 2]
    key_tokens = key_tokens.reshape(total_pairs, 2) + int(cfg.vocab_lo)

    value_lo = int(cfg.vocab_lo) + int(cfg.n_key_tokens)
    value_offsets = torch.arange(total_pairs, dtype=torch.long) % int(
        cfg.n_value_tokens
    )
    value_offsets = value_offsets[torch.randperm(total_pairs, generator=gen)]
    values = value_lo + value_offsets

    n_train = int(cfg.n_train_pairs)
    return ARIntermediatePairTable(
        train_keys=key_tokens[:n_train].contiguous(),
        train_values=values[:n_train].contiguous(),
        held_keys=key_tokens[n_train:].contiguous(),
        held_values=values[n_train:].contiguous(),
        vocab_lo=int(cfg.vocab_lo),
        value_lo=value_lo,
        value_hi=value_lo + int(cfg.n_value_tokens),
        n_value_classes=int(cfg.n_value_classes),
    )


def _steps_to_threshold(
    learning_curve: list[dict[str, float | int]],
    threshold: float,
) -> int | None:
    for row in learning_curve:
        if float(row.get("held_pair_acc") or 0.0) >= float(threshold):
            return int(row["step"])
    return None


def _curve_summary(
    learning_curve: list[dict[str, float | int]],
    *,
    final_step: int,
) -> tuple[float, float, float, float, float]:
    if not learning_curve:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    early = float(learning_curve[0].get("held_pair_acc") or 0.0)
    final = float(learning_curve[-1].get("held_pair_acc") or 0.0)
    best = max(float(row.get("held_pair_acc") or 0.0) for row in learning_curve)
    auc = sum(float(row.get("held_pair_acc") or 0.0) for row in learning_curve) / float(
        len(learning_curve)
    )
    first_step = int(learning_curve[0].get("step") or 0)
    span = max(1, int(final_step) - first_step)
    improvement = final - early
    return early, final, best, auc, improvement * 100.0 / float(span)


def _score(
    held_pair_lift: float,
    held_class_lift: float,
    auc_lift: float,
    improvement_lift: float,
    steps_to_threshold: int | None,
    train_steps: int,
) -> float:
    speed = 0.0
    if steps_to_threshold is not None and int(train_steps) > 0:
        speed = max(0.0, min(1.0, 1.0 - float(steps_to_threshold) / train_steps))
    return round(
        10.0
        * (
            0.45 * clip01(held_pair_lift)
            + 0.20 * clip01(held_class_lift)
            + 0.20 * clip01(auc_lift)
            + 0.10 * clip01(improvement_lift)
            + 0.05 * speed
        ),
        4,
    )


def _err_result(t0: float, status: str, error: str) -> ARIntermediateResult:
    return ARIntermediateResult(
        status=status,
        error=error,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


def ar_intermediate_probe(
    model: nn.Module,
    *,
    cfg: ARIntermediateConfig | None = None,
    device: str = "cuda",
) -> ARIntermediateResult:
    """Run the AR intermediate probe.

    ``cfg.copy_model=True`` preserves the caller's model by training a
    deepcopy. CPU is allowed for tiny configs; production screening should use
    CUDA with the default budget or a calibrated variant.
    """
    cfg = cfg or ARIntermediateConfig()
    t0 = time.perf_counter()
    dev = torch.device(device)
    try:
        probe_model = copy.deepcopy(model).to(dev) if cfg.copy_model else model.to(dev)
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "copy_failed", str(exc))

    try:
        model_vocab = model_vocab_size(probe_model)
        table = build_ar_intermediate_pair_table(cfg)
        sep_token, ans_token = _get_special_tokens(probe_model)
        required_hi = max(table.value_hi, int(sep_token) + 1, int(ans_token) + 1)
        if model_vocab is not None and int(model_vocab) < required_hi:
            return _err_result(
                t0,
                "error",
                f"model_vocab_too_small:{model_vocab}<required:{required_hi}",
            )

        table = _table_to_device(table, dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(cfg.seed))
        opt = make_adamw(
            probe_model.parameters(),
            lr=float(cfg.lr),
            fused_if_available=(dev.type == "cuda"),
        )
        ans_pos = 3 * int(cfg.pairs_per_example) + 3
        eval_every = max(1, int(cfg.eval_every))
        deadline = t0 + float(cfg.timeout_s)
        learning_curve: list[dict[str, float | int]] = []
        steps_done = 0
        status = "ok"
        error = None

        with disable_native_probe_dispatch(probe_model, device=str(dev)):
            for step in range(1, int(cfg.train_steps) + 1):
                if time.perf_counter() > deadline:
                    status = "timeout"
                    break
                ids, targets, _classes = make_ar_intermediate_batch(
                    table,
                    split="train",
                    batch_size=int(cfg.batch_size),
                    pairs_per_example=int(cfg.pairs_per_example),
                    sep_token=sep_token,
                    ans_token=ans_token,
                    device=dev,
                    generator=gen,
                    episodic_values=bool(cfg.episodic_values),
                )
                loss = _train_one_batch(
                    probe_model,
                    ids,
                    targets,
                    opt=opt,
                    table=table,
                    ans_pos=ans_pos,
                )
                if loss is None:
                    status = "error"
                    error = "non_finite_loss"
                    break
                steps_done = step
                if step % eval_every == 0 or step == int(cfg.train_steps):
                    train_acc, _train_class = _evaluate_split(
                        probe_model,
                        table,
                        split="train",
                        n_eval=int(cfg.n_eval),
                        batch_size=int(cfg.batch_size),
                        pairs_per_example=int(cfg.pairs_per_example),
                        sep_token=sep_token,
                        ans_token=ans_token,
                        device=dev,
                        seed=int(cfg.seed) + 10_000 + step,
                        episodic_values=bool(cfg.episodic_values),
                    )
                    held_pair, held_class = _evaluate_split(
                        probe_model,
                        table,
                        split="held",
                        n_eval=int(cfg.n_eval),
                        batch_size=int(cfg.batch_size),
                        pairs_per_example=int(cfg.pairs_per_example),
                        sep_token=sep_token,
                        ans_token=ans_token,
                        device=dev,
                        seed=int(cfg.seed) + 20_000 + step,
                        episodic_values=bool(cfg.episodic_values),
                    )
                    pair_chance = 1.0 / float(table.value_hi - table.value_lo)
                    class_chance = 1.0 / float(table.n_value_classes)
                    learning_curve.append(
                        {
                            "step": step,
                            "loss": round(float(loss.item()), 6),
                            "train_pair_acc": round(train_acc, 4),
                            "held_pair_acc": round(held_pair, 4),
                            "held_class_acc": round(held_class, 4),
                            "held_pair_lift": round(
                                chance_lift(held_pair, pair_chance),
                                4,
                            ),
                            "held_class_lift": round(
                                chance_lift(held_class, class_chance),
                                4,
                            ),
                        }
                    )

            train_acc, _train_class = _evaluate_split(
                probe_model,
                table,
                split="train",
                n_eval=int(cfg.n_eval),
                batch_size=int(cfg.batch_size),
                pairs_per_example=int(cfg.pairs_per_example),
                sep_token=sep_token,
                ans_token=ans_token,
                device=dev,
                seed=int(cfg.seed) + 30_000,
                episodic_values=bool(cfg.episodic_values),
            )
            held_pair, held_class = _evaluate_split(
                probe_model,
                table,
                split="held",
                n_eval=int(cfg.n_eval),
                batch_size=int(cfg.batch_size),
                pairs_per_example=int(cfg.pairs_per_example),
                sep_token=sep_token,
                ans_token=ans_token,
                device=dev,
                seed=int(cfg.seed) + 40_000,
                episodic_values=bool(cfg.episodic_values),
            )

        pair_chance = 1.0 / float(table.value_hi - table.value_lo)
        class_chance = 1.0 / float(table.n_value_classes)
        early, final, best, auc, slope = _curve_summary(
            learning_curve,
            final_step=steps_done,
        )
        improvement = final - early
        held_pair_lift = chance_lift(held_pair, pair_chance)
        held_class_lift = chance_lift(held_class, class_chance)
        auc_lift = chance_lift(auc, pair_chance)
        improvement_lift = chance_lift(final, pair_chance) - chance_lift(
            early,
            pair_chance,
        )
        threshold_step = _steps_to_threshold(learning_curve, float(cfg.threshold))
        return ARIntermediateResult(
            train_pair_acc=round(train_acc, 4),
            held_pair_acc=round(held_pair, 4),
            held_class_acc=round(held_class, 4),
            pair_chance_acc=round(pair_chance, 6),
            class_chance_acc=round(class_chance, 6),
            held_pair_lift=round(held_pair_lift, 4),
            held_class_lift=round(held_class_lift, 4),
            early_held_pair_acc=round(early, 4),
            final_held_pair_acc=round(final, 4),
            best_held_pair_acc=round(best, 4),
            improvement=round(improvement, 4),
            slope_per_100_steps=round(slope, 6),
            auc=round(auc, 4),
            auc_lift=round(auc_lift, 4),
            learning_curve=learning_curve,
            steps_to_threshold=threshold_step,
            score=_score(
                held_pair_lift,
                held_class_lift,
                auc_lift,
                improvement_lift,
                threshold_step,
                int(cfg.train_steps),
            ),
            steps_trained=steps_done,
            status=status,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            error=error,
        )
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "error", str(exc))
    finally:
        del probe_model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


__all__ = [
    "AR_INTERMEDIATE_METRIC_VERSION",
    "ARIntermediateConfig",
    "ARIntermediatePairTable",
    "ARIntermediateResult",
    "build_ar_intermediate_pair_table",
    "make_ar_intermediate_batch",
    "ar_intermediate_probe",
]
