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

import json
import time
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch
import torch.nn as nn

from ._kv_pair import (
    KVPairTable,
    KVProbeRuntime,
    build_kv_pair_table,
    evaluate_kv_probe_checkpoint,
    kv_table_to_device,
    make_kv_pair_batch,
    run_kv_probe_training_loop,
    train_kv_one_batch,
)
from ._probe_runtime import disable_native_probe_dispatch
from ._probe_utils import safe_deepcopy_module
from .associative_recall import _get_special_tokens
from .utils import chance_lift, clip01, make_adamw, model_vocab_size

ARIntermediatePairTable: TypeAlias = KVPairTable
make_ar_intermediate_batch = make_kv_pair_batch
_table_to_device = kv_table_to_device
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
    return build_kv_pair_table(
        seed=int(cfg.seed),
        vocab_lo=int(cfg.vocab_lo),
        n_key_tokens=int(cfg.n_key_tokens),
        n_value_tokens=int(cfg.n_value_tokens),
        n_value_classes=int(cfg.n_value_classes),
        n_train_pairs=int(cfg.n_train_pairs),
        n_held_pairs=int(cfg.n_held_pairs),
        pairs_per_example=int(cfg.pairs_per_example),
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
        probe_model = (
            safe_deepcopy_module(model).to(dev) if cfg.copy_model else model.to(dev)
        )
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
        runtime = KVProbeRuntime(
            n_eval=int(cfg.n_eval),
            batch_size=int(cfg.batch_size),
            pairs_per_example=int(cfg.pairs_per_example),
            sep_token=int(sep_token),
            ans_token=int(ans_token),
            device=dev,
            episodic_values=bool(cfg.episodic_values),
        )
        eval_every = max(1, int(cfg.eval_every))
        deadline = t0 + float(cfg.timeout_s)
        learning_curve: list[dict[str, float | int]] = []

        def _record_eval(
            step: int,
            loss: torch.Tensor,
            train_acc: float,
            held_pair: float,
            held_class: float,
        ) -> None:
            pair_chance = 1.0 / float(table.value_hi - table.value_lo)
            class_chance = 1.0 / float(table.n_value_classes)
            learning_curve.append(
                {
                    "step": step,
                    "loss": round(float(loss.item()), 6),
                    "train_pair_acc": round(train_acc, 4),
                    "held_pair_acc": round(held_pair, 4),
                    "held_class_acc": round(held_class, 4),
                    "held_pair_lift": round(chance_lift(held_pair, pair_chance), 4),
                    "held_class_lift": round(chance_lift(held_class, class_chance), 4),
                }
            )

        with disable_native_probe_dispatch(probe_model, device=str(dev)):
            loop_result = run_kv_probe_training_loop(
                probe_model,
                table,
                runtime=runtime,
                generator=gen,
                opt=opt,
                ans_pos=ans_pos,
                train_steps=int(cfg.train_steps),
                eval_every=eval_every,
                deadline=deadline,
                base_seed=int(cfg.seed),
                monotonic_time=time.perf_counter,
                on_eval=_record_eval,
            )

            train_acc, held_pair, held_class = evaluate_kv_probe_checkpoint(
                probe_model,
                table,
                runtime=runtime,
                base_seed=int(cfg.seed),
            )

        steps_done = loop_result.steps_done
        status = loop_result.status
        error = loop_result.error
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
