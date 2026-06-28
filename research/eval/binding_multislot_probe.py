"""Experimental multi-blank sentence-binding probe.

The probe uses integer-token sentence templates to avoid tokenizer overhead in
CPU tests while preserving the target mechanism:

    entity_1 color object entity_2 color object ... [SEP]
    [QUERY] entity_i [COLOR] [ANS] [QUERY] entity_j [OBJECT] [ANS] ...

Each example contains several entities and two attribute families. The model is
trained to fill multiple query slots from the earlier context in one forward
pass. This stresses distributed binding and entity/attribute mixing more than a
single key-value lookup because the answer positions share one context but ask
for different entities and attributes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._probe_runtime import disable_native_probe_dispatch
from ._probe_utils import (
    probe_curve_summary,
    probe_steps_to_threshold,
    safe_deepcopy_module,
)
from .utils import chance_lift, clip01, clip_grad_norm, make_adamw, model_vocab_size

BINDING_MULTISLOT_METRIC_VERSION = "binding_multislot_probe_v4_three_slot"

DEFAULT_VOCAB_LO = 256
DEFAULT_ENTITIES = 96
DEFAULT_HELD_ENTITIES = 24
DEFAULT_COLOR_VALUES = 32
DEFAULT_OBJECT_VALUES = 32
DEFAULT_BINDINGS_PER_EXAMPLE = 5
DEFAULT_QUERY_SLOTS = 3
DEFAULT_TRAIN_STEPS = 1_000
DEFAULT_EVAL_EVERY = 125
DEFAULT_BATCH_SIZE = 16
DEFAULT_EVAL_EXAMPLES = 256
DEFAULT_LR = 1e-3
DEFAULT_TIMEOUT_S = 240.0
DEFAULT_THRESHOLD = 0.08


@dataclass(frozen=True, slots=True)
class BindingMultislotConfig:
    seed: int = 0
    vocab_lo: int = DEFAULT_VOCAB_LO
    n_entities: int = DEFAULT_ENTITIES
    n_held_entities: int = DEFAULT_HELD_ENTITIES
    n_color_values: int = DEFAULT_COLOR_VALUES
    n_object_values: int = DEFAULT_OBJECT_VALUES
    bindings_per_example: int = DEFAULT_BINDINGS_PER_EXAMPLE
    query_slots: int = DEFAULT_QUERY_SLOTS
    train_steps: int = DEFAULT_TRAIN_STEPS
    eval_every: int = DEFAULT_EVAL_EVERY
    batch_size: int = DEFAULT_BATCH_SIZE
    n_eval: int = DEFAULT_EVAL_EXAMPLES
    lr: float = DEFAULT_LR
    timeout_s: float = DEFAULT_TIMEOUT_S
    threshold: float = DEFAULT_THRESHOLD
    copy_model: bool = True


@dataclass(frozen=True, slots=True)
class MultiBlankLayout:
    train_entities: torch.Tensor
    held_entities: torch.Tensor
    color_values: torch.Tensor
    object_values: torch.Tensor
    sep_token: int
    query_token: int
    color_query_token: int
    object_query_token: int
    ans_token: int
    value_lo: int
    value_hi: int
    color_lo: int
    object_lo: int

    @property
    def required_vocab(self) -> int:
        return int(max(self.value_hi, self.ans_token + 1))


@dataclass(slots=True)
class BindingMultislotResult:
    metric_version: str = BINDING_MULTISLOT_METRIC_VERSION
    train_slot_acc: float = 0.0
    held_entity_slot_acc: float = 0.0
    held_entity_class_acc: float = 0.0
    two_plus_slots_acc: float = 0.0
    all_slots_acc: float = 0.0
    mixed_query_acc: float = 0.0
    mixed_two_plus_slots_acc: float = 0.0
    mixed_all_slots_acc: float = 0.0
    slot_chance_acc: float = 0.0
    class_chance_acc: float = 0.0
    two_plus_slots_chance_acc: float = 0.0
    all_slots_chance_acc: float = 0.0
    held_slot_lift: float = 0.0
    held_class_lift: float = 0.0
    two_plus_slots_lift: float = 0.0
    all_slots_lift: float = 0.0
    mixed_query_lift: float = 0.0
    mixed_two_plus_slots_lift: float = 0.0
    mixed_all_slots_lift: float = 0.0
    early_slot_acc: float = 0.0
    final_slot_acc: float = 0.0
    best_slot_acc: float = 0.0
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
            "binding_multislot_metric_version": self.metric_version,
            "binding_multislot_train_slot_acc": self.train_slot_acc,
            "binding_multislot_held_entity_slot_acc": self.held_entity_slot_acc,
            "binding_multislot_held_entity_class_acc": self.held_entity_class_acc,
            "binding_multislot_two_plus_slots_acc": self.two_plus_slots_acc,
            "binding_multislot_all_slots_acc": self.all_slots_acc,
            "binding_multislot_mixed_query_acc": self.mixed_query_acc,
            "binding_multislot_mixed_two_plus_slots_acc": self.mixed_two_plus_slots_acc,
            "binding_multislot_mixed_all_slots_acc": self.mixed_all_slots_acc,
            "binding_multislot_slot_chance_acc": self.slot_chance_acc,
            "binding_multislot_class_chance_acc": self.class_chance_acc,
            "binding_multislot_two_plus_slots_chance_acc": (
                self.two_plus_slots_chance_acc
            ),
            "binding_multislot_all_slots_chance_acc": self.all_slots_chance_acc,
            "binding_multislot_held_slot_lift": self.held_slot_lift,
            "binding_multislot_held_class_lift": self.held_class_lift,
            "binding_multislot_two_plus_slots_lift": self.two_plus_slots_lift,
            "binding_multislot_all_slots_lift": self.all_slots_lift,
            "binding_multislot_mixed_query_lift": self.mixed_query_lift,
            "binding_multislot_mixed_two_plus_slots_lift": (
                self.mixed_two_plus_slots_lift
            ),
            "binding_multislot_mixed_all_slots_lift": self.mixed_all_slots_lift,
            "binding_multislot_early_slot_acc": self.early_slot_acc,
            "binding_multislot_final_slot_acc": self.final_slot_acc,
            "binding_multislot_best_slot_acc": self.best_slot_acc,
            "binding_multislot_improvement": self.improvement,
            "binding_multislot_slope_per_100_steps": self.slope_per_100_steps,
            "binding_multislot_auc": self.auc,
            "binding_multislot_auc_lift": self.auc_lift,
            "binding_multislot_learning_curve_json": json.dumps(
                self.learning_curve,
                sort_keys=True,
            ),
            "binding_multislot_steps_to_threshold": self.steps_to_threshold,
            "binding_multislot_diagnostic_score": self.score,
            "binding_multislot_steps_trained": self.steps_trained,
            "binding_multislot_status": self.status,
            "binding_multislot_elapsed_ms": self.elapsed_ms,
            "binding_multislot_error": self.error,
        }


def build_multi_blank_layout(cfg: BindingMultislotConfig) -> MultiBlankLayout:
    n_train_entities = int(cfg.n_entities) - int(cfg.n_held_entities)
    if n_train_entities <= 0:
        raise ValueError("n_entities must be larger than n_held_entities")
    if int(cfg.bindings_per_example) < 2:
        raise ValueError("bindings_per_example must be at least 2")
    if int(cfg.query_slots) < 2:
        raise ValueError("query_slots must be at least 2")
    if int(cfg.n_color_values) <= 0 or int(cfg.n_object_values) <= 0:
        raise ValueError("attribute value counts must be positive")

    cursor = int(cfg.vocab_lo)
    entities = torch.arange(cursor, cursor + int(cfg.n_entities), dtype=torch.long)
    cursor += int(cfg.n_entities)
    color_values = torch.arange(
        cursor,
        cursor + int(cfg.n_color_values),
        dtype=torch.long,
    )
    color_lo = cursor
    cursor += int(cfg.n_color_values)
    object_values = torch.arange(
        cursor,
        cursor + int(cfg.n_object_values),
        dtype=torch.long,
    )
    object_lo = cursor
    cursor += int(cfg.n_object_values)
    sep_token = cursor
    query_token = cursor + 1
    color_query_token = cursor + 2
    object_query_token = cursor + 3
    ans_token = cursor + 4
    return MultiBlankLayout(
        train_entities=entities[:n_train_entities].contiguous(),
        held_entities=entities[n_train_entities:].contiguous(),
        color_values=color_values.contiguous(),
        object_values=object_values.contiguous(),
        sep_token=sep_token,
        query_token=query_token,
        color_query_token=color_query_token,
        object_query_token=object_query_token,
        ans_token=ans_token,
        value_lo=color_lo,
        value_hi=object_lo + int(cfg.n_object_values),
        color_lo=color_lo,
        object_lo=object_lo,
    )


def _layout_to_device(
    layout: MultiBlankLayout,
    device: torch.device,
) -> MultiBlankLayout:
    return MultiBlankLayout(
        train_entities=layout.train_entities.to(device),
        held_entities=layout.held_entities.to(device),
        color_values=layout.color_values.to(device),
        object_values=layout.object_values.to(device),
        sep_token=layout.sep_token,
        query_token=layout.query_token,
        color_query_token=layout.color_query_token,
        object_query_token=layout.object_query_token,
        ans_token=layout.ans_token,
        value_lo=layout.value_lo,
        value_hi=layout.value_hi,
        color_lo=layout.color_lo,
        object_lo=layout.object_lo,
    )


def make_multi_blank_batch(
    layout: MultiBlankLayout,
    *,
    split: str,
    batch_size: int,
    bindings_per_example: int,
    query_slots: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if split not in {"train", "held_entity"}:
        raise ValueError("split must be 'train' or 'held_entity'")
    entity_pool = layout.train_entities if split == "train" else layout.held_entities
    if entity_pool.numel() == 0:
        raise ValueError(f"{split} split has no entities")

    batch = int(batch_size)
    n_bind = int(bindings_per_example)
    n_query = int(query_slots)
    if entity_pool.numel() < n_bind:
        raise ValueError("entity split must cover bindings_per_example")

    entity_order = torch.argsort(
        torch.rand((batch, entity_pool.numel()), device=device, generator=generator),
        dim=1,
    )[:, :n_bind]
    entities = entity_pool.index_select(0, entity_order.reshape(-1)).reshape(
        batch,
        n_bind,
    )
    color_idx = torch.randint(
        0,
        layout.color_values.numel(),
        (batch, n_bind),
        device=device,
        generator=generator,
    )
    object_idx = torch.randint(
        0,
        layout.object_values.numel(),
        (batch, n_bind),
        device=device,
        generator=generator,
    )
    colors = layout.color_values.index_select(0, color_idx.reshape(-1)).reshape(
        batch,
        n_bind,
    )
    objects = layout.object_values.index_select(0, object_idx.reshape(-1)).reshape(
        batch,
        n_bind,
    )

    story_len = n_bind * 3
    seq_len = story_len + 1 + n_query * 4
    ids = torch.empty((batch, seq_len), dtype=torch.long, device=device)
    pos = torch.arange(n_bind, device=device)
    ids[:, pos * 3] = entities
    ids[:, pos * 3 + 1] = colors
    ids[:, pos * 3 + 2] = objects
    ids[:, story_len] = int(layout.sep_token)

    q_entity_idx = torch.randint(
        0,
        n_bind,
        (batch, n_query),
        device=device,
        generator=generator,
    )
    q_attr = torch.randint(
        0,
        2,
        (batch, n_query),
        device=device,
        generator=generator,
    )
    q_entities = entities.gather(1, q_entity_idx)
    q_colors = colors.gather(1, q_entity_idx)
    q_objects = objects.gather(1, q_entity_idx)
    targets = torch.where(q_attr == 0, q_colors, q_objects)
    target_classes = q_attr

    q_base = story_len + 1 + torch.arange(n_query, device=device) * 4
    ids[:, q_base] = int(layout.query_token)
    ids[:, q_base + 1] = q_entities
    ids[:, q_base + 2] = torch.where(
        q_attr == 0,
        torch.full_like(q_attr, int(layout.color_query_token)),
        torch.full_like(q_attr, int(layout.object_query_token)),
    )
    ids[:, q_base + 3] = int(layout.ans_token)
    ans_positions = q_base + 3
    return ids, targets, target_classes, ans_positions


def _class_predictions(pred: torch.Tensor, layout: MultiBlankLayout) -> torch.Tensor:
    return (pred >= int(layout.object_lo)).to(torch.long)


@torch.no_grad()
def _evaluate_split(
    model: nn.Module,
    layout: MultiBlankLayout,
    *,
    split: str,
    n_eval: int,
    batch_size: int,
    bindings_per_example: int,
    query_slots: int,
    device: torch.device,
    seed: int,
) -> tuple[float, float, float, float, float, float, float]:
    model.eval()
    counts = torch.zeros(9, dtype=torch.long, device=device)
    (
        slot_correct,
        class_correct,
        two_plus_correct,
        all_slots_correct,
        mixed_correct,
        mixed_two_plus_correct,
        mixed_all_correct,
        mixed_total,
        mixed_examples,
    ) = counts.unbind()
    total_slots = 0
    total_examples = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    remaining = int(n_eval)
    while remaining > 0:
        bs = min(int(batch_size), remaining)
        ids, targets, target_classes, ans_positions = make_multi_blank_batch(
            layout,
            split=split,
            batch_size=bs,
            bindings_per_example=bindings_per_example,
            query_slots=query_slots,
            device=device,
            generator=gen,
        )
        logits = model(ids)
        pred = logits[:, ans_positions, layout.value_lo : layout.value_hi].argmax(-1)
        pred = pred + int(layout.value_lo)
        slot_ok = pred == targets
        per_example_correct = slot_ok.sum(dim=1)
        slot_correct += slot_ok.sum()
        class_correct += (_class_predictions(pred, layout) == target_classes).sum()
        two_plus_correct += (per_example_correct >= 2).sum()
        all_slots_correct += slot_ok.all(dim=1).sum()
        has_mixed = target_classes.min(dim=1).values != target_classes.max(dim=1).values
        n_mixed = has_mixed.sum()
        mixed_correct += slot_ok[has_mixed].sum()
        mixed_total += n_mixed * int(query_slots)
        mixed_two_plus_correct += (per_example_correct[has_mixed] >= 2).sum()
        mixed_all_correct += slot_ok[has_mixed].all(dim=1).sum()
        mixed_examples += n_mixed
        total_slots += int(targets.numel())
        total_examples += bs
        remaining -= bs
    (
        slot_n,
        class_n,
        two_plus_n,
        all_n,
        mixed_n,
        mixed_two_plus_n,
        mixed_all_n,
        mixed_total_n,
        mixed_examples_n,
    ) = counts.tolist()
    return (
        slot_n / max(total_slots, 1),
        class_n / max(total_slots, 1),
        two_plus_n / max(total_examples, 1),
        all_n / max(total_examples, 1),
        mixed_n / max(mixed_total_n, 1),
        mixed_two_plus_n / max(mixed_examples_n, 1),
        mixed_all_n / max(mixed_examples_n, 1),
    )


def _train_one_batch(
    model: nn.Module,
    ids: torch.Tensor,
    targets: torch.Tensor,
    ans_positions: torch.Tensor,
    *,
    opt: torch.optim.Optimizer,
    layout: MultiBlankLayout,
) -> torch.Tensor | None:
    opt.zero_grad(set_to_none=True)
    logits = model(ids)
    pred = logits[:, ans_positions, layout.value_lo : layout.value_hi].float()
    loss = F.cross_entropy(
        pred.reshape(-1, int(layout.value_hi - layout.value_lo)),
        (targets - int(layout.value_lo)).reshape(-1),
    )
    if not torch.isfinite(loss):
        return None
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    opt.step()
    return loss.detach()


def _two_plus_chance(slot_chance: float, query_slots: int) -> float:
    n = max(0, int(query_slots))
    p = clip01(slot_chance)
    if n < 2:
        return 0.0
    p0 = (1.0 - p) ** n
    p1 = n * p * ((1.0 - p) ** (n - 1))
    return max(0.0, min(1.0, 1.0 - p0 - p1))


def _score(
    held_slot_lift: float,
    held_class_lift: float,
    two_plus_slots_lift: float,
    all_slots_lift: float,
    mixed_query_lift: float,
    mixed_two_plus_slots_lift: float,
    mixed_all_slots_lift: float,
    auc_lift: float,
) -> float:
    return round(
        10.0
        * (
            0.25 * clip01(held_slot_lift)
            + 0.05 * clip01(held_class_lift)
            + 0.20 * clip01(two_plus_slots_lift)
            + 0.10 * clip01(all_slots_lift)
            + 0.15 * clip01(mixed_query_lift)
            + 0.10 * clip01(mixed_two_plus_slots_lift)
            + 0.05 * clip01(mixed_all_slots_lift)
            + 0.10 * clip01(auc_lift)
        ),
        4,
    )


def _err_result(t0: float, status: str, error: str) -> BindingMultislotResult:
    return BindingMultislotResult(
        status=status,
        error=error,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


def binding_multislot_probe(
    model: nn.Module,
    *,
    cfg: BindingMultislotConfig | None = None,
    device: str = "cuda",
) -> BindingMultislotResult:
    cfg = cfg or BindingMultislotConfig()
    t0 = time.perf_counter()
    dev = torch.device(device)
    try:
        probe_model = (
            safe_deepcopy_module(model).to(dev) if cfg.copy_model else model.to(dev)
        )
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "copy_failed", str(exc))

    try:
        layout = build_multi_blank_layout(cfg)
        model_vocab = model_vocab_size(probe_model)
        if model_vocab is not None and int(model_vocab) < layout.required_vocab:
            return _err_result(
                t0,
                "error",
                f"model_vocab_too_small:{model_vocab}<required:{layout.required_vocab}",
            )
        layout = _layout_to_device(layout, dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(cfg.seed))
        opt = make_adamw(
            probe_model.parameters(),
            lr=float(cfg.lr),
            fused_if_available=(dev.type == "cuda"),
        )
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
                ids, targets, _classes, ans_positions = make_multi_blank_batch(
                    layout,
                    split="train",
                    batch_size=int(cfg.batch_size),
                    bindings_per_example=int(cfg.bindings_per_example),
                    query_slots=int(cfg.query_slots),
                    device=dev,
                    generator=gen,
                )
                loss = _train_one_batch(
                    probe_model,
                    ids,
                    targets,
                    ans_positions,
                    opt=opt,
                    layout=layout,
                )
                if loss is None:
                    status = "error"
                    error = "non_finite_loss"
                    break
                steps_done = step
                if step % eval_every == 0 or step == int(cfg.train_steps):
                    (
                        train_slot,
                        _train_class,
                        _train_two_plus,
                        _train_all,
                        _train_mixed,
                        _train_mixed_two_plus,
                        _train_mixed_all,
                    ) = _evaluate_split(
                        probe_model,
                        layout,
                        split="train",
                        n_eval=int(cfg.n_eval),
                        batch_size=int(cfg.batch_size),
                        bindings_per_example=int(cfg.bindings_per_example),
                        query_slots=int(cfg.query_slots),
                        device=dev,
                        seed=int(cfg.seed) + 10_000 + step,
                    )
                    (
                        held_slot,
                        held_class,
                        two_plus,
                        all_slots,
                        mixed,
                        mixed_two_plus,
                        mixed_all,
                    ) = _evaluate_split(
                        probe_model,
                        layout,
                        split="held_entity",
                        n_eval=int(cfg.n_eval),
                        batch_size=int(cfg.batch_size),
                        bindings_per_example=int(cfg.bindings_per_example),
                        query_slots=int(cfg.query_slots),
                        device=dev,
                        seed=int(cfg.seed) + 20_000 + step,
                    )
                    slot_chance = 1.0 / float(layout.value_hi - layout.value_lo)
                    class_chance = 0.5
                    two_plus_chance = _two_plus_chance(
                        slot_chance,
                        int(cfg.query_slots),
                    )
                    all_slots_chance = slot_chance ** int(cfg.query_slots)
                    learning_curve.append(
                        {
                            "step": step,
                            "loss": round(float(loss.item()), 6),
                            "train_slot_acc": round(train_slot, 4),
                            "held_entity_slot_acc": round(held_slot, 4),
                            "held_entity_class_acc": round(held_class, 4),
                            "two_plus_slots_acc": round(two_plus, 4),
                            "all_slots_acc": round(all_slots, 4),
                            "mixed_query_acc": round(mixed, 4),
                            "mixed_two_plus_slots_acc": round(mixed_two_plus, 4),
                            "mixed_all_slots_acc": round(mixed_all, 4),
                            "held_slot_lift": round(
                                chance_lift(held_slot, slot_chance),
                                4,
                            ),
                            "held_class_lift": round(
                                chance_lift(held_class, class_chance),
                                4,
                            ),
                            "two_plus_slots_lift": round(
                                chance_lift(two_plus, two_plus_chance),
                                4,
                            ),
                            "all_slots_lift": round(
                                chance_lift(all_slots, all_slots_chance),
                                4,
                            ),
                            "mixed_query_lift": round(
                                chance_lift(mixed, slot_chance),
                                4,
                            ),
                            "mixed_two_plus_slots_lift": round(
                                chance_lift(mixed_two_plus, two_plus_chance),
                                4,
                            ),
                            "mixed_all_slots_lift": round(
                                chance_lift(mixed_all, all_slots_chance),
                                4,
                            ),
                        }
                    )

            (
                train_slot,
                _train_class,
                _train_two_plus,
                _train_all,
                _train_mixed,
                _train_mixed_two_plus,
                _train_mixed_all,
            ) = _evaluate_split(
                probe_model,
                layout,
                split="train",
                n_eval=int(cfg.n_eval),
                batch_size=int(cfg.batch_size),
                bindings_per_example=int(cfg.bindings_per_example),
                query_slots=int(cfg.query_slots),
                device=dev,
                seed=int(cfg.seed) + 30_000,
            )
            (
                held_slot,
                held_class,
                two_plus,
                all_slots,
                mixed,
                mixed_two_plus,
                mixed_all,
            ) = _evaluate_split(
                probe_model,
                layout,
                split="held_entity",
                n_eval=int(cfg.n_eval),
                batch_size=int(cfg.batch_size),
                bindings_per_example=int(cfg.bindings_per_example),
                query_slots=int(cfg.query_slots),
                device=dev,
                seed=int(cfg.seed) + 40_000,
            )

        slot_chance = 1.0 / float(layout.value_hi - layout.value_lo)
        class_chance = 0.5
        two_plus_chance = _two_plus_chance(slot_chance, int(cfg.query_slots))
        all_slots_chance = slot_chance ** int(cfg.query_slots)
        early, final, best, auc, slope = probe_curve_summary(
            learning_curve,
            metric_key="held_entity_slot_acc",
            final_step=steps_done,
        )
        improvement = final - early
        held_slot_lift = chance_lift(held_slot, slot_chance)
        held_class_lift = chance_lift(held_class, class_chance)
        two_plus_slots_lift = chance_lift(two_plus, two_plus_chance)
        all_slots_lift = chance_lift(all_slots, all_slots_chance)
        mixed_query_lift = chance_lift(mixed, slot_chance)
        mixed_two_plus_slots_lift = chance_lift(mixed_two_plus, two_plus_chance)
        mixed_all_slots_lift = chance_lift(mixed_all, all_slots_chance)
        auc_lift = chance_lift(auc, slot_chance)
        return BindingMultislotResult(
            train_slot_acc=round(train_slot, 4),
            held_entity_slot_acc=round(held_slot, 4),
            held_entity_class_acc=round(held_class, 4),
            two_plus_slots_acc=round(two_plus, 4),
            all_slots_acc=round(all_slots, 4),
            mixed_query_acc=round(mixed, 4),
            mixed_two_plus_slots_acc=round(mixed_two_plus, 4),
            mixed_all_slots_acc=round(mixed_all, 4),
            slot_chance_acc=round(slot_chance, 6),
            class_chance_acc=round(class_chance, 6),
            two_plus_slots_chance_acc=round(two_plus_chance, 6),
            all_slots_chance_acc=round(all_slots_chance, 6),
            held_slot_lift=round(held_slot_lift, 4),
            held_class_lift=round(held_class_lift, 4),
            two_plus_slots_lift=round(two_plus_slots_lift, 4),
            all_slots_lift=round(all_slots_lift, 4),
            mixed_query_lift=round(mixed_query_lift, 4),
            mixed_two_plus_slots_lift=round(mixed_two_plus_slots_lift, 4),
            mixed_all_slots_lift=round(mixed_all_slots_lift, 4),
            early_slot_acc=round(early, 4),
            final_slot_acc=round(final, 4),
            best_slot_acc=round(best, 4),
            improvement=round(improvement, 4),
            slope_per_100_steps=round(slope, 6),
            auc=round(auc, 4),
            auc_lift=round(auc_lift, 4),
            learning_curve=learning_curve,
            steps_to_threshold=probe_steps_to_threshold(
                learning_curve,
                metric_key="held_entity_slot_acc",
                threshold=float(cfg.threshold),
            ),
            score=_score(
                held_slot_lift,
                held_class_lift,
                two_plus_slots_lift,
                all_slots_lift,
                mixed_query_lift,
                mixed_two_plus_slots_lift,
                mixed_all_slots_lift,
                auc_lift,
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
    "BINDING_MULTISLOT_METRIC_VERSION",
    "BindingMultislotConfig",
    "BindingMultislotResult",
    "MultiBlankLayout",
    "build_multi_blank_layout",
    "make_multi_blank_batch",
    "binding_multislot_probe",
]
