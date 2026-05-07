"""Champion-scale synthetic associative recall probe.

This probe is intentionally larger than Nano-AR while staying synthetic and
cheap to generate: it uses an integer-token key/value corpus with a default
restricted vocabulary span above 1K tokens, longer examples, and disjoint held
pairs. Batch generation is vectorized so the hot path is model training rather
than Python corpus assembly.
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
from .utils import clip_grad_norm, make_adamw

SMALL_AR_CHAMPION_METRIC_VERSION = "small_ar_champion_story_micro_v1"
INTEGER_SMALL_AR_CHAMPION_METRIC_VERSION = "small_ar_champion_v2_calibrated"
DEFAULT_VOCAB_LO = 1000
DEFAULT_KEY_TOKENS = 1024
DEFAULT_VALUE_TOKENS = 96
DEFAULT_VALUE_CLASSES = 12
DEFAULT_TRAIN_PAIRS = 256
DEFAULT_HELD_PAIRS = 64
DEFAULT_PAIRS_PER_EXAMPLE = 12
DEFAULT_TRAIN_STEPS = 5_000
DEFAULT_EVAL_EVERY = 500
DEFAULT_BATCH_SIZE = 16
DEFAULT_EVAL_EXAMPLES = 256
DEFAULT_LR = 1e-3
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_STORY_BINDINGS = 4
DEFAULT_STORY_NOISE_SENTENCES = 0
DEFAULT_STORY_EVAL_EXAMPLES = 64


@dataclass(frozen=True, slots=True)
class SmallARChampionConfig:
    seed: int = 0
    protocol: str = "story_micro"
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
    episodic_values: bool = True
    story_bindings_per_example: int = DEFAULT_STORY_BINDINGS
    story_noise_sentences_per_example: int = DEFAULT_STORY_NOISE_SENTENCES
    story_eval_examples: int = DEFAULT_STORY_EVAL_EXAMPLES


@dataclass(frozen=True, slots=True)
class SmallARPairTable:
    train_keys: torch.Tensor
    train_values: torch.Tensor
    held_keys: torch.Tensor
    held_values: torch.Tensor
    vocab_lo: int
    value_lo: int
    value_hi: int
    n_value_classes: int

    @property
    def total_token_span(self) -> int:
        return int(self.value_hi - self.vocab_lo)


@dataclass(slots=True)
class SmallARChampionResult:
    metric_version: str = SMALL_AR_CHAMPION_METRIC_VERSION
    final_acc: float = 0.0
    held_pair_match_acc: float = 0.0
    held_class_acc: float = 0.0
    learning_curve: list[dict[str, float | int]] = field(default_factory=list)
    steps_to_floor: int | None = None
    score: float = 0.0
    steps_trained: int = 0
    status: str = "ok"
    elapsed_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "small_ar_champion_metric_version": self.metric_version,
            "small_ar_champion_final_acc": self.final_acc,
            "small_ar_champion_held_pair_match_acc": self.held_pair_match_acc,
            "small_ar_champion_held_class_acc": self.held_class_acc,
            "small_ar_champion_learning_curve_json": json.dumps(
                self.learning_curve,
                sort_keys=True,
            ),
            "small_ar_champion_steps_to_floor": self.steps_to_floor,
            "small_ar_champion_score": self.score,
            "small_ar_champion_status": self.status,
            "small_ar_champion_elapsed_ms": self.elapsed_ms,
        }


def _model_vocab_size(model: nn.Module) -> int | None:
    vocab_size = getattr(model, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return int(module.num_embeddings)
    return None


def build_small_ar_pair_table(cfg: SmallARChampionConfig) -> SmallARPairTable:
    """Build deterministic disjoint train/held key/value pairs."""
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
    return SmallARPairTable(
        train_keys=key_tokens[:n_train].contiguous(),
        train_values=values[:n_train].contiguous(),
        held_keys=key_tokens[n_train:].contiguous(),
        held_values=values[n_train:].contiguous(),
        vocab_lo=int(cfg.vocab_lo),
        value_lo=value_lo,
        value_hi=value_lo + int(cfg.n_value_tokens),
        n_value_classes=int(cfg.n_value_classes),
    )


def _table_to_device(table: SmallARPairTable, device: torch.device) -> SmallARPairTable:
    return SmallARPairTable(
        train_keys=table.train_keys.to(device),
        train_values=table.train_values.to(device),
        held_keys=table.held_keys.to(device),
        held_values=table.held_values.to(device),
        vocab_lo=table.vocab_lo,
        value_lo=table.value_lo,
        value_hi=table.value_hi,
        n_value_classes=table.n_value_classes,
    )


def _value_classes(values: torch.Tensor, table: SmallARPairTable) -> torch.Tensor:
    return (values - int(table.value_lo)).remainder(int(table.n_value_classes))


def make_small_ar_batch(
    table: SmallARPairTable,
    *,
    split: str,
    batch_size: int,
    pairs_per_example: int,
    sep_token: int,
    ans_token: int,
    device: torch.device,
    generator: torch.Generator,
    episodic_values: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one vectorized small-AR batch."""
    if split not in {"train", "held"}:
        raise ValueError("split must be 'train' or 'held'")
    if int(pairs_per_example) < 2:
        raise ValueError("pairs_per_example must be at least 2")

    query_keys = table.train_keys if split == "train" else table.held_keys
    query_values = table.train_values if split == "train" else table.held_values
    if query_keys.numel() == 0:
        raise ValueError(f"{split} split has no pairs")

    batch = int(batch_size)
    n_pairs = int(pairs_per_example)
    value_span = int(table.value_hi - table.value_lo)
    if episodic_values and value_span < n_pairs:
        raise ValueError("episodic value span must cover pairs_per_example")
    q_idx = torch.randint(
        0,
        query_keys.shape[0],
        (batch,),
        device=device,
        generator=generator,
    )
    d_idx = torch.randint(
        0,
        table.train_keys.shape[0],
        (batch, n_pairs - 1),
        device=device,
        generator=generator,
    )

    keys = torch.empty((batch, n_pairs, 2), dtype=torch.long, device=device)
    values = torch.empty((batch, n_pairs), dtype=torch.long, device=device)
    keys[:, 0, :] = query_keys.index_select(0, q_idx)
    flat_d = d_idx.reshape(-1)
    keys[:, 1:, :] = table.train_keys.index_select(0, flat_d).reshape(
        batch,
        n_pairs - 1,
        2,
    )
    if episodic_values:
        value_order = torch.argsort(
            torch.rand((batch, value_span), device=device, generator=generator),
            dim=1,
        )[:, :n_pairs]
        values[:, :] = value_order + int(table.value_lo)
    else:
        values[:, 0] = query_values.index_select(0, q_idx)
        values[:, 1:] = table.train_values.index_select(0, flat_d).reshape(
            batch,
            n_pairs - 1,
        )

    order = torch.argsort(
        torch.rand((batch, n_pairs), device=device, generator=generator),
        dim=1,
    )
    gather_keys = order.unsqueeze(-1).expand(-1, -1, 2)
    keys = keys.gather(1, gather_keys)
    values = values.gather(1, order)

    seq_len = 3 * n_pairs + 4
    ids = torch.empty((batch, seq_len), dtype=torch.long, device=device)
    pair_pos = torch.arange(n_pairs, device=device)
    ids[:, pair_pos * 3] = keys[:, :, 0]
    ids[:, pair_pos * 3 + 1] = keys[:, :, 1]
    ids[:, pair_pos * 3 + 2] = values
    sep_pos = 3 * n_pairs
    ids[:, sep_pos] = int(sep_token)
    ids[:, sep_pos + 1] = query_keys[q_idx, 0]
    ids[:, sep_pos + 2] = query_keys[q_idx, 1]
    ids[:, sep_pos + 3] = int(ans_token)
    query_pos = (order == 0).to(torch.long).argmax(dim=1)
    targets = values.gather(1, query_pos.unsqueeze(1)).squeeze(1)
    return ids, targets, _value_classes(targets, table)


@torch.no_grad()
def _evaluate_split(
    model: nn.Module,
    table: SmallARPairTable,
    *,
    split: str,
    n_eval: int,
    batch_size: int,
    pairs_per_example: int,
    sep_token: int,
    ans_token: int,
    device: torch.device,
    seed: int,
    episodic_values: bool,
) -> tuple[float, float]:
    model.eval()
    exact_total = 0
    class_total = 0
    total = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    ans_pos = 3 * int(pairs_per_example) + 3
    remaining = int(n_eval)
    while remaining > 0:
        bs = min(int(batch_size), remaining)
        ids, targets, target_classes = make_small_ar_batch(
            table,
            split=split,
            batch_size=bs,
            pairs_per_example=pairs_per_example,
            sep_token=sep_token,
            ans_token=ans_token,
            device=device,
            generator=gen,
            episodic_values=episodic_values,
        )
        logits = model(ids)
        pred = logits[:, ans_pos, table.value_lo : table.value_hi].argmax(dim=-1)
        pred = pred + int(table.value_lo)
        exact_total += int((pred == targets).sum().item())
        class_total += int((_value_classes(pred, table) == target_classes).sum().item())
        total += bs
        remaining -= bs
    denom = max(total, 1)
    return exact_total / denom, class_total / denom


def _train_one_batch(
    model: nn.Module,
    ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    opt: torch.optim.Optimizer,
    table: SmallARPairTable,
    ans_pos: int,
) -> float | None:
    opt.zero_grad(set_to_none=True)
    logits = model(ids)
    pred = logits[:, ans_pos, table.value_lo : table.value_hi].float()
    loss = F.cross_entropy(pred, targets - int(table.value_lo))
    if not torch.isfinite(loss):
        return None
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach().item())


def _steps_to_learning_floor(
    learning_curve: list[dict[str, float | int]],
    *,
    n_value_tokens: int,
) -> int | None:
    """Return first step where held-pair accuracy reaches its learned floor."""
    if not learning_curve:
        return None
    chance = 1.0 / max(float(n_value_tokens), 1.0)
    values = [
        (
            int(row["step"]),
            float(row.get("held_pair_match_acc") or 0.0),
        )
        for row in learning_curve
    ]
    best = max(acc for _step, acc in values)
    if best < max(0.05, chance * 5.0):
        return None
    floor_band = max(0.02, best * 0.10)
    floor = max(0.0, best - floor_band)
    for step, acc in values:
        if acc >= floor:
            return step
    return values[-1][0]


def _score(
    held_pair: float, held_class: float, steps_to_floor: int | None, train_steps: int
) -> float:
    speed = 0.0
    if steps_to_floor is not None and int(train_steps) > 0:
        speed = max(0.0, min(1.0, 1.0 - float(steps_to_floor) / float(train_steps)))
    return round(
        6.0 * max(0.0, min(1.0, held_pair))
        + 2.0 * max(0.0, min(1.0, held_class))
        + 2.0 * speed,
        4,
    )


def _steps_to_story_floor(
    learning_curve: list[dict[str, float | int]],
) -> int | None:
    if not learning_curve:
        return None
    values = [
        (
            int(row["step"]),
            float(row.get("held_pair_match_acc") or 0.0),
        )
        for row in learning_curve
    ]
    best = max(acc for _step, acc in values)
    if best < 0.60:
        return None
    floor = max(0.0, best - max(0.03, best * 0.05))
    for step, acc in values:
        if acc >= floor:
            return step
    return values[-1][0]


def _err_result(t0: float, status: str, error: str) -> SmallARChampionResult:
    return SmallARChampionResult(
        status=status,
        error=error,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


def _run_integer_small_ar_champion(
    model: nn.Module,
    *,
    cfg: SmallARChampionConfig | None = None,
    device: str = "cuda",
) -> SmallARChampionResult:
    """Run the integer-token small AR v2 protocol on a deepcopy of ``model``."""
    cfg = cfg or SmallARChampionConfig(protocol="integer_v2")
    t0 = time.perf_counter()
    dev = torch.device(device)
    try:
        probe_model = copy.deepcopy(model).to(dev)
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "copy_failed", str(exc))

    try:
        model_vocab = _model_vocab_size(probe_model)
        table = build_small_ar_pair_table(cfg)
        sep_token, ans_token = _get_special_tokens(probe_model)
        required_hi = max(table.value_hi, int(sep_token) + 1, int(ans_token) + 1)
        if model_vocab is not None and model_vocab < required_hi:
            return _err_result(
                t0,
                "error",
                f"model_vocab_too_small:{model_vocab}<required:{required_hi}",
            )
        table = _table_to_device(table, dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(cfg.seed))
        opt = make_adamw(probe_model.parameters(), lr=float(cfg.lr))
        ans_pos = 3 * int(cfg.pairs_per_example) + 3
        deadline = t0 + float(cfg.timeout_s)
        eval_every = max(1, int(cfg.eval_every))
        learning_curve: list[dict[str, float | int]] = []
        steps_done = 0
        status = "ok"
        error = None

        for step in range(1, int(cfg.train_steps) + 1):
            if time.perf_counter() > deadline:
                status = "timeout"
                break
            ids, targets, _classes = make_small_ar_batch(
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
                in_acc, _in_class = _evaluate_split(
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
                learning_curve.append(
                    {
                        "step": step,
                        "loss": round(loss, 6),
                        "final_acc": round(in_acc, 4),
                        "held_pair_match_acc": round(held_pair, 4),
                        "held_class_acc": round(held_class, 4),
                    }
                )

        final_acc, _ = _evaluate_split(
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
        floor_step = _steps_to_learning_floor(
            learning_curve,
            n_value_tokens=int(cfg.n_value_tokens),
        )
        return SmallARChampionResult(
            metric_version=INTEGER_SMALL_AR_CHAMPION_METRIC_VERSION,
            final_acc=round(final_acc, 4),
            held_pair_match_acc=round(held_pair, 4),
            held_class_acc=round(held_class, 4),
            learning_curve=learning_curve,
            steps_to_floor=floor_step,
            score=_score(held_pair, held_class, floor_step, int(cfg.train_steps)),
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


def _run_story_micro_champion(
    model: nn.Module,
    *,
    cfg: SmallARChampionConfig,
    device: str,
) -> SmallARChampionResult:
    """Run the natural-language micro retrieval story protocol."""
    from research.eval.utils import tokenize_string
    from research.tools import small_ar_story_calibration as story

    t0 = time.perf_counter()
    dev = torch.device(device)
    try:
        probe_model = copy.deepcopy(model).to(dev)
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "copy_failed", str(exc))

    model_vocab = _model_vocab_size(probe_model)
    if model_vocab is None:
        model_vocab = 100_277

    def clipped_encode(_enc: Any, text: str) -> tuple[int, ...]:
        return tuple(
            int(i)
            for i in tokenize_string(
                text,
                int(model_vocab),
                tokenizer="cl100k_base",
            )
        )

    def clipped_answer_suffix(_enc: Any, answer: str) -> tuple[int, ...]:
        ids = clipped_encode(_enc, f" {answer}.")
        if not ids:
            raise ValueError(f"empty answer tokenization for {answer!r}")
        return ids

    old_encode = story._encode_text
    old_answer = story._answer_suffix
    story._encode_text = clipped_encode
    story._answer_suffix = clipped_answer_suffix
    try:
        enc = story._get_tiktoken_encoder(story.TIKTOKEN_ENCODING)
        corpus_cfg = story._preset(
            "micro_retrieval",
            int(cfg.seed),
            bindings_per_story=int(cfg.story_bindings_per_example),
            noise_sentences_per_story=int(cfg.story_noise_sentences_per_example),
        )
        train_items = story.micro_retrieval_items(
            corpus_cfg,
            enc,
            rng=__import__("random").Random(int(cfg.seed) + 101),
            n_stories=max(64, int(cfg.batch_size) * 8),
            split="micro_reference",
        )
        n_eval = int(cfg.story_eval_examples)
        eval_items = story.micro_retrieval_items(
            corpus_cfg,
            enc,
            rng=__import__("random").Random(int(cfg.seed) + 202),
            n_stories=n_eval,
            split="micro_eval",
            story_id_start=10_000,
        )
        eval_items.extend(
            story.micro_retrieval_items(
                corpus_cfg,
                enc,
                rng=__import__("random").Random(int(cfg.seed) + 202),
                n_stories=n_eval,
                split="micro_eval",
                story_id_start=10_000,
                context_mode="counterfactual_target",
                split_suffix="_counterfactual_target",
            )
        )
        opt = make_adamw(probe_model.parameters(), lr=float(cfg.lr))
        rng = __import__("random").Random(int(cfg.seed) + 17)
        learning_curve: list[dict[str, float | int]] = []
        deadline = t0 + float(cfg.timeout_s)
        eval_every = max(1, int(cfg.eval_every))
        steps_done = 0
        status = "ok"
        error = None

        for step_idx in range(1, int(cfg.train_steps) + 1):
            if time.perf_counter() > deadline:
                status = "timeout"
                break
            probe_model.train()
            step_items = story.dynamic_micro_retrieval_train_items(
                corpus_cfg,
                enc,
                rng=rng,
                batch_size=int(cfg.batch_size),
                step=step_idx,
                include_counterfactual_answer=True,
            )
            loss = story._train_step(
                probe_model,
                step_items,
                enc,
                loss_mode="choice",
                score_mode="pmi",
                rng=rng,
                batch_size=int(cfg.batch_size),
                opt=opt,
                device=dev,
            )
            if loss is None:
                status = "error"
                error = "non_finite_loss"
                break
            steps_done = step_idx
            if step_idx % eval_every == 0 or step_idx == int(cfg.train_steps):
                metrics = story.evaluate_choice_rank(
                    probe_model,
                    eval_items,
                    enc,
                    device=dev,
                    score_mode="pmi",
                )
                by_split = metrics["by_split"]
                full = float(by_split.get("micro_eval", {}).get("choice_acc") or 0.0)
                counter = float(
                    by_split.get(
                        "micro_eval_counterfactual_target",
                        {},
                    ).get("choice_acc")
                    or 0.0
                )
                learning_curve.append(
                    {
                        "step": step_idx,
                        "loss": round(loss, 6),
                        "final_acc": round(full, 4),
                        "held_pair_match_acc": round(full, 4),
                        "held_class_acc": round(max(0.0, full - counter), 4),
                        "counterfactual_acc": round(counter, 4),
                    }
                )

        train_final = story.evaluate_choice_rank(
            probe_model,
            train_items,
            enc,
            device=dev,
            score_mode="pmi",
        )
        final = story.evaluate_choice_rank(
            probe_model,
            eval_items,
            enc,
            device=dev,
            score_mode="pmi",
        )
        by_split = final["by_split"]
        full = float(by_split.get("micro_eval", {}).get("choice_acc") or 0.0)
        counter = float(
            by_split.get("micro_eval_counterfactual_target", {}).get("choice_acc")
            or 0.0
        )
        sensitivity = max(0.0, full - counter)
        floor_step = _steps_to_story_floor(learning_curve)
        return SmallARChampionResult(
            final_acc=round(
                float(
                    train_final["by_split"].get("micro_reference", {}).get("choice_acc")
                    or 0.0
                ),
                4,
            ),
            held_pair_match_acc=round(full, 4),
            held_class_acc=round(sensitivity, 4),
            learning_curve=learning_curve,
            steps_to_floor=floor_step,
            score=_score(full, sensitivity, floor_step, int(cfg.train_steps)),
            steps_trained=steps_done,
            status=status,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            error=error,
        )
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "error", str(exc))
    finally:
        story._encode_text = old_encode
        story._answer_suffix = old_answer
        del probe_model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def run_small_ar_champion(
    model: nn.Module,
    *,
    cfg: SmallARChampionConfig | None = None,
    device: str = "cuda",
) -> SmallARChampionResult:
    """Run the configured Small-AR champion protocol on a deepcopy of ``model``."""
    cfg = cfg or SmallARChampionConfig()
    if str(cfg.protocol) == "story_micro":
        return _run_story_micro_champion(model, cfg=cfg, device=device)
    if str(cfg.protocol) == "integer_v2":
        return _run_integer_small_ar_champion(model, cfg=cfg, device=device)
    t0 = time.perf_counter()
    return _err_result(t0, "error", f"unknown_small_ar_protocol:{cfg.protocol}")


__all__ = [
    "SMALL_AR_CHAMPION_METRIC_VERSION",
    "SmallARChampionConfig",
    "SmallARChampionResult",
    "SmallARPairTable",
    "build_small_ar_pair_table",
    "make_small_ar_batch",
    "run_small_ar_champion",
]
