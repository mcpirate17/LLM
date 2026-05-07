#!/usr/bin/env python
"""GPU calibration harness for the natural-language Small-AR story corpus.

This uses ``cl100k_base`` BPE tokenization end to end.  It trains compact
reference models on story-local codebook examples and evaluates in-episode
multiple-choice ranking; no byte-token path is used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.eval.small_ar_champion_corpus import (
    SmallARStoryCorpusConfig,
    StoryBinding,
    StoryExample,
    StoryQuery,
    build_small_ar_story_corpus,
    _key_space,
    _story,
    _value_space,
)
from research.eval.utils import _get_tiktoken_encoder, clip_grad_norm, make_adamw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "research/runtime/small_ar_calibration"
STORY_CALIBRATION_VERSION = "small_ar_story_calibration_v1_bpe"
TIKTOKEN_ENCODING = "cl100k_base"
PAD_ID = 0


@dataclass(frozen=True, slots=True)
class StoryCalibrationConfig:
    seed: int = 0
    train_steps: int = 1_000
    eval_every: int = 250
    batch_size: int = 4
    lr: float = 1e-3
    dim: int = 96
    layers: int = 2
    heads: int = 4
    timeout_s: float = 900.0
    loss_mode: str = "generative"
    score_mode: str = "conditional"
    context_keep_fraction: float = 1.0
    dynamic_train_stories: bool = False
    micro_retrieval: bool = False
    micro_counterfactual_train: bool = True
    include_in_story_unqueried_eval: bool = False
    corpus: SmallARStoryCorpusConfig = SmallARStoryCorpusConfig()


@dataclass(frozen=True, slots=True)
class EncodedStoryItem:
    story_id: int
    split: str
    prompt: str
    answer: str
    choices: tuple[str, ...]
    prefix_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]

    @property
    def full_ids(self) -> tuple[int, ...]:
        return self.prefix_ids + self.answer_ids


class TinyCausalAttentionLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        max_seq_len: int,
        dim: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed = nn.Embedding(self.vocab_size, int(dim))
        self.pos = nn.Embedding(int(max_seq_len), int(dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(dim),
            nhead=int(heads),
            dim_feedforward=int(dim) * 4,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=int(layers))
        self.norm = nn.LayerNorm(int(dim))
        self.head = nn.Linear(int(dim), self.vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = int(input_ids.shape[1])
        if seq_len > int(self.pos.num_embeddings):
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len "
                f"{int(self.pos.num_embeddings)}"
            )
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.embed(input_ids) + self.pos(pos).unsqueeze(0)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.head(self.norm(self.layers(x, mask=mask)))


class NoContextEmbeddingLM(nn.Module):
    def __init__(self, vocab_size: int, *, dim: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed = nn.Embedding(self.vocab_size, int(dim))
        self.head = nn.Linear(int(dim), self.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))


def _encode_text(enc: Any, text: str) -> tuple[int, ...]:
    return tuple(int(i) for i in enc.encode(text, allowed_special=set()))


def _story_context(story: StoryExample) -> str:
    return "\n".join([f"Story {story.story_id}.", *story.context_sentences()])


def _query_context(
    story: StoryExample,
    query: StoryQuery,
    *,
    keep_fraction: float = 1.0,
    context_mode: str = "full",
) -> str:
    sentences = list(story.context_sentences())
    target_binding = next(
        (binding for binding in story.bindings if binding.key.text == query.key.text),
        None,
    )
    if context_mode == "missing_target":
        if target_binding is not None:
            sentences = [
                sentence
                for sentence in sentences
                if sentence != target_binding.sentence()
            ]
    elif context_mode == "counterfactual_target":
        if target_binding is not None:
            wrong = next(choice for choice in query.choices if choice != query.answer)
            sentences = [
                (
                    target_binding.template.format(
                        key=target_binding.key.text,
                        value=wrong,
                    )
                    if sentence == target_binding.sentence()
                    else sentence
                )
                for sentence in sentences
            ]
    elif context_mode != "full":
        raise ValueError(f"unknown context_mode: {context_mode}")
    if float(keep_fraction) >= 0.999:
        return "\n".join([f"Story {story.story_id}.", *sentences])
    if float(keep_fraction) <= 0.0:
        raise ValueError("context_keep_fraction must be positive")
    target_sentence = target_binding.sentence() if target_binding is not None else None
    if context_mode == "missing_target":
        target_sentence = None
    keep_n = max(1, math.ceil(len(sentences) * float(keep_fraction)))
    seed_text = f"{story.story_id}|{query.key.text}|{query.answer}"
    rng = random.Random(sum(ord(ch) for ch in seed_text))
    candidates = [sentence for sentence in sentences if sentence != target_sentence]
    rng.shuffle(candidates)
    selected = set(candidates[: max(0, keep_n - int(target_sentence is not None))])
    if target_sentence is not None:
        selected.add(target_sentence)
    trimmed = [sentence for sentence in sentences if sentence in selected]
    return "\n".join([f"Story {story.story_id}.", *trimmed])


def _answer_suffix(enc: Any, answer: str) -> tuple[int, ...]:
    ids = _encode_text(enc, f" {answer}.")
    if not ids:
        raise ValueError(f"empty answer tokenization for {answer!r}")
    return ids


def _deterministic_choices(
    answer: str,
    story_values: list[str],
    *,
    n_choices: int = 4,
) -> tuple[str, ...]:
    distractors = [value for value in story_values if value != answer]
    out = [answer, *distractors[: max(0, int(n_choices) - 1)]]
    seed = sum(ord(ch) for ch in "|".join([answer, *story_values]))
    random.Random(seed).shuffle(out)
    return tuple(out)


def in_story_unqueried_queries(
    stories: tuple[StoryExample, ...],
    *,
    choices_per_query: int = 4,
) -> tuple[StoryQuery, ...]:
    queries: list[StoryQuery] = []
    for story in stories:
        queried_keys = {query.key.text for query in story.queries}
        story_values = [binding.value for binding in story.bindings]
        for binding in story.bindings:
            if binding.key.text in queried_keys:
                continue
            queries.append(
                StoryQuery(
                    story_id=story.story_id,
                    split="in_story_unqueried",
                    key=binding.key,
                    answer=binding.value,
                    prompt=(
                        f"Question: In Story {story.story_id}, what does the "
                        f"{binding.key.text} mean?"
                    ),
                    choices=_deterministic_choices(
                        binding.value,
                        story_values,
                        n_choices=int(choices_per_query),
                    ),
                )
            )
    return tuple(queries)


def encode_queries_with_context(
    stories: tuple[StoryExample, ...],
    queries: tuple[StoryQuery, ...],
    enc: Any,
    *,
    context_keep_fraction: float = 1.0,
    context_mode: str = "full",
    split_suffix: str = "",
) -> list[EncodedStoryItem]:
    story_by_id = {story.story_id: story for story in stories}
    items: list[EncodedStoryItem] = []
    for query in queries:
        context = _query_context(
            story_by_id[query.story_id],
            query,
            keep_fraction=float(context_keep_fraction),
            context_mode=context_mode,
        )
        prefix = f"{context}\n{query.prompt}\nAnswer:"
        items.append(
            EncodedStoryItem(
                story_id=query.story_id,
                split=f"{query.split}{split_suffix}",
                prompt=query.prompt,
                answer=query.answer,
                choices=query.choices,
                prefix_ids=_encode_text(enc, prefix),
                answer_ids=_answer_suffix(enc, query.answer),
            )
        )
    return items


def encode_story_items(
    stories: tuple[StoryExample, ...],
    enc: Any,
    *,
    context_keep_fraction: float = 1.0,
    context_mode: str = "full",
    split_suffix: str = "",
) -> list[EncodedStoryItem]:
    items: list[EncodedStoryItem] = []
    for story in stories:
        for query in story.queries:
            context = _query_context(
                story,
                query,
                keep_fraction=float(context_keep_fraction),
                context_mode=context_mode,
            )
            prefix = f"{context}\n{query.prompt}\nAnswer:"
            items.append(
                EncodedStoryItem(
                    story_id=story.story_id,
                    split=f"{query.split}{split_suffix}",
                    prompt=query.prompt,
                    answer=query.answer,
                    choices=query.choices,
                    prefix_ids=_encode_text(enc, prefix),
                    answer_ids=_answer_suffix(enc, query.answer),
                )
            )
    return items


def dynamic_story_train_items(
    cfg: SmallARStoryCorpusConfig,
    enc: Any,
    *,
    rng: random.Random,
    batch_size: int,
    step: int,
    context_keep_fraction: float = 1.0,
) -> list[EncodedStoryItem]:
    """Generate fresh story-local bindings and one query per batch item."""
    key_pool = _key_space(cfg)
    value_pool = _value_space(cfg)
    one_query_cfg = replace(cfg, queries_per_story=1)
    items: list[EncodedStoryItem] = []
    for row in range(int(batch_size)):
        keys = rng.sample(key_pool, int(cfg.bindings_per_story))
        values = rng.sample(value_pool, int(cfg.bindings_per_story))
        story = _story(
            story_id=int(step) * max(1, int(batch_size)) + row,
            split="dynamic_train",
            keys=keys,
            values=values,
            rng=rng,
            cfg=one_query_cfg,
        )
        items.extend(
            encode_story_items(
                (story,),
                enc,
                context_keep_fraction=float(context_keep_fraction),
            )
        )
    return items


def _micro_story(
    cfg: SmallARStoryCorpusConfig,
    *,
    rng: random.Random,
    story_id: int,
    split: str,
) -> StoryExample:
    key_pool = _key_space(cfg)
    value_pool = _value_space(cfg)
    keys = rng.sample(key_pool, int(cfg.bindings_per_story))
    values = rng.sample(value_pool, int(cfg.bindings_per_story))
    bindings = tuple(
        StoryBinding(
            key=key,
            value=value,
            template="In this story, the {key} means {value}.",
        )
        for key, value in zip(keys, values)
    )
    noise: list[str] = []
    for idx in range(int(cfg.noise_sentences_per_story)):
        binding = bindings[idx % len(bindings)]
        if idx % 3 == 0:
            noise.append(f"The {binding.key.text} was seen near the quiet shelf.")
        elif idx % 3 == 1:
            noise.append(f"The word {binding.value} was copied onto a side note.")
        else:
            other = bindings[(idx + 1) % len(bindings)]
            noise.append(
                f"The {binding.key.text} and the {other.key.text} appeared together."
            )
    target_idx = rng.randrange(len(bindings))
    target = bindings[target_idx]
    distractor_values = [binding.value for binding in bindings if binding != target]
    choices = _deterministic_choices(
        target.value,
        [target.value, distractor_values[0]],
        n_choices=int(cfg.choices_per_query),
    )
    query = StoryQuery(
        story_id=story_id,
        split=split,
        key=target.key,
        answer=target.value,
        prompt=f"Question: What does the {target.key.text} mean?",
        choices=choices,
    )
    return StoryExample(
        story_id=story_id,
        split=split,
        bindings=bindings,
        noise_sentences=tuple(noise),
        queries=(query,),
    )


def micro_retrieval_items(
    cfg: SmallARStoryCorpusConfig,
    enc: Any,
    *,
    rng: random.Random,
    n_stories: int,
    split: str,
    story_id_start: int = 0,
    context_keep_fraction: float = 1.0,
    context_mode: str = "full",
    split_suffix: str = "",
    include_counterfactual_answer: bool = False,
) -> list[EncodedStoryItem]:
    stories = tuple(
        _micro_story(
            cfg,
            rng=rng,
            story_id=int(story_id_start) + idx,
            split=split,
        )
        for idx in range(int(n_stories))
    )
    items = encode_story_items(
        stories,
        enc,
        context_keep_fraction=float(context_keep_fraction),
        context_mode=context_mode,
        split_suffix=split_suffix,
    )
    if include_counterfactual_answer:
        for story in stories:
            query = story.queries[0]
            wrong = next(choice for choice in query.choices if choice != query.answer)
            context = _query_context(
                story,
                query,
                keep_fraction=float(context_keep_fraction),
                context_mode="counterfactual_target",
            )
            prefix = f"{context}\n{query.prompt}\nAnswer:"
            items.append(
                EncodedStoryItem(
                    story_id=query.story_id,
                    split=f"{query.split}_counterfactual_train",
                    prompt=query.prompt,
                    answer=wrong,
                    choices=query.choices,
                    prefix_ids=_encode_text(enc, prefix),
                    answer_ids=_answer_suffix(enc, wrong),
                )
            )
    return items


def dynamic_micro_retrieval_train_items(
    cfg: SmallARStoryCorpusConfig,
    enc: Any,
    *,
    rng: random.Random,
    batch_size: int,
    step: int,
    context_keep_fraction: float = 1.0,
    include_counterfactual_answer: bool = True,
) -> list[EncodedStoryItem]:
    return micro_retrieval_items(
        cfg,
        enc,
        rng=rng,
        n_stories=int(batch_size),
        split="micro_train",
        story_id_start=int(step) * max(1, int(batch_size)),
        context_keep_fraction=float(context_keep_fraction),
        include_counterfactual_answer=bool(include_counterfactual_answer),
    )


def _pack_training_batch(
    items: list[EncodedStoryItem],
    indices: list[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [items[i] for i in indices]
    max_len = max(len(item.full_ids) for item in selected)
    ids_np = np.full((len(selected), max_len), PAD_ID, dtype=np.int64)
    labels_np = np.full((len(selected), max_len), -100, dtype=np.int64)
    for row, item in enumerate(selected):
        ids = item.full_ids
        ids_np[row, : len(ids)] = ids
        start = len(item.prefix_ids)
        labels_np[row, start : len(ids)] = item.answer_ids
    return (
        torch.from_numpy(ids_np).to(device),
        torch.from_numpy(labels_np).to(device),
    )


def _generative_train_step(
    model: nn.Module,
    items: list[EncodedStoryItem],
    *,
    rng: random.Random,
    batch_size: int,
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> float | None:
    indices = [rng.randrange(len(items)) for _ in range(int(batch_size))]
    ids, labels = _pack_training_batch(items, indices, device=device)
    opt.zero_grad(set_to_none=True)
    logits = model(ids)
    pred = logits[:, :-1, :].contiguous()
    target = labels[:, 1:].contiguous()
    mask = target != -100
    if not bool(mask.any()):
        return None
    loss = F.cross_entropy(pred[mask].float(), target[mask])
    if not torch.isfinite(loss):
        return None
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach().item())


def _pack_choice_batch(
    selected: list[EncodedStoryItem],
    enc: Any,
    *,
    device: torch.device,
    prefix_mode: str = "context",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows: list[tuple[int, ...]] = []
    labels: list[tuple[int, ...]] = []
    candidate_counts: list[int] = []
    target_indices: list[int] = []
    max_len = 0
    for item in selected:
        candidate_counts.append(len(item.choices))
        target_indices.append(item.choices.index(item.answer))
        if prefix_mode == "context":
            prefix_ids = item.prefix_ids
        elif prefix_mode == "query":
            prefix_ids = _encode_text(enc, f"{item.prompt}\nAnswer:")
        else:
            raise ValueError(f"unknown prefix_mode: {prefix_mode}")
        for choice in item.choices:
            suffix = _answer_suffix(enc, choice)
            ids = prefix_ids + suffix
            label = (-100,) * len(prefix_ids) + suffix
            rows.append(ids)
            labels.append(label)
            max_len = max(max_len, len(ids))
    ids_np = np.full((len(rows), max_len), PAD_ID, dtype=np.int64)
    labels_np = np.full((len(rows), max_len), -100, dtype=np.int64)
    for row, ids in enumerate(rows):
        ids_np[row, : len(ids)] = ids
        labels_np[row, : len(labels[row])] = labels[row]
    return (
        torch.from_numpy(ids_np).to(device),
        torch.from_numpy(labels_np).to(device),
        torch.as_tensor(candidate_counts, dtype=torch.long, device=device),
        torch.as_tensor(target_indices, dtype=torch.long, device=device),
    )


def _choice_row_scores(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pred = logits[:, :-1, :].float()
    target = labels[:, 1:].contiguous()
    mask = target != -100
    if not bool(mask.any()):
        raise ValueError("choice batch contains no answer tokens")
    log_probs = F.log_softmax(pred, dim=-1)
    safe_target = target.masked_fill(~mask, 0)
    token_scores = log_probs.gather(2, safe_target.unsqueeze(2)).squeeze(2)
    token_scores = token_scores.masked_fill(~mask, 0.0)
    denom = mask.sum(dim=1).clamp_min(1)
    return token_scores.sum(dim=1) / denom


def _choice_train_step(
    model: nn.Module,
    items: list[EncodedStoryItem],
    enc: Any,
    *,
    rng: random.Random,
    batch_size: int,
    opt: torch.optim.Optimizer,
    device: torch.device,
    score_mode: str,
) -> float | None:
    indices = [rng.randrange(len(items)) for _ in range(int(batch_size))]
    selected = [items[i] for i in indices]
    ids, labels, candidate_counts, target_indices = _pack_choice_batch(
        selected,
        enc,
        device=device,
    )
    if int(candidate_counts.unique().numel()) != 1:
        raise ValueError("choice training currently requires fixed choice count")
    n_choices = int(candidate_counts[0].item())
    opt.zero_grad(set_to_none=True)
    logits = model(ids)
    try:
        scores = _choice_row_scores(logits, labels)
    except ValueError:
        return None
    if score_mode == "pmi":
        prior_ids, prior_labels, _, _ = _pack_choice_batch(
            selected,
            enc,
            device=device,
            prefix_mode="query",
        )
        prior_scores = _choice_row_scores(model(prior_ids), prior_labels)
        scores = scores - prior_scores
    elif score_mode != "conditional":
        raise ValueError(f"unknown score_mode: {score_mode}")
    scores = scores.reshape(len(selected), n_choices)
    loss = F.cross_entropy(scores, target_indices)
    if not torch.isfinite(loss):
        return None
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach().item())


def _train_step(
    model: nn.Module,
    items: list[EncodedStoryItem],
    enc: Any,
    *,
    loss_mode: str,
    score_mode: str,
    rng: random.Random,
    batch_size: int,
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> float | None:
    if loss_mode == "generative":
        return _generative_train_step(
            model,
            items,
            rng=rng,
            batch_size=batch_size,
            opt=opt,
            device=device,
        )
    if loss_mode == "choice":
        return _choice_train_step(
            model,
            items,
            enc,
            rng=rng,
            batch_size=batch_size,
            opt=opt,
            device=device,
            score_mode=score_mode,
        )
    raise ValueError(f"unknown loss_mode: {loss_mode}")


def _candidate_score_for_prefix(
    model: nn.Module,
    prefix_ids: tuple[int, ...],
    candidate: str,
    enc: Any,
    *,
    device: torch.device,
) -> float:
    answer_ids = _answer_suffix(enc, candidate)
    ids = torch.tensor([prefix_ids + answer_ids], dtype=torch.long, device=device)
    logits = model(ids)
    start = len(prefix_ids)
    total = 0.0
    for offset, token_id in enumerate(answer_ids):
        pos = start + offset - 1
        log_probs = F.log_softmax(logits[0, pos].float(), dim=-1)
        total += float(log_probs[int(token_id)].item())
    return total / max(1, len(answer_ids))


def _candidate_score(
    model: nn.Module,
    item: EncodedStoryItem,
    candidate: str,
    enc: Any,
    *,
    device: torch.device,
    score_mode: str,
) -> float:
    context_score = _candidate_score_for_prefix(
        model,
        item.prefix_ids,
        candidate,
        enc,
        device=device,
    )
    if score_mode == "conditional":
        return context_score
    if score_mode == "pmi":
        prior_prefix = _encode_text(enc, f"{item.prompt}\nAnswer:")
        prior_score = _candidate_score_for_prefix(
            model,
            prior_prefix,
            candidate,
            enc,
            device=device,
        )
        return context_score - prior_score
    raise ValueError(f"unknown score_mode: {score_mode}")


def _choice_scores_for_items(
    model: nn.Module,
    selected: list[EncodedStoryItem],
    enc: Any,
    *,
    device: torch.device,
    score_mode: str,
) -> list[dict[str, float]]:
    ids, labels, candidate_counts, _ = _pack_choice_batch(selected, enc, device=device)
    scores = _choice_row_scores(model(ids), labels)
    if score_mode == "pmi":
        prior_ids, prior_labels, _, _ = _pack_choice_batch(
            selected,
            enc,
            device=device,
            prefix_mode="query",
        )
        scores = scores - _choice_row_scores(model(prior_ids), prior_labels)
    elif score_mode != "conditional":
        raise ValueError(f"unknown score_mode: {score_mode}")
    out: list[dict[str, float]] = []
    offset = 0
    scores_list = [float(v) for v in scores.detach().cpu().tolist()]
    for item, n_candidates in zip(selected, candidate_counts.tolist()):
        count = int(n_candidates)
        out.append(
            {
                choice: scores_list[offset + idx]
                for idx, choice in enumerate(item.choices[:count])
            }
        )
        offset += count
    return out


@torch.no_grad()
def evaluate_choice_rank(
    model: nn.Module,
    items: list[EncodedStoryItem],
    enc: Any,
    *,
    device: torch.device,
    limit: int | None = None,
    score_mode: str = "conditional",
) -> dict[str, Any]:
    model.eval()
    eval_items = items[: int(limit)] if limit is not None else items
    counts: dict[str, int] = {}
    correct: dict[str, int] = {}
    margins: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []
    for start in range(0, len(eval_items), 64):
        chunk = eval_items[start : start + 64]
        chunk_scores = _choice_scores_for_items(
            model,
            chunk,
            enc,
            device=device,
            score_mode=score_mode,
        )
        for item, scores in zip(chunk, chunk_scores):
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            pred = ranked[0][0]
            best_wrong = max(
                score for choice, score in scores.items() if choice != item.answer
            )
            margin = float(scores[item.answer] - best_wrong)
            split = item.split
            counts[split] = counts.get(split, 0) + 1
            correct[split] = correct.get(split, 0) + int(pred == item.answer)
            margins.setdefault(split, []).append(margin)
            rows.append(
                {
                    "story_id": item.story_id,
                    "split": split,
                    "answer": item.answer,
                    "pred": pred,
                    "correct": pred == item.answer,
                    "margin": round(margin, 6),
                    "scores": {k: round(v, 6) for k, v in scores.items()},
                }
            )
    total = sum(counts.values())
    total_correct = sum(correct.values())
    by_split = {
        split: {
            "n": counts[split],
            "choice_acc": round(correct.get(split, 0) / max(counts[split], 1), 4),
            "mean_margin": round(
                sum(margins.get(split, [0.0])) / max(len(margins.get(split, [])), 1),
                6,
            ),
        }
        for split in sorted(counts)
    }
    return {
        "n": total,
        "choice_acc": round(total_correct / max(total, 1), 4),
        "by_split": by_split,
        "examples": rows[:10],
    }


def build_model(
    family: str, *, vocab_size: int, max_seq_len: int, cfg: StoryCalibrationConfig
) -> nn.Module:
    if family == "attention":
        return TinyCausalAttentionLM(
            vocab_size,
            max_seq_len=max_seq_len,
            dim=int(cfg.dim),
            layers=int(cfg.layers),
            heads=int(cfg.heads),
        )
    if family == "no_context":
        return NoContextEmbeddingLM(vocab_size, dim=int(cfg.dim))
    raise ValueError(f"unknown family: {family}")


def run_family(
    family: str,
    *,
    cfg: StoryCalibrationConfig,
    train_items: list[EncodedStoryItem],
    eval_items: list[EncodedStoryItem],
    enc: Any,
    vocab_size: int,
    max_seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(cfg.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = build_model(
        family, vocab_size=vocab_size, max_seq_len=max_seq_len, cfg=cfg
    ).to(device)
    opt = make_adamw(model.parameters(), lr=float(cfg.lr))
    rng = random.Random(int(cfg.seed) + 17)
    deadline = time.perf_counter() + float(cfg.timeout_s)
    curve: list[dict[str, Any]] = []
    status = "ok"
    steps_done = 0
    t0 = time.perf_counter()
    try:
        for step in range(1, int(cfg.train_steps) + 1):
            if time.perf_counter() > deadline:
                status = "timeout"
                break
            model.train()
            step_train_items = (
                (
                    dynamic_micro_retrieval_train_items(
                        cfg.corpus,
                        enc,
                        rng=rng,
                        batch_size=int(cfg.batch_size),
                        step=step,
                        context_keep_fraction=float(cfg.context_keep_fraction),
                        include_counterfactual_answer=bool(
                            cfg.micro_counterfactual_train
                        ),
                    )
                    if cfg.micro_retrieval
                    else dynamic_story_train_items(
                        cfg.corpus,
                        enc,
                        rng=rng,
                        batch_size=int(cfg.batch_size),
                        step=step,
                        context_keep_fraction=float(cfg.context_keep_fraction),
                    )
                )
                if cfg.dynamic_train_stories
                else train_items
            )
            loss = _train_step(
                model,
                step_train_items,
                enc,
                loss_mode=str(cfg.loss_mode),
                score_mode=str(cfg.score_mode),
                rng=rng,
                batch_size=int(cfg.batch_size),
                opt=opt,
                device=device,
            )
            if loss is None:
                status = "error"
                break
            steps_done = step
            if step % int(cfg.eval_every) == 0 or step == int(cfg.train_steps):
                train_metrics = evaluate_choice_rank(
                    model,
                    train_items,
                    enc,
                    device=device,
                    limit=min(96, len(train_items)),
                    score_mode=str(cfg.score_mode),
                )
                metrics = evaluate_choice_rank(
                    model,
                    eval_items,
                    enc,
                    device=device,
                    limit=min(96, len(eval_items)),
                    score_mode=str(cfg.score_mode),
                )
                curve.append(
                    {
                        "step": step,
                        "loss": round(loss, 6),
                        "train_choice_acc": train_metrics["choice_acc"],
                        "choice_acc": metrics["choice_acc"],
                        "by_split": metrics["by_split"],
                    }
                )
        train_final = evaluate_choice_rank(
            model,
            train_items,
            enc,
            device=device,
            limit=min(256, len(train_items)),
            score_mode=str(cfg.score_mode),
        )
        final = evaluate_choice_rank(
            model,
            eval_items,
            enc,
            device=device,
            score_mode=str(cfg.score_mode),
        )
        return {
            "family": family,
            "status": status,
            "steps_done": steps_done,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "learning_curve": curve,
            "train_final": train_final,
            "final": final,
        }
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_calibration(
    cfg: StoryCalibrationConfig,
    *,
    families: tuple[str, ...],
    device: str,
) -> dict[str, Any]:
    enc = _get_tiktoken_encoder(TIKTOKEN_ENCODING)
    corpus = build_small_ar_story_corpus(cfg.corpus)
    if cfg.micro_retrieval:
        train_items = micro_retrieval_items(
            cfg.corpus,
            enc,
            rng=random.Random(int(cfg.seed) + 101),
            n_stories=max(64, int(cfg.batch_size) * 8),
            split="micro_reference",
            story_id_start=0,
            context_keep_fraction=float(cfg.context_keep_fraction),
        )
        eval_items = micro_retrieval_items(
            cfg.corpus,
            enc,
            rng=random.Random(int(cfg.seed) + 202),
            n_stories=64,
            split="micro_eval",
            story_id_start=10_000,
            context_keep_fraction=float(cfg.context_keep_fraction),
        )
        eval_items.extend(
            micro_retrieval_items(
                cfg.corpus,
                enc,
                rng=random.Random(int(cfg.seed) + 202),
                n_stories=64,
                split="micro_eval",
                story_id_start=10_000,
                context_keep_fraction=float(cfg.context_keep_fraction),
                context_mode="counterfactual_target",
                split_suffix="_counterfactual_target",
            )
        )
    else:
        train_items = encode_story_items(
            corpus.train_stories,
            enc,
            context_keep_fraction=float(cfg.context_keep_fraction),
        )
        eval_items = encode_story_items(
            corpus.eval_stories,
            enc,
            context_keep_fraction=float(cfg.context_keep_fraction),
        )
        if cfg.include_in_story_unqueried_eval:
            eval_items.extend(
                encode_queries_with_context(
                    corpus.train_stories,
                    in_story_unqueried_queries(
                        corpus.train_stories,
                        choices_per_query=int(cfg.corpus.choices_per_query),
                    ),
                    enc,
                    context_keep_fraction=float(cfg.context_keep_fraction),
                )
            )
    if not train_items or not eval_items:
        raise ValueError("story corpus produced no train/eval items")
    all_items = [*train_items, *eval_items]
    max_seq_len = max(
        max(len(item.full_ids) for item in all_items),
        max(
            len(item.prefix_ids)
            + max(len(_answer_suffix(enc, choice)) for choice in item.choices)
            for item in all_items
        ),
    )
    if cfg.dynamic_train_stories:
        max_seq_len += 128
    vocab_size = int(enc.n_vocab)
    dev = torch.device(device)
    rows = [
        run_family(
            family,
            cfg=cfg,
            train_items=train_items,
            eval_items=eval_items,
            enc=enc,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            device=dev,
        )
        for family in families
    ]
    return {
        "protocol_version": STORY_CALIBRATION_VERSION,
        "tokenizer": TIKTOKEN_ENCODING,
        "config": {
            "seed": cfg.seed,
            "train_steps": cfg.train_steps,
            "eval_every": cfg.eval_every,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "dim": cfg.dim,
            "layers": cfg.layers,
            "heads": cfg.heads,
            "loss_mode": cfg.loss_mode,
            "score_mode": cfg.score_mode,
            "context_keep_fraction": cfg.context_keep_fraction,
            "dynamic_train_stories": cfg.dynamic_train_stories,
            "micro_retrieval": cfg.micro_retrieval,
            "micro_counterfactual_train": cfg.micro_counterfactual_train,
            "include_in_story_unqueried_eval": cfg.include_in_story_unqueried_eval,
            "corpus": asdict(cfg.corpus),
        },
        "corpus": {
            "train_stories": len(corpus.train_stories),
            "eval_stories": len(corpus.eval_stories),
            "train_items": len(train_items),
            "eval_items": len(eval_items),
            "max_seq_len": max_seq_len,
            "sample_train": corpus.train_stories[0].text().splitlines()[:24],
        },
        "rows": rows,
    }


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"small_ar_story_calibration_{stamp}.json"
    md_path = out_dir / f"small_ar_story_calibration_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        f"# Small AR Story Calibration - {stamp}",
        "",
        f"- Protocol: `{report['protocol_version']}`",
        f"- Tokenizer: `{report['tokenizer']}`",
        f"- Max seq len: `{report['corpus']['max_seq_len']}`",
        "",
        "| family | status | steps | train acc | eval acc | in-dist | held-key | cross-story | ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        final = row["final"]
        by_split = final["by_split"]
        in_dist = by_split.get("in_dist", {}).get("choice_acc")
        held = by_split.get("held_key", {}).get("choice_acc")
        cross = by_split.get("cross_story", {}).get("choice_acc")
        lines.append(
            f"| {row['family']} | {row['status']} | {row['steps_done']} | "
            f"{row['train_final']['choice_acc']:.4f} | {final['choice_acc']:.4f} | "
            f"{in_dist if in_dist is not None else ''} | "
            f"{held if held is not None else ''} | "
            f"{cross if cross is not None else ''} | {row['elapsed_ms']:.0f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _preset(
    name: str,
    seed: int,
    queries_per_story: int | None = None,
    bindings_per_story: int | None = None,
    noise_sentences_per_story: int | None = None,
) -> SmallARStoryCorpusConfig:
    n_queries = queries_per_story
    if name == "micro_retrieval":
        bindings = int(bindings_per_story or 4)
        return SmallARStoryCorpusConfig(
            seed=seed,
            n_train_stories=1,
            n_in_dist_eval_stories=1,
            n_held_key_stories=0,
            n_cross_story_groups=0,
            bindings_per_story=bindings,
            noise_sentences_per_story=int(noise_sentences_per_story or 0),
            queries_per_story=1,
            choices_per_query=2,
            n_values=max(16, bindings * 4),
        )
    if name == "curriculum00":
        return SmallARStoryCorpusConfig(
            seed=seed,
            n_train_stories=64,
            n_in_dist_eval_stories=16,
            n_held_key_stories=0,
            n_cross_story_groups=0,
            bindings_per_story=4,
            noise_sentences_per_story=0,
            queries_per_story=int(n_queries or 4),
            choices_per_query=2,
            n_values=16,
        )
    if name == "curriculum0":
        return SmallARStoryCorpusConfig(
            seed=seed,
            n_train_stories=64,
            n_in_dist_eval_stories=16,
            n_held_key_stories=0,
            n_cross_story_groups=0,
            bindings_per_story=8,
            noise_sentences_per_story=0,
            queries_per_story=int(n_queries or 8),
            choices_per_query=2,
            n_values=32,
        )
    if name == "curriculum0_binary":
        return SmallARStoryCorpusConfig(
            seed=seed,
            n_train_stories=64,
            n_in_dist_eval_stories=16,
            n_held_key_stories=0,
            n_cross_story_groups=0,
            bindings_per_story=8,
            noise_sentences_per_story=0,
            queries_per_story=int(n_queries or 8),
            choices_per_query=2,
            n_values=32,
        )
    if name == "curriculum0_unqueried":
        return SmallARStoryCorpusConfig(
            seed=seed,
            n_train_stories=64,
            n_in_dist_eval_stories=0,
            n_held_key_stories=0,
            n_cross_story_groups=0,
            bindings_per_story=8,
            noise_sentences_per_story=0,
            queries_per_story=int(n_queries or 4),
            choices_per_query=2,
            n_values=32,
        )
    if name == "curriculum1":
        return SmallARStoryCorpusConfig(
            seed=seed,
            n_train_stories=64,
            n_in_dist_eval_stories=16,
            n_held_key_stories=16,
            n_cross_story_groups=0,
            bindings_per_story=8,
            noise_sentences_per_story=8,
            queries_per_story=int(n_queries or 8),
            choices_per_query=2,
            n_values=32,
        )
    if name == "curriculum2":
        return SmallARStoryCorpusConfig(
            seed=seed,
            n_train_stories=64,
            n_in_dist_eval_stories=16,
            n_held_key_stories=16,
            n_cross_story_groups=8,
            bindings_per_story=12,
            noise_sentences_per_story=16,
            queries_per_story=int(n_queries or 8),
            choices_per_query=2,
            n_values=48,
        )
    if name == "tiny":
        return SmallARStoryCorpusConfig(
            seed=seed,
            n_train_stories=24,
            n_in_dist_eval_stories=0,
            n_held_key_stories=8,
            n_cross_story_groups=4,
            bindings_per_story=8,
            noise_sentences_per_story=16,
            queries_per_story=int(n_queries or 2),
            n_values=48,
        )
    if name == "v3a":
        return SmallARStoryCorpusConfig(
            seed=seed,
            n_train_stories=64,
            n_in_dist_eval_stories=0,
            n_held_key_stories=16,
            n_cross_story_groups=8,
            bindings_per_story=12,
            noise_sentences_per_story=24,
            queries_per_story=int(n_queries or 2),
            n_values=64,
        )
    if name == "v3b":
        return SmallARStoryCorpusConfig(
            seed=seed,
            queries_per_story=int(
                n_queries or SmallARStoryCorpusConfig().queries_per_story
            ),
        )
    raise ValueError(f"unknown preset: {name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=[
            "micro_retrieval",
            "curriculum00",
            "curriculum0",
            "curriculum0_binary",
            "curriculum0_unqueried",
            "curriculum1",
            "curriculum2",
            "tiny",
            "v3a",
            "v3b",
        ],
        default="v3a",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--loss-mode",
        choices=["generative", "choice"],
        default="generative",
    )
    parser.add_argument(
        "--score-mode",
        choices=["conditional", "pmi"],
        default="conditional",
        help="Use raw conditional answer scores or subtract query-only priors.",
    )
    parser.add_argument(
        "--context-keep-fraction",
        type=float,
        default=1.0,
        help="Keep this fraction of story context per query, preserving target binding.",
    )
    parser.add_argument(
        "--dynamic-train-stories",
        action="store_true",
        help="Generate fresh story/query items for every training step.",
    )
    parser.add_argument(
        "--queries-per-story",
        type=int,
        default=None,
        help="Override corpus query density for calibration runs.",
    )
    parser.add_argument(
        "--bindings-per-story",
        type=int,
        default=None,
        help="Override binding count for micro/curriculum calibration runs.",
    )
    parser.add_argument(
        "--noise-sentences-per-story",
        type=int,
        default=None,
        help="Override harmless related-noise sentence count.",
    )
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["attention", "no_context"],
        choices=["attention", "no_context"],
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    cfg = StoryCalibrationConfig(
        seed=int(args.seed),
        train_steps=int(args.train_steps),
        eval_every=int(args.eval_every),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        dim=int(args.dim),
        layers=int(args.layers),
        heads=int(args.heads),
        timeout_s=float(args.timeout_s),
        loss_mode=str(args.loss_mode),
        score_mode=str(args.score_mode),
        context_keep_fraction=float(args.context_keep_fraction),
        dynamic_train_stories=bool(
            args.dynamic_train_stories or str(args.preset) == "micro_retrieval"
        ),
        micro_retrieval=str(args.preset) == "micro_retrieval",
        include_in_story_unqueried_eval=str(args.preset) == "curriculum0_unqueried",
        corpus=_preset(
            str(args.preset),
            int(args.seed),
            queries_per_story=args.queries_per_story,
            bindings_per_story=args.bindings_per_story,
            noise_sentences_per_story=args.noise_sentences_per_story,
        ),
    )
    report = run_calibration(
        cfg, families=tuple(args.families), device=str(args.device)
    )
    json_path, md_path = write_report(report, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    for row in report["rows"]:
        final = row["final"]
        print(
            f"{row['family']}: choice_acc={final['choice_acc']:.4f} "
            f"train_choice_acc={row['train_final']['choice_acc']:.4f} "
            f"by_split={final['by_split']} status={row['status']}"
        )
    if any(
        not math.isfinite(float(row["final"]["choice_acc"])) for row in report["rows"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
