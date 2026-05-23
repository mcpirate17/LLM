"""Champion-scale synthetic associative recall probe.

**Status as of 2026-05-09: post-champion / optional tool.** Replaced in the
default screening pipeline by ``research.eval.ar_curriculum_probe``, which has
identical rank correlation (Spearman ρ=1.000 across the 4 reference
architectures) but 3.4× greater discrimination spread per second of compute.

Keep this module for:
  * Champion-tier confirmation runs where the larger corpus (vocab_lo=1000,
    n_keys=1024, n_values=96) is desired for an extra stress test.
  * Calibration of new probes against an established baseline.
  * Existing backfilled leaderboard data (``ar_validation_*`` columns) — these
    are still readable and remain a valid ML training target. Going forward,
    new candidates should be screened via ``ar_curriculum_probe`` instead.

Comparison run: ``research/runtime/ar_curriculum_experiment/probe_compare_v1.md``.

This probe is intentionally larger than AR Gate while staying synthetic and
cheap to generate: it uses an integer-token key/value corpus with a default
restricted vocabulary span above 1K tokens, longer examples, and disjoint held
pairs. Batch generation is vectorized so the hot path is model training rather
than Python corpus assembly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from statistics import pstdev
from typing import Any, TypeAlias

import torch
import torch.nn as nn

from ._kv_pair import (
    KVPairTable,
    KVProbeRuntime,
    build_kv_episode_bank,
    build_kv_pair_table,
    evaluate_kv_episode_bank,
    evaluate_kv_probe_checkpoint,
    kv_table_to_device,
    make_kv_pair_batch,
    run_kv_probe_bank_training_loop,
    run_kv_probe_training_loop,
    train_kv_one_batch,
)
from ._probe_utils import safe_deepcopy_module
from .associative_recall import _get_special_tokens
from .utils import make_adamw, model_vocab_size

ARValidationPairTable: TypeAlias = KVPairTable
make_ar_validation_batch = make_kv_pair_batch
_table_to_device = kv_table_to_device
_train_one_batch = train_kv_one_batch

AR_VALIDATION_METRIC_VERSION = "ar_validation_story_micro_v1"
INTEGER_AR_VALIDATION_METRIC_VERSION = "ar_validation_v2_easy25"
STABLE_AR_VALIDATION_METRIC_VERSION = "ar_validation_v3_stable_size_budget"
DEFAULT_AR_VALIDATION_PROTOCOL = "integer_v2"
STABLE_AR_VALIDATION_PROTOCOL = "integer_v3_stable"
DEFAULT_VOCAB_LO = 1000
DEFAULT_KEY_TOKENS = 1024
DEFAULT_VALUE_TOKENS = 96
DEFAULT_VALUE_CLASSES = 12
DEFAULT_TRAIN_PAIRS = 256
DEFAULT_HELD_PAIRS = 64
DEFAULT_PAIRS_PER_EXAMPLE = 9
DEFAULT_TRAIN_STEPS = 5_000
DEFAULT_EVAL_EVERY = 500
DEFAULT_BATCH_SIZE = 16
DEFAULT_EVAL_EXAMPLES = 256
DEFAULT_LR = 1e-3
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_STABLE_SEED_COUNT = 3
DEFAULT_STABLE_TRAIN_BANK_EXAMPLES = 8192
DEFAULT_STORY_BINDINGS = 4
DEFAULT_STORY_NOISE_SENTENCES = 0
DEFAULT_STORY_EVAL_EXAMPLES = 64


@dataclass(frozen=True, slots=True)
class ARValidationBudget:
    size_bucket: str
    train_steps: int
    n_eval: int
    seed_count: int
    timeout_s: float
    train_bank_examples: int

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "size_bucket": self.size_bucket,
            "train_steps": int(self.train_steps),
            "n_eval": int(self.n_eval),
            "seed_count": int(self.seed_count),
            "timeout_s": float(self.timeout_s),
            "train_bank_examples": int(self.train_bank_examples),
        }


@dataclass(frozen=True, slots=True)
class ARValidationConfig:
    seed: int = 0
    protocol: str = DEFAULT_AR_VALIDATION_PROTOCOL
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
    copy_model: bool = True
    seed_count: int = 1
    auto_size_budget: bool = False
    size_bucket: str | None = None
    param_count: int | None = None
    train_bank_examples: int = DEFAULT_STABLE_TRAIN_BANK_EXAMPLES
    deterministic_episode_bank: bool = False
    story_bindings_per_example: int = DEFAULT_STORY_BINDINGS
    story_noise_sentences_per_example: int = DEFAULT_STORY_NOISE_SENTENCES
    story_eval_examples: int = DEFAULT_STORY_EVAL_EXAMPLES


@dataclass(slots=True)
class ARValidationResult:
    metric_version: str = INTEGER_AR_VALIDATION_METRIC_VERSION
    final_acc: float = 0.0
    held_pair_acc: float = 0.0
    held_class_acc: float = 0.0
    learning_curve: list[dict[str, float | int]] = field(default_factory=list)
    steps_to_floor: int | None = None
    score: float = 0.0
    steps_trained: int = 0
    status: str = "ok"
    elapsed_ms: float = 0.0
    error: str | None = None
    size_bucket: str | None = None
    param_count: int | None = None
    seed_count: int | None = None
    seed_scores: list[dict[str, Any]] = field(default_factory=list)
    rank_score_mean: float | None = None
    rank_score_std: float | None = None
    rank_score_stable: float | None = None
    held_pair_acc_mean: float | None = None
    held_pair_acc_std: float | None = None
    held_class_acc_mean: float | None = None
    held_class_acc_std: float | None = None
    budget: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None
    stage_status: str | None = None
    stage_elapsed_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "ar_validation_metric_version": self.metric_version,
            "ar_validation_final_acc": self.final_acc,
            "ar_validation_held_pair_acc": self.held_pair_acc,
            "ar_validation_held_class_acc": self.held_class_acc,
            "ar_validation_learning_curve_json": json.dumps(
                self.learning_curve,
                sort_keys=True,
            ),
            "ar_validation_steps_to_floor": self.steps_to_floor,
            "ar_validation_rank_score": self.score,
            "ar_validation_status": self.status,
            "ar_validation_elapsed_ms": self.elapsed_ms,
        }
        optional = {
            "ar_validation_size_bucket": self.size_bucket,
            "ar_validation_param_count": self.param_count,
            "ar_validation_seed_count": self.seed_count,
            "ar_validation_seed_scores_json": (
                json.dumps(self.seed_scores, sort_keys=True, separators=(",", ":"))
                if self.seed_scores
                else None
            ),
            "ar_validation_rank_score_mean": self.rank_score_mean,
            "ar_validation_rank_score_std": self.rank_score_std,
            "ar_validation_rank_score_stable": self.rank_score_stable,
            "ar_validation_held_pair_acc_mean": self.held_pair_acc_mean,
            "ar_validation_held_pair_acc_std": self.held_pair_acc_std,
            "ar_validation_held_class_acc_mean": self.held_class_acc_mean,
            "ar_validation_held_class_acc_std": self.held_class_acc_std,
            "ar_validation_budget_json": (
                json.dumps(self.budget, sort_keys=True, separators=(",", ":"))
                if self.budget
                else None
            ),
            "ar_validation_checkpoint_path": self.checkpoint_path,
            "ar_validation_stage_status": self.stage_status,
            "ar_validation_stage_elapsed_ms": self.stage_elapsed_ms,
        }
        data.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return data


def count_model_parameters(model: nn.Module) -> int:
    return int(sum(int(param.numel()) for param in model.parameters()))


def ar_validation_size_bucket(param_count: int | None) -> str:
    if param_count is None:
        return "unknown"
    n_params = int(param_count)
    if n_params < 15_000_000:
        return "10m"
    if n_params < 25_000_000:
        return "20m"
    if n_params < 45_000_000:
        return "30m"
    if n_params < 80_000_000:
        return "60m"
    return "100m_plus"


def ar_validation_budget_for_param_count(param_count: int | None) -> ARValidationBudget:
    bucket = ar_validation_size_bucket(param_count)
    budgets = {
        "10m": ARValidationBudget(bucket, 5_000, 1024, 3, 900.0, 8192),
        "20m": ARValidationBudget(bucket, 7_500, 1024, 3, 1200.0, 8192),
        "30m": ARValidationBudget(bucket, 10_000, 1024, 3, 1800.0, 12_288),
        "60m": ARValidationBudget(bucket, 15_000, 1536, 3, 2700.0, 16_384),
        "100m_plus": ARValidationBudget(bucket, 25_000, 2048, 3, 5400.0, 24_576),
        "unknown": ARValidationBudget(bucket, 5_000, 1024, 3, 900.0, 8192),
    }
    return budgets[bucket]


def stable_ar_validation_config(
    *,
    param_count: int | None = None,
    seed: int = 0,
    copy_model: bool = True,
    timeout_s: float | None = None,
    train_steps: int | None = None,
) -> ARValidationConfig:
    budget = ar_validation_budget_for_param_count(param_count)
    effective_steps = (
        int(train_steps) if train_steps is not None else budget.train_steps
    )
    return ARValidationConfig(
        seed=int(seed),
        protocol=STABLE_AR_VALIDATION_PROTOCOL,
        train_steps=effective_steps,
        eval_every=max(500, effective_steps // 10),
        n_eval=int(budget.n_eval),
        timeout_s=float(timeout_s)
        if timeout_s is not None
        else float(budget.timeout_s),
        copy_model=bool(copy_model),
        seed_count=int(budget.seed_count),
        size_bucket=budget.size_bucket,
        param_count=int(param_count) if param_count is not None else None,
        train_bank_examples=int(budget.train_bank_examples),
        deterministic_episode_bank=True,
    )


def resolve_stable_ar_validation_config(
    cfg: ARValidationConfig,
    *,
    model: nn.Module | None = None,
    param_count: int | None = None,
) -> ARValidationConfig:
    if str(cfg.protocol) != STABLE_AR_VALIDATION_PROTOCOL:
        return cfg
    effective_param_count = (
        int(param_count)
        if param_count is not None
        else (count_model_parameters(model) if model is not None else cfg.param_count)
    )
    if not cfg.auto_size_budget:
        bucket = cfg.size_bucket or ar_validation_size_bucket(effective_param_count)
        return replace(
            cfg,
            size_bucket=bucket,
            param_count=effective_param_count,
            deterministic_episode_bank=True,
            seed_count=max(1, int(cfg.seed_count)),
        )
    budget_cfg = stable_ar_validation_config(
        param_count=effective_param_count,
        seed=int(cfg.seed),
        copy_model=bool(cfg.copy_model),
        timeout_s=(
            float(cfg.timeout_s)
            if float(cfg.timeout_s) != float(DEFAULT_TIMEOUT_S)
            else None
        ),
        train_steps=(
            int(cfg.train_steps)
            if int(cfg.train_steps) != int(DEFAULT_TRAIN_STEPS)
            else None
        ),
    )
    return replace(
        budget_cfg,
        lr=float(cfg.lr),
        batch_size=int(cfg.batch_size),
        vocab_lo=int(cfg.vocab_lo),
        n_key_tokens=int(cfg.n_key_tokens),
        n_value_tokens=int(cfg.n_value_tokens),
        n_value_classes=int(cfg.n_value_classes),
        n_train_pairs=int(cfg.n_train_pairs),
        n_held_pairs=int(cfg.n_held_pairs),
        pairs_per_example=int(cfg.pairs_per_example),
        episodic_values=bool(cfg.episodic_values),
    )


def build_ar_validation_pair_table(cfg: ARValidationConfig) -> ARValidationPairTable:
    return build_kv_pair_table(
        seed=int(cfg.seed),
        vocab_lo=int(cfg.vocab_lo),
        n_key_tokens=int(cfg.n_key_tokens),
        n_value_tokens=int(cfg.n_value_tokens),
        n_value_classes=int(cfg.n_value_classes),
        n_train_pairs=int(cfg.n_train_pairs),
        n_held_pairs=int(cfg.n_held_pairs),
    )


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
            float(row.get("held_pair_acc") or 0.0),
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
            float(row.get("held_pair_acc") or 0.0),
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


def _metric_version_for_protocol(cfg: ARValidationConfig) -> str:
    if str(cfg.protocol) == "story_micro":
        return AR_VALIDATION_METRIC_VERSION
    if str(cfg.protocol) == STABLE_AR_VALIDATION_PROTOCOL:
        return STABLE_AR_VALIDATION_METRIC_VERSION
    return INTEGER_AR_VALIDATION_METRIC_VERSION


def _err_result(
    t0: float,
    status: str,
    error: str,
    *,
    metric_version: str = INTEGER_AR_VALIDATION_METRIC_VERSION,
) -> ARValidationResult:
    return ARValidationResult(
        metric_version=metric_version,
        status=status,
        error=error,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


def _run_integer_ar_validation(
    model: nn.Module,
    *,
    cfg: ARValidationConfig | None = None,
    device: str = "cuda",
) -> ARValidationResult:
    """Run the integer-token AR validation v2 protocol."""
    cfg = cfg or ARValidationConfig(protocol="integer_v2")
    t0 = time.perf_counter()
    dev = torch.device(device)
    try:
        probe_model = (
            safe_deepcopy_module(model).to(dev) if cfg.copy_model else model.to(dev)
        )
    except Exception as exc:  # noqa: BLE001
        return _err_result(
            t0,
            "copy_failed",
            str(exc),
            metric_version=_metric_version_for_protocol(cfg),
        )

    try:
        model_vocab = model_vocab_size(probe_model)
        table = build_ar_validation_pair_table(cfg)
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
        runtime = KVProbeRuntime(
            n_eval=int(cfg.n_eval),
            batch_size=int(cfg.batch_size),
            pairs_per_example=int(cfg.pairs_per_example),
            sep_token=int(sep_token),
            ans_token=int(ans_token),
            device=dev,
            episodic_values=bool(cfg.episodic_values),
        )
        deadline = t0 + float(cfg.timeout_s)
        eval_every = max(1, int(cfg.eval_every))
        learning_curve: list[dict[str, float | int]] = []
        use_episode_bank = bool(cfg.deterministic_episode_bank) or (
            str(cfg.protocol) == STABLE_AR_VALIDATION_PROTOCOL
        )

        def _record_eval(
            step: int,
            loss: torch.Tensor,
            in_acc: float,
            held_pair: float,
            held_class: float,
        ) -> None:
            learning_curve.append(
                {
                    "step": step,
                    "loss": round(float(loss.item()), 6),
                    "final_acc": round(in_acc, 4),
                    "held_pair_acc": round(held_pair, 4),
                    "held_class_acc": round(held_class, 4),
                }
            )

        if use_episode_bank:
            train_bank = build_kv_episode_bank(
                table,
                split="train",
                n_examples=max(int(cfg.batch_size), int(cfg.train_bank_examples)),
                batch_size=int(cfg.batch_size),
                pairs_per_example=int(cfg.pairs_per_example),
                sep_token=int(sep_token),
                ans_token=int(ans_token),
                device=dev,
                seed=int(cfg.seed) + 50_000,
                episodic_values=bool(cfg.episodic_values),
            )
            train_eval_bank = build_kv_episode_bank(
                table,
                split="train",
                n_examples=max(int(cfg.batch_size), int(cfg.n_eval)),
                batch_size=int(cfg.batch_size),
                pairs_per_example=int(cfg.pairs_per_example),
                sep_token=int(sep_token),
                ans_token=int(ans_token),
                device=dev,
                seed=int(cfg.seed) + 30_000,
                episodic_values=bool(cfg.episodic_values),
            )
            held_eval_bank = build_kv_episode_bank(
                table,
                split="held",
                n_examples=max(int(cfg.batch_size), int(cfg.n_eval)),
                batch_size=int(cfg.batch_size),
                pairs_per_example=int(cfg.pairs_per_example),
                sep_token=int(sep_token),
                ans_token=int(ans_token),
                device=dev,
                seed=int(cfg.seed) + 40_000,
                episodic_values=bool(cfg.episodic_values),
            )
            loop_result = run_kv_probe_bank_training_loop(
                probe_model,
                table,
                train_bank=train_bank,
                train_eval_bank=train_eval_bank,
                held_eval_bank=held_eval_bank,
                batch_size=int(cfg.batch_size),
                opt=opt,
                ans_pos=ans_pos,
                train_steps=int(cfg.train_steps),
                eval_every=eval_every,
                deadline=deadline,
                monotonic_time=time.perf_counter,
                on_eval=_record_eval,
            )
            final_acc, _train_class = evaluate_kv_episode_bank(
                probe_model,
                table,
                train_eval_bank,
                batch_size=int(cfg.batch_size),
                ans_pos=ans_pos,
            )
            held_pair, held_class = evaluate_kv_episode_bank(
                probe_model,
                table,
                held_eval_bank,
                batch_size=int(cfg.batch_size),
                ans_pos=ans_pos,
            )
        else:
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

            final_acc, held_pair, held_class = evaluate_kv_probe_checkpoint(
                probe_model,
                table,
                runtime=runtime,
                base_seed=int(cfg.seed),
            )

        floor_step = _steps_to_learning_floor(
            learning_curve,
            n_value_tokens=int(cfg.n_value_tokens),
        )
        budget = {}
        if str(cfg.protocol) == STABLE_AR_VALIDATION_PROTOCOL:
            budget = {
                "size_bucket": cfg.size_bucket
                or ar_validation_size_bucket(cfg.param_count),
                "param_count": cfg.param_count,
                "train_steps": int(cfg.train_steps),
                "eval_every": int(eval_every),
                "batch_size": int(cfg.batch_size),
                "n_eval": int(cfg.n_eval),
                "seed_count": int(cfg.seed_count),
                "train_bank_examples": int(cfg.train_bank_examples),
                "deterministic_episode_bank": bool(use_episode_bank),
            }
        return ARValidationResult(
            metric_version=_metric_version_for_protocol(cfg),
            final_acc=round(final_acc, 4),
            held_pair_acc=round(held_pair, 4),
            held_class_acc=round(held_class, 4),
            learning_curve=learning_curve,
            steps_to_floor=floor_step,
            score=_score(held_pair, held_class, floor_step, int(cfg.train_steps)),
            steps_trained=loop_result.steps_done,
            status=loop_result.status,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            error=loop_result.error,
            size_bucket=cfg.size_bucket,
            param_count=cfg.param_count,
            seed_count=1
            if str(cfg.protocol) == STABLE_AR_VALIDATION_PROTOCOL
            else None,
            rank_score_mean=(
                _score(held_pair, held_class, floor_step, int(cfg.train_steps))
                if str(cfg.protocol) == STABLE_AR_VALIDATION_PROTOCOL
                else None
            ),
            rank_score_std=(
                0.0 if str(cfg.protocol) == STABLE_AR_VALIDATION_PROTOCOL else None
            ),
            rank_score_stable=(
                _score(held_pair, held_class, floor_step, int(cfg.train_steps))
                if str(cfg.protocol) == STABLE_AR_VALIDATION_PROTOCOL
                else None
            ),
            held_pair_acc_mean=(
                round(held_pair, 4)
                if str(cfg.protocol) == STABLE_AR_VALIDATION_PROTOCOL
                else None
            ),
            held_pair_acc_std=(
                0.0 if str(cfg.protocol) == STABLE_AR_VALIDATION_PROTOCOL else None
            ),
            held_class_acc_mean=(
                round(held_class, 4)
                if str(cfg.protocol) == STABLE_AR_VALIDATION_PROTOCOL
                else None
            ),
            held_class_acc_std=(
                0.0 if str(cfg.protocol) == STABLE_AR_VALIDATION_PROTOCOL else None
            ),
            budget=budget,
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
    cfg: ARValidationConfig,
    device: str,
) -> ARValidationResult:
    """Run the natural-language micro retrieval story protocol."""
    from research.eval.utils import tokenize_string
    from research.tools import ar_validation_story_calibration as story

    t0 = time.perf_counter()
    dev = torch.device(device)
    try:
        probe_model = (
            safe_deepcopy_module(model).to(dev) if cfg.copy_model else model.to(dev)
        )
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "copy_failed", str(exc))

    model_vocab = model_vocab_size(probe_model)
    if model_vocab is None:
        model_vocab = 100_277

    def clipped_encode(_enc: Any, text: str) -> tuple[int, ...]:
        return tuple(
            tokenize_string(
                text,
                int(model_vocab),
                tokenizer="cl100k_base",
            ).tolist()
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
                        "held_pair_acc": round(full, 4),
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
        return ARValidationResult(
            metric_version=AR_VALIDATION_METRIC_VERSION,
            final_acc=round(
                float(
                    train_final["by_split"].get("micro_reference", {}).get("choice_acc")
                    or 0.0
                ),
                4,
            ),
            held_pair_acc=round(full, 4),
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


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _round_mean(values: list[float]) -> float:
    return round(_mean(values), 4)


def _round_std(values: list[float]) -> float:
    return round(pstdev(values), 4) if len(values) > 1 else 0.0


def _aggregate_learning_curves(
    results: list[ARValidationResult],
) -> list[dict[str, float | int]]:
    by_step: dict[int, dict[str, list[float]]] = {}
    for result in results:
        for row in result.learning_curve:
            step = int(row["step"])
            bucket = by_step.setdefault(step, {})
            for key in ("loss", "final_acc", "held_pair_acc", "held_class_acc"):
                if key in row:
                    bucket.setdefault(key, []).append(float(row[key]))
    curve: list[dict[str, float | int]] = []
    for step in sorted(by_step):
        out: dict[str, float | int] = {"step": step}
        for key, values in by_step[step].items():
            out[key] = round(_mean(values), 6 if key == "loss" else 4)
        curve.append(out)
    return curve


def _aggregate_stable_results(
    cfg: ARValidationConfig,
    *,
    seed_results: list[tuple[int, ARValidationResult]],
    t0: float,
) -> ARValidationResult:
    ok_results = [result for _seed, result in seed_results if result.status == "ok"]
    if not ok_results:
        first = seed_results[0][1] if seed_results else None
        return ARValidationResult(
            metric_version=STABLE_AR_VALIDATION_METRIC_VERSION,
            status="error",
            error=first.error if first else "no_seed_results",
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            size_bucket=cfg.size_bucket,
            param_count=cfg.param_count,
            seed_count=int(cfg.seed_count),
            budget={
                "size_bucket": cfg.size_bucket,
                "param_count": cfg.param_count,
                "train_steps": int(cfg.train_steps),
                "eval_every": int(cfg.eval_every),
                "batch_size": int(cfg.batch_size),
                "n_eval": int(cfg.n_eval),
                "seed_count": int(cfg.seed_count),
                "train_bank_examples": int(cfg.train_bank_examples),
                "deterministic_episode_bank": True,
            },
        )

    scores = [float(result.score) for result in ok_results]
    held_pair = [float(result.held_pair_acc) for result in ok_results]
    held_class = [float(result.held_class_acc) for result in ok_results]
    final_acc = [float(result.final_acc) for result in ok_results]
    aggregate_curve = _aggregate_learning_curves(ok_results)
    floor_step = _steps_to_learning_floor(
        aggregate_curve,
        n_value_tokens=int(cfg.n_value_tokens),
    )
    score_mean = _round_mean(scores)
    score_std = _round_std(scores)
    seed_scores = [
        {
            "seed": int(seed),
            "status": result.status,
            "score": result.score,
            "final_acc": result.final_acc,
            "held_pair_acc": result.held_pair_acc,
            "held_class_acc": result.held_class_acc,
            "steps_to_floor": result.steps_to_floor,
            "elapsed_ms": result.elapsed_ms,
            "error": result.error,
        }
        for seed, result in seed_results
    ]
    return ARValidationResult(
        metric_version=STABLE_AR_VALIDATION_METRIC_VERSION,
        final_acc=_round_mean(final_acc),
        held_pair_acc=_round_mean(held_pair),
        held_class_acc=_round_mean(held_class),
        learning_curve=aggregate_curve,
        steps_to_floor=floor_step,
        score=score_mean,
        steps_trained=max(int(result.steps_trained) for result in ok_results),
        status="ok" if len(ok_results) == len(seed_results) else "partial",
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        error=None if len(ok_results) == len(seed_results) else "partial_seed_failure",
        size_bucket=cfg.size_bucket,
        param_count=cfg.param_count,
        seed_count=int(cfg.seed_count),
        seed_scores=seed_scores,
        rank_score_mean=score_mean,
        rank_score_std=score_std,
        rank_score_stable=round(max(0.0, score_mean - score_std), 4),
        held_pair_acc_mean=_round_mean(held_pair),
        held_pair_acc_std=_round_std(held_pair),
        held_class_acc_mean=_round_mean(held_class),
        held_class_acc_std=_round_std(held_class),
        budget={
            "size_bucket": cfg.size_bucket,
            "param_count": cfg.param_count,
            "train_steps": int(cfg.train_steps),
            "eval_every": int(cfg.eval_every),
            "batch_size": int(cfg.batch_size),
            "n_eval": int(cfg.n_eval),
            "seed_count": int(cfg.seed_count),
            "train_bank_examples": int(cfg.train_bank_examples),
            "deterministic_episode_bank": True,
        },
    )


def _run_stable_integer_ar_validation(
    model: nn.Module,
    *,
    cfg: ARValidationConfig,
    device: str,
) -> ARValidationResult:
    t0 = time.perf_counter()
    cfg = resolve_stable_ar_validation_config(cfg, model=model)
    seed_count = max(1, int(cfg.seed_count))
    seed_results: list[tuple[int, ARValidationResult]] = []
    for idx in range(seed_count):
        seed = int(cfg.seed) + idx * 10_007
        seed_cfg = replace(
            cfg,
            seed=seed,
            seed_count=1,
            auto_size_budget=False,
            deterministic_episode_bank=True,
            copy_model=bool(cfg.copy_model),
        )
        seed_results.append(
            (
                seed,
                _run_integer_ar_validation(model, cfg=seed_cfg, device=device),
            )
        )
    return _aggregate_stable_results(cfg, seed_results=seed_results, t0=t0)


def run_ar_validation(
    model: nn.Module,
    *,
    cfg: ARValidationConfig | None = None,
    device: str = "cuda",
) -> ARValidationResult:
    """Run the configured AR Validation champion protocol."""
    cfg = cfg or ARValidationConfig()
    t0 = time.perf_counter()
    dev = torch.device(device)
    if dev.type != "cuda":
        return _err_result(
            t0,
            "missing_accelerator",
            "ar_validation_requires_cuda",
            metric_version=_metric_version_for_protocol(cfg),
        )
    if not torch.cuda.is_available():
        return _err_result(
            t0,
            "missing_accelerator",
            "cuda_unavailable",
            metric_version=_metric_version_for_protocol(cfg),
        )
    if str(cfg.protocol) == "story_micro":
        return _run_story_micro_champion(model, cfg=cfg, device=device)
    if str(cfg.protocol) == STABLE_AR_VALIDATION_PROTOCOL:
        return _run_stable_integer_ar_validation(model, cfg=cfg, device=device)
    if str(cfg.protocol) == "integer_v2":
        return _run_integer_ar_validation(model, cfg=cfg, device=device)
    return _err_result(t0, "error", f"unknown_ar_validation_protocol:{cfg.protocol}")


__all__ = [
    "AR_VALIDATION_METRIC_VERSION",
    "INTEGER_AR_VALIDATION_METRIC_VERSION",
    "STABLE_AR_VALIDATION_METRIC_VERSION",
    "DEFAULT_AR_VALIDATION_PROTOCOL",
    "STABLE_AR_VALIDATION_PROTOCOL",
    "ARValidationBudget",
    "ARValidationConfig",
    "ARValidationResult",
    "ARValidationPairTable",
    "ar_validation_budget_for_param_count",
    "ar_validation_size_bucket",
    "build_ar_validation_pair_table",
    "count_model_parameters",
    "make_ar_validation_batch",
    "resolve_stable_ar_validation_config",
    "run_ar_validation",
    "stable_ar_validation_config",
]
