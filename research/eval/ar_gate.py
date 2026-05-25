"""AR Gate-INV — investigation-tier associative-recall probe.

Pattern mirrors ``research.eval.nano_bind`` (deepcopy + fine-tune + eval),
but trains/grades against a 3-slot binding corpus (noun → adj + object).

Two operating modes:

  - ``from_s1=True`` (production / investigation): ``model`` is the live
    Stage-1-trained model. We deepcopy it and fine-tune the copy on the
    AR gate corpus, leveraging the wikitext language priors the backbone
    already learned.
  - ``from_s1=False`` (pilot / standalone): ``graph_json`` is provided
    instead. We compile a fresh-init model and train it directly on the
    AR gate corpus — no wikitext warmup. Used for offline pilots.

Score: ``in_dist_exact_acc`` is the headline metric (top-1 adj AND top-1
object both correct, in-distribution prompts). Held-out class and held-out
exact accuracies are stored as audit channels.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from research.eval.ar_gate_corpus import (
    DEFAULT_HELD_OUT_NOUNS,
    DEFAULT_N_DISTRACTORS,
    DEFAULT_N_PAIRS_PER_NOUN,
    DEFAULT_REPS,
    OBJECTS,
    CorpusSpec,
    Fact,
    build_corpus,
    query_prompt,
)
from research.eval._probe_utils import safe_deepcopy_module
from research.tools.nano_corpus_v0 import ADJECTIVES

logger = logging.getLogger(__name__)

AR_GATE_METRIC_VERSION = "ar_gate_v0"
TIKTOKEN_ENCODING = "cl100k_base"
PAD_ID = 0

DEFAULT_TRAIN_STEPS = 800
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 32
DEFAULT_TIMEOUT_S = 90.0


@dataclass(frozen=True, slots=True)
class ARGateConfig:
    """Sweep parameters for one AR gate-INV run.

    For pilot mode (``from_s1=False``) we simulate the investigation-tier
    backbone by training on wikitext for ``wikitext_warmup_steps`` *before*
    fine-tuning on the AR gate corpus for ``finetune_steps``. This mirrors
    the production path where the live S1 model already has language priors.
    """

    seed: int = 0
    n_pairs_per_noun: int = DEFAULT_N_PAIRS_PER_NOUN
    reps: int = DEFAULT_REPS
    n_distractors: int = DEFAULT_N_DISTRACTORS
    n_adjectives: int = 20
    n_objects: int = 25
    held_out_nouns: tuple[str, ...] = DEFAULT_HELD_OUT_NOUNS
    finetune_steps: int = 400  # AR gate fine-tune budget
    wikitext_warmup_steps: int = 500  # pilot-only; ignored when from_s1=True
    wikitext_warmup_seq_len: int = 256
    wikitext_warmup_batch_size: int = 4
    wikitext_warmup_lr: float = 3e-4
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH
    timeout_s: float = DEFAULT_TIMEOUT_S
    from_s1: bool = True


@dataclass(slots=True)
class ARGateResult:
    """Outcome of one probe run.

    ``in_dist_pair_acc`` is the headline score (range 0..1) — fraction of
    in-dist nouns where the predicted (adj, obj) is one of the noun's trained
    pairs. ``in_dist_class_acc`` is the looser "any-adj + any-obj" check.
    Held-out fields are diagnostics for compositional generalization.
    """

    metric_version: str = AR_GATE_METRIC_VERSION
    in_dist_pair_acc: float = 0.0
    in_dist_class_acc: float = 0.0
    held_pair_acc: float = 0.0
    held_class_acc: float = 0.0
    n_in_dist: int = 0
    n_held: int = 0
    wikitext_warmup_steps_done: int = 0
    finetune_steps_done: int = 0
    elapsed_ms: float = 0.0
    status: str = "ok"
    error: str | None = None
    per_prompt_top1: list[dict[str, Any]] = field(default_factory=list)
    sweep_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ar_gate_metric_version": self.metric_version,
            "ar_gate_in_dist_pair_acc": self.in_dist_pair_acc,
            "ar_gate_in_dist_class_acc": self.in_dist_class_acc,
            "ar_gate_held_pair_acc": self.held_pair_acc,
            "ar_gate_held_class_acc": self.held_class_acc,
            "ar_gate_n_in_dist": self.n_in_dist,
            "ar_gate_n_held": self.n_held,
            "ar_gate_wikitext_warmup_steps_done": self.wikitext_warmup_steps_done,
            "ar_gate_finetune_steps_done": self.finetune_steps_done,
            "ar_gate_elapsed_ms": self.elapsed_ms,
            "ar_gate_status": self.status,
            "ar_gate_error": self.error,
            "ar_gate_per_prompt_top1": list(self.per_prompt_top1),
            "ar_gate_sweep_metadata": self.sweep_metadata,
        }


# Single source of truth for the AR-gate score + hard no-go recipe.
# Mirrors nano_bind's persistent-zero rule: an ``ok`` run whose in-dist
# pair-match AND held-out class accuracy both collapse below 0.10 is a
# frequency-collapse degenerate (no-go). Transient failures (timeout /
# non-finite) are NOT flagged — only ``status == 'ok'`` runs.
AR_GATE_NO_GO_PAIR_THRESHOLD: float = 0.10
AR_GATE_NO_GO_HELD_CLASS_THRESHOLD: float = 0.10


def ar_gate_score(result: "ARGateResult") -> float:
    """Headline AR-gate score: 0.6 * in-dist pair-match + 0.4 * held class."""
    return round(0.6 * result.in_dist_pair_acc + 0.4 * result.held_class_acc, 4)


def ar_gate_is_no_go(result: "ARGateResult") -> bool:
    """True iff an ``ok`` run is a frequency-collapse degenerate (hard no-go)."""
    return (
        result.status == "ok"
        and result.in_dist_pair_acc < AR_GATE_NO_GO_PAIR_THRESHOLD
        and result.held_class_acc < AR_GATE_NO_GO_HELD_CLASS_THRESHOLD
    )


def _get_encoder():
    from research.eval.utils import _get_tiktoken_encoder

    return _get_tiktoken_encoder(TIKTOKEN_ENCODING)


def _tokenize_words(enc, words: list[str]) -> list[int]:
    from research.eval.utils import tokenize_words_serial

    return tokenize_words_serial(enc, words, encoding_name=TIKTOKEN_ENCODING)


def _build_train_tensor(enc, sentences: list[str], device) -> torch.Tensor:
    from research.eval.utils import pack_token_rows

    rows = [_tokenize_words(enc, s.split()) for s in sentences]
    return pack_token_rows(rows, device, pad_id=PAD_ID)


def _build_prompt_tensor(
    enc, facts: tuple[Fact, ...], device
) -> tuple[torch.Tensor, list[int]]:
    from research.eval.utils import pack_token_rows

    rows = [_tokenize_words(enc, query_prompt(f).split()) for f in facts]
    out = pack_token_rows(rows, device, pad_id=PAD_ID)
    last_pos = [len(r) - 1 for r in rows]
    return out, last_pos


_WIKITEXT_PATH = "/home/tim/Projects/LLM/research/corpus/wikitext103_train.npy"
_WIKITEXT_MEMMAP_CACHE: dict[str, Any] = {}
_TENSOR_CACHE: dict[
    tuple[Any, ...], tuple[CorpusSpec, torch.Tensor, torch.Tensor, list[int]]
] = {}
_TOKEN_ID_CACHE: dict[tuple[Any, ...], tuple[dict[str, int], dict[str, int]]] = {}


def _load_wikitext_memmap():
    """Memory-map the canonical wikitext-103 corpus once per process."""
    import numpy as np

    cached = _WIKITEXT_MEMMAP_CACHE.get(_WIKITEXT_PATH)
    if cached is not None:
        return cached
    arr = np.load(_WIKITEXT_PATH, mmap_mode="r")
    _WIKITEXT_MEMMAP_CACHE[_WIKITEXT_PATH] = arr
    return arr


def _train_on_wikitext(
    model: nn.Module,
    *,
    n_steps: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    seed: int,
    deadline: float,
    device: "torch.device",
) -> tuple[int, str, str | None]:
    """Brief wikitext warmup mirroring stage1: random windows + AdamW + CE.

    Reuses the canonical wikitext-103 memmap and the same tokenizer/sampling
    convention as the live stage1 trainer. Not a parallel implementation —
    just a thin wrapper around the standard ``np.random.randint`` window
    sampling pattern used elsewhere in the pipeline.
    """
    if n_steps <= 0:
        return 0, "ok", None
    import numpy as np
    import torch.nn.functional as F

    from research.eval.utils import clip_grad_norm, make_adamw

    arr = _load_wikitext_memmap()
    n_total = int(arr.shape[0])
    if n_total < seq_len + 1:
        return 0, "error", "wikitext_corpus_too_short"

    opt = make_adamw(model.parameters(), lr=lr)
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(
        0,
        n_total - seq_len - 1,
        size=(int(n_steps), int(batch_size)),
        dtype=np.int64,
    )
    offsets = np.arange(int(seq_len) + 1, dtype=np.int64)
    batch_np = arr[starts[..., None] + offsets]
    all_batches = torch.from_numpy(batch_np.astype("int64", copy=False)).to(device)
    steps_done = 0
    model.train()
    for step in range(1, int(n_steps) + 1):
        if time.perf_counter() > deadline:
            return steps_done, "timeout", None
        batch = all_batches[step - 1]
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        opt.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)
        )
        if not torch.isfinite(loss):
            return steps_done, "error", "wikitext_warmup_non_finite_loss"
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step()
        steps_done = step
    return steps_done, "ok", None


def _train_one_step(
    model: nn.Module,
    train_ids: torch.Tensor,
    opt: torch.optim.Optimizer,
    rng: torch.Generator,
    batch_size: int,
) -> bool:
    n = train_ids.shape[0]
    idx = torch.randint(
        0, n, (int(batch_size),), generator=rng, device=train_ids.device
    )
    batch = train_ids.index_select(0, idx)
    return _train_one_batch(model, batch, opt)


def _train_one_batch(
    model: nn.Module,
    batch: torch.Tensor,
    opt: torch.optim.Optimizer,
) -> bool:
    import torch.nn.functional as F

    from research.eval.utils import clip_grad_norm

    opt.zero_grad(set_to_none=True)
    logits = model(batch)
    targets = batch[:, 1:].contiguous()
    pred = logits[:, :-1, :].contiguous()
    mask = targets != PAD_ID
    if not bool(mask.any()):
        return True
    loss = F.cross_entropy(pred[mask].float(), targets[mask])
    if not torch.isfinite(loss):
        return False
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    opt.step()
    return True


@torch.no_grad()
def _greedy_decode_two_tokens(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    last_pos: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy-decode adj at ``last_pos[i]`` then obj one position later."""
    model.eval()
    n = prompt_ids.shape[0]
    row_idx = torch.arange(n, device=prompt_ids.device)
    last_pos_tensor = torch.as_tensor(
        last_pos, dtype=torch.long, device=prompt_ids.device
    )
    logits = model(prompt_ids)
    adj_top1 = logits[row_idx, last_pos_tensor, :].argmax(dim=-1)

    extended = torch.full(
        (n, prompt_ids.shape[1] + 1),
        PAD_ID,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    extended[:, : prompt_ids.shape[1]] = prompt_ids
    obj_pos = last_pos_tensor + 1
    extended[row_idx, obj_pos] = adj_top1

    logits_ext = model(extended)
    obj_top1 = logits_ext[row_idx, obj_pos, :].argmax(dim=-1)
    return adj_top1, obj_top1


def _grade_predictions(
    facts: tuple[Fact, ...],
    adj_top1: torch.Tensor,
    obj_top1: torch.Tensor,
    *,
    adj_token_ids: dict[str, int],
    obj_token_ids: dict[str, int],
    trained_pairs_by_noun: dict[str, frozenset[tuple[str, str]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pair-match + class grading split by held-out vs in-dist.

    Headline metric is **in_dist_pair**: did the predicted (adj, obj)
    match ANY trained pair for that noun? With multi-pair-per-noun corpora
    this measures combinatorial retrieval rather than lexical-specific recall.
    """
    adj_id_set = set(adj_token_ids.values())
    obj_id_set = set(obj_token_ids.values())
    inv_adj = {v: k for k, v in adj_token_ids.items()}
    inv_obj = {v: k for k, v in obj_token_ids.items()}
    pairs_by_noun = trained_pairs_by_noun or {}
    per_prompt: list[dict[str, Any]] = []
    counts = {
        "n_in": 0,
        "n_held": 0,
        "in_pair": 0,
        "in_class": 0,
        "held_pair": 0,
        "held_class": 0,
    }
    adj_preds = adj_top1.detach().cpu().tolist()
    obj_preds = obj_top1.detach().cpu().tolist()
    for i, fact in enumerate(facts):
        a_pred = int(adj_preds[i])
        o_pred = int(obj_preds[i])
        adj_ok = a_pred in adj_id_set
        obj_ok = o_pred in obj_id_set
        pred_pair = (inv_adj.get(a_pred), inv_obj.get(o_pred))
        accepted = pairs_by_noun.get(fact.noun, frozenset())
        pair_match = adj_ok and obj_ok and (pred_pair[0], pred_pair[1]) in accepted
        per_prompt.append(
            {
                "noun": fact.noun,
                "accepted_pairs": sorted(accepted),
                "pred_adj": pred_pair[0],
                "pred_obj": pred_pair[1],
                "pred_adj_id": a_pred,
                "pred_obj_id": o_pred,
                "class_ok": adj_ok and obj_ok,
                "pair_match": pair_match,
                "held_out": fact.held_out,
            }
        )
        clazz = adj_ok and obj_ok
        bucket = "held" if fact.held_out else "in"
        counts[f"n_{bucket}"] += 1
        counts[f"{bucket}_pair"] += int(pair_match)
        counts[f"{bucket}_class"] += int(clazz)
    return per_prompt, counts


def _acquire_probe_model(
    model: nn.Module | None,
    graph_json: str | None,
    dev,
    seed: int,
) -> nn.Module:
    """Deepcopy the live model or compile fresh from graph_json."""
    if model is not None:
        return safe_deepcopy_module(model).to(dev)
    from research.scientist.native_runner import (
        compile_model_native_first as compile_model,
    )
    from research.synthesis.serializer import graph_from_json

    graph = graph_from_json(graph_json)
    torch.manual_seed(int(seed))
    return compile_model([graph]).to(dev)


def _run_training(
    probe_model: nn.Module,
    train_ids: torch.Tensor,
    *,
    opt: torch.optim.Optimizer,
    rng: torch.Generator,
    batch_size: int,
    train_steps: int,
    deadline: float,
) -> tuple[int, str, str | None]:
    steps_done = 0
    probe_model.train()
    if train_steps <= 0:
        return 0, "ok", None
    batch_indices = torch.randint(
        0,
        train_ids.shape[0],
        (int(train_steps), int(batch_size)),
        generator=rng,
        device=train_ids.device,
    )
    for step in range(1, int(train_steps) + 1):
        if time.perf_counter() > deadline:
            return steps_done, "timeout", None
        batch = train_ids.index_select(0, batch_indices[step - 1])
        if not _train_one_batch(probe_model, batch, opt):
            return steps_done, "error", "non_finite_loss"
        steps_done = step
    return steps_done, "ok", None


def _err_result(t0: float, status: str, msg: str, **extra) -> ARGateResult:
    return ARGateResult(
        status=status,
        error=msg,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        **extra,
    )


def _setup_corpus_and_tensors(
    enc, dev, cfg: ARGateConfig
) -> tuple[CorpusSpec, torch.Tensor, torch.Tensor, list[int]]:
    cache_key = (
        str(dev),
        int(cfg.seed),
        int(cfg.n_pairs_per_noun),
        int(cfg.reps),
        int(cfg.n_distractors),
        tuple(cfg.held_out_nouns),
        int(cfg.n_adjectives),
        int(cfg.n_objects),
    )
    cached = _TENSOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    spec = build_corpus(
        seed=cfg.seed,
        n_pairs_per_noun=cfg.n_pairs_per_noun,
        reps=cfg.reps,
        n_distractors=cfg.n_distractors,
        held_out_nouns=cfg.held_out_nouns,
        n_adjectives=cfg.n_adjectives,
        n_objects=cfg.n_objects,
    )
    train_ids = _build_train_tensor(enc, list(spec.train_sentences), dev)
    prompt_ids, last_pos = _build_prompt_tensor(enc, spec.test_facts, dev)
    cached = (spec, train_ids, prompt_ids, last_pos)
    _TENSOR_CACHE[cache_key] = cached
    return cached


def _get_class_token_ids(
    enc, cfg: ARGateConfig
) -> tuple[dict[str, int], dict[str, int]]:
    """Cached batch-encode of class vocabulary (adj + obj single-token IDs)."""
    cache_key = (TIKTOKEN_ENCODING, int(cfg.n_adjectives), int(cfg.n_objects))
    cached = _TOKEN_ID_CACHE.get(cache_key)
    if cached is not None:
        return cached
    adj_words = list(ADJECTIVES[: cfg.n_adjectives])
    obj_words = list(OBJECTS[: cfg.n_objects])
    adj_ids = _tokenize_words(enc, adj_words)
    obj_ids = _tokenize_words(enc, obj_words)
    adj_token_ids = dict(zip(adj_words, adj_ids))
    obj_token_ids = dict(zip(obj_words, obj_ids))
    cached = (adj_token_ids, obj_token_ids)
    _TOKEN_ID_CACHE[cache_key] = cached
    return cached


def _assemble_result(
    *,
    t0,
    c,
    per_prompt,
    warmup_done,
    finetune_done,
    status,
    error,
    cfg: ARGateConfig,
) -> ARGateResult:
    return ARGateResult(
        in_dist_pair_acc=round(c["in_pair"] / max(c["n_in"], 1), 4),
        in_dist_class_acc=round(c["in_class"] / max(c["n_in"], 1), 4),
        held_pair_acc=round(c["held_pair"] / max(c["n_held"], 1), 4),
        held_class_acc=round(c["held_class"] / max(c["n_held"], 1), 4),
        n_in_dist=c["n_in"],
        n_held=c["n_held"],
        wikitext_warmup_steps_done=warmup_done,
        finetune_steps_done=finetune_done,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        status=status,
        error=error,
        per_prompt_top1=per_prompt,
        sweep_metadata={
            "from_s1": bool(cfg.from_s1),
            "n_pairs_per_noun": cfg.n_pairs_per_noun,
            "reps": cfg.reps,
            "n_distractors": cfg.n_distractors,
            "n_adjectives": cfg.n_adjectives,
            "n_objects": cfg.n_objects,
            "finetune_steps": cfg.finetune_steps,
            "wikitext_warmup_steps": cfg.wikitext_warmup_steps,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "held_out_nouns": list(cfg.held_out_nouns),
            "tokenizer": TIKTOKEN_ENCODING,
        },
    )


def _train_decode_grade(
    probe_model: nn.Module,
    *,
    spec: CorpusSpec,
    train_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    last_pos: list[int],
    cfg: ARGateConfig,
    deadline: float,
    dev,
    adj_token_ids: dict[str, int],
    obj_token_ids: dict[str, int],
) -> tuple[int, int, str, str | None, list[dict[str, Any]], dict[str, int]]:
    from research.eval.utils import make_adamw

    warmup_done = 0
    if not cfg.from_s1 and cfg.wikitext_warmup_steps > 0:
        warmup_done, status, error = _train_on_wikitext(
            probe_model,
            n_steps=cfg.wikitext_warmup_steps,
            seq_len=cfg.wikitext_warmup_seq_len,
            batch_size=cfg.wikitext_warmup_batch_size,
            lr=cfg.wikitext_warmup_lr,
            seed=cfg.seed,
            deadline=deadline,
            device=dev,
        )
        if status != "ok":
            return (
                warmup_done,
                0,
                status,
                error,
                [],
                {
                    "n_in": 0,
                    "n_held": 0,
                    "in_pair": 0,
                    "in_class": 0,
                    "held_pair": 0,
                    "held_class": 0,
                },
            )

    rng = torch.Generator(device=train_ids.device)
    rng.manual_seed(int(cfg.seed))
    opt = make_adamw(probe_model.parameters(), lr=cfg.lr)
    finetune_done, status, error = _run_training(
        probe_model,
        train_ids,
        opt=opt,
        rng=rng,
        batch_size=cfg.batch_size,
        train_steps=cfg.finetune_steps,
        deadline=deadline,
    )
    adj_top1, obj_top1 = _greedy_decode_two_tokens(probe_model, prompt_ids, last_pos)
    per_prompt, counts = _grade_predictions(
        spec.test_facts,
        adj_top1,
        obj_top1,
        adj_token_ids=adj_token_ids,
        obj_token_ids=obj_token_ids,
        trained_pairs_by_noun=spec.trained_pairs_by_noun,
    )
    return warmup_done, finetune_done, status, error, per_prompt, counts


def ar_gate(
    *,
    model: nn.Module | None = None,
    graph_json: str | None = None,
    device: str = "cuda",
    cfg: ARGateConfig | None = None,
) -> ARGateResult:
    """Run AR gate-INV. Provide either ``model`` (live) or ``graph_json``."""
    cfg = cfg or ARGateConfig()
    t0 = time.perf_counter()
    if (model is None) == (graph_json is None):
        return _err_result(
            t0, "error", "exactly one of model or graph_json must be provided"
        )
    enc = _get_encoder()
    dev = torch.device(device)
    try:
        probe_model = _acquire_probe_model(model, graph_json, dev, cfg.seed)
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "compile_failed", str(exc))

    try:
        spec, train_ids, prompt_ids, last_pos = _setup_corpus_and_tensors(enc, dev, cfg)
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "error", f"corpus_build:{exc}")

    adj_token_ids, obj_token_ids = _get_class_token_ids(enc, cfg)
    deadline = t0 + float(cfg.timeout_s)
    try:
        warmup_done, finetune_done, status, error, per_prompt, c = _train_decode_grade(
            probe_model,
            spec=spec,
            train_ids=train_ids,
            prompt_ids=prompt_ids,
            last_pos=last_pos,
            cfg=cfg,
            deadline=deadline,
            dev=dev,
            adj_token_ids=adj_token_ids,
            obj_token_ids=obj_token_ids,
        )
    except Exception as exc:  # noqa: BLE001
        return _err_result(t0, "error", f"train_or_eval:{exc}")
    finally:
        del probe_model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return _assemble_result(
        t0=t0,
        c=c,
        per_prompt=per_prompt,
        warmup_done=warmup_done,
        finetune_done=finetune_done,
        status=status,
        error=error,
        cfg=cfg,
    )


__all__ = [
    "AR_GATE_METRIC_VERSION",
    "ARGateConfig",
    "ARGateResult",
    "ar_gate",
    "DEFAULT_TRAIN_STEPS",
    "DEFAULT_LR",
    "DEFAULT_BATCH",
]
