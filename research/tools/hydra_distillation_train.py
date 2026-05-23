"""HYDRA distillation trainer — fine-tune a checkpoint on teacher completions.

Companion to ``research/eval/hydra_compositional_eval.py``. The eval grades a
model's ability to discriminate the correct natural-language answer from
distractors at AR-style compositional retrieval. This trainer is the
production-scale analogue of the AR-curriculum probe's fine-tune phase: same
idea (give the model dedicated steps on a task-shaped signal), but using
distilled teacher completions from claude-sonnet-4 instead of synthetic
at-distance pairs.

Key choices
-----------
- **Completion-only loss masking.** Loss is computed only over the
  teacher_completion tokens, never over the prompt. Without this the model
  spends most of its gradient on copying the prompt back, which is not what
  we want to fine-tune.
- **Held-out matches the eval.** The last 5% of each source file is reserved
  for ``hydra_compositional_eval``; training uses everything before that. Same
  files (``distill_math.jsonl``, ``distill_reasoning.jsonl``, or
  ``distill_all.jsonl``) and the same chronological split.
- **Reuses mixer_fingerprint primitives.** ``_make_optimizer``,
  ``_WarmupCosineSchedule``, ``_configure_torch_performance``,
  ``_resolve_lane_factories``, and ``_build_tinylm`` are imported, not
  re-implemented. No new optimizer/scheduler/lane code.
- **No probe involved.** The model is fine-tuned, then *separately* evaluated
  by ``hydra_compositional_eval`` (which is measurement-only). No mid-training
  probe is run.

Not in scope
------------
- KL/logit distillation from teacher logits. HYDRA stores teacher text only,
  not logits; we treat completions as gold tokens and use vanilla CE.
- Curriculum scheduling over prompt difficulty / length. Could be layered on
  later if the simple version doesn't lift the eval.
- LoRA / PEFT — full-parameter fine-tune; the models are small (76M-145M).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.nn import functional as F
import tiktoken

from research.defaults import VOCAB_SIZE
from research.tools.mixer_fingerprint import (
    _WarmupCosineSchedule,
    _configure_torch_performance,
    _make_optimizer,
    _resolve_lane_factories,
)
from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm


HYDRA_ROOT = Path("/home/tim/Projects/LLM/HYDRA/data")
IGNORE_INDEX = -100  # F.cross_entropy default ignore_index


@dataclass(slots=True)
class _Sample:
    """One distillation example, already tokenized + loss-masked."""

    input_ids: list[int]
    labels: list[
        int
    ]  # IGNORE_INDEX on prompt + framing tokens; teacher tokens elsewhere


def _load_train_slice(
    files: tuple[str, ...],
    held_out_frac: float,
) -> list[dict]:
    """Return every example NOT in the eval hold-out slice."""
    out: list[dict] = []
    for fname in files:
        path = HYDRA_ROOT / fname
        with path.open() as fh:
            lines = fh.readlines()
        n_total = len(lines)
        n_held = max(1, int(n_total * held_out_frac))
        train_lines = lines[: n_total - n_held]
        for ln in train_lines:
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if d.get("prompt") and d.get("teacher_completion"):
                out.append(d)
    return out


def _tokenize_sample(
    tokenizer: tiktoken.Encoding,
    prompt: str,
    completion: str,
    max_seq_len: int,
) -> _Sample | None:
    """Encode one example. Returns None if completion empty post-truncation."""
    # Match the eval's framing exactly so train/eval contexts are aligned.
    prompt_ids = tokenizer.encode(prompt + " The answer is ")
    comp_ids = tokenizer.encode(" " + completion.strip())
    # If completion alone overruns max_seq_len, drop the example — masking
    # would zero every loss-contributing token anyway.
    if len(comp_ids) >= max_seq_len:
        return None
    keep_prompt = max_seq_len - len(comp_ids)
    prompt_ids = prompt_ids[-keep_prompt:]  # left-truncate prompt, preserve completion
    input_ids = list(prompt_ids) + list(comp_ids)
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(comp_ids)
    return _Sample(input_ids=input_ids, labels=labels)


def _pack_batch(
    samples: list[_Sample], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch of variable-length samples."""
    max_len = max(len(s.input_ids) for s in samples)
    bs = len(samples)
    input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
    labels = torch.full((bs, max_len), IGNORE_INDEX, dtype=torch.long, device=device)
    for i, s in enumerate(samples):
        input_ids[i, : len(s.input_ids)] = torch.tensor(
            s.input_ids, dtype=torch.long, device=device
        )
        labels[i, : len(s.labels)] = torch.tensor(
            s.labels, dtype=torch.long, device=device
        )
    return input_ids, labels


def _iter_batches(
    samples: list[_Sample],
    batch_size: int,
    pad_id: int,
    device: torch.device,
    rng: random.Random,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Shuffle once per epoch, then bucket-sort within a window to reduce padding."""
    order = list(range(len(samples)))
    rng.shuffle(order)
    # Length-bucket within a sliding window for less padding waste.
    BUCKET = 32 * batch_size
    for win_start in range(0, len(order), BUCKET):
        window = order[win_start : win_start + BUCKET]
        window.sort(key=lambda i: len(samples[i].input_ids))
        for batch_start in range(0, len(window), batch_size):
            idxs = window[batch_start : batch_start + batch_size]
            batch = [samples[i] for i in idxs]
            yield _pack_batch(batch, pad_id, device)


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shift logits/labels, then CE with IGNORE_INDEX masking on prompt tokens."""
    # logits[:, t, :] predicts token at position t+1. Shift so labels[:, 1:] align.
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def train(
    *,
    model: torch.nn.Module,
    samples: list[_Sample],
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    min_lr: float,
    pad_id: int,
    device: torch.device,
    grad_clip: float = 1.0,
    log_every: int = 50,
    seed: int = 0,
) -> dict:
    """Fine-tune with completion-only CE. Returns a small training summary."""
    rng = random.Random(seed)
    opt, _ = _make_optimizer(model, learning_rate=learning_rate, device=device)
    sched = _WarmupCosineSchedule(
        opt,
        learning_rate=learning_rate,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        total_steps=n_steps,
    )

    model.train()
    step = 0
    losses: list[float] = []
    while step < n_steps:
        for input_ids, labels in _iter_batches(
            samples, batch_size, pad_id, device, rng
        ):
            sched.apply(step)
            logits = model(input_ids)
            loss = _causal_lm_loss(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.item()))
            if step % log_every == 0 or step == n_steps - 1:
                # Active-tokens-per-step for sanity (varies due to masking)
                active = int((labels != IGNORE_INDEX).sum().item())
                print(
                    f"step={step:5d}/{n_steps}  loss={loss.item():.4f}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}  active_tok={active}",
                    flush=True,
                )
            step += 1
            if step >= n_steps:
                break
    model.eval()
    return {
        "n_steps": step,
        "final_loss": losses[-1] if losses else None,
        "mean_last_100_loss": (
            sum(losses[-100:]) / max(1, min(100, len(losses))) if losses else None
        ),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-in",
        required=True,
        type=Path,
        help="Existing model state_dict (from mixer_fingerprint or earlier finetune).",
    )
    p.add_argument(
        "--checkpoint-out",
        required=True,
        type=Path,
        help="Where to save the fine-tuned state_dict (compatible with the eval).",
    )
    p.add_argument(
        "--lane",
        required=True,
        type=str,
        help="Lane name passed to scaling_blimp_study (e.g. local_ssm_diff or 'interleaved').",
    )
    p.add_argument(
        "--pattern", default=None, type=str, help="Required if --lane=interleaved."
    )
    p.add_argument("--dim", default=384, type=int)
    p.add_argument("--n-blocks", default=12, type=int)
    p.add_argument(
        "--files",
        nargs="+",
        default=["distill_math.jsonl", "distill_reasoning.jsonl"],
        help="HYDRA jsonl filenames to draw training data from. "
        "Use 'distill_all.jsonl' for the full 49,905-pair mix.",
    )
    p.add_argument(
        "--held-out-frac",
        default=0.05,
        type=float,
        help="MUST match the eval's hold-out fraction to avoid train/eval leakage.",
    )
    p.add_argument("--max-seq-len", default=512, type=int)
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--steps", default=5000, type=int)
    p.add_argument(
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Conservative default: fine-tune lr typically 3-10x lower than pretraining lr.",
    )
    p.add_argument("--min-lr", default=1e-6, type=float)
    p.add_argument("--warmup-steps", default=500, type=int)
    p.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Default 'cpu' on purpose so this can't OOM a 5090 mid-training-run. "
        "Set to 'cuda' only when GPU is idle.",
    )
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--tokenizer", default="cl100k_base", type=str)
    return p


def _load_model_with_checkpoint(args, device):
    if args.lane == "interleaved":
        if not args.pattern:
            raise SystemExit("--pattern required when --lane=interleaved")
        model_factory, _ = _resolve_lane_factories("interleaved", args.pattern)
    else:
        model_factory = _build_lane_factory(args.lane)
    model = _build_tinylm(model_factory, dim=args.dim, n_blocks=args.n_blocks).to(
        device
    )
    state = torch.load(args.checkpoint_in, map_location=device, weights_only=False)  # nosec B614 - locally-produced checkpoint, not network-sourced
    model.load_state_dict(
        state["model"] if isinstance(state, dict) and "model" in state else state
    )
    return model


def _tokenize_train_samples(args) -> list[_Sample]:
    print(
        f"Loading HYDRA train data: {args.files} (holding out {args.held_out_frac:.1%})"
    )
    raw = _load_train_slice(tuple(args.files), held_out_frac=args.held_out_frac)
    print(f"  raw examples: {len(raw)}")
    tok = tiktoken.get_encoding(args.tokenizer)
    samples: list[_Sample] = []
    for d in raw:
        s = _tokenize_sample(
            tok, d["prompt"], d["teacher_completion"], args.max_seq_len
        )
        if s is not None:
            samples.append(s)
    if not samples:
        raise SystemExit(
            "No usable HYDRA samples after tokenization. Check max_seq_len."
        )
    print(
        f"  usable after tokenization (max_seq_len={args.max_seq_len}): {len(samples)}"
    )
    total_active_tok = sum(
        sum(1 for L in s.labels if L != IGNORE_INDEX) for s in samples
    )
    print(f"  active (teacher) tokens in train set: {total_active_tok:,}")
    return samples


def main() -> None:
    args = _build_arg_parser().parse_args()
    _configure_torch_performance()
    device = torch.device(args.device)
    model = _load_model_with_checkpoint(args, device)
    samples = _tokenize_train_samples(args)
    # cl100k has no canonical pad — VOCAB_SIZE-1 is unused in our embedding range
    # (TinyLM allocates VOCAB_SIZE entries), so it's safe to use as padding.
    pad_id = VOCAB_SIZE - 1
    summary = train(
        model=model,
        samples=samples,
        n_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        pad_id=pad_id,
        device=device,
        seed=args.seed,
    )
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.checkpoint_out)
    print(f"\nSaved fine-tuned checkpoint -> {args.checkpoint_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
