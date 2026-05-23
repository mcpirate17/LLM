"""HYDRA compositional eval — AR-style real-text evaluator for trained models.

Design intent
-------------
The AR-curriculum probe (research/eval/ar_curriculum_probe.py) is a *screening*
tool: it gives a nano-scale model 1200 fine-tune steps on synthetic
at-distance retrieval data and grades how far it gets. The probe is the
trainer; at production scale (76M+ params, well-pretrained), the probe's
budget can't move the model meaningfully and the metric saturates.

This eval is the production-scale analogue. It measures the same underlying
capability — at-distance retrieval + composition — but on real
prompt/completion pairs distilled from claude-sonnet-4. No fine-tuning at
eval time. Pure log-likelihood discrimination on held-out data.

Method (logit discrimination — works for tiny models that can't generate)
------------------------------------------------------------------------
For each held-out example, compute log p_model(expected_answer | prompt, lead)
where ``lead`` is a fixed framing string (``" The answer is "``). Compare to
log-probs of N distractor answers drawn from other held-out examples (or a
calibrated random set). Score:

  - top1: correct answer ranks #1 among distractors
  - MRR: mean reciprocal rank of the correct answer
  - margin: log p(correct) - mean log p(distractors)

Stratify by:
  - category: math vs reasoning vs general
  - prompt length tertile (a proxy for "retrieval distance" — more setup
    tokens between facts and the question = harder composition)
  - expected_answer kind: numeric / string / multi-token

The math example bb0b8d5856da (gold-graph) requires retrieving
{5 people, 3 days, 2 + 1 bars/day} from a 436-char prompt and composing them.
That is structurally identical to AR-curriculum's at-distance pair retrieval —
just with semantic content instead of synthetic vocab tokens.

Output
------
JSONL with per-example scores + a summary report. No DB writes; no training.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import tiktoken

from research.defaults import VOCAB_SIZE


HYDRA_ROOT = Path("/home/tim/Projects/LLM/HYDRA/data")
LEAD = " The answer is "  # framing tail glued onto prompt before scoring answer


@dataclass(slots=True)
class HydraExample:
    prompt: str
    expected_answer: str
    category: str
    type_: str
    src_idx: int


def _is_clean_example(d: dict) -> bool:
    """Drop rows where the labeled `expected_answer` disagrees with the teacher.

    Discovered 2026-05-22 while auditing `distill_math.jsonl`: ~5-10% of rows
    have `expected_answer` mismatching the value the teacher actually reaches
    in `teacher_completion` (e.g. labeled "441111" while the boxed answer is
    "111111111111"). Keeping those rows pollutes top-1 with examples the model
    is being graded *wrong* for getting *right*.

    Rule: `expected_answer` must appear as a substring of `teacher_completion`.
    Cheap, conservative, catches the label/rationale drift without trying to
    parse boxed-LaTeX or numeric equivalence.
    """
    ans = str(d.get("expected_answer", "")).strip()
    if not ans:
        return False
    completion = d.get("teacher_completion", "")
    if not isinstance(completion, str):
        return False
    return ans in completion


def load_held_out(
    files: tuple[str, ...] = ("distill_math.jsonl",),
    held_out_frac: float = 0.05,
    max_examples: int | None = 500,
    seed: int = 0,
) -> list[HydraExample]:
    """Pull the last `held_out_frac` of each file (chronological hold-out).

    Math-only by default: `distill_math.jsonl` is the only HYDRA distill file
    with `expected_answer` labels (audited 2026-05-22: 0/19990 in reasoning,
    0/23784 in teacher_completions). Discrimination scoring requires a known
    correct answer, so non-math files belong to the future
    `hydra_distillation_train` path, not this eval.
    """
    rng = random.Random(seed)
    out: list[HydraExample] = []
    for fname in files:
        path = HYDRA_ROOT / fname
        with path.open() as fh:
            lines = fh.readlines()
        n_total = len(lines)
        n_held = max(1, int(n_total * held_out_frac))
        held = lines[-n_held:]
        for i, ln in enumerate(held):
            d = json.loads(ln)
            if not _is_clean_example(d):
                continue
            out.append(
                HydraExample(
                    prompt=d["prompt"],
                    expected_answer=str(d["expected_answer"]).strip(),
                    category=d.get("category", "?"),
                    type_=d.get("type", "?"),
                    src_idx=n_total - n_held + i,
                )
            )
    rng.shuffle(out)
    if max_examples is not None:
        out = out[:max_examples]
    return out


def _length_bucket(prompt_chars: int) -> str:
    if prompt_chars < 200:
        return "short"
    if prompt_chars < 500:
        return "medium"
    return "long"


def _answer_kind(ans: str) -> str:
    s = ans.strip().lstrip("-")
    if s.replace(".", "", 1).isdigit():
        return "numeric"
    if len(ans.split()) == 1:
        return "single_token_ish"
    return "multi_word"


def _build_scoring_batch(
    tokenizer: tiktoken.Encoding,
    context: str,
    completions: list[str],
    max_context_tokens: int,
) -> tuple[list[list[int]], list[int]]:
    """Encode (context + each completion) and return (sequences, start_positions).

    Left-truncates the context to keep each completion intact. Returned
    `start_positions[i]` is the index into `sequences[i]` such that scoring the
    span ``[start_positions[i] + 1, len(seq_i) - 1]`` covers all completion
    tokens — matching the contract of `batched_span_mean_log_probs`.
    """
    ctx_ids_full = tokenizer.encode(context)
    sequences: list[list[int]] = []
    start_positions: list[int] = []
    for completion in completions:
        comp_ids = tokenizer.encode(completion)
        keep_ctx = max_context_tokens - len(comp_ids) - 1
        if keep_ctx <= 0:
            # Answer alone exceeds the model's max context — emit a 2-token
            # stub that batched_span_mean_log_probs flags as -inf.
            sequences.append([0, 0])
            start_positions.append(0)
            continue
        ctx_ids = ctx_ids_full[-keep_ctx:]
        sequences.append(list(ctx_ids) + list(comp_ids))
        # span_mean_log_probs scores positions [start+1, len-1], so start = ctx_end-1
        # gives us the prediction *of* comp_ids[0] (token at ctx_end) onward.
        start_positions.append(len(ctx_ids) - 1)
    return sequences, start_positions


def _pick_distractors(
    pool: list[str],
    correct: str,
    n_distractors: int,
    rng: random.Random,
) -> list[str]:
    """Sample N distractors of the same answer-kind when possible."""
    ex_kind = _answer_kind(correct)
    same_kind = [a for a in pool if a != correct and _answer_kind(a) == ex_kind]
    if len(same_kind) < n_distractors:
        same_kind = [a for a in pool if a != correct]
    return rng.sample(same_kind, min(n_distractors, len(same_kind)))


@torch.no_grad()
def _score_one_example(
    model: torch.nn.Module,
    ex: HydraExample,
    distractors: list[str],
    tok: tiktoken.Encoding,
    *,
    device: str | torch.device,
    max_context_tokens: int,
    vocab_size: int,
) -> dict:
    """One batched forward pass over (correct + distractors); returns row dict.

    Wrapped in ``torch.no_grad`` because ``model.eval()`` does not stop
    autograd from retaining activations — without this the eval doubles its
    peak memory for a backward pass that never runs.
    """
    from .utils import batched_span_mean_log_probs

    completions = [ex.expected_answer] + distractors
    sequences, starts = _build_scoring_batch(
        tok, ex.prompt + LEAD, completions, max_context_tokens
    )
    mean_lps = batched_span_mean_log_probs(
        model, sequences, starts, vocab_size=vocab_size, device=device
    ).tolist()

    lp_correct = mean_lps[0]
    lp_distractors = mean_lps[1:]
    rank = 1 + sum(1 for lp in lp_distractors if lp > lp_correct)
    lp_dist_mean = sum(lp_distractors) / len(lp_distractors) if lp_distractors else None
    margin = lp_correct - (lp_dist_mean if lp_dist_mean is not None else 0.0)

    return {
        "src_idx": ex.src_idx,
        "category": ex.category,
        "kind": _answer_kind(ex.expected_answer),
        "prompt_chars": len(ex.prompt),
        "length_bucket": _length_bucket(len(ex.prompt)),
        "expected_answer": ex.expected_answer,
        "lp_correct": lp_correct,
        "lp_distractor_mean": lp_dist_mean,
        "rank": rank,
        "rr": 1.0 / rank,
        "margin": margin,
        "n_distractors": len(distractors),
    }


def _agg(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "top1": sum(1 for r in rows if r["rank"] == 1) / len(rows),
        "top3": sum(1 for r in rows if r["rank"] <= 3) / len(rows),
        "mrr": sum(r["rr"] for r in rows) / len(rows),
        "mean_margin": sum(r["margin"] for r in rows) / len(rows),
    }


def evaluate(
    model: torch.nn.Module,
    examples: list[HydraExample],
    *,
    device: str | torch.device,
    n_distractors: int = 8,
    max_context_tokens: int = 1024,
    tokenizer_name: str = "cl100k_base",
    vocab_size: int = VOCAB_SIZE,
    seed: int = 0,
) -> dict:
    """Score each example by log-likelihood discrimination against N distractors.

    Mean log-prob (not sum) is used so answers of different lengths compare
    fairly. Distractors are drawn same-kind-first from the held-out answer pool.
    """
    tok = tiktoken.get_encoding(tokenizer_name)
    rng = random.Random(seed)
    pool = list({e.expected_answer for e in examples})

    model.eval()
    per_example = [
        _score_one_example(
            model,
            ex,
            _pick_distractors(pool, ex.expected_answer, n_distractors, rng),
            tok,
            device=device,
            max_context_tokens=max_context_tokens,
            vocab_size=vocab_size,
        )
        for ex in examples
    ]

    summary = {"overall": _agg(per_example)}
    for key in ("category", "length_bucket", "kind"):
        buckets: dict[str, list[dict]] = {}
        for r in per_example:
            buckets.setdefault(r[key], []).append(r)
        summary[f"by_{key}"] = {k: _agg(v) for k, v in buckets.items()}
    return {"per_example": per_example, "summary": summary}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a model state_dict .pt produced by mixer_fingerprint.",
    )
    p.add_argument(
        "--lane",
        required=True,
        type=str,
        help="Lane name (e.g. local_ssm_diff). Used to rebuild the model "
        "architecture before loading the checkpoint.",
    )
    p.add_argument(
        "--pattern",
        default=None,
        type=str,
        help="If --lane=interleaved, the pattern to expand.",
    )
    p.add_argument("--dim", default=384, type=int)
    p.add_argument("--n-blocks", default=12, type=int)
    p.add_argument(
        "--n-examples",
        default=5000,
        type=int,
        help="Eval pool size. Default 5000 saturates the clean math pool "
        "(5158 rows) for tightest std (~0.005 top1 at 3 seeds). Drop to "
        "2500 if you only need ±0.7 pp resolution and want faster runs.",
    )
    p.add_argument("--n-distractors", default=8, type=int)
    p.add_argument("--max-context-tokens", default=1024, type=int)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: 50 examples × 4 distractors × 512 ctx tokens.",
    )
    p.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Path to write per-seed + aggregated JSON. Optional.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Default 'cpu' to avoid OOM when a training run is GPU-resident.",
    )
    p.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Base distractor seed. Runs 1..N-1 use seed+1, seed+2, ...",
    )
    p.add_argument(
        "--n-seeds",
        default=3,
        type=int,
        help="Number of distractor seeds. Default 3 because at n=251 "
        "the distractor variance is ~1.5 pp top1 and a single seed misranks.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.quick:
        args.n_examples = min(args.n_examples, 50)
        args.n_distractors = min(args.n_distractors, 4)
        args.max_context_tokens = min(args.max_context_tokens, 512)

    device = torch.device(args.device)
    model = _load_checkpoint_model(args, device)

    # Held-out pool is fixed across seeds (same examples scored each time);
    # variance comes from distractor sampling inside evaluate().
    examples = load_held_out(
        max_examples=args.n_examples, seed=args.seed, held_out_frac=1.0
    )
    print(
        f"Loaded {len(examples)} eval examples; running {args.n_seeds} seed(s)...",
        flush=True,
    )

    result = _run_multi_seed(model, examples, args, device=device)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(result, fh, indent=2)

    print(json.dumps(result["aggregated"], indent=2))


def _run_multi_seed(
    model: torch.nn.Module,
    examples: list[HydraExample],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict:
    """Run evaluate() N times with consecutive seeds; return per-seed + aggregated."""
    import statistics

    per_seed = []
    for i in range(int(args.n_seeds)):
        seed = int(args.seed) + i
        r = evaluate(
            model,
            examples,
            device=device,
            n_distractors=args.n_distractors,
            max_context_tokens=args.max_context_tokens,
            seed=seed,
        )
        per_seed.append({"seed": seed, "summary": r["summary"]})
        ov = r["summary"]["overall"]
        print(
            f"  seed={seed} top1={ov['top1']:.4f} top3={ov['top3']:.4f} "
            f"mrr={ov['mrr']:.4f} margin={ov['mean_margin']:.4f}",
            flush=True,
        )

    def ms(values: list[float]) -> dict:
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }

    aggregated: dict = {"overall": {}}
    for metric in ("top1", "top3", "mrr", "mean_margin"):
        aggregated["overall"][metric] = ms(
            [s["summary"]["overall"][metric] for s in per_seed]
        )
    for axis in ("by_category", "by_length_bucket", "by_kind"):
        bucket_keys = set()
        for s in per_seed:
            bucket_keys.update(s["summary"].get(axis, {}).keys())
        axis_agg: dict = {}
        for bk in sorted(bucket_keys):
            axis_agg[bk] = {
                m: ms(
                    [
                        s["summary"][axis][bk][m]
                        for s in per_seed
                        if bk in s["summary"].get(axis, {})
                        and m in s["summary"][axis][bk]
                    ]
                )
                for m in ("top1", "top3", "mrr", "mean_margin")
            }
        aggregated[axis] = axis_agg

    return {"per_seed": per_seed, "aggregated": aggregated}


def _load_checkpoint_model(
    args: argparse.Namespace, device: torch.device
) -> torch.nn.Module:
    """Rebuild the lane architecture and load weights from `args.checkpoint`."""
    from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm
    from research.tools.mixer_fingerprint import _resolve_lane_factories

    if args.lane == "interleaved":
        model_factory, _ = _resolve_lane_factories("interleaved", args.pattern)
    else:
        model_factory = _build_lane_factory(args.lane)
    model = _build_tinylm(model_factory, dim=args.dim, n_blocks=args.n_blocks).to(
        device
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)  # nosec B614 - locally-produced checkpoint, not network-sourced
    model.load_state_dict(_extract_state_dict(state))
    return model


def _extract_state_dict(state: object) -> dict:
    """Return the model parameter dict from a torch.load result.

    Accepts the mixer_fingerprint wrapped format ``{"model_state_dict": ...,
    "step": ...}`` (5 usages in research/), the older sketch format
    ``{"model": ...}``, or a bare state_dict.
    """
    if isinstance(state, dict):
        for key in ("model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    if isinstance(state, dict):
        return state
    raise TypeError(f"unrecognized checkpoint payload type: {type(state)!r}")


if __name__ == "__main__":
    main()
