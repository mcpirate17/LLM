"""Capability A/B for staged training regimes (Workstream E).

Train the SAME non-QKV carrier at identical init and token budget while varying
only the staged-training genotype. Evaluation stays on the natural validation
stream. A regime wins only if it improves matched-budget capability/convergence
over ``all_train``; lower training loss alone is not enough.

Default conditions are valid for a standalone carrier:

* ``all_train`` — normal full-model training baseline.
* ``embed_warm_then_all`` — train embedding/head first, then all params.
* ``body_warm_then_all`` — train non-embedding body first, then all params.

Pair-specific regimes such as ``carrier_warm_then_all`` and
``router_warm_then_all`` are implemented in the genotype and can be requested
explicitly when the model has matching parameter names.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from research.defaults import MAX_SEQ_LEN, N_LAYERS
from research.synthesis.training_regime_grammar import (
    TrainStageSpec,
    TrainingRegimeSpec,
    implemented_training_regimes,
    training_regime_to_axes,
)
from research.tools.embed_warmup_ab import (
    _DEFAULT_CARRIER_RID,
    _OUT_DIR,
    _build_carrier,
    _carrier_graph_json,
)
from research.tools.loss_monster_screen import (
    _CORPUS_TRAIN,
    _CORPUS_VAL,
    _RUNS_DB,
    _sample_batch,
    evaluate,
)
from research.training._optimizer_factory import build_optimizer
from research.training.staged_training import apply_train_stage, trainable_parameters

_DEFAULT_CONDITIONS = ("all_train", "embed_warm_then_all", "body_warm_then_all")


def _rescale_stage_steps(
    regime: TrainingRegimeSpec, total_steps: int
) -> TrainingRegimeSpec:
    """Scale a regime's stage proportions to the requested total step budget."""

    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    raw_total = max(1, regime.total_steps)
    scaled: list[TrainStageSpec] = []
    used = 0
    for index, stage in enumerate(regime.stages):
        if index == len(regime.stages) - 1:
            steps = total_steps - used
        else:
            steps = max(1, int(round(total_steps * stage.steps / raw_total)))
            remaining_stages = len(regime.stages) - index - 1
            steps = min(steps, total_steps - used - remaining_stages)
        used += steps
        scaled.append(replace(stage, steps=steps))
    return replace(regime, stages=tuple(scaled))


def _make_optimizer(
    params: list[torch.nn.Parameter],
    regime: TrainingRegimeSpec,
    stage: TrainStageSpec,
    base_lr: float,
) -> torch.optim.Optimizer:
    return build_optimizer(
        params,
        optimizer_type=regime.optimizer,
        lr=base_lr * stage.lr_scale,
        weight_decay=regime.weight_decay,
    )


def _train_curve_staged(
    model: torch.nn.Module,
    train: np.ndarray,
    val: np.ndarray,
    regime: TrainingRegimeSpec,
    *,
    seq: int,
    batch: int,
    steps: int,
    lr: float,
    device: str,
    eval_every: int,
    eval_batches: int,
    allow_empty_targets: bool = False,
) -> list[dict[str, float | str | int]]:
    """Train with staged freeze masks; evaluate on natural val stream."""

    active = _rescale_stage_steps(regime, steps)
    gen = np.random.default_rng(1234)
    curve: list[dict[str, float | str | int]] = []
    global_step = 0
    current_stage = active.stages[0]
    report = apply_train_stage(
        model, current_stage, allow_empty=allow_empty_targets
    )
    params = trainable_parameters(model)
    opt = _make_optimizer(params, active, current_stage, lr)

    stage_ends: list[int] = []
    running = 0
    for stage in active.stages:
        running += stage.steps
        stage_ends.append(running)

    def maybe_eval(stage_name: str) -> None:
        if global_step % eval_every == 0 or global_step == steps:
            m = evaluate(
                model, val, batch=batch, seq=seq, n_batches=eval_batches, device=device
            )
            curve.append(
                {
                    "step": global_step,
                    "stage": stage_name,
                    "trainable_params": report.trainable_param_count,
                    **m,
                }
            )

    maybe_eval(current_stage.target)
    for stage_index, stage_end in enumerate(stage_ends):
        current_stage = active.stages[stage_index]
        if stage_index > 0:
            report = apply_train_stage(
                model, current_stage, allow_empty=allow_empty_targets
            )
            params = trainable_parameters(model)
            opt = _make_optimizer(params, active, current_stage, lr)
            maybe_eval(current_stage.target)
        while global_step < stage_end:
            x, y = _sample_batch(train, batch, seq, gen, device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, active.max_grad_norm)
            opt.step()
            global_step += 1
            maybe_eval(current_stage.target)
    return curve


def _steps_to_threshold(curve: list[dict[str, Any]], target: float) -> int | None:
    for pt in curve:
        if float(pt["val_loss"]) <= target:
            return int(pt["step"])
    return None


def run_condition(
    graph_json: str,
    condition: str,
    regime: TrainingRegimeSpec,
    seed: int,
    train: np.ndarray,
    val: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    t0 = time.time()
    model = _build_carrier(graph_json, args.n_layers, args.seq, args.device, seed)
    active_regime = _rescale_stage_steps(regime, args.steps)
    curve = _train_curve_staged(
        model,
        train,
        val,
        regime,
        seq=args.seq,
        batch=args.batch,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        allow_empty_targets=args.allow_empty_targets,
    )
    final = curve[-1]
    print(
        f"    [{condition} seed{seed}] final val_loss={final['val_loss']:.4f} "
        f"top1={final['top1_acc']:.4f}  ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return {
        "condition": condition,
        "training_regime": training_regime_to_axes(active_regime),
        "seed": seed,
        "curve": curve,
        "final": final,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [r["final"]["val_loss"] for r in results if r["condition"] == "all_train"]
    target = float(np.mean(baseline)) if baseline else float("inf")
    summary: dict[str, Any] = {
        "baseline_all_train_val_loss": round(target, 4),
        "by_condition": {},
    }
    for cond in dict.fromkeys(r["condition"] for r in results):
        runs = [r for r in results if r["condition"] == cond]
        finals = [r["final"]["val_loss"] for r in runs]
        top1s = [r["final"]["top1_acc"] for r in runs]
        s2t = [
            s
            for r in runs
            if (s := _steps_to_threshold(r["curve"], target)) is not None
        ]
        summary["by_condition"][cond] = {
            "mean_final_val_loss": round(float(np.mean(finals)), 4),
            "mean_final_top1": round(float(np.mean(top1s)), 4),
            "mean_steps_to_all_train_final": (
                round(float(np.mean(s2t)), 1) if s2t else None
            ),
            "n_reached_baseline": f"{len(s2t)}/{len(runs)}",
        }
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    regimes = implemented_training_regimes()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carrier-rid", default=_DEFAULT_CARRIER_RID)
    ap.add_argument(
        "--conditions",
        nargs="*",
        default=list(_DEFAULT_CONDITIONS),
        help=f"subset of {list(regimes)}",
    )
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--seq", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-empty-targets", action="store_true")
    ap.add_argument("--out", default=str(_OUT_DIR / "training_regime_ab.json"))
    return ap


def main() -> int:
    args = _build_argparser().parse_args()
    regimes = implemented_training_regimes()
    unknown = [c for c in args.conditions if c not in regimes]
    if unknown:
        print(f"Unknown conditions {unknown}; valid={list(regimes)}")
        return 1
    graph_json = _carrier_graph_json(_RUNS_DB, args.carrier_rid)
    train = np.load(_CORPUS_TRAIN, mmap_mode="r")
    val = np.load(_CORPUS_VAL, mmap_mode="r")

    print(
        f"Carrier rid={args.carrier_rid} regimes={args.conditions} "
        f"seeds={args.seeds} steps={args.steps}"
    )
    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        for cond in args.conditions:
            results.append(
                run_condition(
                    graph_json, cond, regimes[cond], seed, train, val, args
                )
            )

    summary = _summarize(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
