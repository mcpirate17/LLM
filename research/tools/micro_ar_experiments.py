"""Short AR-transfer micro-experiments.

Sweeps a handful of small (dim=512, ~10-20M params, 3K-step wikitext) runs
and prints ar_curriculum + binding + induction probe scores per variant.
Built to be cheap enough to iterate on the "why does AR not transfer" question
without burning 100K-step hybrid runs.

Each run:
  - builds TinyLM(lane, dim=512, n_blocks=K) with the chosen interleaved pattern
  - trains on wikitext-103 for N steps (default 3000)
  - runs ar_curriculum + binding_intermediate + induction_intermediate probes
  - logs JSONL row + a short stdout summary

Output: research/reports/mixer_fingerprint/micro_ar_experiments.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from research.defaults import VOCAB_SIZE
from research.tools.mixer_fingerprint import (
    _configure_torch_performance,
    _resolve_lane_factories,
)
from research.tools.scaling_blimp_study import (
    _RandomWindowBatcher,
    _build_tinylm,
    _causal_lm_loss,
    _load_wikitext_tokens,
)

OUTDIR = Path(__file__).resolve().parents[1] / "reports" / "mixer_fingerprint"
OUTDIR.mkdir(parents=True, exist_ok=True)
JSONL = OUTDIR / "micro_ar_experiments.jsonl"


def _train(
    model: nn.Module,
    *,
    train_batcher: _RandomWindowBatcher,
    n_steps: int,
    lr: float,
    device: torch.device,
    log_every: int = 200,
) -> dict[str, Any]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=(device.type == "cuda"))
    model.train()
    t0 = time.perf_counter()
    losses: list[float] = []
    for step in range(1, n_steps + 1):
        ids = train_batcher.next()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(ids)
            loss = _causal_lm_loss(logits, ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            losses.append(float(loss.detach().item()))
    torch.cuda.synchronize() if device.type == "cuda" else None
    return {
        "wall_s": round(time.perf_counter() - t0, 1),
        "final_loss": losses[-1] if losses else float("nan"),
        "loss_curve": losses,
    }


def _eval_probes(
    model: nn.Module, *, device: torch.device, full_ar: bool = False
) -> dict[str, Any]:
    """Run AR curriculum + binding_intermediate + induction_intermediate probes."""
    from research.eval.ar_curriculum_probe import (
        ar_curriculum_probe,
        ARCurriculumConfig,
    )
    from research.eval.binding_intermediate_probe import run_binding_intermediate
    from research.eval.induction_intermediate_probe import run_induction_intermediate

    out: dict[str, Any] = {}
    model.eval()

    t0 = time.perf_counter()
    try:
        steps_per = 1000 if full_ar else 400
        res = ar_curriculum_probe(
            model,
            cfg=ARCurriculumConfig(
                seed=0,
                steps_per_stage=steps_per,
                batch_size=16,
                eval_batches=16,
                mode="cumulative",
            ),
            device=str(device),
        )
        d = res.to_dict() if hasattr(res, "to_dict") else res
        out["ar_curriculum_pair_final"] = d.get("ar_curriculum_auc_pair_final")
        out["ar_curriculum_max_stage"] = d.get("ar_curriculum_max_passing_stage")
        out["ar_curriculum_status"] = d.get("ar_curriculum_status")
    except Exception as e:
        out["ar_curriculum_status"] = f"err: {type(e).__name__}: {e}"[:200]
    out["_t_ar_curriculum"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    try:
        res = run_binding_intermediate(
            model,
            n_train_steps=300,
            n_eval=128,
            train_batch_size=8,
            eval_batch_size=8,
            device=str(device),
        )
        out["binding_v2_auc"] = res.auc
        out["binding_v2_status"] = res.status
    except Exception as e:
        out["binding_v2_status"] = f"err: {type(e).__name__}: {e}"[:200]
    out["_t_binding"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    try:
        res = run_induction_intermediate(
            model, n_train_steps=300, n_eval=128, batch_size=8, device=str(device)
        )
        out["induction_intermediate_auc"] = res.auc
        out["induction_intermediate_status"] = res.status
    except Exception as e:
        out["induction_intermediate_status"] = f"err: {type(e).__name__}: {e}"[:200]
    out["_t_induction"] = round(time.perf_counter() - t0, 1)

    # binding_multislot — the metric that regressed in yesterday's 2way hybrid.
    # Tracking it explicitly is necessary to see whether ensemble-at-bottom
    # keeps multi-slot binding alive or just AR.
    t0 = time.perf_counter()
    try:
        from research.eval.binding_multislot_probe import (
            binding_multislot_probe,
            BindingMultislotConfig,
        )

        res = binding_multislot_probe(
            model,
            cfg=BindingMultislotConfig(train_steps=300, batch_size=8, n_eval=96),
            device=str(device),
        )
        d = res.to_dict() if hasattr(res, "to_dict") else {}
        out["binding_multislot_auc"] = d.get("binding_multislot_auc")
        out["binding_multislot_status"] = d.get("binding_multislot_status")
    except Exception as e:
        out["binding_multislot_status"] = f"err: {type(e).__name__}: {e}"[:200]
    out["_t_binding_multislot"] = round(time.perf_counter() - t0, 1)

    return out


def _run_variant(
    *,
    label: str,
    mixer: str,
    pattern: str | None,
    dim: int,
    n_blocks: int,
    n_steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    seed: int,
    train_tokens: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model_factory, _ = _resolve_lane_factories(mixer, pattern)
    model = _build_tinylm(
        model_factory, dim=dim, n_blocks=n_blocks, vocab_size=VOCAB_SIZE, use_ffn=True
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    batcher = _RandomWindowBatcher(
        train_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        device=str(device),
        seed=seed,
    )
    train_meta = _train(
        model,
        train_batcher=batcher,
        n_steps=n_steps,
        lr=lr,
        device=device,
    )
    probes = _eval_probes(model, device=device)

    row = {
        "label": label,
        "mixer": mixer,
        "pattern": pattern,
        "dim": dim,
        "n_blocks": n_blocks,
        "n_params": n_params,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "lr": lr,
        "seed": seed,
        "train": train_meta,
        "probes": probes,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with JSONL.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _fmt(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "  err"


def _print_row(row: dict[str, Any]) -> None:
    p = row["probes"]
    n_params_m = row["n_params"] / 1e6
    print(
        f"  [{row['label']:36s}] {n_params_m:5.1f}M  "
        f"ar={_fmt(p.get('ar_curriculum_pair_final'))}  "
        f"bind={_fmt(p.get('binding_v2_auc'))}  "
        f"mslot={_fmt(p.get('binding_multislot_auc'))}  "
        f"ind={_fmt(p.get('induction_intermediate_auc'))}  "
        f"wall={row['train']['wall_s']:.0f}s  "
        f"loss={row['train']['final_loss']:.2f}",
        flush=True,
    )


def run_experiment_d_multi_capability(
    *, train_tokens, device, dim=384, n_steps=3000
) -> None:
    """Exp D: can we keep AR + binding + induction simultaneously?

    Fixed depth=5. Vary ensemble:three_lane ratio. Ensemble at bottom (Exp C
    showed this is the only position that preserves AR in heterogeneous
    stacks). All patterns at the same n_blocks so per-step compute is similar.
    """
    print(
        f"=== Experiment D — multi-capability ratio sweep "
        f"(dim={dim}, depth=5, steps={n_steps}) ==="
    )
    # Ratio sweep at fixed depth=5, ensemble at bottom
    patterns = [
        ("D_5tl_control", "interleaved", "three_lane:5"),
        ("D_1ens_4tl", "interleaved", "ensemble_top_ar_2way:1,three_lane:4"),
        ("D_2ens_3tl", "interleaved", "ensemble_top_ar_2way:2,three_lane:3"),
        ("D_3ens_2tl", "interleaved", "ensemble_top_ar_2way:3,three_lane:2"),
        ("D_4ens_1tl", "interleaved", "ensemble_top_ar_2way:4,three_lane:1"),
        ("D_5ens_control", "ensemble_top_ar_2way", None),
        # Interleaving variants at 2-ensemble budget
        (
            "D_2ens_1tl_1conv_1tl",
            "interleaved",
            "ensemble_top_ar_2way:2,three_lane:1,conv:1,three_lane:1",
        ),
        (
            "D_1ens_2tl_1conv_1tl",
            "interleaved",
            "ensemble_top_ar_2way:1,three_lane:2,conv:1,three_lane:1",
        ),
    ]
    for label, mixer, pattern in patterns:
        n_blocks = 5
        row = _run_variant(
            label=label,
            mixer=mixer,
            pattern=pattern,
            dim=dim,
            n_blocks=n_blocks,
            n_steps=n_steps,
            batch_size=16,
            seq_len=256,
            lr=3e-4,
            seed=0,
            train_tokens=train_tokens,
            device=device,
        )
        _print_row(row)


def run_experiment_a_depth_scaling(
    *, train_tokens, device, dim=512, n_steps=3000
) -> None:
    """Exp A: does AR survive at depth > 1 when EVERY block is the ensemble?"""
    print(
        f"=== Experiment A — pure-ensemble depth scaling (dim={dim}, steps={n_steps}) ==="
    )
    for n in [1, 2, 4]:
        row = _run_variant(
            label=f"A_pure_ens4way_n{n}",
            mixer="ensemble_top_ar_4way",
            pattern=None,
            dim=dim,
            n_blocks=n,
            n_steps=n_steps,
            batch_size=16,
            seq_len=256,
            lr=3e-4,
            seed=0,
            train_tokens=train_tokens,
            device=device,
        )
        _print_row(row)


def run_experiment_b_single_vs_ensemble(
    *, train_tokens, device, dim=512, n_steps=3000
) -> None:
    """Exp B: at depth=2, does ensemble averaging help or hurt vs single graph?"""
    print(
        f"=== Experiment B — single vs ensemble at depth=2 (dim={dim}, steps={n_steps}) ==="
    )
    for label, mixer in [
        ("B_single_topar_d2", "top_ar_block"),
        ("B_ens2way_d2", "ensemble_top_ar_2way"),
        ("B_ens4way_d2", "ensemble_top_ar_4way"),
    ]:
        row = _run_variant(
            label=label,
            mixer=mixer,
            pattern=None,
            dim=dim,
            n_blocks=2,
            n_steps=n_steps,
            batch_size=16,
            seq_len=256,
            lr=3e-4,
            seed=0,
            train_tokens=train_tokens,
            device=device,
        )
        _print_row(row)


def run_experiment_c_position(*, train_tokens, device, dim=512, n_steps=3000) -> None:
    """Exp C: at fixed n_blocks=5 (3 conv + 2 ensemble), does position matter?"""
    print(
        f"=== Experiment C — ensemble position in hybrid (dim={dim}, steps={n_steps}) ==="
    )
    for label, pattern in [
        ("C_ens_top", "ensemble_top_ar_2way:2,conv:3"),
        ("C_ens_bottom", "conv:3,ensemble_top_ar_2way:2"),
        ("C_ens_middle", "conv:2,ensemble_top_ar_2way:2,conv:1"),
    ]:
        row = _run_variant(
            label=label,
            mixer="interleaved",
            pattern=pattern,
            dim=dim,
            n_blocks=5,
            n_steps=n_steps,
            batch_size=16,
            seq_len=256,
            lr=3e-4,
            seed=0,
            train_tokens=train_tokens,
            device=device,
        )
        _print_row(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiments",
        nargs="+",
        default=["A", "B", "C"],
        choices=["A", "B", "C", "D"],
    )
    p.add_argument("--steps", default=3000, type=int)
    p.add_argument("--dim", default=512, type=int)
    args = p.parse_args()

    _configure_torch_performance()
    os.environ.setdefault("SYNTHESIS_ENSEMBLE_EXECUTOR", "compiled")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    print("loading wikitext-103 tokens...")
    train_tokens, _, n_train, _ = _load_wikitext_tokens(
        variant="wikitext-103-raw-v1",
        vocab_size=VOCAB_SIZE,
        max_chars_train=50_000_000,
        max_chars_val=2_000_000,
    )
    print(f"  loaded {n_train:,} tokens")

    if "A" in args.experiments:
        run_experiment_a_depth_scaling(
            train_tokens=train_tokens, device=device, dim=args.dim, n_steps=args.steps
        )
    if "B" in args.experiments:
        run_experiment_b_single_vs_ensemble(
            train_tokens=train_tokens, device=device, dim=args.dim, n_steps=args.steps
        )
    if "C" in args.experiments:
        run_experiment_c_position(
            train_tokens=train_tokens, device=device, dim=args.dim, n_steps=args.steps
        )
    if "D" in args.experiments:
        run_experiment_d_multi_capability(
            train_tokens=train_tokens, device=device, dim=args.dim, n_steps=args.steps
        )


if __name__ == "__main__":
    main()
