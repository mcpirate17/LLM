"""Arch-only v3 AR-validation sweep over mixer_fingerprint lane patterns.

Companion to ``backfill_ar_validation``. The DB-backfill tool reconstructs
fresh models from ``graph_json`` rows. This tool does the same thing for the
lane-pattern architectures used by ``mixer_fingerprint`` (conv6_threelane6,
hybrid_conv5_3lane5_ens2way2, etc.) — which have no DB row.

What it does for each architecture:
  1. Build a FRESH ``TinyLM`` from (lane, pattern, dim, n_blocks, vocab_size).
     The .pt weights of the original training run are NOT loaded — this
     measures the architecture's AR-affinity, not any specific run's weights.
  2. Pretrain that model from scratch for --pretrain-steps on the wikitext103
     corpus (tokens projected mod vocab_size).
  3. Run the v3 stable AR-validation protocol on the pretrained model
     (multi-seed, deterministic episode bank, size-budgeted).
  4. Tabulate ``rank_score_mean ± rank_score_std``, held_pair, held_class.

This is intentionally a separate tool — no DB writes, no notebook entries.
Output is a JSON summary in ``research/reports/``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from research.eval.ar_validation import (
    STABLE_AR_VALIDATION_PROTOCOL,
    ARValidationConfig,
    run_ar_validation,
)
from research.tools.run_ar_validation_fingerprint_sweep import (
    DEFAULT_CORPUS,
    _load_projected_corpus,
    _pretrain_lm,
)


PROJECT_ROOT = Path("/home/tim/Projects/LLM")
DEFAULT_OUT = PROJECT_ROOT / "research/reports/hydra_eval_2026-05-22/ar_arch_sweep.json"

# Standardized 6-layer probes derived from the mixer_fingerprint architectures.
# Pattern proportions are halved from the production runs so the v3 size budget
# is comparable across arches; ensemble_top_ar_4way is left at n_blocks=1
# because that's what the original n_blocks=1 model used.
DEFAULT_ARCHS = [
    # label, lane, pattern, n_blocks
    ("conv_only", "causal_conv", None, 6),
    ("three_lane_only", "tropical_sparsemax_wavelet_three_lane", None, 6),
    ("sparsemax_only", "sparsemax_attention", None, 6),
    ("tropical_only", "tropical_attention", None, 6),
    ("conv_three_lane", "interleaved", "conv:3,three_lane:3", 6),
    ("conv_sparsemax", "interleaved", "conv:3,sparsemax:3", 6),
    (
        "hybrid_conv2_3lane3_ens2way1",
        "interleaved",
        "conv:2,three_lane:3,ensemble_top_ar_2way:1",
        6,
    ),
    ("ensemble_top_ar_4way_n1", "ensemble_top_ar_4way", None, 1),
]


def _build_probe_model(
    lane: str,
    pattern: str | None,
    *,
    dim: int,
    n_blocks: int,
    vocab_size: int,
    max_seq_len: int,
) -> torch.nn.Module:
    """Build a fresh TinyLM probe model from a lane spec (no checkpoint loaded)."""
    from research.tools.mixer_fingerprint import _resolve_lane_factories
    from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm

    if lane == "interleaved":
        if not pattern:
            raise ValueError("interleaved lane requires a pattern")
        lane_factory, _probe_factory = _resolve_lane_factories("interleaved", pattern)
    else:
        lane_factory = _build_lane_factory(lane)

    return _build_tinylm(
        lane_factory,
        dim=int(dim),
        n_blocks=int(n_blocks),
        vocab_size=int(vocab_size),
        max_seq_len=int(max_seq_len),
    )


def _run_one_arch(
    label: str,
    lane: str,
    pattern: str | None,
    n_blocks: int,
    *,
    dim: int,
    vocab_size: int,
    pretrain_steps: int,
    pretrain_batch_size: int,
    pretrain_seq_len: int,
    pretrain_lr: float,
    cfg: ARValidationConfig,
    corpus_tokens: torch.Tensor,
    device: torch.device,
    init_seed: int,
) -> dict:
    """Pretrain a fresh probe + run v3 stable AR-validation; return summary row."""
    t0 = time.perf_counter()
    torch.manual_seed(int(init_seed))
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(init_seed))

    model = _build_probe_model(
        lane,
        pattern,
        dim=dim,
        n_blocks=n_blocks,
        vocab_size=vocab_size,
        max_seq_len=max(512, pretrain_seq_len),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{label}] built fresh probe: lane={lane!r} pattern={pattern!r} "
        f"n_blocks={n_blocks} dim={dim} vocab={vocab_size} params={n_params / 1e6:.1f}M",
        flush=True,
    )

    pre_loss, pre_ms = _pretrain_lm(
        model,
        corpus_tokens,
        device=device,
        steps=int(pretrain_steps),
        batch_size=int(pretrain_batch_size),
        seq_len=int(pretrain_seq_len),
        lr=float(pretrain_lr),
        seed=int(init_seed) + 1009,
        progress_every=max(int(pretrain_steps) // 5, 1),
        progress_label={"arch": label},
    )
    print(
        f"[{label}] pretrain done: final_loss={pre_loss} elapsed_s={pre_ms / 1000:.1f}",
        flush=True,
    )

    result = run_ar_validation(model, cfg=cfg, device=str(device))
    print(
        f"[{label}] ar_val v3 done: score_mean={getattr(result, 'rank_score_mean', None)} "
        f"score_std={getattr(result, 'rank_score_std', None)} status={result.status}",
        flush=True,
    )

    row = _build_result_row(
        label,
        lane,
        pattern,
        n_blocks,
        dim,
        vocab_size,
        n_params,
        pretrain_steps,
        pretrain_batch_size,
        pretrain_seq_len,
        pretrain_lr,
        pre_loss,
        pre_ms,
        result,
        t0,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _build_result_row(
    label: str,
    lane: str,
    pattern: str | None,
    n_blocks: int,
    dim: int,
    vocab_size: int,
    n_params: int,
    pretrain_steps: int,
    pretrain_batch_size: int,
    pretrain_seq_len: int,
    pretrain_lr: float,
    pre_loss: float | None,
    pre_ms: float,
    result,
    t0: float,
) -> dict:
    return {
        "arch": label,
        "lane": lane,
        "pattern": pattern,
        "n_blocks": int(n_blocks),
        "dim": int(dim),
        "vocab_size": int(vocab_size),
        "n_params": int(n_params),
        "pretrain": {
            "steps": int(pretrain_steps),
            "batch_size": int(pretrain_batch_size),
            "seq_len": int(pretrain_seq_len),
            "lr": float(pretrain_lr),
            "final_loss": pre_loss,
            "elapsed_ms": round(pre_ms, 1),
        },
        "ar_validation": result.to_dict(),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT, help="JSON output path.")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument(
        "--dim",
        default=384,
        type=int,
        help="Probe model hidden dim. v3 size budget self-adjusts.",
    )
    p.add_argument("--vocab-size", default=32000, type=int)
    p.add_argument(
        "--layers",
        default=6,
        type=int,
        help="Default n_blocks for archs that don't specify (overridden per-arch).",
    )
    p.add_argument("--pretrain-steps", default=5000, type=int)
    p.add_argument("--pretrain-batch-size", default=32, type=int)
    p.add_argument("--pretrain-seq-len", default=512, type=int)
    p.add_argument("--pretrain-lr", default=3e-4, type=float)
    p.add_argument(
        "--seed-count",
        default=3,
        type=int,
        help="v3 stable: number of distractor seeds for ar_validation aggregation.",
    )
    p.add_argument(
        "--init-seed", default=0, type=int, help="Seed for torch + pretrain RNG."
    )
    p.add_argument(
        "--archs",
        nargs="*",
        default=None,
        help="Optional subset of arch labels to run (default: all in DEFAULT_ARCHS).",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but cuda not available")

    print(
        f"loading corpus from {args.corpus_path} (mod vocab={args.vocab_size})...",
        flush=True,
    )
    corpus_tokens = _load_projected_corpus(
        args.corpus_path,
        vocab_size=int(args.vocab_size),
        device=device,
    )
    print(f"  corpus tokens: {corpus_tokens.numel():,}", flush=True)

    cfg = ARValidationConfig(
        protocol=STABLE_AR_VALIDATION_PROTOCOL,
        seed=int(args.init_seed),
        seed_count=int(args.seed_count),
        auto_size_budget=True,
        deterministic_episode_bank=True,
        copy_model=True,
    )

    selected = (
        DEFAULT_ARCHS
        if not args.archs
        else [a for a in DEFAULT_ARCHS if a[0] in set(args.archs)]
    )
    if not selected:
        raise ValueError(f"no archs matched: {args.archs}")
    print(f"running {len(selected)} arches: {[a[0] for a in selected]}", flush=True)

    rows = []
    for label, lane, pattern, n_blocks in selected:
        try:
            row = _run_one_arch(
                label,
                lane,
                pattern,
                n_blocks,
                dim=int(args.dim),
                vocab_size=int(args.vocab_size),
                pretrain_steps=int(args.pretrain_steps),
                pretrain_batch_size=int(args.pretrain_batch_size),
                pretrain_seq_len=int(args.pretrain_seq_len),
                pretrain_lr=float(args.pretrain_lr),
                cfg=cfg,
                corpus_tokens=corpus_tokens,
                device=device,
                init_seed=int(args.init_seed),
            )
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append(
                {
                    "arch": label,
                    "lane": lane,
                    "pattern": pattern,
                    "n_blocks": n_blocks,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
        # Incremental save after each arch — a kill mid-sweep won't lose completed work
        _write_output(args, cfg, rows)
        print(f"[{label}] partial output flushed to {args.output}", flush=True)

    _write_output(args, cfg, rows)
    _print_summary(rows)
    print(f"\nwrote {args.output}")


def _write_output(
    args: argparse.Namespace, cfg: ARValidationConfig, rows: list[dict]
) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(
            {
                "config": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()
                },
                "ar_validation_cfg": asdict(cfg),
                "rows": rows,
            },
            fh,
            indent=2,
        )


def _print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 110)
    print(
        f"{'arch':>34} {'params':>9} {'score m±s':>22} {'held_pair m±s':>22} {'held_class m±s':>22}"
    )
    print("=" * 110)
    for r in rows:
        if "error" in r:
            err = str(r.get("error", ""))[:80]
            print(f"{r['arch']:>34} {'-':>9} {'FAILED':>22} {err}")
            continue
        av = r["ar_validation"]
        rs_mean = av.get(
            "ar_validation_rank_score_mean", av.get("ar_validation_rank_score") or 0
        )
        rs_std = av.get("ar_validation_rank_score_std", 0.0)
        hpm = av.get(
            "ar_validation_held_pair_acc_mean", av.get("ar_validation_held_pair_acc", 0)
        )
        hps = av.get("ar_validation_held_pair_acc_std", 0.0)
        hcm = av.get(
            "ar_validation_held_class_acc_mean",
            av.get("ar_validation_held_class_acc", 0),
        )
        hcs = av.get("ar_validation_held_class_acc_std", 0.0)
        params_str = f"{r['n_params'] / 1e6:.1f}M"
        score_str = f"{rs_mean:.4f} ± {rs_std:.4f}"
        hp_str = f"{hpm:.4f} ± {hps:.4f}"
        hc_str = f"{hcm:.4f} ± {hcs:.4f}"
        print(
            f"{r['arch']:>34} {params_str:>9} {score_str:>22} {hp_str:>22} {hc_str:>22}"
        )


if __name__ == "__main__":
    main()
