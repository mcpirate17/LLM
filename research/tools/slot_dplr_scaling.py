"""THE comparison: does DPLR beat the FIXED slot table, and does the edge hold as
size grows? Fixed slots beat softmax but saturate at scale; DPLR is meant to fix
that. Runs fixed (base) vs DPLR (all) at increasing widths, same from-scratch
binding protocol, and reports the DPLR-minus-fixed gap per size. If the gap holds
or grows with size -> DPLR scales better. If it shrinks -> DPLR also saturates.
All CPU; streams after every probe.

    python -m research.tools.slot_dplr_scaling --dims 128,256,512 --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from component_fab.harness.tiny_lm import TinyLM, TinyLMConfig
from research.eval.binding_multislot_probe import (
    BindingMultislotConfig,
    binding_multislot_probe,
    build_multi_blank_layout,
)
from research.tools.grade_slot_dplr_nano import _ablation_factory

VARIANTS = (("fixed", "base"), ("dplr", "all"))
LADDER = (1000, 2000, 5000, 10000)


def _probe(
    variant: str, *, dim: int, steps: int, seed: int, device: str, timeout: float
) -> dict:
    cfg = BindingMultislotConfig(seed=seed, train_steps=steps, timeout_s=timeout)
    vocab = build_multi_blank_layout(cfg).required_vocab + 1
    torch.manual_seed(seed)
    model = TinyLM(
        _ablation_factory(variant),
        TinyLMConfig(vocab_size=vocab, dim=dim, n_blocks=2, use_ffn=True),
    )
    nonembed = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "embed" not in n.lower() and "lm_head" not in n.lower()
    )
    r = binding_multislot_probe(model, cfg=cfg, device=device).to_dict()
    return {
        "steps": steps,
        "all_slots": round(float(r["binding_multislot_all_slots_acc"]), 4),
        "two_plus": round(float(r["binding_multislot_two_plus_slots_acc"]), 4),
        "status": r.get("binding_multislot_status", "?"),
        "nonembed_m": round(nonembed / 1e6, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="128,256,512")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--timeout", type=float, default=6000.0)
    ap.add_argument("--out", default="research/reports/slot_dplr_scaling.json")
    args = ap.parse_args()
    dims = [int(d) for d in args.dims.split(",") if d.strip()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    started = time.perf_counter()

    def flush() -> None:
        out.write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "ladder": list(LADDER),
                    "results": results,
                    "elapsed_s": round(time.perf_counter() - started, 1),
                },
                indent=2,
            )
        )

    for dim in dims:
        results[str(dim)] = {label: [] for label, _ in VARIANTS}
        for label, variant in VARIANTS:
            for steps in LADDER:
                row = _probe(
                    variant,
                    dim=dim,
                    steps=steps,
                    seed=args.seed,
                    device=args.device,
                    timeout=args.timeout,
                )
                results[str(dim)][label].append(row)
                flush()
                print(
                    f"[dim{dim} {label:5s} steps={steps:>6}] all_slots={row['all_slots']:.4f} "
                    f"({row['status']}, {row['nonembed_m']}M)",
                    flush=True,
                )
        f = results[str(dim)]["fixed"][-1]["all_slots"]
        d = results[str(dim)]["dplr"][-1]["all_slots"]
        print(
            f"  >> dim{dim} @ {LADDER[-1]} steps: fixed {f:.4f} vs DPLR {d:.4f}  (DPLR-fixed {d - f:+.4f})",
            flush=True,
        )

    flush()
    print("\n=== SCALING SUMMARY (DPLR - fixed gap at top step budget) ===")
    for dim in dims:
        f = results[str(dim)]["fixed"][-1]["all_slots"]
        d = results[str(dim)]["dplr"][-1]["all_slots"]
        print(f"  dim{dim:>4}: fixed {f:.4f} | DPLR {d:.4f} | gap {d - f:+.4f}")
    print(f"DONE. wrote {out} ({round(time.perf_counter() - started, 1)}s)")


if __name__ == "__main__":
    main()
