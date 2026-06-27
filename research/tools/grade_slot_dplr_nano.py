"""Run-1 nano grade: does the DPLR/forget/learnable-slot upgrade move the
compositional-binding wall (binding_multislot all_slots=0.0) off the floor?

Compares the locked `slot_table_mh` lane against the upgraded `slot_table_mh_dplr`
lane (content-aware per-slot eviction + DPLR low-rank value + learnt slot identity)
on the binding_multislot probe. Single-lane TinyLM so any movement is attributable
to the slot mechanism alone. Cheap (CPU/GPU, ~1k probe steps); gate before the 40M.

    python -m research.tools.grade_slot_dplr_nano --device cuda --seeds 0,1
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
from research.tools._scaling_lanes import _build_lane_factory

LANES = ("slot_table_mh", "slot_table_mh_dplr")
KEYS = (
    "binding_multislot_all_slots_acc",
    "binding_multislot_two_plus_slots_acc",
    "binding_multislot_held_entity_slot_acc",
    "binding_multislot_held_entity_class_acc",
    "binding_multislot_mixed_all_slots_acc",
)


_ABLATION = {
    "base": dict(content_forget=False, dplr_value_rank=0, learnable_slots=False),
    "forget_only": dict(content_forget=True, dplr_value_rank=0, learnable_slots=False),
    "dplr_only": dict(content_forget=False, dplr_value_rank=16, learnable_slots=False),
    "slots_only": dict(content_forget=False, dplr_value_rank=0, learnable_slots=True),
    "all": dict(content_forget=True, dplr_value_rank=16, learnable_slots=True),
}


def _ablation_factory(variant: str):
    """Single-lever variants (same base config as slot_table_mh_dplr) so the gain
    can be attributed to content_forget vs DPLR-value vs learnable-slots."""
    from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane

    return lambda d: MultiHeadSlotTableMemoryLane(
        d,
        memory_dim=max(4, ((7 * d) // 32) * 4),
        n_heads=max(4, d // 64),
        n_slots=8,
        use_delta_update=True,
        route_from_input=True,
        normalize_slot_values=True,
        refine_write_route=True,
        consolidate_slots=True,
        **_ABLATION[variant],
    )


def _grade_one(
    lane: str,
    *,
    dim: int,
    n_blocks: int,
    steps: int,
    seed: int,
    device: str,
    bindings: int,
    query_slots: int,
) -> dict:
    factory = (
        _ablation_factory(lane) if lane in _ABLATION else _build_lane_factory(lane)
    )
    cfg = BindingMultislotConfig(
        seed=seed,
        train_steps=steps,
        bindings_per_example=bindings,
        query_slots=query_slots,
    )
    vocab = build_multi_blank_layout(cfg).required_vocab + 1
    torch.manual_seed(seed)
    model = TinyLM(
        factory,
        TinyLMConfig(vocab_size=vocab, dim=dim, n_blocks=n_blocks, use_ffn=True),
    )
    n_params = sum(p.numel() for p in model.parameters())
    result = binding_multislot_probe(model, cfg=cfg, device=device).to_dict()
    out = {k: round(float(result.get(k, 0.0)), 4) for k in KEYS}
    out["status"] = result.get("binding_multislot_status", "ok")
    out["n_params"] = n_params
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bindings", type=int, default=5, help="bindings_per_example")
    ap.add_argument("--query-slots", type=int, default=3)
    ap.add_argument(
        "--lanes",
        default=",".join(LANES),
        help="comma list; lane names or ablation variants (base/forget_only/dplr_only/slots_only/all)",
    )
    ap.add_argument(
        "--out",
        default="research/reports/slot_dplr_nano_grade.json",
    )
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    lanes = tuple(s.strip() for s in args.lanes.split(",") if s.strip())
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    t0 = time.perf_counter()
    rows: dict[str, list[dict]] = {lane: [] for lane in lanes}
    for lane in lanes:
        for seed in seeds:
            row = _grade_one(
                lane,
                dim=args.dim,
                n_blocks=args.n_blocks,
                steps=args.steps,
                seed=seed,
                device=args.device,
                bindings=args.bindings,
                query_slots=args.query_slots,
            )
            rows[lane].append(row)
            print(f"[{lane} seed={seed}] {row}")

    # Mean over seeds per lane.
    summary = {}
    for lane in lanes:
        agg = {
            k: round(sum(r[k] for r in rows[lane]) / len(rows[lane]), 4) for k in KEYS
        }
        agg["n_params"] = rows[lane][0]["n_params"]
        summary[lane] = agg

    # Baseline = first lane, treatment = last lane (works for the headline 2-lane
    # run and for ablation sweeps base,forget_only,... ,all).
    base, dplr = summary[lanes[0]], summary[lanes[-1]]
    deltas = {k: round(dplr[k] - base[k], 4) for k in KEYS}
    verdict = {
        "all_slots_moved_off_floor": dplr["binding_multislot_all_slots_acc"] > 0.02,
        "all_slots_delta": deltas["binding_multislot_all_slots_acc"],
        "two_plus_delta": deltas["binding_multislot_two_plus_slots_acc"],
    }

    report = {
        "config": vars(args),
        "seeds": seeds,
        "per_seed": rows,
        "summary": summary,
        "dplr_minus_baseline": deltas,
        "verdict": verdict,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n=== SUMMARY (mean over seeds) ===")
    print(
        f"baseline all_slots={base['binding_multislot_all_slots_acc']}  "
        f"two_plus={base['binding_multislot_two_plus_slots_acc']}"
    )
    print(
        f"DPLR     all_slots={dplr['binding_multislot_all_slots_acc']}  "
        f"two_plus={dplr['binding_multislot_two_plus_slots_acc']}"
    )
    print(f"VERDICT: {verdict}")
    print(f"wrote {args.out}  ({report['elapsed_s']}s)")


if __name__ == "__main__":
    main()
