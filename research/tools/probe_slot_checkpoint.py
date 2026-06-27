"""Intermediate read on a live slot run: load ONE checkpoint and run just the
compositional-binding probe (binding_multislot all_slots = the Run-1 verdict),
optionally binding_range. Fast + checkpoint-only, so it can be run mid-training
without waiting for the 100K post-eval. Default device cpu = does not contend
with the GPU training job.

    python -m research.tools.probe_slot_checkpoint \
        research/reports/native_adaptive_hydra_ckpts/slot_dplr_run1_40m_slot_table_mh_dplr_step021000.pt
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from research.eval.binding_multislot_probe import (
    BindingMultislotConfig,
    binding_multislot_probe,
)
from research.tools.eval_trained_checkpoint import _load_model

_STEP = re.compile(r"step(\d+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--mixer", default="slot_table_mh_dplr")
    ap.add_argument("--dim", type=int, default=640)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument(
        "--device", default="cpu", help="cpu = non-invasive vs GPU training"
    )
    ap.add_argument(
        "--steps", type=int, default=600, help="probe train steps (read budget)"
    )
    ap.add_argument("--bindings", type=int, default=5)
    ap.add_argument("--query-slots", type=int, default=3)
    ap.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help="probe internal wall budget; raise it for slow CPU full-budget reads",
    )
    args = ap.parse_args()

    step = int(m.group(1)) if (m := _STEP.search(args.checkpoint.name)) else -1
    t0 = time.perf_counter()
    model = _load_model(
        mixer=args.mixer,
        dim=args.dim,
        n_blocks=args.n_blocks,
        use_ffn=True,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    cfg = BindingMultislotConfig(
        train_steps=args.steps,
        bindings_per_example=args.bindings,
        query_slots=args.query_slots,
        timeout_s=args.timeout,
    )
    r = binding_multislot_probe(model, cfg=cfg, device=args.device).to_dict()
    print(
        f"step={step}  all_slots={r['binding_multislot_all_slots_acc']:.4f}  "
        f"two_plus={r['binding_multislot_two_plus_slots_acc']:.4f}  "
        f"held_slot={r['binding_multislot_held_entity_slot_acc']:.4f}  "
        f"held_class={r['binding_multislot_held_entity_class_acc']:.4f}  "
        f"({r.get('binding_multislot_status', '?')}, {time.perf_counter() - t0:.0f}s)"
    )
    print(
        "  reference: native 100K all_slots=0.0 (the wall) | nano slot_dplr ~0.32 | "
        "nano baseline ~0.25 (so >~0.1 here = clearing the wall, building)"
    )


if __name__ == "__main__":
    main()
