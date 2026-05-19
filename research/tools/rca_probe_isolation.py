"""Isolate which hard probe is the true assert source under CUDA_LAUNCH_BLOCKING=1.

Loads the canonical 100K reference, runs the 4 broken probes ONE AT A TIME with
fresh CUDA contexts between them, so we can localize which kernel really OOBs.
"""

from __future__ import annotations

import argparse
import traceback

import torch

from research.synthesis.reference_checkpoints import resolve_reference_checkpoint
from research.tools.mixer_fingerprint import _build_model_and_batchers


def _load_ref_model(device: str = "cuda"):
    ckpt = resolve_reference_checkpoint("mixer_interleaved_conv6_three_lane6_50m_100k")
    model, _factory, _train_batcher, _val_batches, n_params = _build_model_and_batchers(
        mixer="interleaved",
        pattern="conv:6,three_lane:6",
        batch_size=16,
        seq_len=256,
        device=device,
        n_eval_batches=4,
        dim=320,
        n_blocks=12,
    )
    payload = torch.load(ckpt, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    print(
        f"loaded {n_params / 1e6:.1f}M-param ref @ step {payload.get('step')}",
        flush=True,
    )
    return model


def _try_probe(name: str, fn):
    print(f"\n=== {name} ===", flush=True)
    try:
        out = fn()
        print(f"  OK: {out!r}"[:500], flush=True)
    except Exception:
        print("  FAILED:", flush=True)
        traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe",
        choices=[
            "all",
            "pipeline",
            "binding_range",
            "binding_multislot",
            "induction_validation",
            "ar_validation",
        ],
        default="all",
    )
    args = parser.parse_args()

    model = _load_ref_model("cuda")

    if args.probe == "pipeline":
        from research.tools.mixer_fingerprint import _expensive_evals

        out = _expensive_evals(model=model, device=torch.device("cuda"), seed=0)
        for probe_name, result in out.items():
            if not isinstance(result, dict):
                print(f"  {probe_name}: {str(result)[:120]}", flush=True)
                continue
            status = (
                result.get("status")
                or result.get(f"{probe_name}_status")
                or result.get("binding_status")
                or "<no status field>"
            )
            print(f"  {probe_name}: {str(status)[:200]}", flush=True)
        return 0

    if args.probe in ("all", "binding_range"):
        from research.eval.binding_range import binding_range_profile

        _try_probe("binding_range", lambda: binding_range_profile(model, device="cuda"))

    if args.probe in ("all", "binding_multislot"):
        from research.eval.binding_multislot_probe import binding_multislot_probe

        _try_probe(
            "binding_multislot", lambda: binding_multislot_probe(model, device="cuda")
        )

    if args.probe in ("all", "induction_validation"):
        from research.eval.induction_validation_probe import (
            run_induction_validation_champion,
        )

        _try_probe(
            "induction_validation",
            lambda: run_induction_validation_champion(model, device="cuda"),
        )

    if args.probe in ("all", "ar_validation"):
        from research.eval.ar_validation import run_ar_validation, ARValidationConfig

        _try_probe(
            "ar_validation",
            lambda: run_ar_validation(model, cfg=ARValidationConfig(), device="cuda"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
