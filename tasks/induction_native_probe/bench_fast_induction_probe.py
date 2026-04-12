from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from research.eval.reference_training import BaselineTransformer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fast_induction_probe import (
        NativeProbeConfig,
        induction_score_fast,
        load_native_induction_probe,
    )
else:
    from .fast_induction_probe import (
        NativeProbeConfig,
        induction_score_fast,
        load_native_induction_probe,
    )

from research.eval.induction_probe import induction_score


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pool-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_native_induction_probe()
    model = BaselineTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.layers,
    ).to(device)

    variants = [
        (
            "baseline_python",
            lambda seed: induction_score(
                model,
                n_train_steps=args.steps,
                n_eval=args.eval,
                batch_size=args.batch_size,
                device=device,
                seed=seed,
            ),
        ),
        (
            "native_generator",
            lambda seed: induction_score_fast(
                model,
                config=NativeProbeConfig(
                    n_train_steps=args.steps,
                    n_eval=args.eval,
                    batch_size=args.batch_size,
                    device=device,
                    seed=seed,
                    pool_size=0,
                    use_native_generator=True,
                ),
            ),
        ),
        (
            f"native_pool_{args.pool_size}",
            lambda seed: induction_score_fast(
                model,
                config=NativeProbeConfig(
                    n_train_steps=args.steps,
                    n_eval=args.eval,
                    batch_size=args.batch_size,
                    device=device,
                    seed=seed,
                    pool_size=args.pool_size,
                    use_native_generator=True,
                ),
            ),
        ),
    ]

    results = []
    for label, fn in variants:
        wall_samples = []
        auc_samples = []
        last = None
        for rep in range(args.repeats):
            seed = 123 + rep
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            last = fn(seed)
            if device == "cuda":
                torch.cuda.synchronize()
            wall_samples.append((time.perf_counter() - t0) * 1000)
            auc_samples.append(last.auc)
        results.append(
            {
                "label": label,
                "median_wall_ms": round(statistics.median(wall_samples), 1),
                "all_wall_ms": [round(x, 1) for x in wall_samples],
                "median_auc": round(statistics.median(auc_samples), 4),
                "all_auc": auc_samples,
                "last_gap_accuracies": last.gap_accuracies if last is not None else {},
                "last_status": last.status if last is not None else "unknown",
            }
        )

    print(
        json.dumps(
            {
                "device": device,
                "config": {
                    "steps": args.steps,
                    "eval": args.eval,
                    "batch_size": args.batch_size,
                    "d_model": args.d_model,
                    "layers": args.layers,
                    "vocab_size": args.vocab_size,
                    "repeats": args.repeats,
                    "pool_size": args.pool_size,
                },
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
