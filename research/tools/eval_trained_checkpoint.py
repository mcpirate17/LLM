"""Post-hoc evaluation of a mixer_fingerprint checkpoint.

Loads a saved TinyLM checkpoint and runs the full probe suite with
``copy_model=True`` — each probe trains on its own deepcopy, so the suite
runs without cross-probe contamination. Graph-derived lanes (CompiledLayer)
work here because the probes use ``safe_deepcopy_module`` (research/eval/
_probe_utils.py), which materializes inference tensors and detaches non-leaf
attribute caches before deepcopying.

Usage:
    python -m research.tools.eval_trained_checkpoint \\
        --mixer ensemble_top_ar_4way --no-use-ffn \\
        --dim 640 --n-blocks 1 \\
        --checkpoint research/reports/mixer_fingerprint/ensemble_top_ar_4way_dim640_n1_100k_2026-05-20_step100000.pt \\
        --output research/reports/mixer_fingerprint/ensemble_top_ar_4way_post_eval.json
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any

import torch


class _ProbeTimeout(Exception):
    """Raised when a single probe exceeds its per-probe wall-clock budget."""


def _alarm_handler(signum, frame):  # noqa: ANN001 — signal handler signature
    raise _ProbeTimeout()


from research.defaults import VOCAB_SIZE
from research.eval.ar_curriculum_probe import ar_curriculum_probe, ARCurriculumConfig
from research.eval.associative_recall import associative_recall_score
from research.eval.binding_curriculum import curriculum_binding_range_profile
from research.eval.binding_intermediate_probe import run_binding_intermediate
from research.eval.binding_multislot_probe import (
    binding_multislot_probe,
    BindingMultislotConfig,
)
from research.eval.binding_range import binding_range_profile
from research.eval.induction_intermediate_probe import run_induction_intermediate
from research.eval.induction_validation_probe import run_induction_validation_champion
from research.eval.ar_validation import run_ar_validation, ARValidationConfig
from research.tools.eval_checkpoints_blimp import _infer_ffn_kind
from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm


def _load_model(
    *,
    mixer: str,
    dim: int,
    n_blocks: int,
    use_ffn: bool,
    checkpoint_path: Path,
    device: str,
):
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    state_dict = (
        payload.get("model_state_dict") if isinstance(payload, dict) else payload
    )
    ffn_kind = _infer_ffn_kind(state_dict) if use_ffn else "gelu"
    factory = _build_lane_factory(mixer)
    model = _build_tinylm(
        factory,
        dim=dim,
        n_blocks=n_blocks,
        vocab_size=VOCAB_SIZE,
        use_ffn=use_ffn,
        ffn_kind=ffn_kind,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _dispatch_recall_probes(
    safe,  # callable(label: str, fn: Callable) -> None
    model: torch.nn.Module,
    dev: torch.device,
    device: str,
    seed: int,
) -> None:
    """Fire induction and AR probes (first half of the standard battery)."""
    safe(
        "induction_intermediate",
        lambda: run_induction_intermediate(
            model, n_train_steps=300, n_eval=128, batch_size=8, device=dev
        ),
    )
    safe(
        "ar_legacy",
        lambda: associative_recall_score(
            model, n_train_steps=300, n_eval=128, batch_size=8, device=dev
        ),
    )
    safe(
        "ar_curriculum",
        lambda: ar_curriculum_probe(
            model,
            cfg=ARCurriculumConfig(
                seed=seed,
                steps_per_stage=1000,
                batch_size=16,
                eval_batches=32,
                mode="cumulative",
            ),
            device=device,
        ),
    )
    safe(
        "induction_validation",
        lambda: run_induction_validation_champion(
            model, n_train_steps=2000, device=dev
        ),
    )
    safe(
        "ar_validation",
        lambda: run_ar_validation(model, cfg=ARValidationConfig(), device=device),
    )


def _dispatch_binding_probes(
    safe,  # callable(label: str, fn: Callable) -> None
    model: torch.nn.Module,
    dev: torch.device,
) -> None:
    """Fire binding probes (second half of the standard battery)."""
    safe(
        "binding_v2",
        lambda: run_binding_intermediate(
            model,
            n_train_steps=300,
            n_eval=128,
            train_batch_size=8,
            eval_batch_size=8,
            device=dev,
        ),
    )
    safe(
        "binding_range",
        lambda: binding_range_profile(
            model,
            distances=(8, 16, 32, 64, 128, 256),
            n_eval=128,
            seq_len=320,
            batch_size=8,
            device=dev,
        ),
    )
    safe(
        "binding_curriculum",
        lambda: curriculum_binding_range_profile(
            model,
            distances=(4, 8, 16, 32, 64),
            n_train_steps=300,
            n_eval=128,
            train_batch_size=8,
            eval_batch_size=8,
            device=dev,
        ),
    )
    safe(
        "binding_multislot",
        lambda: binding_multislot_probe(
            model,
            cfg=BindingMultislotConfig(train_steps=400, batch_size=8, n_eval=128),
            device=dev,
        ),
    )


def _dispatch_probes(
    safe,  # callable(label: str, fn: Callable) -> None
    model: torch.nn.Module,
    dev: torch.device,
    device: str,
    seed: int,
) -> None:
    """Call ``safe`` for every probe in the standard battery (order is significant)."""
    _dispatch_recall_probes(safe, model, dev, device, seed)
    _dispatch_binding_probes(safe, model, dev)


def _run_probes(
    model: torch.nn.Module, device: str, seed: int, probe_timeout: int = 1800
) -> dict[str, Any]:
    """Run the full probe suite; each probe deepcopies the model internally.

    ``probe_timeout`` (seconds, 0 = unlimited) is a generous PER-PROBE wall-clock
    budget enforced with SIGALRM: the slow AR / induction probes get plenty of
    time on a large model (the 50K run's 0.0 scores were an EXTERNAL kill, not a
    real failure), while a genuinely hung probe is cut so the battery always
    completes (and any downstream shutdown can fire).
    """
    out: dict[str, Any] = {}
    dev = torch.device(device)
    use_alarm = probe_timeout > 0 and hasattr(signal, "SIGALRM")

    def _safe(label: str, fn):
        t0 = time.monotonic()
        if use_alarm:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(int(probe_timeout))
        try:
            res = fn()
            if hasattr(res, "to_dict"):
                res = res.to_dict()
            out[label] = res
            out[f"_t_{label}"] = round(time.monotonic() - t0, 2)
            print(f"  {label}: OK ({out[f'_t_{label}']}s)")
        except _ProbeTimeout:
            out[label] = {"status": "timeout", "limit_s": probe_timeout}
            out[f"_t_{label}"] = round(time.monotonic() - t0, 2)
            print(f"  {label}: TIMEOUT after {probe_timeout}s — moving on")
        except Exception as e:
            out[label] = {"status": "error", "error": str(e)[:240]}
            out[f"_t_{label}"] = round(time.monotonic() - t0, 2)
            print(f"  {label}: FAILED — {type(e).__name__}: {str(e)[:120]}")
        finally:
            if use_alarm:
                signal.alarm(0)

    _dispatch_probes(_safe, model, dev, device, seed)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mixer", required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--n-blocks", type=int, required=True)
    p.add_argument("--use-ffn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--probe-timeout",
        type=int,
        default=1800,
        help="per-probe wall-clock budget in seconds (0 = unlimited)",
    )
    args = p.parse_args()

    print(
        f"Loading model: mixer={args.mixer} dim={args.dim} n_blocks={args.n_blocks} use_ffn={args.use_ffn}"
    )
    model = _load_model(
        mixer=args.mixer,
        dim=args.dim,
        n_blocks=args.n_blocks,
        use_ffn=args.use_ffn,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded {n_params:,} params from {args.checkpoint}")

    print(
        "Running probes (each probe deepcopies the model via safe_deepcopy_module)..."
    )
    probes = _run_probes(
        model=model,
        device=args.device,
        seed=args.seed,
        probe_timeout=args.probe_timeout,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(
            {
                "mixer": args.mixer,
                "dim": args.dim,
                "n_blocks": args.n_blocks,
                "use_ffn": args.use_ffn,
                "checkpoint": str(args.checkpoint),
                "n_params": n_params,
                "seed": args.seed,
                "probes": probes,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
