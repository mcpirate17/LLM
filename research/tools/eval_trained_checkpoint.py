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
from research.eval.gmqar import score_model_gmqar
from research.eval.ar_curriculum_probe import ar_curriculum_probe, ARCurriculumConfig
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


def _fine_tune_budget(probe_timeout: int) -> float:
    """Per-probe wall-clock budget handed to each fine-tuning probe's internal cap.

    The fine-tuning probes (induction/AR/binding) each carry an internal timeout
    calibrated for nano models (90-300s at ~0.08 s/step). A 144M+ model trains
    ~2x slower, so those caps fire mid-training and write 0.0 — a false negative,
    not a capability result (the induction_validation 0.0 on the 144M run was
    exactly this). Hand every probe the same generous budget the outer SIGALRM
    already grants (minus a small margin so the probe's own graceful stop wins),
    so the internal cap only ever bounds a genuinely-hung probe. This only ever
    RAISES caps, so no probe that completes today can be cut shorter.
    """
    return float(probe_timeout) - 30.0 if probe_timeout and probe_timeout > 0 else 1e9


# gMQAR KV ids are drawn from a well-trained low-id range so the probe measures
# the binding MECHANISM, not embedding quality of rare tokens (matches the
# calibrated_ar_probe convention). Candidate scoring already restricts the argmax
# to in-context values.
_GMQAR_TOKEN_POOL = 2048


def _gmqar_recall(model: torch.nn.Module, device: str) -> dict[str, Any]:
    """PRIMARY associative-recall metric: zero-shot graded multi-query AR.

    No fine-tuning and no deepcopy, so it is free of the optimization artifacts
    that distort the fine-tuned ar_validation / ar_curriculum probes on annealed
    checkpoints (the reason ar_legacy was retired). Reports AUDC (area under the
    difficulty curve) and D50 (largest KV-pair count still recalled >= 50%)."""
    res = score_model_gmqar(
        model, vocab_size=VOCAB_SIZE, device=device, token_pool=_GMQAR_TOKEN_POOL
    )
    return {
        "gmqar_audc": res.audc,
        "gmqar_d50": res.d50,
        "gmqar_chance": res.chance,
        "gmqar_token_pool": _GMQAR_TOKEN_POOL,
        "scoring": "candidate",
        "cells": res.cells,
    }


def _dispatch_recall_probes(
    safe,  # callable(label: str, fn: Callable) -> None
    model: torch.nn.Module,
    dev: torch.device,
    device: str,
    seed: int,
    probe_timeout: int = 1800,
) -> None:
    """Fire induction and AR probes (first half of the standard battery).

    gMQAR runs FIRST and is the recall metric of record: zero-shot, cheap, and
    artifact-free. The fine-tuned ar_curriculum / ar_validation probes that follow
    are kept as secondary learnability diagnostics (and ar_validation remains a
    required write-gate field), but they are no longer the headline recall number.
    """
    budget = _fine_tune_budget(probe_timeout)
    safe("gmqar", lambda: _gmqar_recall(model, device))
    safe(
        "induction_intermediate",
        lambda: run_induction_intermediate(
            model,
            n_train_steps=300,
            n_eval=128,
            batch_size=8,
            device=dev,
            timeout_s=budget,
        ),
    )
    # ar_legacy probe retired 2026-06-18: measurement artifact (a softmax positive
    # control also floors it; full-vocab argmax + deepcopy FT + 300-step harness «
    # Zoology's ~8K). gMQAR (research/eval/gmqar.py) is the AR metric of record.
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
                timeout_s=budget,
            ),
            device=device,
        ),
    )
    safe(
        "induction_validation",
        lambda: run_induction_validation_champion(
            model,
            n_train_steps=2000,
            device=dev,
            timeout_s=budget,
        ),
    )
    safe(
        "ar_validation",
        lambda: run_ar_validation(
            model, cfg=ARValidationConfig(timeout_s=budget), device=device
        ),
    )


def _dispatch_binding_probes(
    safe,  # callable(label: str, fn: Callable) -> None
    model: torch.nn.Module,
    dev: torch.device,
    probe_timeout: int = 1800,
) -> None:
    """Fire binding probes (second half of the standard battery)."""
    budget = _fine_tune_budget(probe_timeout)
    safe(
        "binding_v2",
        lambda: run_binding_intermediate(
            model,
            n_train_steps=300,
            n_eval=128,
            train_batch_size=8,
            eval_batch_size=8,
            device=dev,
            timeout_s=budget,
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
            cfg=BindingMultislotConfig(
                train_steps=400, batch_size=8, n_eval=128, timeout_s=budget
            ),
            device=dev,
        ),
    )


def _dispatch_probes(
    safe,  # callable(label: str, fn: Callable) -> None
    model: torch.nn.Module,
    dev: torch.device,
    device: str,
    seed: int,
    probe_timeout: int = 1800,
) -> None:
    """Call ``safe`` for every probe in the standard battery (order is significant)."""
    _dispatch_recall_probes(safe, model, dev, device, seed, probe_timeout)
    _dispatch_binding_probes(safe, model, dev, probe_timeout)


def _preflight_check(model: torch.nn.Module, device: str) -> dict[str, Any]:
    """Cheap forward+backward sanity check run BEFORE the expensive battery.

    Validates the contract every probe relies on: ``model(ids)`` returns logits of
    shape ``(B, S, VOCAB_SIZE)``, all-finite, and a backward populates finite
    gradients (the fine-tuned probes deepcopy then train). Catches the silent
    failure class — wrong vocab dim, NaN forward, dead gradient — in ~1s on a real
    eval-length sequence instead of after minutes of wasted probes. It does NOT
    validate full-scale memory at every probe's batch size; it exercises one
    eval-length sequence so seq-dependent shape/OOM bugs surface early."""
    dev = torch.device(device)
    b, s = 4, 320
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, VOCAB_SIZE, (b, s), generator=g).to(dev)
    try:
        out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out
    except Exception as e:
        return {
            "ok": False,
            "stage": "forward",
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }
    shape = tuple(logits.shape)
    if (logits.dim(), shape[0], shape[1], shape[-1]) != (3, b, s, VOCAB_SIZE):
        return {
            "ok": False,
            "stage": "shape",
            "logits_shape": shape,
            "expected": (b, s, VOCAB_SIZE),
        }
    if not bool(torch.isfinite(logits).all()):
        return {"ok": False, "stage": "finite_forward", "logits_shape": shape}
    try:
        logits.float().pow(2).mean().backward()
    except Exception as e:
        model.zero_grad(set_to_none=True)
        return {
            "ok": False,
            "stage": "backward",
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }
    n_grad = sum(
        1
        for p in model.parameters()
        if p.grad is not None and bool(torch.isfinite(p.grad).all())
    )
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    model.zero_grad(set_to_none=True)
    if n_grad == 0:
        return {"ok": False, "stage": "no_gradient", "n_trainable": n_trainable}
    return {
        "ok": True,
        "logits_shape": shape,
        "n_params_with_grad": n_grad,
        "n_trainable": n_trainable,
    }


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
    out["_preflight"] = _preflight_check(model, device)
    if not out["_preflight"]["ok"]:
        print(
            f"  PREFLIGHT FAILED (stage={out['_preflight'].get('stage')}) — "
            "skipping the probe battery (model contract is broken)"
        )
        return out
    print(f"  preflight OK (logits {out['_preflight']['logits_shape']})")
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

    _dispatch_probes(_safe, model, dev, device, seed, probe_timeout)
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
    # Fail loud: a broken model contract makes the whole battery meaningless, so
    # exit non-zero even though the diagnostic artifact was written.
    if not probes.get("_preflight", {}).get("ok", False):
        print("EXIT non-zero: preflight failed — battery skipped")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
