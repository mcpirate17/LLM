"""Causality (anti-causal leak) sweep over every sequence-mixing op in the NAS.

Background
----------
2026-05-23: ``adjacent_token_merge`` was found anti-causal — it merged token p
INTO p-1, so ``output[p-1]`` depended on ``x[p]`` (a one-step next-token leak).
Because binding_range / screening / curriculum probes are *causal* next-token
tasks scored at position i against ``input[i+1]``, the leak handed them the
label and inflated the binding scores of every model containing that op.

This tool sweeps every op that MIXES or SHIFTS across the sequence dimension and
applies the exact causality invariant that pinned the fix:

    For an op on a hidden-state sequence x of shape [B, S, D]:
      output[:, i, :] must depend ONLY on input positions <= i.

Test (the ``_max_earlier_delta`` sweep from
``research/tests/test_adjacent_token_merge_causality.py``):
    base = op(x). For each t in [1, S): perturb a COPY at position t only
    (xp[:, t, :] = large value), recompute, and measure
    ``max|out[:, :t] - base[:, :t]|``. If that exceeds the tolerance for ANY t
    the op LEAKS future information backward (anti-causal).

How ops are instantiated
------------------------
We build each op exactly as the COMPILER builds it in production:

  * Primary path: ``CompiledOp(op_name, config, in_shape, out_shape, model_dim)``
    — the production module that self-initialises its parameters from the
    primitive definition (compiled_op_params.py) and runs the production
    dispatch fn (compiler_registry.OP_DISPATCH). We feed it a float [B,S,D]
    hidden-state tensor directly (NOT token ids) so no token embedding can mask
    a leak.

  * Fallback path: for ops that live in OP_DISPATCH but have no entry in
    PRIMITIVE_REGISTRY (e.g. ``roll_seq`` / ``roll_neg``), we call the dispatch
    fn directly with a minimal telemetry-sink module.

Run::

    source /home/tim/venvs/llm/bin/activate
    python -m research.tools.causality_op_sweep            # full sweep
    python -m research.tools.causality_op_sweep --validate  # harness self-test only

The harness self-validates BEFORE trusting any result:
  * ``adjacent_token_merge`` MUST report CAUSAL (it was just fixed).
  * ``relu`` MUST report CAUSAL (per-position true-negative control).
  * ``roll_neg`` MUST report LEAK (future-shift true-positive control).
If any control disagrees, the harness aborts — a false +/- here is worse than
no answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from research.mathspaces.registry import register_all_mathspaces
from research.synthesis.compiled_op import CompiledOp
from research.synthesis.compiler_registry import OP_DISPATCH
from research.synthesis.graph import ShapeInfo
from research.synthesis.primitives import PRIMITIVE_REGISTRY

# Tolerance: the original regression test used 1e-6; the task spec uses 1e-5.
# We report the raw delta and classify against 1e-5 (the looser, task-mandated
# bar) so float32 round-off in long associative scans isn't mislabelled a leak.
LEAK_TOL = 1e-5


# ── Op selection ──────────────────────────────────────────────────────────
#
# Only ops that CAN move information across the sequence dim are worth testing.
# Pure per-position (elementwise / channel-only) ops cannot leak; a couple are
# included below as explicit true-negative controls.

_SEQ_MIXING_CATEGORIES = {"mixing", "sequence", "frequency"}

# PARAMETERIZED / FUNCTIONAL / MATH_SPACE ops that mix or shift across S.
# (Their category is generic, so we name them explicitly.)
_EXTRA_SEQ_OPS = frozenset(
    {
        # conv / ssm / recurrent
        "conv1d_seq",
        "conv_only",
        "long_conv_hyena",
        "long_conv_hyen",
        "selective_scan",
        "state_space",
        "rwkv_time_mixing",
        "rwkv_channel",
        "mlstm_cell",
        "gated_delta",
        "gated_linear_attention",
        # spectral / wavelet / integral
        "spectral_filter",
        "chebyshev_spectral_mix",
        "integral_kernel",
        # math-space attention family
        "tropical_attention",
        "clifford_attention",
        "ultrametric_attention",
        "stdp_attention",
        "mla_attention",
        # scans / reductions that could broadcast across S
        "cumsum",
        "cumprod_safe",
        "sum_last",
        "mean_last",
        "max_last",
        "norm_last",
        # token-routing family
        "adjacent_token_merge",
        "token_merge",
        "depth_token_mask",
        "mod_topk",
        "confidence_token_gate",
        "learned_token_gate",
        "hybrid_token_gate",
        "route_topk",
        "speculative",
        "gather_topk",
        "sparse_span_builder",
        # sequence masks
        "causal_mask",
        "sliding_window_mask",
        "softmax_last",
    }
)

# Dispatch-only ops with no PRIMITIVE_REGISTRY entry (cannot build a CompiledOp).
# roll_seq (+1, causal) / roll_neg (-1, anti-causal) are the shift controls.
_DISPATCH_ONLY_OPS = frozenset({"roll_seq", "roll_neg"})

# True-negative pointwise controls (must report CAUSAL).
_POINTWISE_CONTROLS = frozenset({"relu", "gelu", "add", "mul"})

# Ops declared n_inputs=2 that are still seq-mixers and accept the SAME hidden
# state for both inputs in production (mla_attention falls back to inputs[0];
# gather_topk takes a score tensor we can derive from x itself). For these we
# feed [x, x] instead of skipping — otherwise their causality goes untested.
_DUPLICATE_INPUT_OPS = frozenset({"mla_attention", "gather_topk"})

# Config overrides per op so the sweep exercises a representative production
# setting. Anything not listed uses the primitive's defaults.
_OP_CONFIG: dict[str, dict] = {
    "adjacent_token_merge": {"n_keep": 8},
    "token_merge": {"n_keep": 8},
    "local_window_attn": {"window_size": 4},
    "sliding_window_mask": {"window_size": 4},
    "depth_token_mask": {"threshold": 0.5},
    "mod_topk": {"threshold": 0.5},
    "confidence_token_gate": {"threshold": 0.5},
    "learned_token_gate": {"threshold": 0.5},
    "hybrid_token_gate": {"threshold": 0.5},
    "route_topk": {"k": 4},
    "gather_topk": {"k": 4},
    "sparse_span_builder": {"span_width": 4, "fallback_behavior": "identity"},
    "speculative": {"threshold": 0.5},
}


class _TelemetrySink:
    """Minimal stand-in module for dispatch-only ops (no params needed)."""

    training = False


@dataclass
class _Verdict:
    op: str
    category: str
    binding_range: str
    verdict: str  # CAUSAL | LEAK | ERROR | SKIP
    max_delta: float = 0.0
    leak_positions: list[int] = field(default_factory=list)
    note: str = ""


def _max_earlier_delta(
    fn: Callable[[torch.Tensor], torch.Tensor],
    S: int,
    D: int,
    seed: int = 0,
) -> tuple[float, list[int]]:
    """Max change at output positions STRICTLY BEFORE a perturbed input pos.

    Mirrors research/tests/test_adjacent_token_merge_causality.py::
    _max_earlier_delta. Returns (worst_delta, leaking_perturbation_positions).

    A position t is a "leak position" if perturbing input[t] moved any output
    position < t by more than LEAK_TOL. For a causal map this set is empty.
    """
    torch.manual_seed(seed)
    x = torch.randn(1, S, D)
    base = fn(x)
    if not isinstance(base, torch.Tensor):
        raise TypeError(f"op returned {type(base)}, not a tensor")
    # Compare on the overlapping seq prefix; some ops change D but must keep S.
    if base.dim() != 3 or base.shape[1] != S:
        raise ValueError(
            f"op changed the sequence dim (out shape {tuple(base.shape)} vs S={S}); "
            "position-aligned causality test is not applicable"
        )
    worst = 0.0
    leak_positions: list[int] = []
    for t in range(1, S):
        xp = x.clone()
        xp[:, t, :] = 99.0  # large, unambiguous perturbation
        out = fn(xp)
        delta = (out[:, :t] - base[:, :t]).abs().max().item()
        if delta > worst:
            worst = delta
        if delta > LEAK_TOL:
            leak_positions.append(t)
    return worst, leak_positions


def _build_runner(op_name: str) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    """Return (fn, path_note) that runs the op on a float [1,S,D] tensor.

    Primary path uses the production CompiledOp; dispatch-only ops fall back to
    calling the raw OP_DISPATCH fn with a telemetry-sink module.
    """
    cfg = dict(_OP_CONFIG.get(op_name, {}))
    dup = op_name in _DUPLICATE_INPUT_OPS

    if op_name in PRIMITIVE_REGISTRY:
        # Build the production module once; reuse it across perturbations so the
        # randomly-initialised weights stay fixed (a re-init per call would make
        # base != perturbed for reasons unrelated to causality).
        def _make() -> Callable[[torch.Tensor], torch.Tensor]:
            def fn(x: torch.Tensor) -> torch.Tensor:
                D = x.shape[-1]
                in_shape = ShapeInfo(seq="S", dim=D)
                out_shape = ShapeInfo(seq="S", dim=D)
                # Module is cached on first call against the tensor's D.
                mod = fn._mod  # type: ignore[attr-defined]
                if mod is None:
                    mod = CompiledOp(op_name, cfg, in_shape, out_shape, model_dim=D)
                    mod.eval()
                    fn._mod = mod  # type: ignore[attr-defined]
                with torch.no_grad():
                    return mod(x, x) if dup else mod(x)

            fn._mod = None  # type: ignore[attr-defined]
            return fn

        return _make(), "CompiledOp"

    if op_name in OP_DISPATCH:
        dispatch_fn = OP_DISPATCH[op_name]

        def fn(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                ins = [x, x] if dup else [x]
                return dispatch_fn(_TelemetrySink(), ins, cfg)

        return fn, "dispatch"

    raise KeyError(f"op {op_name!r} not in PRIMITIVE_REGISTRY or OP_DISPATCH")


def _classify(op_name: str, S: int = 16, D: int = 16) -> _Verdict:
    """Run the sweep for one op and return a verdict."""
    op = PRIMITIVE_REGISTRY.get(op_name)
    category = op.category.value if op is not None else "(dispatch-only)"
    binding = op.binding_range_class if op is not None else "?"

    # Multi-input ops can't generally be driven with a single hidden-state
    # tensor — skip and flag. Exception: ops that take the same hidden state for
    # both inputs in production (see _DUPLICATE_INPUT_OPS) are driven with [x,x].
    if op is not None and op.n_inputs > 1 and op_name not in _DUPLICATE_INPUT_OPS:
        return _Verdict(
            op_name,
            category,
            binding,
            "SKIP",
            note=f"n_inputs={op.n_inputs}; single-tensor causality probe N/A",
        )

    try:
        fn, path = _build_runner(op_name)
    except Exception as exc:  # noqa: BLE001 — report, don't crash the sweep
        return _Verdict(op_name, category, binding, "ERROR", note=f"build: {exc}")

    try:
        worst, leaks = _max_earlier_delta(fn, S, D)
    except Exception as exc:  # noqa: BLE001
        return _Verdict(op_name, category, binding, "ERROR", note=f"{path}: {exc}")

    verdict = "LEAK" if worst > LEAK_TOL else "CAUSAL"
    return _Verdict(
        op_name,
        category,
        binding,
        verdict,
        max_delta=worst,
        leak_positions=leaks,
        note=path,
    )


def _collect_target_ops() -> list[str]:
    """Ops to sweep: all seq-mixing categories + explicit extras + controls."""
    register_all_mathspaces()
    targets: set[str] = set()
    for name, op in PRIMITIVE_REGISTRY.items():
        if op.category.value in _SEQ_MIXING_CATEGORIES:
            targets.add(name)
    targets |= {n for n in _EXTRA_SEQ_OPS if n in PRIMITIVE_REGISTRY}
    targets |= {n for n in _DISPATCH_ONLY_OPS if n in OP_DISPATCH}
    targets |= {n for n in _POINTWISE_CONTROLS if n in PRIMITIVE_REGISTRY}
    return sorted(targets)


def _validate_harness() -> bool:
    """True-negative and true-positive controls. Aborts the sweep on failure."""
    print("=== HARNESS SELF-VALIDATION ===")
    checks = [
        ("adjacent_token_merge", "CAUSAL"),  # just fixed — must be causal
        ("relu", "CAUSAL"),  # pointwise true-negative
        ("roll_neg", "LEAK"),  # future-shift (shifts=-1) true-positive
        # roll_seq is torch.roll(shifts=+1) which is CIRCULAR: output[0] wraps
        # to input[S-1], a genuine future leak at position 0. So the expected
        # verdict is LEAK (not the naive "past shift = causal" intuition). This
        # control proves the sweep catches wraparound leaks too.
        ("roll_seq", "LEAK"),
    ]
    ok = True
    for op_name, expected in checks:
        v = _classify(op_name)
        status = "PASS" if v.verdict == expected else "FAIL"
        if v.verdict != expected:
            ok = False
        print(
            f"  [{status}] {op_name:22s} expect={expected:6s} got={v.verdict:6s} "
            f"max_delta={v.max_delta:.3e} note={v.note}"
        )
    print(f"=== VALIDATION {'OK' if ok else 'FAILED'} ===\n")
    return ok


def _print_table(results: list[_Verdict]) -> None:
    # Sort: leaks first (by severity desc), then errors/skips, then causal.
    order = {"LEAK": 0, "ERROR": 1, "SKIP": 2, "CAUSAL": 3}
    results = sorted(results, key=lambda v: (order[v.verdict], -v.max_delta, v.op))
    hdr = f"{'op_name':32s} {'category':14s} {'binding':8s} {'verdict':7s} {'max_delta':>11s}  leaking_positions / note"
    print(hdr)
    print("-" * len(hdr))
    for v in results:
        if v.verdict == "LEAK":
            pos = str(v.leak_positions[:8]) + (
                "..." if len(v.leak_positions) > 8 else ""
            )
            extra = pos
        elif v.verdict in ("ERROR", "SKIP"):
            extra = v.note
        else:
            extra = ""
        print(
            f"{v.op:32s} {v.category:14s} {v.binding_range:8s} "
            f"{v.verdict:7s} {v.max_delta:11.3e}  {extra}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--validate",
        action="store_true",
        help="run only the harness self-validation controls and exit",
    )
    ap.add_argument(
        "--json",
        type=str,
        default=None,
        help="optional path to dump full results as JSON",
    )
    ap.add_argument("--seq", type=int, default=16, help="sequence length S")
    ap.add_argument("--dim", type=int, default=16, help="model dim D")
    args = ap.parse_args(argv)

    if not _validate_harness():
        print(
            "HARNESS VALIDATION FAILED — refusing to report results.", file=sys.stderr
        )
        return 2

    if args.validate:
        return 0

    targets = _collect_target_ops()
    print(f"Sweeping {len(targets)} ops (S={args.seq}, D={args.dim})...\n")
    results = [_classify(op, S=args.seq, D=args.dim) for op in targets]
    _print_table(results)

    leaks = [v for v in results if v.verdict == "LEAK"]
    errs = [v for v in results if v.verdict == "ERROR"]
    skips = [v for v in results if v.verdict == "SKIP"]
    print(
        f"\nSummary: {len(leaks)} LEAK, {len(errs)} ERROR, {len(skips)} SKIP, "
        f"{len(results) - len(leaks) - len(errs) - len(skips)} CAUSAL "
        f"(of {len(results)} tested)."
    )
    if leaks:
        print("ANTI-CAUSAL ops:", ", ".join(v.op for v in leaks))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                [
                    {
                        "op": v.op,
                        "category": v.category,
                        "binding_range": v.binding_range,
                        "verdict": v.verdict,
                        "max_delta": v.max_delta,
                        "leak_positions": v.leak_positions,
                        "note": v.note,
                    }
                    for v in results
                ],
                fh,
                indent=2,
            )
        print(f"\nWrote JSON results to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
