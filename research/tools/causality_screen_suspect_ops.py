"""Deliverable 0 — causality screen for compiled-op primitives.

For each suspect op, build the minimal `_Mod` it needs, then run the
"perturb input position t, check output positions < t" sweep used in
`research/tests/test_adjacent_token_merge_causality.py`.

Output:
  research/reports/arch_component_analysis_2026-05-23/causality_screen.json

Usage:
  python -m research.tools.causality_screen_suspect_ops
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from research.synthesis.compiler import OP_DISPATCH


REPO = Path(__file__).resolve().parents[2]
OUT_PATH = (
    REPO
    / "research"
    / "reports"
    / "arch_component_analysis_2026-05-23"
    / "causality_screen.json"
)


def _max_earlier_delta(fn: Callable, S: int, D: int, seed: int = 0) -> float:
    torch.manual_seed(seed)
    x = torch.randn(1, S, D)
    base = fn(x)
    if not isinstance(base, torch.Tensor):
        raise RuntimeError(f"non-tensor output {type(base)}")
    worst = 0.0
    # Sweep odd + boundary indices to catch off-by-ones.
    for t in (1, 2, 3, S // 2, S // 2 + 1, S - 2, S - 1):
        if t <= 0 or t >= S:
            continue
        xp = x.clone()
        xp[:, t, :] = 99.0
        out = fn(xp)
        diff = (out[:, :t] - base[:, :t]).abs().max().item()
        if diff > worst:
            worst = diff
    return worst


def _attn_mod(D: int, n_heads: int = 4) -> nn.Module:
    m = nn.Module()
    hd = D // n_heads
    m.n_heads = n_heads
    m.head_dim = hd
    m.attn_scale = 1.0 / math.sqrt(hd)
    m.q_proj = nn.Linear(D, D, bias=False)
    m.k_proj = nn.Linear(D, D, bias=False)
    m.v_proj = nn.Linear(D, D, bias=False)
    m.o_proj = nn.Linear(D, D, bias=False)
    return m


def _linear_mod(D: int) -> nn.Module:
    m = nn.Module()
    m.weight = nn.Parameter(torch.randn(D, D) * 0.02)
    m.sparse_kernel_ready = False  # forces dense fallback (CPU-safe)
    return m


def _latent_mod(D: int) -> nn.Module:
    m = nn.Module()
    r = max(8, D // 4)
    # _safe_linear(x, W) expects W shape (out, in).
    m.kv_compress = nn.Parameter(torch.randn(r, D) * 0.02)
    m.kv_up = nn.Parameter(torch.randn(2 * D, r) * 0.02)
    return m


def _conv1d_mod(D: int, k: int = 4) -> nn.Module:
    m = nn.Module()
    m.conv_weight = nn.Parameter(torch.randn(D, 1, k) * 0.02)
    m.conv_bias = nn.Parameter(torch.zeros(D))
    m.kernel_size = k
    m.groups = D
    return m


def _selective_scan_mod(D: int) -> nn.Module:
    m = nn.Module()
    m.A_log = nn.Parameter(torch.zeros(D))
    m.D_param = nn.Parameter(torch.ones(D))
    # `dt_proj` is sliced `[:D]` as a tensor in the op body.
    m.dt_proj = nn.Parameter(torch.zeros(D))
    m.B_proj = nn.Linear(D, D, bias=False)
    m.C_proj = nn.Linear(D, D, bias=False)
    return m


def _spectral_mod(D: int) -> nn.Module:
    m = nn.Module()
    m.freq_mask = nn.Parameter(torch.ones(D // 2 + 1))
    return m


def _rope_mod(_: int) -> nn.Module:
    return nn.Module()


def _swiglu_mod(D: int) -> nn.Module:
    m = nn.Module()
    h = 2 * D
    # Op accesses `.weight` and `.bias` on each projection.
    m.gate_proj = nn.Linear(D, h, bias=False)
    m.up_proj = nn.Linear(D, h, bias=False)
    m.down_proj = nn.Linear(h, D, bias=False)
    return m


def _trivial_mod(_: int) -> nn.Module:
    return nn.Module()


@dataclass
class OpSpec:
    name: str
    make_mod: Callable[[int], nn.Module]
    config: dict
    D: int = 16
    S: int = 16
    # Some ops (matmul) require ≥2 inputs. `n_inputs=2` runs the screen with
    # `inputs=[x, x]` so the perturbation propagates through both arms — still
    # exposes any future-leak.
    n_inputs: int = 1


SPECS: list[OpSpec] = [
    OpSpec("rope_rotate", _rope_mod, {}, D=32),
    OpSpec("semi_structured_2_4_linear", _linear_mod, {}),
    OpSpec("softmax_attention", _attn_mod, {}, D=32),
    OpSpec("linear_attention", _attn_mod, {}, D=32),
    OpSpec("local_window_attn", _trivial_mod, {"window_size": 4}, D=32),
    OpSpec("sliding_window_mask", _trivial_mod, {"window_size": 4}, D=32),
    OpSpec("spectral_filter", _spectral_mod, {}, D=32),
    OpSpec("latent_attention_compressor", _latent_mod, {}, D=32),
    OpSpec("token_entropy", _trivial_mod, {}, D=16),
    OpSpec("conv1d_seq", _conv1d_mod, {}, D=16),
    OpSpec("selective_scan", _selective_scan_mod, {}, D=16),
    OpSpec("swiglu_mlp", _swiglu_mod, {}, D=16),
    OpSpec("relu", _trivial_mod, {}, D=16),
    OpSpec("gelu", _trivial_mod, {}, D=16),
    OpSpec("nm_sparse_linear", _linear_mod, {}, D=16),
    OpSpec("matmul", _trivial_mod, {}, D=16, n_inputs=2),
    OpSpec(
        "adjacent_token_merge", _trivial_mod, {"n_keep": 8}, D=16
    ),  # post-fix control
]


def screen(spec: OpSpec) -> dict:
    fn = OP_DISPATCH.get(spec.name)
    if fn is None:
        return {"op": spec.name, "status": "missing"}
    mod = spec.make_mod(spec.D)
    mod.eval()

    def wrapped(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return fn(mod, [x], dict(spec.config))

    try:
        delta = _max_earlier_delta(wrapped, spec.S, spec.D)
    except Exception as exc:
        return {"op": spec.name, "status": "error", "error": repr(exc)[:200]}
    return {
        "op": spec.name,
        "status": "ok",
        "max_earlier_delta": delta,
        "anti_causal": delta > 1e-5,
        "S": spec.S,
        "D": spec.D,
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = [screen(s) for s in SPECS]
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO)}")
    for r in results:
        if r.get("status") == "ok":
            tag = "LEAK" if r["anti_causal"] else "ok  "
            print(
                f"  {tag}  {r['op']:32s}  max_earlier_delta={r['max_earlier_delta']:.3e}"
            )
        else:
            print(f"  ????  {r['op']:32s}  {r.get('status')}: {r.get('error', '')}")


if __name__ == "__main__":
    main()
