"""Cross-family fairness audit (#7): are the matrix rankings confounded?

The cross-axis matrix (cross_axis_architecture_matrix_2026-06-07.md) compares
mixers that are NOT matched on the things that drive capability: head count
(novel attn ops ran 1-head via `d_in//64` vs gpt2 4-head `d_in//16`), parameter
count, and compute (attention is O(L^2 d), memory/SSM is O(L d m)). Before any
ranking is defensible, this measures those confounds per lane at the cohort scale
(dim64) and reports accuracy-per-param and accuracy-per-MFLOP next to raw accuracy.

FLOPs are matmul-class only (torch FlopCounterMode counts mm/bmm/sdpa/conv, not
elementwise logsumexp/exp/sigmoid) — the dominant term, labelled honestly. Training
budget (3000 steps, Adam lr 3e-3) is identical across the cohort, so it's a constant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from component_fab.harness.tiny_lm import lane_factory_for_baseline
from research.tools.grade_named_lanes_tier2 import _attention_mixer_factory
from research.tools.grade_ssm_fair_cohort import MODELS as _SSM_MODELS

_REPORT = Path(__file__).resolve().parents[2] / "research" / "reports"

# Recorded matrix accuracy (cross_axis note): bind = mean eval-acc on 6 recall
# tasks @3000; s_trk = state-tracking loss-reduction. None where not measured.
_MATRIX: dict[str, dict[str, float | None]] = {
    "gpt2": {"bind": 0.870, "s_trk": 3.02},
    "softmax_attention": {"bind": 0.683, "s_trk": 2.70},
    "mamba2": {"bind": 0.585, "s_trk": 2.60},
    "mamba": {"bind": 0.154, "s_trk": 3.30},
    "semiring": {"bind": 0.605, "s_trk": 3.04},
    "reciprocal": {"bind": 0.615, "s_trk": 3.04},
    "semiring_reciprocal": {"bind": 0.603, "s_trk": 3.04},
    "hier_compress": {"bind": 0.477, "s_trk": 3.03},
    "fast_weight": {"bind": 0.380, "s_trk": 2.58},
    "power_semiring": {"bind": 0.363, "s_trk": 2.64},
    "legendre_ssm": {"bind": 0.324, "s_trk": 2.65},
    "ddecay_memory": {"bind": 0.302, "s_trk": 2.59},
    "sparse_mor_conv": {"bind": 0.008, "s_trk": 3.14},
}


def _registry() -> dict[str, tuple[str, Callable[[int], nn.Module]]]:
    nonqkv = {
        k: ("non-qkv", v[1]) for k, v in _SSM_MODELS.items() if v[0] == "candidate"
    }
    return {
        "gpt2": ("frontier-attn", lane_factory_for_baseline("gpt2")),
        "softmax_attention": (
            "frontier-attn",
            lane_factory_for_baseline("softmax_attention"),
        ),
        "mamba": ("frontier-ssm", lane_factory_for_baseline("mamba")),
        "mamba2": ("frontier-ssm", lane_factory_for_baseline("mamba2")),
        "semiring": (
            "novel-attn",
            _attention_mixer_factory("learnable_semiring_attention"),
        ),
        "reciprocal": (
            "novel-attn",
            _attention_mixer_factory("reciprocal_rank_attention"),
        ),
        "semiring_reciprocal": (
            "novel-attn",
            _attention_mixer_factory("reciprocal_semiring_attention"),
        ),
        **nonqkv,
    }


def _find_heads(module: nn.Module) -> int | None:
    for _, mod in module.named_modules():
        for attr in ("n_heads", "num_heads"):
            if hasattr(mod, attr):
                return int(getattr(mod, attr))
    return None


def _pos_enc_in_lane(module: nn.Module) -> str:
    names = " ".join(n.lower() for n, _ in module.named_modules())
    if "rope" in names or "rotary" in names:
        return "rope"
    return "none"  # TinyLM adds a uniform abs-pos-embed outside the lane


def _matmul_flops(module: nn.Module, seq_len: int, dim: int) -> int:
    x = torch.randn(1, seq_len, dim)
    module.eval()
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            module(x)
        return int(fc.get_total_flops())
    except Exception:  # noqa: BLE001 - some lanes use unsupported ops; report -1
        return -1


def main() -> int:
    dim = 64
    rows: list[dict[str, Any]] = []
    for name, (cls, factory) in _registry().items():
        m = factory(dim)
        params = sum(p.numel() for p in m.parameters())
        heads = _find_heads(m)
        f64 = _matmul_flops(m, 64, dim)
        f256 = _matmul_flops(factory(dim), 256, dim)
        acc = _MATRIX.get(name, {})
        bind = acc.get("bind")
        rows.append(
            {
                "model": name,
                "class": cls,
                "params": params,
                "heads": heads,
                "pos_enc_lane": _pos_enc_in_lane(m),
                "mflops_L64": round(f64 / 1e6, 2) if f64 >= 0 else None,
                "mflops_L256": round(f256 / 1e6, 2) if f256 >= 0 else None,
                "bind": bind,
                "s_trk": acc.get("s_trk"),
                "bind_per_kparam": round(bind / (params / 1000), 4) if bind else None,
                "bind_per_mflop_L256": (
                    round(bind / (f256 / 1e6), 4) if bind and f256 > 0 else None
                ),
            }
        )

    rows.sort(key=lambda r: (r["bind"] is None, -(r["bind"] or 0)))
    out = _REPORT / "fairness_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))

    print(
        f"\n{'model':20s}{'class':14s}{'params':>8s}{'head':>5s}{'pos':>5s}"
        f"{'MFLOP@256':>10s}{'bind':>6s}{'b/kP':>7s}{'b/MFL':>7s}"
    )
    print("-" * 82)
    for r in rows:
        b = f"{r['bind']:.3f}" if r["bind"] is not None else "  -  "
        bkp = f"{r['bind_per_kparam']:.3f}" if r["bind_per_kparam"] else "  -  "
        bmf = f"{r['bind_per_mflop_L256']:.3f}" if r["bind_per_mflop_L256"] else "  -  "
        mfl = f"{r['mflops_L256']:.1f}" if r["mflops_L256"] is not None else "n/a"
        h = str(r["heads"]) if r["heads"] is not None else "-"
        print(
            f"{r['model']:20s}{r['class']:14s}{r['params']:>8d}{h:>5s}"
            f"{r['pos_enc_lane']:>5s}{mfl:>10s}{b:>6s}{bkp:>7s}{bmf:>7s}"
        )
    print(
        f"\n[report -> {out}]  (FLOPs = matmul-class only; budget=3000 steps/Adam 3e-3, matched)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
