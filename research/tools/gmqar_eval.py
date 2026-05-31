"""Run the graded MQAR scaling probe (research/eval/gmqar.py) on real TinyLM
checkpoints, reusing eval_checkpoints_blimp's loader so the architecture is
reconstructed from the state_dict exactly as for the BLiMP eval.

This is the "does gMQAR rank models across scale" check from
tasks/scaling_test_validation_plan.md Phase 3 (the nano + 100M rungs that exist
locally; HYDRA-700M needs its own loader and is out of scope here).

Read-only: loads checkpoints, runs zero-shot gMQAR, writes a JSON + prints a
table. No training, no DB writes.

Usage:
    python -m research.tools.gmqar_eval --device cpu \
        --out research/reports/gmqar_ladder.json CKPT [CKPT ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.defaults import VOCAB_SIZE
from research.eval.gmqar import score_model_gmqar
from research.tools.eval_checkpoints_blimp import (
    _build_lane_factory,
    _build_tinylm,
    _hybrid_spec,
    _infer_arch,
    _lane_from_name,
)


# Campaign checkpoints (research/checkpoints/novel_mixer_campaign/, discrim_sweep)
# use names eval_checkpoints_blimp's _lane_from_name doesn't recognise. Map them
# to the canonical single-mixer lanes accepted by _build_lane_factory. Substring
# match, longest/most-specific first.
_CAMPAIGN_LANE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("reciprocal_rank_attention", "reciprocal_rank_attention"),
    ("recip_100m", "reciprocal_rank_attention"),
    ("softmax_100m", "softmax_attention"),
    ("discrim_sweep_30m_softmax", "softmax_attention"),
    ("_softmax_", "softmax_attention"),
)


def _resolve_lane_name(name: str) -> str | None:
    """Canonical lane name for a checkpoint filename, or None if unknown.

    Tries eval_checkpoints_blimp._lane_from_name first (winner/standard lanes),
    then the campaign-name fallback map above.
    """
    std = _lane_from_name(name)
    if std is not None:
        return std[0]
    n = name.lower()
    for pat, lane in _CAMPAIGN_LANE_PATTERNS:
        if pat in n:
            return lane
    return None


def _load_model(ckpt: Path, device: str):
    """Reconstruct + load a TinyLM checkpoint (same path as eval_checkpoints_blimp).

    Returns (model, info) or (None, info) if the arch can't be inferred.
    """
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = payload.get("model_state_dict") or payload.get("state_dict")
    step = int(payload.get("step", 0) or 0)
    if sd is None:
        return None, {"ckpt": ckpt.name, "status": "skip", "reason": "no state_dict"}
    dim, n_blocks, use_ffn = _infer_arch(sd)
    coverage = 1.0
    lane_name = _resolve_lane_name(ckpt.name)
    if lane_name is not None:
        model = _build_tinylm(
            _build_lane_factory(lane_name), dim=dim, n_blocks=n_blocks, use_ffn=use_ffn
        )
        model.load_state_dict(sd)  # strict
    else:
        spec = _hybrid_spec(ckpt.name)
        if spec is None:
            return None, {"ckpt": ckpt.name, "status": "skip", "reason": "unknown lane"}
        from research.tools.mixer_fingerprint import _resolve_lane_factories

        mixer, pattern = spec
        factories = _resolve_lane_factories(mixer, pattern)
        model = _build_tinylm(factories, dim=dim, n_blocks=n_blocks, use_ffn=use_ffn)
        inc = model.load_state_dict(sd, strict=False)
        total = len(sd)
        missing = len(inc.missing_keys)
        coverage = round((total - missing) / total, 3) if total else 0.0
    model.to(device).eval()
    info = {
        "ckpt": ckpt.name,
        "status": "ok",
        "step": step,
        "dim": dim,
        "n_blocks": n_blocks,
        "use_ffn": use_ffn,
        "coverage": coverage,
    }
    return model, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpts", nargs="+", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument(
        "--token-pool",
        type=int,
        default=0,
        help="restrict KV ids to this many low (well-trained) token "
        "ids so gMQAR measures binding, not embedding quality "
        "(0 = full vocab).",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/gmqar_ladder.json")
    )
    args = ap.parse_args()

    rows = []
    for ckpt in args.ckpts:
        if not ckpt.exists():
            rows.append({"ckpt": str(ckpt), "status": "missing"})
            print(f"  MISSING {ckpt}")
            continue
        model, info = _load_model(ckpt, args.device)
        if model is None:
            rows.append(info)
            print(f"  SKIP {info['ckpt']}: {info.get('reason')}")
            continue
        res = score_model_gmqar(
            model,
            vocab_size=args.vocab_size,
            device=args.device,
            token_pool=args.token_pool,
        )
        info.update(
            {"audc": res.audc, "d50": res.d50, "chance": res.chance, "cells": res.cells}
        )
        rows.append(info)
        print(
            f"  {info['ckpt']:50s} dim={info['dim']:4d} nblk={info['n_blocks']:2d} "
            f"AUDC={res.audc:.4f} D50={res.d50:2d} (chance {res.chance:.2e})"
        )
        del model

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"vocab_size": args.vocab_size, "rows": rows}, indent=2)
    )
    print(f"\nwrote {args.out}")
    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: (r.get("audc", 0), r.get("d50", 0)), reverse=True)
    print("\nRANKING by AUDC (higher = binds across more difficulty):")
    for r in ok:
        print(f"  {r['audc']:.4f}  D50={r['d50']:2d}  {r['ckpt']}")


if __name__ == "__main__":
    main()
