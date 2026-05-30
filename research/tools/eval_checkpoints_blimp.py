"""Full-BLiMP evaluation of the most-trained mixer_fingerprint checkpoints.

Reconstructs each TinyLM checkpoint's architecture from its state_dict (dim from
``embed.weight``, n_blocks from ``blocks.N`` indices, use_ffn from presence of
``blocks.N.mlp``), STRICT-loads the weights (a wrong arch fails loudly rather
than producing a misleading score), and runs the full BLiMP benchmark
(``n_per_subtask=1000``, all 67 subtasks) with a 12-category rollup.

Usage:
    python -m research.tools.eval_checkpoints_blimp --n-per-subtask 1000 \
        --out research/reports/full_blimp_checkpoints.json CKPT [CKPT ...]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from research.defaults import VOCAB_SIZE
from research.eval.blimp_eval import evaluate_blimp
from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm

# BLiMP subtask -> 12 linguistic categories (UID prefixes per the dataset paper).
_CATEGORY_KEYS = (
    ("anaphor_agreement", "anaphor agreement"),
    ("argument_structure", "argument structure"),
    ("binding", "binding"),
    ("control_raising", "control/raising"),
    ("determiner_noun_agreement", "determiner-noun agr"),
    ("ellipsis", "ellipsis"),
    ("filler_gap", "filler-gap"),
    ("irregular_forms", "irregular forms"),
    ("island_effects", "island effects"),
    ("npi_licensing", "NPI licensing"),
    ("quantifiers", "quantifiers"),
    ("subject_verb", "subject-verb agr"),
)


def _lane_from_name(name: str) -> tuple[str, bool] | None:
    """(lane_name, use_ffn) inferred from the checkpoint filename, or None."""
    n = name.lower()
    # winner lanes carry their own norm/FFN internally -> use_ffn=False
    if "pq_rope" in n:
        return "pq_rope_winner", False
    if "semiring_ffw_real" in n or "semiring_chinchilla" in n:
        return "semiring_winner", False
    for lane in (
        "anisotropic_semiring_reciprocal",
        "fixed_rank_reciprocal",
        "hetero_semiring_reciprocal",
        "semiring_reciprocal_attention",
        "reciprocal_rank_attention",
        "phase_lock_attention",
        "tempered_tropical",
        "softmax_attention",
    ):
        if lane in n:
            return lane, True
    return None


def _infer_arch(sd: dict) -> tuple[int, int, bool]:
    dim = sd["embed.weight"].shape[1]
    blocks = {
        int(m.group(1)) for k in sd for m in [re.match(r"blocks\.(\d+)\.", k)] if m
    }
    n_blocks = max(blocks) + 1 if blocks else 0
    use_ffn = any(".mlp.fc1.weight" in k for k in sd)
    return int(dim), int(n_blocks), bool(use_ffn)


def _rollup(subtask_acc: dict) -> dict:
    cats: dict[str, list] = defaultdict(list)
    for sub, acc in subtask_acc.items():
        for key, label in _CATEGORY_KEYS:
            if sub.startswith(key):
                cats[label].append(acc)
                break
    return {label: round(sum(v) / len(v), 4) for label, v in cats.items() if v}


def _hybrid_spec(name: str) -> tuple[str, str | None] | None:
    """(mixer, pattern) for interleaved/three-lane/ensemble hybrids, or None."""
    n = name.lower()
    m = re.search(r"conv(\d+)_three_?lane(\d+)", n)
    if m:  # interleaved conv:X,three_lane:Y
        return "interleaved", f"conv:{m.group(1)},three_lane:{m.group(2)}"
    m = re.search(r"ensemble_top_ar_(\d+)way", n)
    if m:
        return f"ensemble_top_ar_{m.group(1)}way", None
    for lane in (
        "reciprocal_phase_tropical_three_lane",
        "reciprocal_sparsemax_wavelet_three_lane",
        "tropical_sparsemax_wavelet_three_lane",
    ):
        if lane in n:
            return lane, None
    if "pure_three_lane" in n:
        return "tropical_sparsemax_wavelet_three_lane", None
    return None


def _eval_one(ckpt: Path, n_per_subtask: int, device: str) -> dict:
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = payload.get("model_state_dict") or payload.get("state_dict")
    step = int(payload.get("step", 0) or 0)
    if sd is None:
        return {"ckpt": ckpt.name, "status": "skip", "reason": "no state_dict"}
    dim, n_blocks, ffn_detected = _infer_arch(sd)
    use_ffn = ffn_detected
    lane = _lane_from_name(ckpt.name)
    coverage = 1.0
    if lane is not None:
        lane_name, _ = lane
        model = _build_tinylm(
            _build_lane_factory(lane_name), dim=dim, n_blocks=n_blocks, use_ffn=use_ffn
        )
        model.load_state_dict(sd)  # strict: wrong arch raises here
    else:
        spec = _hybrid_spec(ckpt.name)
        if spec is None:
            return {"ckpt": ckpt.name, "status": "skip", "reason": "unknown lane"}
        from research.tools.mixer_fingerprint import _resolve_lane_factories

        mixer, pattern = spec
        lane_name = mixer if pattern is None else f"interleaved[{pattern}]"
        model_factory, _ = _resolve_lane_factories(mixer, pattern)
        model = _build_tinylm(
            model_factory, dim=dim, n_blocks=n_blocks, use_ffn=use_ffn
        )
        inc = model.load_state_dict(sd, strict=False)  # hybrid: report coverage
        nk = len(list(model.state_dict().keys()))
        coverage = round((nk - len(inc.missing_keys)) / nk, 4)
        if coverage < 0.98:
            return {
                "ckpt": ckpt.name,
                "status": "low_coverage",
                "lane": lane_name,
                "coverage": coverage,
                "missing": len(inc.missing_keys),
                "unexpected": len(inc.unexpected_keys),
                "missing_sample": inc.missing_keys[:6],
            }
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    t = time.monotonic()
    res = evaluate_blimp(
        model, vocab_size=VOCAB_SIZE, device=device, n_per_subtask=n_per_subtask
    )
    sub = dict(getattr(res, "subtask_accuracies", {}) or {})
    return {
        "ckpt": ckpt.name,
        "status": "ok",
        "lane": lane_name,
        "coverage": coverage,
        "dim": dim,
        "n_blocks": n_blocks,
        "use_ffn": use_ffn,
        "step": step,
        "n_params_m": round(n_params / 1e6, 1),
        "blimp_overall": round(float(res.overall_accuracy or 0.0), 4),
        "n_subtasks": len(sub),
        "n_per_subtask": n_per_subtask,
        "by_category": _rollup(sub),
        "elapsed_s": round(time.monotonic() - t, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--n-per-subtask", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results = []
    for c in args.checkpoints:
        cp = Path(c)
        try:
            r = _eval_one(cp, args.n_per_subtask, args.device)
        except Exception as e:  # report, keep going
            r = {"ckpt": cp.name, "status": "error", "reason": str(e)[:200]}
        results.append(r)
        print(json.dumps(r), flush=True)
        if args.out:
            args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
