"""Compare in-context associative-recall capacity (MQAR) of fab candidates vs the
softmax baseline. The discriminating question: does a novel non-QKV mechanism hold
MORE key->value pairs in-context (higher K crossover) than softmax?

Builds each candidate lane from its ledger math_axes (generate_module) and the
softmax baseline (multi-head causal attention), runs the recall sweep over K, and
reports each lane's capacity (largest K with recall >= threshold).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

from component_fab.generator.code_generator import generate_module
from research.tools.nano_recall_sweep import _train_and_recall

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CANDS = ("improve_improve_tropical", "improve_cross_cross")


def _axes_for(prefixes: tuple[str, ...]) -> dict[str, dict]:
    """Latest math_axes per candidate name-prefix, across ledger rotations."""
    found: dict[str, dict] = {}
    for f in sorted(glob.glob("component_fab/catalog/ledger.jsonl*")):
        for ln in Path(f).read_text().splitlines():
            if '"event": "grade"' not in ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            name = d.get("name", "")
            for pre in prefixes:
                if name.startswith(pre):
                    ax = (d.get("metadata") or {}).get("math_axes")
                    if ax:
                        found[name[:46]] = ax
    return found


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--loads", nargs="*", type=int, default=[1, 2, 4, 8])
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--max-candidates", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=str(_REPO / "reports" / "recall_candidates.json"))
    args = p.parse_args(argv)

    cand_axes = _axes_for(_DEFAULT_CANDS)
    cands = dict(list(cand_axes.items())[: args.max_candidates])
    lanes: dict[str, object] = {"softmax_multihead": None}  # None -> default gpt2 lane
    for name, ax in cands.items():
        lanes[name] = (lambda a: (lambda dim: generate_module(a, dim=dim)))(ax)
    print(f"lanes: {list(lanes)} | dim={args.dim} loads={args.loads} steps={args.steps}")

    report = {"dim": args.dim, "loads": args.loads, "steps": args.steps, "lanes": {}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for lane_name, factory in lanes.items():
        print(f"\n=== {lane_name} ===")
        rows, cap = [], None
        for k in sorted(args.loads):
            recs = []
            for s in range(args.seeds):
                try:
                    r = _train_and_recall(
                        args.dim, args.n_blocks, k, device=args.device, seed=s,
                        steps=args.steps, lr=3e-3, batch_size=128, eval_batch=512,
                        lane_factory=factory,
                    )
                    recs.append(r["recall"])
                except Exception as e:  # noqa: BLE001 - one lane must not abort the rest
                    print(f"  K={k} FAILED: {type(e).__name__}: {e}")
            rec = sum(recs) / len(recs) if recs else float("nan")
            mark = "" if rec >= args.threshold else "  <below>"
            print(f"  K={k:<3} recall={rec:.3f}{mark}")
            rows.append({"k": k, "recall": round(rec, 3)})
            if recs and rec >= args.threshold:
                cap = k
        report["lanes"][lane_name] = {"rows": rows, "capacity_K": cap}
        out.write_text(json.dumps(report, indent=2))
        print(f"  capacity (largest K, recall>={args.threshold}): {cap}")

    print(f"\nreport: {out}")
    sm = report["lanes"].get("softmax_multihead", {}).get("capacity_K")
    print(f"softmax capacity_K={sm}; candidates beating it:",
          [n for n, d in report["lanes"].items()
           if n != "softmax_multihead" and (d["capacity_K"] or 0) > (sm or 0)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
