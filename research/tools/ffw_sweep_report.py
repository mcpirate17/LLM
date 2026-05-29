"""Rank FineFineWeb mixer_fingerprint lane results by nano priority metrics.

Reads ``research/reports/mixer_fingerprint/<lane>.jsonl`` files, pulls the final
checkpoint's capability metrics (the user's priority set: nano_induction_nearest,
ni05, nb05, BLiMP), and prints a ranked table. PPL is shown last (deprioritized).

Usage:
    python -m research.tools.ffw_sweep_report
    python -m research.tools.ffw_sweep_report --lanes reciprocal_rank_attention phase_lock_attention
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_DIR = Path("research/reports/mixer_fingerprint")


def _final_checkpoint(path: Path) -> dict | None:
    ck = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip() and '"event": "checkpoint"' in line
    ]
    return ck[-1] if ck else None


def _max_acc(cheap: dict, key: str) -> float | None:
    v = cheap.get(key)
    if isinstance(v, dict):
        return v.get("max_accuracy")
    return v if isinstance(v, (int, float)) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=_DIR)
    ap.add_argument(
        "--lanes", nargs="*", default=None, help="restrict to these lane names"
    )
    args = ap.parse_args()

    rows = []
    for path in sorted(args.dir.glob("*.jsonl")):
        lane = path.stem
        if args.lanes and lane not in args.lanes:
            continue
        ck = _final_checkpoint(path)
        if not ck:
            continue
        c = ck.get("cheap", {})
        rows.append(
            {
                "lane": lane,
                "step": ck.get("step"),
                "blimp": c.get("blimp_overall"),
                "nano_ind_near": _max_acc(c, "nano_induction_nearest"),
                "ni05": _max_acc(c, "ni05"),
                "nb05": _max_acc(c, "nb05"),
                "ind_scr": c.get("induction_screening_auc"),
                "ppl": c.get("wikitext_ppl"),
            }
        )

    if not rows:
        print("(no mixer_fingerprint jsonl results found)")
        return

    # Composite capability rank: induction-nearest + ni05 + nb05 + (blimp-0.5).
    def score(r):
        return (
            (r["nano_ind_near"] or 0)
            + (r["ni05"] or 0)
            + (r["nb05"] or 0)
            + max(0.0, (r["blimp"] or 0) - 0.5)
        )

    rows.sort(key=score, reverse=True)
    hdr = f"{'lane':44s} {'step':>5s} {'BLiMP':>6s} {'indNear':>7s} {'ni05':>6s} {'nb05':>6s} {'indScr':>6s} {'ppl':>7s}"
    print(hdr)
    print("-" * len(hdr))

    def f(v, p=4):
        return f"{v:.{p}f}" if isinstance(v, (int, float)) else "  -  "

    for r in rows:
        print(
            f"{r['lane']:44s} {str(r['step']):>5s} {f(r['blimp']):>6s} "
            f"{f(r['nano_ind_near'], 3):>7s} {f(r['ni05'], 3):>6s} {f(r['nb05'], 3):>6s} "
            f"{f(r['ind_scr'], 3):>6s} {f(r['ppl'], 0):>7s}"
        )
    print(
        "\nranked by capability composite (indNear + ni05 + nb05 + blimp-over-random); ppl ignored"
    )


if __name__ == "__main__":
    main()
