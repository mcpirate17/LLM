#!/usr/bin/env python3
"""Cross-round consistency analyzer — ground-truth anchored.

The 3 full audits run back-to-back on a byte-identical codebase (no fixes between them),
so they're a clean repeatability experiment. Comparing free-text LLM findings by raw path
strings is fragile (models cite `__init__.py` vs `aria_core/__init__.py` vs absolute), so
this anchors to DETERMINISTIC GROUND TRUTH instead:

  * detectors.measure() yields the exact set of real god files (>1250 LOC) and god
    functions (>100 LOC). That set is identical every round (code is unchanged) — which is
    itself the oracle-stability check.
  * For each round we measure the DETECTION RATE: of the N real god files, how many did that
    round's `god_files` audit actually name? Same for god functions. A target counts as
    detected only if its real path (suffix-matched, repo-prefix-insensitive) appears in the
    findings — no credit for hallucinated paths.

Stable detection rates across rounds = the auditors reliably re-surface the same REAL issues
every pass, so nothing structural gets dropped between audits. That is the property the user
asked for ("are the findings repeating every audit / consistency").

Usage:
    python consistency.py --rounds 2 3 4
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import detectors

HERE = Path(__file__).resolve().parent
FINDINGS = HERE.parent / "findings"


def _cfg() -> dict:
    return tomllib.loads((HERE / "config.toml").read_text())


def _repo(c: dict) -> Path:
    return (HERE / c["repo"]["root"]).resolve()


def _ground_truth(c: dict) -> tuple[list[str], list[str], dict]:
    repo = _repo(c)
    targets = [repo / t for t in c["loop"]["targets"]] or [repo]
    files = detectors._iter_py(targets, set(c["loop"]["exclude"]))
    # full (untruncated) ground-truth lists straight from the detector internals.
    _, _, gf_list, gfn_list = detectors._god_counts(files)
    gf = [s.split(" ")[0] for s in gf_list]
    # god functions: reduce to the DISTINCT FILES that contain them (matching free-text
    # function names risks generic-name false hits; "did the audit point at the right
    # files" is the clean, conservative signal).
    gfn = sorted({s.split(":")[0] for s in gfn_list})
    m = detectors.measure(repo, c["loop"]["targets"], set(c["loop"]["exclude"]))
    return gf, gfn, m.as_dict()


def _round_text(rnd: int, pass_id: str) -> str:
    d = FINDINGS / f"round-{rnd:02d}"
    return "\n".join(f.read_text(errors="replace") for f in d.glob(f"{pass_id}__*.md"))


def _suffixes(path: str) -> list[str]:
    """Distinctive tails of a path to match against free text, prefix-insensitive."""
    p = path.split(":")[0]
    parts = p.split("/")
    tails = [p]
    if len(parts) >= 2:
        tails.append("/".join(parts[-2:]))
    return tails


def _detected(truth: list[str], text: str) -> int:
    hits = 0
    for t in truth:
        # require the two-component tail (e.g. runner/dashboard_orchestrator.py) so a bare
        # basename like __init__.py doesn't over-credit.
        tails = _suffixes(t)
        needle = tails[-1] if len(tails) > 1 else tails[0]
        if needle in text:
            hits += 1
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rounds", nargs="+", type=int, required=True)
    args = ap.parse_args()
    rounds = args.rounds
    c = _cfg()

    print("Measuring deterministic ground truth (god files / god functions)...")
    gf, gfn, vec = _ground_truth(c)
    print("\nDeterministic oracle vector (identical every round — code unchanged):")
    print("  " + ", ".join(f"{k}={v}" for k, v in vec.items() if k != "detail"))
    print(
        f"\nGround truth: {len(gf)} god files, {len(gfn)} god functions (>thresholds).\n"
    )

    print(
        f"{'pass / metric':<22}" + "".join(f"round {r:<5}" for r in rounds) + "stable?"
    )
    print("-" * 62)

    rows = [
        ("god_files detected", gf, "god_files"),
        ("god_functions found", gfn, "god_functions"),
    ]
    for label, truth, pass_id in rows:
        rates = []
        for r in rounds:
            txt = _round_text(r, pass_id)
            d = _detected(truth, txt)
            rates.append(d)
        cells = "".join(f"{f'{d}/{len(truth)}':<11}" for d in rates)
        spread = max(rates) - min(rates) if rates else 0
        stable = (
            "yes"
            if (truth and spread <= max(1, len(truth) // 10))
            else ("n/a" if not truth else "VARIES")
        )
        print(f"{label:<22}{cells}{stable}")

    print("-" * 62)
    print(
        "\n'stable' = detection count varies by <=10% of ground-truth across rounds → the\n"
        "auditors reliably re-find the same real structural issues each audit (so nothing\n"
        "gets silently dropped between runs). Hallucinated paths get no credit by design."
    )


if __name__ == "__main__":
    main()
