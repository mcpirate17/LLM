"""Phase C.3 — Auto-demote trial templates that underperform production.

Reads `_template_trial` flag aggregations via
`scientist.analytics._exp_weights._WeightsMixin.compute_trial_template_stats`
and proposes weight-floor demotions for trial templates that hit n_trial >= 30
with s1_trial_rate materially below their production peers.

V1 scope: emit JSON proposals; optional `--apply` writes weight edits to
`research/synthesis/templates.py` / sibling manifests in-place. Human review
is the default; apply is opt-in.

Demotion criteria (default — overridable via CLI):
  - n_trial >= 30
  - s1_trial_rate < 0.30 (absolute floor) OR
    s1_trial_rate < 0.50 * s1_prod_rate (relative)
  - n_prod >= 30 for the relative-comparison branch (else absolute only)

Outputs:
  research/reports/trial_template_demotions.json — proposals (always)
  +/- in-place weight edits when --apply is passed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.scientist.notebook import LabNotebook  # noqa: E402
from research.scientist.analytics import ExperimentAnalytics  # noqa: E402

REPORTS = REPO / "research/reports"
DB_PATH = REPO / "research/runs.db"

DEFAULT_MIN_N_TRIAL = 30
DEFAULT_MIN_N_PROD_FOR_RELATIVE = 30
DEFAULT_ABSOLUTE_FLOOR = 0.30
DEFAULT_RELATIVE_FACTOR = 0.50
WEIGHT_FLOOR = 0.5


def _decide(stats: dict, args: argparse.Namespace) -> tuple[bool, str]:
    n_trial = int(stats.get("n_trial", 0))
    n_prod = int(stats.get("n_prod", 0))
    s1_trial = float(stats.get("s1_trial_rate", 0.0))
    s1_prod = float(stats.get("s1_prod_rate", 0.0))
    if n_trial < args.min_n_trial:
        return False, f"insufficient_n_trial={n_trial}"
    if s1_trial < args.absolute_floor:
        return True, f"s1_trial={s1_trial:.2f}<{args.absolute_floor}"
    if (
        n_prod >= args.min_n_prod
        and s1_prod > 0
        and s1_trial < args.relative_factor * s1_prod
    ):
        return (
            True,
            f"s1_trial={s1_trial:.2f}<{args.relative_factor}*s1_prod={s1_prod:.2f}",
        )
    return False, "stats_acceptable"


def collect_proposals(stats: dict[str, dict], args: argparse.Namespace) -> list[dict]:
    proposals = []
    for tpl, s in sorted(stats.items()):
        demote, reason = _decide(s, args)
        proposals.append(
            {
                "template_name": tpl,
                "demote": demote,
                "reason": reason,
                "n_trial": s["n_trial"],
                "n_prod": s["n_prod"],
                "s1_trial_rate": round(s["s1_trial_rate"], 4),
                "s1_prod_rate": round(s["s1_prod_rate"], 4),
                "proposed_floor_weight": WEIGHT_FLOOR if demote else None,
            }
        )
    return proposals


def _find_weight_file(template_name: str) -> tuple[Path, str] | None:
    """Locate which manifest defines this template's static weight."""
    candidates = [
        REPO / "research/synthesis/templates.py",
        REPO / "research/synthesis/_template_attention_manifest.py",
        REPO / "research/synthesis/_template_routing_manifest.py",
        REPO / "research/synthesis/_template_research_manifest.py",
        REPO / "research/synthesis/_template_role_slot_manifest.py",
        REPO / "research/synthesis/_templates_attention_tail.py",
    ]
    needle = f'"{template_name}":'
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text()
        for line in text.splitlines():
            if needle in line and ":" in line:
                # Filter to weight-dict lines (value is a float-like)
                tail = line.split(needle, 1)[1].strip().rstrip(",").strip()
                try:
                    float(tail.split("#", 1)[0].strip())
                    return path, line
                except ValueError:
                    continue
    return None


def apply_demotion(template_name: str) -> tuple[bool, str]:
    found = _find_weight_file(template_name)
    if not found:
        return False, f"no weight definition found for {template_name}"
    path, line = found
    text = path.read_text()
    if line not in text:
        return False, "line drifted between locate and apply"
    new_line = line.split(":", 1)[0] + f": {WEIGHT_FLOOR},"
    text2 = text.replace(line, new_line)
    if text == text2:
        return False, "no replacement applied"
    path.write_text(text2)
    return True, f"{path.name}: {line.strip()} -> {new_line.strip()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-n-trial", type=int, default=DEFAULT_MIN_N_TRIAL)
    parser.add_argument(
        "--min-n-prod", type=int, default=DEFAULT_MIN_N_PROD_FOR_RELATIVE
    )
    parser.add_argument("--absolute-floor", type=float, default=DEFAULT_ABSOLUTE_FLOOR)
    parser.add_argument(
        "--relative-factor", type=float, default=DEFAULT_RELATIVE_FACTOR
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write weight-floor edits in place (default: proposals only)",
    )
    args = parser.parse_args()

    nb = LabNotebook(str(DB_PATH), read_only=True)
    analytics = ExperimentAnalytics(nb)
    stats = analytics.compute_trial_template_stats(min_used=1)
    print(f"Trial templates observed: {len(stats)}", file=sys.stderr)
    proposals = collect_proposals(stats, args)
    n_demote = sum(1 for p in proposals if p["demote"])
    print(f"Demotion proposals: {n_demote}/{len(proposals)}", file=sys.stderr)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS / "trial_template_demotions.json"
    out_path.write_text(json.dumps({"proposals": proposals}, indent=2))
    print(f"Wrote {out_path}", file=sys.stderr)

    if args.apply:
        for p in proposals:
            if not p["demote"]:
                continue
            ok, msg = apply_demotion(p["template_name"])
            print(
                ("APPLIED" if ok else "SKIP   ") + f": {p['template_name']}: {msg}",
                file=sys.stderr,
            )
    nb.close()


if __name__ == "__main__":
    main()
