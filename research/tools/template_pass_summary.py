"""Generate research/reports/template_pass_2026XXXX_summary.md.

Acceptance criterion 9 of the deep-dive doc: per-template one-liner with
empirical justification (n, mean, CI). Reads template_classification.csv.
"""

from __future__ import annotations

import csv
import datetime
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "research/reports"
CLASSIFICATION = REPORTS / "template_classification.csv"

BUCKET_ORDER = ("A+", "A", "C", "D", "B", "HOLD", "E")
BUCKET_DESCRIPTIONS = {
    "A+": "high-capability promotion (v2_max_auc ≥ 0.95)",
    "A": "keep (mean_sa ≥ 0.80, dom_slot, mixer_op)",
    "C": "rescue (slot fill discriminates pass/fail)",
    "D": "mine (exotic_functional + mixer pair in pass cohort)",
    "B": "cull to weight floor 0.5",
    "HOLD": "no automatic decision (review by hand)",
    "E": "insufficient data (n < 20 or slot-opaque)",
}


def _format_row(r: dict[str, str]) -> str:
    n = int(r["n"])
    mean_sa = float(r["mean_sa"]) if r["mean_sa"] else 0.0
    ci_lo = float(r["ci_low"]) if r["ci_low"] else 0.0
    ci_hi = float(r["ci_high"]) if r["ci_high"] else 0.0
    pass_rate = float(r["pass_rate"]) if r["pass_rate"] else 0.0
    name = r["template_name"]
    reason = r["primary_reason"]
    if len(reason) > 110:
        reason = reason[:107] + "..."
    return (
        f"- `{name}` — n={n}, mean_sa={mean_sa:.2f}, "
        f"pass_rate={pass_rate:.2f} CI[{ci_lo:.2f},{ci_hi:.2f}] | {reason}"
    )


def main() -> None:
    today = datetime.date.today().strftime("%Y%m%d")
    out_path = REPORTS / f"template_pass_{today}_summary.md"

    rows = list(csv.DictReader(open(CLASSIFICATION)))
    bucket_groups: dict[str, list[dict[str, str]]] = {b: [] for b in BUCKET_ORDER}
    for r in rows:
        bucket_groups.setdefault(r["bucket"], []).append(r)

    lines = []
    lines.append(f"# Template-pass classification — {today}")
    lines.append("")
    lines.append("Inputs:")
    lines.append("- `template_classification.csv` (per-template buckets)")
    lines.append("- `slot_mixer_credit_v2.csv` (9 dominant-mixing slots)")
    lines.append(
        "- `op_mixer_certification_v2.csv` (6 mixer / 8 underperforming / 6 exotic_functional / 91 non-mixer / 50 insufficient)"
    )
    lines.append("- `slot_realization.csv` (per-slot motif pass rates, n≥20 published)")
    lines.append(
        "- `high_capability_slot_fills.csv` (induction_intermediate_auc per slot)"
    )
    lines.append("")
    lines.append(
        "Pass cohort: `language_control_s05_sentence_assoc_score >= 0.95 AND failure_op != 'nano_bind'`"
    )
    lines.append(
        "Fail cohort: `language_control_s05_sentence_assoc_score < 0.30 OR failure_op = 'nano_bind'`"
    )
    lines.append("")
    lines.append("## Bucket counts")
    lines.append("")
    lines.append("| Bucket | Count | Description |")
    lines.append("|---|---|---|")
    for b in BUCKET_ORDER:
        lines.append(
            f"| {b} | {len(bucket_groups.get(b, []))} | {BUCKET_DESCRIPTIONS[b]} |"
        )
    lines.append("")

    for b in BUCKET_ORDER:
        members = bucket_groups.get(b, [])
        if not members:
            continue
        lines.append(f"## Bucket {b} — {BUCKET_DESCRIPTIONS[b]} ({len(members)})")
        lines.append("")
        members_sorted = sorted(members, key=lambda r: -int(r["n"]))
        for r in members_sorted:
            lines.append(_format_row(r))
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}", file=sys.stderr)
    print(
        f"  {len(rows)} templates classified into {len(bucket_groups)} buckets",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
