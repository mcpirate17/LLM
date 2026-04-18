#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from research.scientist.api_routes._ml_influence_status import build_ml_influence_status
from research.scientist.notebook import LabNotebook

DEFAULT_DB = "research/lab_notebook.db"
DEFAULT_STRENGTH_REPORT = "research/reports/model_strength/model_strength_report.json"
DEFAULT_OUT_DIR = "research/docs"


def _load_strength_report(path: str | Path) -> Dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def _priority_watchlist(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _score(row: Dict[str, Any]) -> tuple[int, int, float]:
        evidence = str(row.get("evidence_level") or "")
        evidence_rank = {"insufficient": 0, "sparse": 1, "building": 2}.get(evidence, 3)
        n_used = int(row.get("n_used") or 0)
        return (evidence_rank, n_used, float(row.get("avg_loss_ratio") or 1e9))

    watch = [
        row
        for row in rows
        if str(row.get("evidence_level") or "")
        in {"insufficient", "sparse", "building"}
    ]
    watch.sort(key=_score)
    return watch


def build_plan_payload(
    *,
    db_path: str,
    strength_report_path: str,
) -> Dict[str, Any]:
    status = build_ml_influence_status()
    nb = LabNotebook(db_path)
    try:
        observability = nb.get_template_slot_observability(limit=20)
    finally:
        nb.close()

    all_templates = list(observability.get("all_templates") or [])
    top_templates = list(observability.get("top_templates") or [])
    watchlist = _priority_watchlist(all_templates)

    strength = _load_strength_report(strength_report_path)
    weak_templates = list(
        (strength.get("rankings") or {}).get("weak_templates_overall") or []
    )

    return {
        "ml_influence": status,
        "top_reference_templates": top_templates[:8],
        "backfill_priority_templates": watchlist[:20],
        "weak_templates_overall": weak_templates[:20],
        "data_recommendations": [
            "Collect more slot-level evidence for sparse/building attention templates before any positive weighting.",
            "Expand induction and binding coverage for families that pass S1 but remain weak on non-perplexity probes.",
            "Prefer backfills with complete screening provenance so candidate/signal weighting can be validated on trusted rows.",
            "Use reference families with repeated low-loss survivors to anchor nearby template backfills rather than globally raising weak families.",
        ],
    }


def _render_markdown(payload: Dict[str, Any]) -> str:
    status = payload["ml_influence"]
    lines = [
        "# ML Hardening Plan",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## ML Influence Status",
        "",
    ]
    for name, info in (status.get("components") or {}).items():
        lines.append(
            f"- `{name}`: quality={info.get('quality_tier')} requested={info.get('requested')} allowed={info.get('allowed')} reason={info.get('reason')}"
        )
    lines.extend(
        [
            "",
            "## Backfill Priority Templates",
            "",
        ]
    )
    for row in payload.get("backfill_priority_templates", []):
        lines.append(
            f"- `{row.get('name')}`: evidence={row.get('evidence_level')} n_used={row.get('n_used')} s1_rate={row.get('s1_rate')} actions={'; '.join(row.get('actions') or [])}"
        )
    lines.extend(
        [
            "",
            "## Reference Families",
            "",
        ]
    )
    for row in payload.get("top_reference_templates", []):
        lines.append(
            f"- `{row.get('name')}`: n_used={row.get('n_used')} s1_rate={row.get('s1_rate')} best_loss={row.get('best_loss_ratio')}"
        )
    lines.extend(
        [
            "",
            "## Data Recommendations",
            "",
        ]
    )
    for item in payload.get("data_recommendations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML hardening plan/report")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--strength-report", default=DEFAULT_STRENGTH_REPORT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    payload = build_plan_payload(
        db_path=args.db,
        strength_report_path=args.strength_report,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ml_hardening_plan_{date.today().isoformat()}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
