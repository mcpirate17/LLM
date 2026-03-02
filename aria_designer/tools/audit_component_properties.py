#!/usr/bin/env python3
"""Generate component property coverage report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.app.property_audit import audit_components


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components-root", default="components")
    parser.add_argument("--json-out", default="docs/component_property_audit.json")
    parser.add_argument("--md-out", default="docs/component_property_audit.md")
    args = parser.parse_args()

    root = Path(args.components_root).resolve()
    report = audit_components(root)

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = []
    s = report["summary"]
    md_lines.append("# Component Property Audit")
    md_lines.append("")
    md_lines.append(f"- total: {s['total_components']}")
    md_lines.append(f"- ok: {s['ok']}")
    md_lines.append(f"- warnings: {s['warnings']}")
    md_lines.append(f"- errors: {s['errors']}")
    md_lines.append("")
    md_lines.append("## Components With Issues")
    md_lines.append("")

    for row in report["components"]:
        if row["status"] == "ok":
            continue
        md_lines.append(f"### {row['id']} ({row['status']})")
        md_lines.append(f"- category: {row['category']}")
        md_lines.append(f"- properties: {row['property_count']}")
        for issue in row["issues"]:
            md_lines.append(f"- [{issue['severity']}] {issue['code']}: {issue['message']}")
        md_lines.append("")

    Path(args.md_out).write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {json_path} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
