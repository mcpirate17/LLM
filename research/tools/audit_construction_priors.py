#!/usr/bin/env python3
"""Audit construction-prior snapshots before activation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.scientist.construction_priors import (  # noqa: E402
    DEFAULT_ACTIVATION_MAX_RISK_RATIO,
    DEFAULT_ACTIVATION_MIN_CONTEXTS,
    DEFAULT_ACTIVATION_MIN_WEIGHT_USED,
    audit_construction_prior_payload,
    filter_construction_prior_payload_for_activation,
    record_construction_prior_snapshot,
)
from research.scientist.notebook import LabNotebook  # noqa: E402


DB_PATH = PROJECT_ROOT / "research/lab_notebook.db"
RUNTIME_DIR = PROJECT_ROOT / "research/runtime"


def _load_snapshot(nb: LabNotebook, version: str | None) -> dict[str, Any]:
    if version:
        row = nb.conn.execute(
            """
            SELECT version, is_active, payload_json, summary_json, notes
            FROM construction_prior_snapshots
            WHERE version = ?
            """,
            (version,),
        ).fetchone()
    else:
        row = nb.conn.execute(
            """
            SELECT version, is_active, payload_json, summary_json, notes
            FROM construction_prior_snapshots
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        raise SystemExit(
            f"construction prior snapshot not found: {version or 'latest'}"
        )
    payload = json.loads(row["payload_json"])
    summary = json.loads(row["summary_json"]) if row["summary_json"] else {}
    return {
        "version": row["version"],
        "is_active": bool(row["is_active"]),
        "payload": payload,
        "summary": summary,
        "notes": row["notes"] or "",
    }


def _summary_for_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rules = [r for r in (payload.get("rules") or []) if isinstance(r, dict)]
    source_counts = dict(payload.get("source_counts") or {})
    return {
        "version": payload.get("version"),
        "n_rules": len(rules),
        "n_use": sum(1 for r in rules if r.get("verdict") == "use"),
        "n_avoid": sum(1 for r in rules if r.get("verdict") == "avoid"),
        "n_mixed": sum(1 for r in rules if r.get("verdict") == "mixed"),
        "n_local_edit_observations": source_counts.get("local_edit_observations", 0),
        "n_v2_observations": source_counts.get("v2_observations", 0),
        "n_risk_rows": source_counts.get("risk_rows", 0),
        "n_op_weights": len(payload.get("op_weights") or {}),
        "n_slot_motif_multipliers": sum(
            len(v) for v in (payload.get("slot_motif_multipliers") or {}).values()
        ),
        "n_slot_motif_denylist": sum(
            len(v) for v in (payload.get("slot_motif_denylist") or {}).values()
        ),
        "activation_filter": payload.get("activation_filter") or {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--version", default="")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--min-contexts", type=int, default=DEFAULT_ACTIVATION_MIN_CONTEXTS
    )
    parser.add_argument(
        "--max-risk-ratio",
        type=float,
        default=DEFAULT_ACTIVATION_MAX_RISK_RATIO,
    )
    parser.add_argument(
        "--min-weight-used",
        type=float,
        default=DEFAULT_ACTIVATION_MIN_WEIGHT_USED,
    )
    parser.add_argument(
        "--output",
        default=str(RUNTIME_DIR / "construction_prior_audit.json"),
    )
    parser.add_argument(
        "--write-filtered-snapshot",
        action="store_true",
        help="Record a filtered snapshot; default only writes the audit report.",
    )
    parser.add_argument(
        "--activate-filtered-snapshot",
        action="store_true",
        help="Activate the filtered snapshot. Implies --write-filtered-snapshot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    nb = LabNotebook(str(args.db), use_native=False)
    filtered_version = None
    try:
        snapshot = _load_snapshot(nb, str(args.version or "") or None)
        report = audit_construction_prior_payload(
            snapshot["payload"],
            min_contexts=max(1, int(args.min_contexts)),
            max_risk_ratio=max(0.0, float(args.max_risk_ratio)),
            min_weight_used=max(0.0, float(args.min_weight_used)),
            top_n=max(1, int(args.top_n)),
        )
        filtered_payload = filter_construction_prior_payload_for_activation(
            snapshot["payload"],
            min_contexts=max(1, int(args.min_contexts)),
            max_risk_ratio=max(0.0, float(args.max_risk_ratio)),
            min_weight_used=max(0.0, float(args.min_weight_used)),
        )
        filtered_payload["version"] = (
            f"{snapshot['version']}-filtered-{time.strftime('%H%M%S')}"
        )
        filtered_payload["computed_at"] = time.time()
        filtered_summary = _summary_for_payload(filtered_payload)
        should_write = bool(
            args.write_filtered_snapshot or args.activate_filtered_snapshot
        )
        if should_write:
            filtered_version = record_construction_prior_snapshot(
                nb,
                {"payload": filtered_payload, "summary": filtered_summary},
                activate=bool(args.activate_filtered_snapshot),
                notes=(
                    "risk-filtered construction prior derived from "
                    f"{snapshot['version']}"
                ),
            )
        audit = {
            "created_at": time.time(),
            "source_snapshot": {
                "version": snapshot["version"],
                "is_active": snapshot["is_active"],
                "summary": snapshot["summary"],
            },
            "audit": report,
            "filtered_summary": filtered_summary,
            "filtered_snapshot_version": filtered_version,
        }
    finally:
        nb.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "source_version": audit["source_snapshot"]["version"],
                "eligible_rules": audit["audit"]["eligible_rules"],
                "blocked_rules": audit["audit"]["blocked_rules"],
                "filtered_snapshot_version": filtered_version,
                "filtered_summary": filtered_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
