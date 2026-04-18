#!/usr/bin/env python
"""Audit scaffold coverage and native-bridge exposure across component ops."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from research.synthesis.native_support import BOUND_PARAM_OPS, BOUND_POINTWISE_OPS
from research.synthesis.primitives import OP_NAME_ALIASES, PRIMITIVE_REGISTRY
from research.tools.profile_component_scaffolds import (
    canonical_missing_profile_ops,
    recommended_scaffold_family,
)


def _load_profiled_ops(db_path: Path) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [row[0] for row in conn.execute("SELECT op_name FROM op_profiles")]
    finally:
        conn.close()


def build_audit_report(db_path: Path) -> dict[str, object]:
    profiled_ops = _load_profiled_ops(db_path)
    profiled_set = set(profiled_ops)
    primitive_ops = sorted(PRIMITIVE_REGISTRY)
    primitive_set = set(primitive_ops)
    missing_raw = sorted(primitive_set - profiled_set)
    canonical_missing = canonical_missing_profile_ops(profiled_ops)
    alias_only_profiled = sorted(
        op
        for op in missing_raw
        if OP_NAME_ALIASES.get(op, op) not in missing_raw
        and OP_NAME_ALIASES.get(op, op)
        in {OP_NAME_ALIASES.get(p, p) for p in profiled_set}
    )

    scaffold_map = {
        op: recommended_scaffold_family(op)
        for op in sorted(set(missing_raw) | set(canonical_missing))
    }
    scaffoldable_missing = {
        op: fam for op, fam in scaffold_map.items() if fam is not None
    }
    unscaffolded_missing = sorted(op for op, fam in scaffold_map.items() if fam is None)

    bound_ops = sorted((BOUND_PARAM_OPS | BOUND_POINTWISE_OPS) & primitive_set)
    bound_profiled = sorted(set(bound_ops) & profiled_set)
    bound_canonical_missing = sorted(set(bound_ops) & set(canonical_missing))

    return {
        "profiling_db": str(db_path),
        "counts": {
            "primitive_ops_raw": len(primitive_ops),
            "profiled_ops_raw": len(profiled_ops),
            "missing_ops_raw": len(missing_raw),
            "primitive_ops_canonical": len(
                {OP_NAME_ALIASES.get(op, op) for op in primitive_set}
            ),
            "profiled_ops_canonical": len(
                {OP_NAME_ALIASES.get(op, op) for op in profiled_set}
            ),
            "missing_ops_canonical": len(canonical_missing),
            "alias_only_profiled": len(alias_only_profiled),
            "bound_native_ops": len(bound_ops),
            "bound_native_profiled_ops": len(bound_profiled),
            "bound_native_missing_canonical_ops": len(bound_canonical_missing),
        },
        "missing_raw_ops": missing_raw,
        "alias_only_profiled_ops": alias_only_profiled,
        "missing_canonical_ops": canonical_missing,
        "scaffoldable_missing_ops": scaffoldable_missing,
        "unscaffolded_missing_ops": unscaffolded_missing,
        "bound_native_profiled_ops": bound_profiled,
        "bound_native_missing_canonical_ops": bound_canonical_missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit component scaffold coverage")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("research/profiling/component_profiles.db"),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = build_audit_report(args.db)
    payload = json.dumps(report, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
