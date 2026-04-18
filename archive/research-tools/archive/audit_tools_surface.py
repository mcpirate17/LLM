#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

ROOT = Path("research/tools")
DEFAULT_OUT = Path("research/reports/tools_surface_audit.csv")

_REGISTRY_ROW = re.compile(
    r"^\|\s*`(?P<tool>[^`]+)`\s*\|\s*(?P<category>[^|]+?)\s*\|\s*(?P<desc>[^|]+?)\s*\|\s*(?P<entry>[^|]+?)\s*\|\s*(?P<expiry>[^|]+?)\s*\|$"
)


def _load_registry() -> dict[str, dict[str, str]]:
    registry_path = ROOT / "REGISTRY.md"
    entries: dict[str, dict[str, str]] = {}
    if not registry_path.exists():
        return entries
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        match = _REGISTRY_ROW.match(line.strip())
        if not match:
            continue
        data = {k: v.strip() for k, v in match.groupdict().items()}
        entries[data["tool"]] = data
    return entries


def _doc_summary(tree: ast.AST) -> str:
    return (
        (ast.get_docstring(tree) or "").strip().splitlines()[0]
        if ast.get_docstring(tree)
        else ""
    )


def _has_main(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {"main", "_parse_args"}:
            return True
    return False


def _emit_entrypoint(path: Path) -> str:
    rel = path.as_posix()
    if rel.endswith(".py"):
        module = rel[:-3].replace("/", ".")
        return f"python -m {module}"
    return ""


def _feature_flags(text: str) -> dict[str, int]:
    lower = text.lower()
    return {
        "touches_db": int("sqlite" in lower or "lab_notebook" in lower),
        "touches_program_results": int("program_results" in lower),
        "touches_leaderboard": int("leaderboard" in lower),
        "touches_induction": int("induction" in lower),
        "touches_binding": int("binding" in lower),
        "touches_hellaswag": int("hellaswag" in lower),
        "touches_backfill": int("backfill" in lower or "backpopulate" in lower),
        "touches_ml": int(
            "predictor" in lower
            or "ml " in lower
            or "graphpredictor" in lower
            or "induction_metrics_v2" in lower
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the research.tools script surface."
    )
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    root = Path(args.root)
    registry = _load_registry()
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = (
            path.relative_to(Path.cwd()).as_posix()
            if path.is_absolute()
            else path.as_posix()
        )
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = ast.parse("")
        registered = registry.get(path.name, {})
        row: dict[str, object] = {
            "path": rel,
            "tool": path.name,
            "registered": int(path.name in registry),
            "registry_category": registered.get("category", ""),
            "registry_entry": registered.get("entry", ""),
            "registry_expiry": registered.get("expiry", ""),
            "entrypoint": _emit_entrypoint(path),
            "has_main": int(_has_main(tree)),
            "doc_summary": _doc_summary(tree),
        }
        row.update(_feature_flags(text))
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"rows={len(rows)} out={out_path}")


if __name__ == "__main__":
    main()
