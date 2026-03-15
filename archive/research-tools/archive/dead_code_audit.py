from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class DeadCodeAuditReport:
    workspace: str
    dashboard_orphans: List[str] = field(default_factory=list)
    python_possible_orphans: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _dashboard_component_orphans(root: Path) -> List[str]:
    comp_dir = root / "dashboard" / "src" / "components"
    if not comp_dir.exists():
        return []

    component_files = sorted(p for p in comp_dir.glob("*.js") if p.is_file())
    names = {p.stem for p in component_files}
    if not names:
        return []

    used: Set[str] = set()
    js_files = list((root / "dashboard" / "src").rglob("*.js"))
    import_re = re.compile(r"import\s+([A-Za-z_][A-Za-z0-9_]*)\s+from\s+['\"](\./|\.\./).*?['\"]")

    for file_path in js_files:
        content = _read_text(file_path)
        for match in import_re.finditer(content):
            symbol = match.group(1)
            if symbol in names:
                used.add(symbol)

    # Entry component App is consumed by index.js; keep list conservative.
    return sorted(name for name in names if name not in used)


def _python_possible_orphans(root: Path) -> List[str]:
    ignored_tokens = (
        "/__pycache__/",
        "/tests/",
        "/dashboard/",
        "/tools/",
        "/venv/",
        "/.venv/",
        "/site-packages/",
        "/node_modules/",
    )

    all_py_files = [
        p for p in root.rglob("*.py")
        if not any(token in p.as_posix() for token in ignored_tokens)
    ]

    corpus: Dict[Path, str] = {p: _read_text(p) for p in all_py_files}
    all_text = "\n".join(corpus.values())

    import_from_re = re.compile(r"from\s+([\.\w]+)\s+import\s+([^\n]+)")
    import_re = re.compile(r"import\s+([\.\w]+)")

    imported_modules: Set[str] = set()
    imported_symbols: Set[str] = set()

    for mod, sym_list in import_from_re.findall(all_text):
        imported_modules.add(mod.lstrip("."))
        for raw in sym_list.split(","):
            sym = raw.strip()
            if not sym:
                continue
            # handle aliases: "foo as bar"
            sym = sym.split(" as ")[0].strip()
            # avoid wildcard noise
            if sym == "*":
                continue
            imported_symbols.add(sym)
    for mod in import_re.findall(all_text):
        imported_modules.add(mod.lstrip("."))

    py_files = [p for p in all_py_files if p.name != "__init__.py"]

    orphans: List[str] = []

    for path in py_files:
        if path.name in {"__main__.py", "dead_code_audit.py"}:
            continue

        module_stem = path.stem
        rel = path.relative_to(root).as_posix().replace("/", ".")[:-3]  # e.g. scientist.runner
        package = rel.rsplit(".", 1)[0] if "." in rel else ""

        aliases = {
            rel,
            f"research.{rel}",
            module_stem,
            package,
            f"research.{package}" if package else "",
        }
        aliases = {a for a in aliases if a}

        referenced = (
            module_stem in imported_symbols
            or any(a in imported_modules for a in aliases)
        )

        if not referenced:
            orphans.append(path.relative_to(root).as_posix())

    return sorted(orphans)


def run_audit(workspace: Path) -> DeadCodeAuditReport:
    report = DeadCodeAuditReport(workspace=str(workspace))
    report.dashboard_orphans = _dashboard_component_orphans(workspace)
    report.python_possible_orphans = _python_possible_orphans(workspace)
    report.notes = [
        "This report is conservative and non-destructive.",
        "'python_possible_orphans' are candidates for manual review, not auto-delete targets.",
        "Dynamic imports and runtime entrypoints may create false positives.",
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-destructive dead code audit")
    parser.add_argument("--workspace", default="research", help="Workspace root path")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    report = run_audit(workspace)

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(f"Workspace: {report.workspace}")
        print(f"Dashboard orphan components: {len(report.dashboard_orphans)}")
        for name in report.dashboard_orphans:
            print(f"  - {name}")
        print(f"Python possible orphans: {len(report.python_possible_orphans)}")
        for name in report.python_possible_orphans[:30]:
            print(f"  - {name}")
        if len(report.python_possible_orphans) > 30:
            print(f"  ... and {len(report.python_possible_orphans) - 30} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
