from __future__ import annotations

import ast
import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARIA_ROOT = PROJECT_ROOT / "aria_designer"
KERNEL_FALLBACKS = sorted(ARIA_ROOT.glob("components/**/kernel_fallback.py"))
IMPORT_HYGIENE_TARGETS = [
    *KERNEL_FALLBACKS,
    *sorted((ARIA_ROOT / "runtime").glob("*.py")),
    *sorted((ARIA_ROOT / "tools").glob("*.py")),
    *sorted((ARIA_ROOT / "api" / "app" / "routers").glob("*.py")),
    PROJECT_ROOT / "research" / "scientist" / "api_routes" / "programs_bp.py",
]

FORBIDDEN_ROOTS = {"runtime", "components"}


def _legacy_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FORBIDDEN_ROOTS:
                    matches.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in FORBIDDEN_ROOTS:
                matches.append(node.module)
    return matches


def test_no_legacy_top_level_imports() -> None:
    offenders = {
        str(path.relative_to(PROJECT_ROOT)): imports
        for path in IMPORT_HYGIENE_TARGETS
        for imports in [_legacy_imports(path)]
        if imports
    }
    assert not offenders, offenders


def test_all_kernel_fallback_modules_import_cleanly() -> None:
    failures: dict[str, str] = {}
    for path in KERNEL_FALLBACKS:
        module_name = ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - failure path only
            failures[module_name] = repr(exc)
    assert not failures, failures
