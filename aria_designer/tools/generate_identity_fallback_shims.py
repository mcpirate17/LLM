#!/usr/bin/env python3
"""Convert legacy identity fallbacks into shared shim handlers.

A legacy identity fallback is detected when kernel_fallback.py contains:
- `return nn.Identity()` in build()
- reads `x = inputs["x"]`
- returns `{"y": x}`

These are replaced with:
    from runtime.fallback_templates import make_identity_handler
    ComponentHandler = make_identity_handler("<category>/<component>")
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _is_legacy_identity_fallback(text: str) -> bool:
    return (
        "return nn.Identity()" in text
        and 'x = inputs["x"]' in text
        and 'return {"y": x}' in text
        and "make_identity_handler(" not in text
    )


def _shim_text(component_type: str) -> str:
    return (
        f'"""Fallback kernel shim for {component_type}."""\n'
        "from runtime.fallback_templates import make_identity_handler\n\n"
        f'ComponentHandler = make_identity_handler("{component_type}")\n'
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--components-root",
        default=str(Path(__file__).resolve().parents[1] / "components"),
        help="Path to aria_designer/components",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag, runs in dry-run mode.",
    )
    args = parser.parse_args()

    root = Path(args.components_root)
    changed = 0
    for path in sorted(root.rglob("kernel_fallback.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not _is_legacy_identity_fallback(text):
            continue
        rel = path.relative_to(root)
        component_type = f"{rel.parts[0]}/{rel.parts[1]}"
        print(f"{'UPDATE' if args.apply else 'WOULD UPDATE'}: {path}")
        if args.apply:
            path.write_text(_shim_text(component_type), encoding="utf-8")
        changed += 1

    print(f"legacy_identity_fallbacks={'updated' if args.apply else 'found'}:{changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

