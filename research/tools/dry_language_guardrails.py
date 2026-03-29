#!/usr/bin/env python3
"""DRY + language-principle guardrails with baseline-aware strict mode.

This script checks:
1) Non-canonical component types (alias usage) in workflow/example JSON files.
2) Duplicate exported C symbols across aria_core CPU source files.
3) Duplicate kernel fallback templates in aria_designer components.
4) Python loop hotspots in runtime dispatch functions expected to be native/tensorized.

Default mode is report-only (always exit 0).
Use --strict to fail when metrics regress beyond baseline.
Use --update-baseline to snapshot current metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = (
    REPO_ROOT / "research" / "tools" / "baselines" / "dry_guardrails_baseline.json"
)


ALIASES: Dict[str, str] = {
    "normalization/layernorm": "normalization/layernorm_pre",
    "normalization/rmsnorm": "normalization/rmsnorm_pre",
    "sequence/selective_scan": "linear_algebra/selective_scan",
}


@dataclass(slots=True)
class CanonicalViolation:
    file: str
    node_id: str
    component_type: str
    canonical_type: str


def _collect_json_files() -> List[Path]:
    targets = [
        REPO_ROOT / "aria_designer" / "ui" / "public" / "examples",
        REPO_ROOT / "aria_designer" / "workflows",
    ]
    out: List[Path] = []
    for root in targets:
        if not root.exists():
            continue
        out.extend(sorted(root.rglob("*.json")))
    return out


def _canonical_component_violations() -> List[CanonicalViolation]:
    violations: List[CanonicalViolation] = []
    for file_path in _collect_json_files():
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, list):
            nodes = payload
        elif isinstance(payload, dict):
            nodes = payload.get("nodes")
        else:
            continue
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict):
                continue
            component_type = node.get("component_type")
            if not isinstance(component_type, str):
                continue
            canonical = ALIASES.get(component_type)
            if canonical is None:
                continue
            node_id = str(node.get("id", "<unknown>"))
            violations.append(
                CanonicalViolation(
                    file=str(file_path.relative_to(REPO_ROOT)),
                    node_id=node_id,
                    component_type=component_type,
                    canonical_type=canonical,
                )
            )
    return violations


def _duplicate_c_symbols() -> Dict[str, List[str]]:
    cpu_root = REPO_ROOT / "aria_core" / "src" / "cpu"
    files = sorted(list(cpu_root.glob("*.c")) + list(cpu_root.glob("*.cpp")))
    symbol_re = re.compile(
        r"^\s*(?:void|int|float|double|int32_t|int64_t|size_t|bool)\s+"
        r"(aria_[A-Za-z0-9_]+)\s*\(",
    )

    owners: Dict[str, set[str]] = {}
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            m = symbol_re.match(line)
            if not m:
                continue
            # Skip declarations/prototypes ending with ';'
            if line.strip().endswith(";"):
                continue
            sym = m.group(1)
            owners.setdefault(sym, set()).add(str(path.relative_to(REPO_ROOT)))
    duplicates: Dict[str, List[str]] = {}
    for sym, paths in owners.items():
        if len(paths) > 1:
            duplicates[sym] = sorted(paths)
    return duplicates


def _normalize_fallback(text: str, slug: str) -> str:
    lines = []
    for line in text.splitlines():
        s = line.rstrip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        # Remove top-level module docstring line noise
        if (
            "Python fallback kernel for" in s
            or "Auto-generated Python fallback kernel for" in s
        ):
            continue
        lines.append(s)
    normalized = "\n".join(lines)
    # Normalize obvious component-specific names
    normalized = normalized.replace(slug, "__COMPONENT__")
    camel = "".join(part.capitalize() for part in slug.split("_"))
    normalized = normalized.replace(camel, "__COMPONENT__")
    normalized = re.sub(r"class\s+[A-Za-z0-9_]+Module", "class __MODULE__", normalized)
    normalized = re.sub(r"class\s+ComponentHandler", "class __HANDLER__", normalized)
    return normalized.strip()


def _duplicate_fallback_templates() -> Dict[str, List[str]]:
    roots = [REPO_ROOT / "aria_designer" / "components"]
    files: List[Path] = []
    for root in roots:
        files.extend(sorted(root.rglob("kernel_fallback.py")))

    buckets: Dict[str, List[str]] = {}
    for path in files:
        slug = path.parent.name
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        # Thin shims that delegate to shared runtime templates are intentionally
        # duplicated wrappers and should not count as DRY violations.
        if "runtime.fallback_templates" in text:
            continue
        norm = _normalize_fallback(text, slug)
        if not norm:
            continue
        digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        buckets.setdefault(digest, []).append(str(path.relative_to(REPO_ROOT)))

    # Keep only multi-file duplicates.
    return {k: sorted(v) for k, v in buckets.items() if len(v) > 1}


def _extract_function_block(text: str, fn_name: str) -> str:
    lines = text.splitlines()
    start = None
    base_indent = 0
    for i, line in enumerate(lines):
        if re.match(rf"^\s*def\s+{re.escape(fn_name)}\s*\(", line):
            start = i
            base_indent = len(line) - len(line.lstrip(" "))
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= base_indent and line.lstrip().startswith("def "):
            end = j
            break
    return "\n".join(lines[start:end])


def _hotpath_python_loop_count() -> int:
    dispatch = REPO_ROOT / "aria_designer" / "runtime" / "dispatch.py"
    try:
        text = dispatch.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0
    count = 0
    for fn in ("tropical_attention", "tropical_gate"):
        block = _extract_function_block(text, fn)
        if not block:
            continue
        has_loop = ("for " in block) and ("range(" in block)
        has_numpy = "np." in block
        if has_loop and has_numpy:
            count += 1
    return count


def _load_baseline(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in metrics.items():
        if isinstance(v, int):
            out[k] = v
    return out


def _write_baseline(path: Path, metrics: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "metrics": metrics,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="Fail on regressions against baseline."
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write current metrics to baseline file.",
    )
    parser.add_argument(
        "--baseline-file", default=str(DEFAULT_BASELINE), help="Path to baseline JSON."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable output."
    )
    args = parser.parse_args()

    canonical_violations = _canonical_component_violations()
    duplicate_symbols = _duplicate_c_symbols()
    duplicate_fallback = _duplicate_fallback_templates()
    hotpath_loops = _hotpath_python_loop_count()

    duplicate_fallback_files = sum(len(v) for v in duplicate_fallback.values())

    metrics = {
        "canonical_alias_occurrences": len(canonical_violations),
        "duplicate_c_symbols": len(duplicate_symbols),
        "fallback_duplicate_groups": len(duplicate_fallback),
        "fallback_duplicate_files": duplicate_fallback_files,
        "hotpath_python_loops": hotpath_loops,
    }

    baseline_path = Path(args.baseline_file)
    baseline = _load_baseline(baseline_path)

    regressions: List[Tuple[str, int, int]] = []
    for key, cur in metrics.items():
        limit = baseline.get(key, 0)
        if cur > limit:
            regressions.append((key, cur, limit))

    if args.update_baseline:
        _write_baseline(baseline_path, metrics)

    if args.json:
        print(
            json.dumps(
                {
                    "metrics": metrics,
                    "baseline": baseline,
                    "regressions": [
                        {"metric": k, "current": cur, "baseline": base}
                        for (k, cur, base) in regressions
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("DRY/Language Guardrails Report")
        print(
            f"- canonical alias occurrences: {metrics['canonical_alias_occurrences']}"
        )
        print(f"- duplicate C symbols: {metrics['duplicate_c_symbols']}")
        print(
            f"- duplicate fallback template groups: {metrics['fallback_duplicate_groups']}"
        )
        print(
            f"- duplicate fallback template files: {metrics['fallback_duplicate_files']}"
        )
        print(f"- hotpath python loops: {metrics['hotpath_python_loops']}")
        if canonical_violations:
            print("\nTop canonical-name violations:")
            for v in canonical_violations[:20]:
                print(
                    f"  {v.file}: node={v.node_id} uses {v.component_type} "
                    f"(canonical: {v.canonical_type})"
                )
        if duplicate_symbols:
            print("\nDuplicate C symbols:")
            for sym, paths in sorted(duplicate_symbols.items()):
                print(f"  {sym}")
                for p in paths:
                    print(f"    - {p}")
        if regressions:
            print("\nRegressions vs baseline:")
            for key, cur, base in regressions:
                print(f"  {key}: current={cur} baseline={base}")

    if args.strict and regressions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
