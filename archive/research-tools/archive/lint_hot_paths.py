#!/usr/bin/env python3
"""Performance lint for synthesis/ hot paths.

Flags patterns that cause GPU→CPU synchronization, Python-level loops
over tensor dimensions, or O(S²) allocations in compiler ops.

Run: python -m research.tools.lint_hot_paths
Exit code: 0 if clean, 1 if violations found.
"""

import re
import sys
from pathlib import Path

SYNTHESIS_DIR = Path(__file__).resolve().parents[1] / "synthesis"

# Files containing forward-pass hot paths
HOT_PATH_FILES = [
    "compiler.py",
    "compiler_ops_math.py",
    "compiler_ops_attention.py",
    "compiler_ops_routing.py",
    "compiler_ops_mathspaces.py",
    "true_routing_ops.py",
    "ir_executor.py",
]

# Patterns that indicate GPU→CPU sync in hot paths
SYNC_PATTERNS = [
    (
        re.compile(r"\.item\(\)"),
        "GPU→CPU sync via .item()",
        # Allowed contexts: telemetry gated by collect_telemetry, init code, cached sampling
        [
            "collect_telemetry",
            "_profile",
            "_sparse_density_sampled",
            "_sparse_density_counter",
            "def __init__",
            "# init",
        ],
    ),
    (
        re.compile(r"\.tolist\(\)"),
        "tensor→Python list conversion",
        # Allowed: bincount().tolist() in sequential dispatch (CPU-only path),
        # IRExecutor init pre-conversion (intentional optimization)
        ["_moe_sequential_dispatch", "expert_counts", "Pre-convert", "def __init__"],
    ),
]

# Patterns that indicate Python loops over tensor dimensions
LOOP_PATTERNS = [
    (
        re.compile(r"for\s+\w+\s+in\s+(range\(.*\)|drop_positions|expert_ids)"),
        "Python loop potentially over tensor dimension",
        # Allowed: init loops, small fixed loops, cached weight stacking,
        # CPU-only sequential dispatch, IRExecutor init/forward (pre-optimized)
        [
            "def __init__",
            "def _init_",
            "def _moe_get_stacked",
            "def _moe_sequential",
            "def _dispatch_to_experts",
            "range(top_k)",
            "range(n_iters)",
            "range(2, K)",
            "range(K)",
            "range(n_ways)",
            "range(n_experts)",
            "range(n_actual)",
            "range(max_depth)",
            "range(len(self.op_codes))",
            "range(n_nodes)",
            "isqrt",
            "# init",
            "CHUNK",
            "class IRExecutor",
        ],
    ),
]


def lint_file(filepath: Path) -> list[str]:
    """Check a file for performance antipatterns. Returns list of violations."""
    violations = []
    try:
        lines = filepath.read_text().splitlines()
    except Exception:
        return []

    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for pattern, description, allowed_contexts in SYNC_PATTERNS + LOOP_PATTERNS:
            if pattern.search(line):
                # Check if any allowed context appears nearby (within 20 lines)
                context_window = "\n".join(lines[max(0, line_no - 20) : line_no + 5])
                if any(ctx in context_window for ctx in allowed_contexts):
                    continue
                violations.append(
                    f"  {filepath.name}:{line_no}: {description}\n    {stripped}"
                )

    return violations


def main() -> int:
    total_violations = []
    for filename in HOT_PATH_FILES:
        filepath = SYNTHESIS_DIR / filename
        if filepath.exists():
            violations = lint_file(filepath)
            total_violations.extend(violations)

    if total_violations:
        print(
            f"PERF LINT: {len(total_violations)} violation(s) in synthesis/ hot paths:\n"
        )
        for v in total_violations:
            print(v)
        print(
            "\nFix: move .item()/.tolist() out of hot path, or gate behind "
            "collect_telemetry flag.\n"
            "If intentional, add an allowed context comment nearby."
        )
        return 1

    print("PERF LINT: synthesis/ hot paths clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
