#!/usr/bin/env python3
"""Bootstrap mathspace component manifests from research/mathspaces.

Generates manifest.yaml + kernel.c + minimal contract tests under
components/math_space/<op>/.

Usage:
    python tools/bootstrap_mathspace_components.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

COMPONENTS_DIR = Path(__file__).resolve().parent.parent / "components"

SHAPE_RULE_TO_PORTS: Dict[str, Tuple[List[Dict], List[Dict]]] = {
    "identity": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
    "binary_broadcast": (
        [
            {"name": "a", "dtype": "tensor", "shape": ["B", "S", "D"]},
            {"name": "b", "dtype": "tensor", "shape": ["B", "S", "D"]},
        ],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
    "reduce_last": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "1"]}],
    ),
    "reduce_seq": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "1", "D"]}],
    ),
    "matmul": (
        [
            {"name": "a", "dtype": "tensor", "shape": ["B", "S", "D"]},
            {"name": "b", "dtype": "tensor", "shape": ["B", "D", "K"]},
        ],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "K"]}],
    ),
    "outer": (
        [
            {"name": "a", "dtype": "tensor", "shape": ["B", "S", "D"]},
            {"name": "b", "dtype": "tensor", "shape": ["B", "S", "D"]},
        ],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
}


def _config_keys_to_params(config_keys: Tuple[str, ...]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key in config_keys:
        params[key] = {"type": "integer", "default": 0}
    return params


def build_manifest(op) -> Dict[str, Any]:
    inputs, outputs = SHAPE_RULE_TO_PORTS.get(
        op.shape_rule,
        (
            [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
            [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        ),
    )

    tags = ["math_space"]
    if op.has_params:
        tags.append("learnable")
    if op.numerically_risky:
        tags.append("numerically_risky")

    manifest = {
        "id": op.name,
        "version": "1.0.0",
        "name": op.description or op.name.replace("_", " ").title(),
        "category": "math_space",
        "tags": tags,
        "status": "draft",
        "description": op.description,
        "inputs": inputs,
        "outputs": outputs,
        "params": _config_keys_to_params(op.config_keys),
        "implementation": {
            "native": "kernel.c",
            "rust": None,
            "cython": None,
            "python": None,
        },
        "performance": {
            "has_params": op.has_params,
            "param_formula": op.param_formula,
            "preserves_gradient": op.preserves_gradient,
            "numerically_risky": op.numerically_risky,
        },
        "limits": {
            "deterministic": True,
        },
    }
    return manifest


def generate_kernel_stub(component_id: str) -> str:
    return """
#include <stdint.h>
#include <string.h>

// Minimal kernel stub for {component_id}
// TODO: Implement high-performance kernel.

typedef struct {{
    float* data;
    int64_t* shape;
    int ndim;
    int dtype;  // 0=f32, 1=f16, 2=bf16
}} TensorView;

typedef struct {{
    const char* json_config;
}} ComponentConfig;

int component_validate(const ComponentConfig* config, char* error_buf, int buf_size) {{
    (void)config;
    const char* msg = "not implemented";
    if (error_buf && buf_size > 0) {{
        strncpy(error_buf, msg, (size_t)buf_size - 1);
        error_buf[buf_size - 1] = '\0';
    }}
    return -1;
}}

int component_forward(const TensorView* inputs, int n_inputs,
                      TensorView* outputs, int n_outputs,
                      const ComponentConfig* config) {{
    (void)inputs; (void)n_inputs; (void)outputs; (void)n_outputs; (void)config;
    return -1;
}}

void component_cleanup(void) {{
}}
""".lstrip().format(component_id=component_id)


def write_component(manifest: Dict[str, Any], dry_run: bool = False) -> Path:
    comp_dir = COMPONENTS_DIR / manifest["category"] / manifest["id"]
    manifest_path = comp_dir / "manifest.yaml"
    kernel_path = comp_dir / "kernel.c"
    test_dir = comp_dir / "tests"

    if dry_run:
        print(f"  [DRY RUN] Would create: {comp_dir}")
        return comp_dir

    comp_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, width=120)

    with open(kernel_path, "w") as f:
        f.write(generate_kernel_stub(manifest["id"]))

    test_path = test_dir / f"test_{manifest['id']}.py"
    with open(test_path, "w") as f:
        f.write(f'"""Contract tests for {manifest["id"]}."""\n')
        f.write("import yaml\n")
        f.write("from pathlib import Path\n\n\n")
        f.write("def test_manifest_valid():\n")
        f.write('    manifest_path = Path(__file__).parent.parent / "manifest.yaml"\n')
        f.write("    with open(manifest_path) as f:\n")
        f.write("        manifest = yaml.safe_load(f)\n")
        f.write(f'    assert manifest["id"] == "{manifest["id"]}"\n')
        f.write('    assert manifest["category"] == "math_space"\n')
        f.write('    assert manifest["version"] == "1.0.0"\n')
        f.write('    assert len(manifest["outputs"]) >= 1\n')
        f.write('    assert manifest["limits"]["deterministic"] is True\n')
        f.write('    assert "numerically_risky" in manifest["performance"]\n')

    return comp_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap mathspace components")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created")
    args = parser.parse_args()

    from research.synthesis.primitives import list_primitives, OpCategory
    from research.mathspaces.registry import register_all_mathspaces

    register_all_mathspaces()
    ops = list_primitives(OpCategory.MATH_SPACE)

    print(f"Mathspace ops: {len(ops)}")

    existing_ids = set()
    for path in COMPONENTS_DIR.rglob("manifest.yaml"):
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        if "id" in data:
            existing_ids.add(data["id"])

    created = 0
    for op in ops:
        if op.name in existing_ids:
            print(f"  [SKIP] {op.name} (already exists)")
            continue
        manifest = build_manifest(op)
        write_component(manifest, dry_run=args.dry_run)
        created += 1
        print(f"  [math_space] {op.name}")

    print(f"Created {created} components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
