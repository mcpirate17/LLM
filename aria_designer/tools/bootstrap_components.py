#!/usr/bin/env python3
"""Bootstrap component manifests from research/ primitives and arch_builder.

Reads PRIMITIVE_REGISTRY from research/synthesis/primitives.py and morphological
box dimensions from research/morphological_box.py, then generates manifest.yaml
for each component under aria_designer/components/.

Usage:
    python -m tools.bootstrap_components [--dry-run]
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COMPONENTS_DIR = Path(__file__).resolve().parent.parent / "components"

# ── Category mapping from OpCategory to designer categories ──────────

OP_CATEGORY_MAP = {
    "elementwise_unary": "math",
    "elementwise_binary": "math",
    "reduction": "math",
    "linear_algebra": "linear_algebra",
    "structural": "structural",
    "parameterized": "linear_algebra",
    "sequence": "sequence",
    "frequency": "frequency",
    "math_space": "math_space",
    "functional": "functional",
}

# ── Shape rule to port shape mapping ──────────────────────────────────

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
    "transpose_seq_dim": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "D", "S"]}],
    ),
    "split": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [
            {"name": "y0", "dtype": "tensor", "shape": ["B", "S", "D_split"]},
            {"name": "y1", "dtype": "tensor", "shape": ["B", "S", "D_split"]},
        ],
    ),
    "concat": (
        [
            {"name": "a", "dtype": "tensor", "shape": ["B", "S", "D_a"]},
            {"name": "b", "dtype": "tensor", "shape": ["B", "S", "D_b"]},
        ],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D_out"]}],
    ),
    "linear": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D_in"]}],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D_out"]}],
    ),
    "roll": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
    "gather": (
        [
            {"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]},
            {"name": "idx", "dtype": "index", "shape": ["B", "S"]},
        ],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
    "scatter": (
        [
            {"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]},
            {"name": "idx", "dtype": "index", "shape": ["B", "S"]},
        ],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
    "rfft": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [{"name": "y", "dtype": "complex_tensor", "shape": ["B", "S_half", "D"]}],
    ),
    "irfft": (
        [{"name": "x", "dtype": "complex_tensor", "shape": ["B", "S_half", "D"]}],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
    "sort": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [
            {"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]},
            {"name": "idx", "dtype": "index", "shape": ["B", "S"]},
        ],
    ),
    "unsort": (
        [
            {"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]},
            {"name": "idx", "dtype": "index", "shape": ["B", "S"]},
        ],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
    "cumulative": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
    "softmax": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
    "causal_mask": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
    "scale": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
    "bias": (
        [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
    ),
}

# ── Morphological box dimension → designer category mapping ──────────

MORPH_DIM_CATEGORY = {
    "token_representation": "representation",
    "weight_storage": "linear_algebra",
    "token_mixing": "mixing",
    "channel_mixing": "channel_mixing",
    "compute_routing": "routing",
    "architecture_topology": "topology",
    "normalization": "normalization",
    "positional_encoding": "positional",
}


def _config_keys_to_params(config_keys: Tuple[str, ...]) -> Dict[str, Any]:
    """Convert PrimitiveOp config_keys to manifest param definitions."""
    params = {}
    defaults = {
        "n_splits": {
            "type": "integer",
            "default": 2,
            "constraints": {"min": 2, "max": 8},
        },
        "n_heads": {
            "type": "integer",
            "default": 8,
            "constraints": {"min": 1, "max": 128},
        },
        "window_size": {
            "type": "integer",
            "default": 64,
            "constraints": {"min": 4, "max": 4096},
        },
        "out_dim": {
            "type": "integer",
            "default": None,
            "description": "Output dim (null=same as input)",
        },
        "kernel_scale": {
            "type": "float",
            "default": 1.0,
            "constraints": {"min": 0.01, "max": 100.0},
        },
        "n_iters": {
            "type": "integer",
            "default": 3,
            "constraints": {"min": 1, "max": 20},
        },
        "damping": {
            "type": "float",
            "default": 0.5,
            "constraints": {"min": 0.0, "max": 1.0},
        },
        "n": {"type": "integer", "default": 2, "constraints": {"min": 1, "max": 4}},
        "m": {"type": "integer", "default": 4, "constraints": {"min": 2, "max": 8}},
        "block_size": {
            "type": "integer",
            "default": 32,
            "constraints": {"min": 8, "max": 128},
        },
        "block_density": {
            "type": "float",
            "default": 0.25,
            "constraints": {"min": 0.05, "max": 1.0},
        },
    }
    for key in config_keys:
        params[key] = defaults.get(key, {"type": "integer", "default": 0})
    return params


def build_primitive_manifest(op) -> Dict[str, Any]:
    """Build a manifest dict from a PrimitiveOp."""
    category = OP_CATEGORY_MAP.get(
        op.category.value if hasattr(op.category, "value") else str(op.category), "math"
    )
    shape_key = op.shape_rule
    inputs, outputs = SHAPE_RULE_TO_PORTS.get(
        shape_key,
        (
            [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
            [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
        ),
    )

    tags = [op.category.value if hasattr(op.category, "value") else str(op.category)]
    if op.has_params:
        tags.append("learnable")
    if op.numerically_risky:
        tags.append("numerically_risky")

    manifest = {
        "id": op.name,
        "version": "1.0.0",
        "name": op.description or op.name.replace("_", " ").title(),
        "category": category,
        "tags": tags,
        "status": "approved" if category != "math_space" else "draft",
        "description": op.description,
        "inputs": inputs,
        "outputs": outputs,
        "params": _config_keys_to_params(op.config_keys),
        "implementation": {
            "native": "kernel.c" if category == "math_space" else None,
            "rust": None,
            "cython": None,
            "python": "kernel_fallback.py",
        },
        "performance": {
            "has_params": op.has_params,
            "param_formula": op.param_formula,
            "preserves_gradient": op.preserves_gradient,
            "numerically_risky": op.numerically_risky,
        },
    }
    if category == "math_space":
        manifest["limits"] = {"deterministic": True}
    return manifest


def build_morph_manifest(dim_name: str, option) -> Dict[str, Any]:
    """Build a manifest dict from a morphological box dimension option."""
    category = MORPH_DIM_CATEGORY.get(dim_name, "blocks")
    tags = list(option.tags) if option.tags else []
    tags.append(dim_name)

    # Standard ports for arch-level components
    inputs = [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}]
    outputs = [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}]

    manifest = {
        "id": option.name,
        "version": "1.0.0",
        "name": option.name.replace("_", " ").title(),
        "category": category,
        "tags": tags,
        "status": "approved",
        "description": option.description,
        "inputs": inputs,
        "outputs": outputs,
        "params": {},
        "implementation": {
            "native": None,
            "rust": None,
            "cython": None,
            "python": "kernel_fallback.py",
        },
        "performance": {
            "has_params": True,
            "param_formula": "D*D",
            "preserves_gradient": True,
            "numerically_risky": False,
        },
        "constraints": {
            "incompatible_with": list(option.incompatible_with)
            if option.incompatible_with
            else [],
        },
    }
    return manifest


def write_component(manifest: Dict[str, Any], dry_run: bool = False) -> Path:
    """Write a component manifest without generating placeholder runtimes."""
    comp_dir = COMPONENTS_DIR / manifest["category"] / manifest["id"]
    manifest_path = comp_dir / "manifest.yaml"

    if dry_run:
        print(f"  [DRY RUN] Would create: {comp_dir}")
        return comp_dir

    comp_dir.mkdir(parents=True, exist_ok=True)
    manifest_to_write = deepcopy(manifest)
    implementation = dict(manifest_to_write.get("implementation") or {})
    implementation["native"] = None
    implementation["python"] = None
    manifest_to_write["implementation"] = implementation

    with open(manifest_path, "w") as f:
        yaml.dump(
            manifest_to_write, f, default_flow_style=False, sort_keys=False, width=120
        )

    return comp_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap component manifests from research/"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be created"
    )
    return parser.parse_args()


def _load_research_sources():
    from research.mathspaces.registry import register_all_mathspaces
    from research.morphological_box import DIMENSIONS
    from research.synthesis.primitives import PRIMITIVE_REGISTRY

    return PRIMITIVE_REGISTRY, DIMENSIONS, register_all_mathspaces


def _print_header() -> None:
    print("=" * 60)
    print("Bootstrapping components from research/")
    print("=" * 60)


def _bootstrap_primitives(primitive_registry, dry_run: bool) -> int:
    print(f"\n--- Primitive Operations ({len(primitive_registry)} ops) ---")
    prim_count = 0
    for name, op in primitive_registry.items():
        if name == "input":
            continue
        manifest = build_primitive_manifest(op)
        write_component(manifest, dry_run=dry_run)
        prim_count += 1
        print(f"  [{manifest['category']:>16s}] {name}")
    return prim_count


def _bootstrap_morph_options(
    dimensions,
    seen_ids: set[str],
    dry_run: bool,
) -> int:
    print(f"\n--- Morphological Box Dimensions ({len(dimensions)} dims) ---")
    morph_count = 0
    for dim in dimensions:
        print(f"\n  Dimension: {dim.name}")
        for option in dim.options:
            if option.name in seen_ids:
                print(f"    [SKIP] {option.name} (already exists as primitive)")
                continue
            seen_ids.add(option.name)
            manifest = build_morph_manifest(dim.name, option)
            write_component(manifest, dry_run=dry_run)
            morph_count += 1
            print(f"    [{manifest['category']:>16s}] {option.name}")
    return morph_count


def _io_component_manifests() -> list[dict[str, Any]]:
    return [
        {
            "id": "input",
            "version": "1.0.0",
            "name": "Input",
            "category": "io",
            "tags": ["io", "source"],
            "status": "approved",
            "description": "Model input (token embeddings)",
            "inputs": [],
            "outputs": [{"name": "y", "dtype": "tensor", "shape": ["B", "S", "D"]}],
            "params": {
                "dim": {
                    "type": "integer",
                    "default": 256,
                    "constraints": {"min": 1, "max": 65536},
                },
                "vocab_size": {"type": "integer", "default": 32000},
            },
            "implementation": {
                "native": None,
                "rust": None,
                "cython": None,
                "python": "kernel_fallback.py",
            },
            "performance": {"has_params": True, "param_formula": "vocab_size * D"},
        },
        {
            "id": "output_head",
            "version": "1.0.0",
            "name": "Output Head",
            "category": "io",
            "tags": ["io", "sink"],
            "status": "approved",
            "description": "Model output projection to vocabulary",
            "inputs": [{"name": "x", "dtype": "tensor", "shape": ["B", "S", "D"]}],
            "outputs": [
                {"name": "logits", "dtype": "tensor", "shape": ["B", "S", "V"]}
            ],
            "params": {
                "vocab_size": {"type": "integer", "default": 32000},
                "tie_weights": {"type": "boolean", "default": True},
            },
            "implementation": {
                "native": None,
                "rust": None,
                "cython": None,
                "python": "kernel_fallback.py",
            },
            "performance": {"has_params": True, "param_formula": "D * vocab_size"},
        },
    ]


def _bootstrap_io_components(seen_ids: set[str], dry_run: bool) -> int:
    print("\n--- IO Components ---")
    io_count = 0
    for manifest in _io_component_manifests():
        if manifest["id"] not in seen_ids:
            write_component(manifest, dry_run=dry_run)
            seen_ids.add(manifest["id"])
            io_count += 1
            print(f"  [{manifest['category']:>16s}] {manifest['id']}")
    return io_count


def _print_summary(
    prim_count: int, morph_count: int, io_count: int, dry_run: bool
) -> None:
    print(f"\n{'=' * 60}")
    print(
        f"Total: {prim_count} primitives + {morph_count} morph options + {io_count} io = {prim_count + morph_count + io_count} components"
    )
    if dry_run:
        print("(DRY RUN — no files written)")
    print(f"{'=' * 60}")


def main():
    args = _parse_args()
    primitive_registry, dimensions, register_all_mathspaces = _load_research_sources()
    _print_header()
    register_all_mathspaces()
    prim_count = _bootstrap_primitives(primitive_registry, args.dry_run)
    seen_ids = set(primitive_registry.keys())
    morph_count = _bootstrap_morph_options(dimensions, seen_ids, args.dry_run)
    io_count = _bootstrap_io_components(seen_ids, args.dry_run)
    _print_summary(prim_count, morph_count, io_count, args.dry_run)


if __name__ == "__main__":
    main()
