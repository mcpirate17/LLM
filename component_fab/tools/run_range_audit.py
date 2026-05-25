"""Audit promoted fab components on sparse/long-range binding.

Re-scores the saved winners (``catalog/saved_winners.json``) with the
component-tier ``range_binding_gate`` — the capability nothing was tested on
before it existed (the old gates only exercised mixing to seq_len<=32). Surfaces
promoted components that bind locally but collapse across a long sparse gap.

Reconstruction fidelity caveat (honest): ``generate_module`` is driven purely by
``math_axes``. For block-template winners whose saved ``math_axes`` omit the
slot composition (``op_block_slot_b`` / ``op_block_slot_c`` /
``op_block_inner_template``), the rebuild falls back to default lanes and may NOT
match the named/promoted architecture. Those rows are flagged ``fidelity=low``
and their numbers must not be trusted as a measurement of the named component —
fix the catalog to store the full spec before relying on them.

Usage:
    python -m component_fab.tools.run_range_audit
    python -m component_fab.tools.run_range_audit --steps 600 --name-filter top_ar
    python -m component_fab.tools.run_range_audit --seeds 3 --include-refs
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from component_fab.generator.code_generator import generate_module
from component_fab.generator.memory_primitives import (
    PadicSurpriseMemoryLane,
    TropicalSurpriseMemoryLane,
)
from component_fab.generator.primitive_templates import (
    LinearStateSpaceLane,
    TropicalAttention,
)
from component_fab.harness.range_binding_probe import (
    DEFAULT_DISTANCES,
    range_binding_gate,
)
from component_fab.harness.standard_block import _LocalConvLane

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_WINNERS = _REPO / "component_fab" / "catalog" / "saved_winners.json"
DEFAULT_OUT = _REPO / "component_fab" / "catalog" / "range_audit_latest.json"

_BLOCK_SLOT_KEYS = ("op_block_slot_b", "op_block_slot_c", "op_block_inner_template")


def reconstruction_fidelity(math_axes: dict[str, Any]) -> tuple[str, str]:
    """Return (fidelity, reason). ``low`` when a block template is requested but
    its slot composition is absent from the axes (rebuild uses default lanes)."""
    if math_axes.get("op_block_template") and not any(
        math_axes.get(k) for k in _BLOCK_SLOT_KEYS
    ):
        return "low", "block_template without stored slot composition"
    return "ok", ""


def classify(effective_distance: int) -> str:
    if effective_distance >= 128:
        return "FULL-RANGE"
    if effective_distance >= 32:
        return "mid-range"
    if effective_distance >= 8:
        return "short"
    return "no-bind"


def _audit_one(
    make: Callable[[], nn.Module],
    *,
    dim: int,
    seeds: list[int],
    steps: int,
    distances: tuple[int, ...],
    device: str,
) -> dict[str, Any]:
    effs: list[int] = []
    curves: dict[int, list[float]] = {d: [] for d in distances}
    t0 = time.perf_counter()
    try:
        for seed in seeds:
            torch.manual_seed(seed)
            lane = make().to(device)
            res = range_binding_gate(
                lane, dim=dim, distances=distances, n_train_steps=steps, seed=seed
            )
            effs.append(res.effective_distance)
            for d, acc in res.per_distance_accuracy.items():
                curves[d].append(acc)
    except Exception as exc:  # noqa: BLE001 — record the failure, keep auditing
        return {"error": f"{type(exc).__name__}: {exc}"}
    eff = int(st.median(effs))
    return {
        "effective_distance": eff,
        "classification": classify(eff),
        "per_distance": {str(d): round(st.median(curves[d]), 3) for d in distances},
        "seconds_per_seed": round((time.perf_counter() - t0) / len(seeds), 1),
        "error": None,
    }


def _reference_targets(dim: int) -> dict[str, Callable[[], nn.Module]]:
    return {
        "REF:tropical_attention": lambda: TropicalAttention(dim),
        "REF:linear_ssm": lambda: LinearStateSpaceLane(dim),
        "REF:causal_conv": lambda: _LocalConvLane(dim),
        "INV:tropical_surprise": lambda: TropicalSurpriseMemoryLane(dim),
        "INV:padic_surprise": lambda: PadicSurpriseMemoryLane(dim),
    }


def run_audit(
    *,
    winners_path: Path = DEFAULT_WINNERS,
    dim: int = 32,
    seeds: list[int] | None = None,
    steps: int = 400,
    distances: tuple[int, ...] = DEFAULT_DISTANCES,
    include_refs: bool = True,
    name_filter: str = "",
    device: str | None = None,
) -> dict[str, Any]:
    seeds = seeds or [0, 1, 2]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    winners = json.loads(winners_path.read_text())["winners"]

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for win in winners:
        name = win["name"]
        if name in seen or (name_filter and name_filter not in name):
            continue
        seen.add(name)
        axes = dict(win["math_axes"])
        fidelity, reason = reconstruction_fidelity(axes)
        row = {
            "component": f"WIN:{name}",
            "fidelity": fidelity,
            "fidelity_reason": reason,
            **_audit_one(
                lambda axes=axes: generate_module(dict(axes), dim=dim),
                dim=dim,
                seeds=seeds,
                steps=steps,
                distances=distances,
                device=device,
            ),
        }
        results.append(row)

    if include_refs and not name_filter:
        for label, make in _reference_targets(dim).items():
            results.append(
                {
                    "component": label,
                    "fidelity": "ok",
                    "fidelity_reason": "",
                    **_audit_one(
                        make,
                        dim=dim,
                        seeds=seeds,
                        steps=steps,
                        distances=distances,
                        device=device,
                    ),
                }
            )

    results.sort(key=lambda r: -(r.get("effective_distance") or -1))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "dim": dim,
        "seeds": seeds,
        "train_steps": steps,
        "distances": list(distances),
        "results": results,
    }


def _print_table(report: dict[str, Any]) -> None:
    print(
        f"\n===== RANGE AUDIT (dim{report['dim']}, {len(report['seeds'])} seeds, "
        f"{report['train_steps']} steps, median) ====="
    )
    dists = report["distances"]
    header = f"{'component':54}{'eff_d':>6}{'fid':>5} {'class':>11}   per-distance acc"
    print(header)
    print("-" * (len(header) + 30))
    for row in report["results"]:
        comp = row["component"]
        if row.get("error"):
            print(f"{comp:54}{'ERR':>6}{row['fidelity']:>5}   {row['error']}")
            continue
        cs = "  ".join(f"{d}:{row['per_distance'][str(d)]:.2f}" for d in dists)
        print(
            f"{comp:54}{row['effective_distance']:>6}{row['fidelity']:>5} "
            f"{row['classification']:>11}   {cs}  [{row['seconds_per_seed']}s]"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="sparse/long-range binding audit of fab winners"
    )
    p.add_argument("--winners", type=Path, default=DEFAULT_WINNERS)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--no-refs", dest="include_refs", action="store_false")
    p.add_argument(
        "--name-filter", default="", help="Audit only winners whose name contains this."
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    report = run_audit(
        winners_path=args.winners,
        dim=args.dim,
        seeds=list(range(args.seeds)),
        steps=args.steps,
        include_refs=args.include_refs,
        name_filter=args.name_filter,
    )
    _print_table(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote: {args.out}")
    low = [r["component"] for r in report["results"] if r["fidelity"] == "low"]
    if low:
        print(f"NOTE: low-fidelity rebuilds (numbers unreliable): {', '.join(low)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
