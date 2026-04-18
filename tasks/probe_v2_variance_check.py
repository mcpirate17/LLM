"""Phase 1 variance check for probe v2 integration (2026-04-18).

Reuses the architecture zoo + probe runners from probe_calibration_sweep.py.
Runs each (arch × probe × seed) combination and reports coefficient of
variation per architectural family. Gate: CoV < 0.05 inside families means
the step budget is stable enough to anchor a scoring gate.

Fixed budgets under test:
  - induction v2: 500 mixed-gap steps (recommended by PROBE_CALIBRATION doc)
  - binding:      2400 steps (bumped from doc's 1600 — attn_2l/hybrid_2l
                  showed convergence anomalies at 1600)

Outputs:
  tasks/probe_calibration_results/variance_induction_v2.csv
  tasks/probe_calibration_results/variance_binding_2400.csv
  tasks/probe_calibration_results/variance_summary.md
"""

from __future__ import annotations

import csv
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_calibration_sweep import (  # type: ignore[import]
    ARCHITECTURES,
    DEVICE,
    run_binding_curriculum,
    run_induction,
)

RESULTS_DIR = Path(__file__).resolve().parent / "probe_calibration_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INDUCTION_CSV = RESULTS_DIR / "variance_induction_v2.csv"
BINDING_CSV = RESULTS_DIR / "variance_binding_2400.csv"
SUMMARY_MD = RESULTS_DIR / "variance_summary.md"

SEEDS: List[int] = [11, 23, 47, 89, 131]
INDUCTION_STEPS = 500
BINDING_STEPS = 2400

FAMILY_OF = {
    "attn_1l": "attention",
    "attn_2l": "attention",
    "attn_4l": "attention",
    "conv3_2l": "conv",
    "conv7_2l": "conv",
    "conv7_4l": "conv",
    "ssm_2l": "ssm",
    "ssm_4l": "ssm",
    "rwkv_2l": "rwkv",
    "hybrid_2l": "hybrid",
    "hybrid_4l": "hybrid",
}


def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _init_csv(path: Path, fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def _append_csv(path: Path, row: Dict[str, object], fieldnames: List[str]) -> None:
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def run_induction_variance() -> List[Dict[str, object]]:
    fields = ["arch", "family", "seed", "auc", "max_gap_acc", "elapsed_s", "status"]
    _init_csv(INDUCTION_CSV, fields)
    rows: List[Dict[str, object]] = []
    for arch_name, factory in ARCHITECTURES.items():
        for seed in SEEDS:
            _seed_all(seed)
            base = factory()
            t0 = time.perf_counter()
            res = run_induction(
                base,
                n_train_steps=INDUCTION_STEPS,
                train_mode="mixed",
            )
            row = {
                "arch": arch_name,
                "family": FAMILY_OF[arch_name],
                "seed": seed,
                "auc": res["auc"],
                "max_gap_acc": res.get("max_gap_acc", 0.0),
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "status": res["status"],
            }
            rows.append(row)
            _append_csv(INDUCTION_CSV, row, fields)
            print(
                f"[ind] {arch_name} seed={seed} auc={res['auc']:.4f} ({row['elapsed_s']}s)"
            )
            del base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def run_binding_variance() -> List[Dict[str, object]]:
    fields = ["arch", "family", "seed", "auc", "elapsed_s", "status"]
    _init_csv(BINDING_CSV, fields)
    rows: List[Dict[str, object]] = []
    for arch_name, factory in ARCHITECTURES.items():
        for seed in SEEDS:
            _seed_all(seed)
            base = factory()
            t0 = time.perf_counter()
            res = run_binding_curriculum(base, n_train_steps=BINDING_STEPS)
            row = {
                "arch": arch_name,
                "family": FAMILY_OF[arch_name],
                "seed": seed,
                "auc": res["auc"],
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "status": res["status"],
            }
            rows.append(row)
            _append_csv(BINDING_CSV, row, fields)
            print(
                f"[bind] {arch_name} seed={seed} auc={res['auc']:.4f} ({row['elapsed_s']}s)"
            )
            del base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def _summarize(rows: List[Dict[str, object]], probe_name: str) -> str:
    by_arch: Dict[str, List[float]] = {}
    for r in rows:
        by_arch.setdefault(r["arch"], []).append(float(r["auc"]))  # type: ignore[arg-type]

    arch_lines = [
        f"### {probe_name} — per-architecture variance",
        "",
        "| arch | family | n | mean | median | std | CoV | min | max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arch, vals in by_arch.items():
        mean = statistics.mean(vals)
        med = statistics.median(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        cov = std / mean if mean > 1e-6 else float("inf")
        arch_lines.append(
            f"| `{arch}` | {FAMILY_OF[arch]} | {len(vals)} | "
            f"{mean:.4f} | {med:.4f} | {std:.4f} | "
            f"{cov:.3f} | {min(vals):.4f} | {max(vals):.4f} |"
        )

    # Family-level rollup
    by_family: Dict[str, List[float]] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(float(r["auc"]))  # type: ignore[arg-type]
    fam_lines = [
        "",
        f"### {probe_name} — per-family separation",
        "",
        "| family | n | min | median | max |",
        "|---|---|---|---|---|",
    ]
    for fam, vals in sorted(by_family.items()):
        fam_lines.append(
            f"| {fam} | {len(vals)} | {min(vals):.4f} | "
            f"{statistics.median(vals):.4f} | {max(vals):.4f} |"
        )

    return "\n".join(arch_lines + fam_lines)


def write_summary(
    ind_rows: List[Dict[str, object]], bind_rows: List[Dict[str, object]]
) -> None:
    parts = [
        "# Probe v2 variance check (Phase 1)",
        "",
        f"Seeds: {SEEDS}",
        f"Induction v2: {INDUCTION_STEPS} mixed-gap steps",
        f"Binding:      {BINDING_STEPS} steps",
        "",
        "## Gate criteria",
        "",
        "- Attention family CoV < 0.05 at high-AUC archs (attn_2l, attn_4l)",
        "- Family ordering holds: attention ≥ hybrid > conv > ssm/rwkv",
        "- No arch flips family rank across seeds",
        "",
        _summarize(ind_rows, "Induction v2"),
        "",
        _summarize(bind_rows, "Binding"),
    ]
    SUMMARY_MD.write_text("\n".join(parts))


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Archs: {len(ARCHITECTURES)}")
    print(f"Total runs: {2 * len(ARCHITECTURES) * len(SEEDS)}")

    t0 = time.perf_counter()
    print("\n=== Induction v2 (500 mixed-gap steps) ===")
    ind_rows = run_induction_variance()
    print(f"\nInduction total: {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    print("\n=== Binding (2400 steps) ===")
    bind_rows = run_binding_variance()
    print(f"\nBinding total: {time.perf_counter() - t1:.1f}s")

    write_summary(ind_rows, bind_rows)
    print(f"\nSummary: {SUMMARY_MD}")
    print(f"Total wall: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
