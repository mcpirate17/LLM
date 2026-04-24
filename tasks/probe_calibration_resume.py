"""Resume script for the probe calibration sweep.

The initial sweep crashed on a broken HybridConvAttnLM (nn.Parameter inside
nn.ModuleDict). The build was fixed in probe_calibration_sweep.py, but the
running process had the broken class loaded, so the hybrid rows never landed.

This script reruns only the missing combinations:
  - induction main: hybrid_2l, hybrid_4l × full step/mode grid
  - induction extended: hybrid_2l + any others the main run missed
    (the main run crashed before running ssm_4l/rwkv_2l/conv7_4l in extended)
  - binding curriculum: hybrid_2l, hybrid_4l
  - associative recall: hybrid_2l, hybrid_4l

Appends to existing CSVs.
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from pathlib import Path
from typing import Set, Tuple

# Import the (fixed) harness — must be after path setup for the import to find
# the local tasks package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tasks.probe_calibration_sweep import (
    ARCHITECTURES,
    INDUCTION_CSV,
    INDUCTION_EXTENDED_CSV,
    BINDING_CURR_CSV,
    AR_CSV,
    _init_csv,
    _param_count,
    run_induction,
    run_binding_curriculum,
    run_associative_recall,
    DEVICE,
)

import torch


def _seen_keys(path: Path, key_cols: Tuple[str, ...]) -> Set[Tuple[str, ...]]:
    seen: Set[Tuple[str, ...]] = set()
    if not path.exists():
        return seen
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            seen.add(tuple(str(row.get(c, "")) for c in key_cols))
    return seen


def resume_induction():
    fieldnames = [
        "arch",
        "n_params",
        "n_train_steps",
        "train_mode",
        "auc",
        "max_gap_acc",
        "min_gap_acc",
        "acc_4",
        "acc_8",
        "acc_16",
        "acc_32",
        "acc_64",
        "first_loss",
        "last_loss",
        "status",
        "elapsed_s",
    ]
    f, w = _init_csv(INDUCTION_CSV, fieldnames)
    seen = _seen_keys(INDUCTION_CSV, ("arch", "n_train_steps", "train_mode"))
    archs = ["hybrid_2l", "hybrid_4l"]
    steps_grid = (100, 250, 500, 1000, 2000)
    modes = ("fixed8", "mixed")
    print("\n=== Resume induction (hybrid) ===")
    for arch in archs:
        base = ARCHITECTURES[arch]()
        n_params = _param_count(base)
        for steps in steps_grid:
            for mode in modes:
                key = (arch, str(steps), mode)
                if key in seen:
                    continue
                try:
                    res = run_induction(base, n_train_steps=steps, train_mode=mode)
                except Exception as e:
                    res = {
                        "status": f"exc: {e}",
                        "auc": 0.0,
                        "gap_acc": {},
                        "max_gap_acc": 0.0,
                        "min_gap_acc": 0.0,
                        "first_loss": 0.0,
                        "last_loss": 0.0,
                        "elapsed_s": 0.0,
                    }
                gap_acc = res.get("gap_acc", {})
                row = {
                    "arch": arch,
                    "n_params": n_params,
                    "n_train_steps": steps,
                    "train_mode": mode,
                    "auc": res["auc"],
                    "max_gap_acc": res.get("max_gap_acc", 0.0),
                    "min_gap_acc": res.get("min_gap_acc", 0.0),
                    "acc_4": gap_acc.get(4, 0.0),
                    "acc_8": gap_acc.get(8, 0.0),
                    "acc_16": gap_acc.get(16, 0.0),
                    "acc_32": gap_acc.get(32, 0.0),
                    "acc_64": gap_acc.get(64, 0.0),
                    "first_loss": res.get("first_loss", 0.0),
                    "last_loss": res.get("last_loss", 0.0),
                    "status": res["status"],
                    "elapsed_s": res["elapsed_s"],
                }
                w.writerow(row)
                f.flush()
                print(
                    f"  {arch:<12} steps={steps:4} {mode:<6} "
                    f"auc={res['auc']:.3f} "
                    f"peak={res.get('max_gap_acc', 0):.3f} "
                    f"time={res['elapsed_s']}s"
                )
        del base
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    f.close()


def resume_induction_extended():
    fieldnames = [
        "arch",
        "n_params",
        "n_train_steps",
        "train_mode",
        "auc",
        "max_gap_acc",
        "min_gap_acc",
        "acc_4",
        "acc_8",
        "acc_16",
        "acc_32",
        "acc_64",
        "last_loss",
        "status",
        "elapsed_s",
    ]
    f, w = _init_csv(INDUCTION_EXTENDED_CSV, fieldnames)
    seen = _seen_keys(INDUCTION_EXTENDED_CSV, ("arch", "n_train_steps", "train_mode"))
    target_archs = [
        "hybrid_2l",
        "ssm_4l",
        "rwkv_2l",
        "conv7_4l",  # missed when main crashed
    ]
    steps_grid = (50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400)
    print("\n=== Resume induction extended ===")
    for arch in target_archs:
        base = ARCHITECTURES[arch]()
        n_params = _param_count(base)
        for steps in steps_grid:
            key = (arch, str(steps), "mixed")
            if key in seen:
                continue
            try:
                res = run_induction(base, n_train_steps=steps, train_mode="mixed")
            except Exception as e:
                res = {
                    "status": f"exc: {e}",
                    "auc": 0.0,
                    "gap_acc": {},
                    "max_gap_acc": 0.0,
                    "min_gap_acc": 0.0,
                    "last_loss": 0.0,
                    "elapsed_s": 0.0,
                }
            gap_acc = res.get("gap_acc", {})
            row = {
                "arch": arch,
                "n_params": n_params,
                "n_train_steps": steps,
                "train_mode": "mixed",
                "auc": res["auc"],
                "max_gap_acc": res.get("max_gap_acc", 0.0),
                "min_gap_acc": res.get("min_gap_acc", 0.0),
                "acc_4": gap_acc.get(4, 0.0),
                "acc_8": gap_acc.get(8, 0.0),
                "acc_16": gap_acc.get(16, 0.0),
                "acc_32": gap_acc.get(32, 0.0),
                "acc_64": gap_acc.get(64, 0.0),
                "last_loss": res.get("last_loss", 0.0),
                "status": res["status"],
                "elapsed_s": res["elapsed_s"],
            }
            w.writerow(row)
            f.flush()
            print(
                f"  {arch:<12} steps={steps:4} auc={res['auc']:.3f} "
                f"peak={res.get('max_gap_acc', 0):.3f} "
                f"time={res['elapsed_s']}s"
            )
        del base
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    f.close()


def resume_binding_curriculum():
    fieldnames = [
        "arch",
        "n_params",
        "n_train_steps",
        "auc",
        "acc_4",
        "acc_8",
        "acc_16",
        "acc_32",
        "first_loss",
        "last_loss",
        "status",
        "elapsed_s",
    ]
    f, w = _init_csv(BINDING_CURR_CSV, fieldnames)
    seen = _seen_keys(BINDING_CURR_CSV, ("arch", "n_train_steps"))
    archs = ["hybrid_2l", "hybrid_4l"]
    steps_grid = (200, 400, 800, 1600)
    print("\n=== Resume binding curriculum (hybrid) ===")
    for arch in archs:
        base = ARCHITECTURES[arch]()
        n_params = _param_count(base)
        for steps in steps_grid:
            if (arch, str(steps)) in seen:
                continue
            try:
                res = run_binding_curriculum(base, n_train_steps=steps)
            except Exception as e:
                res = {
                    "status": f"exc: {e}",
                    "auc": 0.0,
                    "dist_acc": {},
                    "first_loss": 0.0,
                    "last_loss": 0.0,
                    "elapsed_s": 0.0,
                }
            dist_acc = res.get("dist_acc", {})
            row = {
                "arch": arch,
                "n_params": n_params,
                "n_train_steps": steps,
                "auc": res["auc"],
                "acc_4": dist_acc.get(4, 0.0),
                "acc_8": dist_acc.get(8, 0.0),
                "acc_16": dist_acc.get(16, 0.0),
                "acc_32": dist_acc.get(32, 0.0),
                "first_loss": res.get("first_loss", 0.0),
                "last_loss": res.get("last_loss", 0.0),
                "status": res["status"],
                "elapsed_s": res["elapsed_s"],
            }
            w.writerow(row)
            f.flush()
            print(
                f"  {arch:<12} steps={steps:4} auc={res['auc']:.3f} "
                f"time={res['elapsed_s']}s"
            )
        del base
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    f.close()


def resume_ar():
    fieldnames = [
        "arch",
        "n_params",
        "n_train_steps",
        "auc",
        "final_acc",
        "first_loss",
        "last_loss",
        "status",
        "elapsed_s",
    ]
    f, w = _init_csv(AR_CSV, fieldnames)
    seen = _seen_keys(AR_CSV, ("arch", "n_train_steps"))
    archs = ["hybrid_2l", "hybrid_4l"]
    steps_grid = (500, 1000, 2000)
    print("\n=== Resume associative recall (hybrid) ===")
    for arch in archs:
        base = ARCHITECTURES[arch]()
        n_params = _param_count(base)
        for steps in steps_grid:
            if (arch, str(steps)) in seen:
                continue
            try:
                res = run_associative_recall(base, n_train_steps=steps)
            except Exception as e:
                res = {
                    "status": f"exc: {e}",
                    "auc": 0.0,
                    "final_acc": 0.0,
                    "first_loss": 0.0,
                    "last_loss": 0.0,
                    "elapsed_s": 0.0,
                }
            row = {
                "arch": arch,
                "n_params": n_params,
                "n_train_steps": steps,
                "auc": res["auc"],
                "final_acc": res["final_acc"],
                "first_loss": res.get("first_loss", 0.0),
                "last_loss": res.get("last_loss", 0.0),
                "status": res["status"],
                "elapsed_s": res["elapsed_s"],
            }
            w.writerow(row)
            f.flush()
            print(
                f"  {arch:<12} steps={steps:4} auc={res['auc']:.3f} "
                f"final={res['final_acc']:.3f} time={res['elapsed_s']}s"
            )
        del base
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    f.close()


if __name__ == "__main__":
    t0 = time.perf_counter()
    for step in (
        resume_induction,
        resume_induction_extended,
        resume_binding_curriculum,
        resume_ar,
    ):
        try:
            step()
        except Exception as e:
            print(f"{step.__name__} failed: {e}")
            traceback.print_exc()
    print(f"\nResume complete in {(time.perf_counter() - t0) / 60:.1f} min")
