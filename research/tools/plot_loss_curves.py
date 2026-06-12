"""Professional training dynamics visualization for JSONL logs.

Renders high-quality multi-panel charts:
- Loss (Linear & Log)
- PPL (Linear & Log)
- Learning Rate
- Gradient Norm (Log scale + health markers)
- MoR Depth / MoD Utilization
- Grad Clip Pressure

Features:
- Faint gray raw data with bold EMA-smoothed lines.
- Dual-scale panels (normal and log) for primary metrics.
- Multi-run comparison with automated merging of resumed runs.
- Robust metric auto-detection.

Usage:
  python -m research.tools.plot_loss_curves research/reports/mor_*.jsonl --out reports/dashboard.png
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path):
    """Loads all metrics from a JSONL file.

    Resumed runs that reset their step counter (loaded from a checkpoint but
    log ``first_step==1``) are shifted onto the absolute training timeline by
    offsetting every step by ``loaded_step``. Runs that already log absolute
    steps (``first_step > loaded_step``) are left untouched.
    """
    data = OrderedDict()
    label = path.stem
    step_offset = 0

    for ln in path.open():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue

        # Priority 1: run_label + resume offset from start event
        ev = r.get("event")
        if ev == "start":
            label = r.get("run_label") or label
            loaded = r.get("loaded_step")
            first = r.get("first_step")
            # Counter reset on resume: steps restart at 1 (or <= loaded_step)
            # instead of continuing from the checkpoint's step.
            if loaded and (first is None or first <= loaded):
                step_offset = loaded

        # Priority 2: Any event with a 'step' key can contain metrics
        s = r.get("step")
        if s is not None:
            s += step_offset
            # Metrics to extract directly
            for k in ["loss", "lr", "grad_norm", "tokens_per_sec", "vram_gb"]:
                v = r.get(k)
                if v is not None:
                    data.setdefault(k, {})[s] = v

            # Nested metrics: PPL
            ppl = (r.get("eval") or {}).get("ppl")
            if ppl is not None:
                data.setdefault("ppl", {})[s] = ppl

            # MoR Depth: can be under 'depth' or 'mor'
            depth_obj = r.get("depth") or r.get("mor")
            if isinstance(depth_obj, dict):
                md = depth_obj.get("mean_depth")
                if md is not None:
                    data.setdefault("depth", {})[s] = md

            # Grad clip pressure
            pre = r.get("grad_norm_pre_clip")
            post = r.get("grad_norm")
            if pre and post:
                data.setdefault("grad_pressure", {})[s] = post / max(pre, 1e-9)

        # Legacy standalone eval events
        if ev == "eval":
            es = r.get("step")
            ppl = r.get("ppl")
            if ppl is not None and es is not None:
                data.setdefault("ppl", {})[es + step_offset] = ppl

    return label, data


def _ema(xs: np.ndarray, alpha: float) -> np.ndarray:
    """Standard EMA implementation."""
    if len(xs) == 0:
        return xs
    out = np.zeros_like(xs)
    m = xs[0]
    for i, x in enumerate(xs):
        # Handle cases where m might be None or NaN from a previous step
        if not np.isfinite(m):
            m = x
        m = alpha * x + (1.0 - alpha) * m
        out[i] = m
    return out


def _parse_args() -> argparse.Namespace:
    """CLI definition."""
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="JSONL files or globs")
    ap.add_argument("--out", type=Path, required=True, help="Output PNG path")
    ap.add_argument("--smooth", type=float, default=0.05, help="EMA alpha (0-1)")
    ap.add_argument("--ppl", action="store_true", help="Show PPL panel")
    ap.add_argument("--lr", action="store_true", help="Show LR panel")
    ap.add_argument("--grad", action="store_true", help="Show Gradient Norm panel")
    ap.add_argument("--depth", action="store_true", help="Show MoR depth panel")
    ap.add_argument("--log-loss", action="store_true", help="Add Log Loss panel")
    ap.add_argument("--log-ppl", action="store_true", help="Add Log PPL panel")
    ap.add_argument("--all", action="store_true", help="Show all detected metrics")
    ap.add_argument("--title", default="Training Dynamics Dashboard")
    return ap.parse_args()


def _aggregate_runs(run_args: list[str]):
    """Expand globs, load each file, and merge metrics by run_label."""
    paths: list[Path] = []
    for r in run_args:
        matched = sorted(glob.glob(r))
        paths += [Path(p) for p in matched] if matched else [Path(r)]

    runs: OrderedDict[str, dict] = OrderedDict()
    available_metrics: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        label, data = _load(p)
        if not data:
            continue
        run_data = runs.setdefault(label, {})
        for k, steps in data.items():
            run_data.setdefault(k, {}).update(steps)
            available_metrics.add(k)
    return runs, available_metrics


def _select_panels(a: argparse.Namespace, available_metrics: set[str]) -> list[str]:
    """Resolve which panels to render from flags + detected metrics."""
    user_panels = a.ppl or a.lr or a.grad or a.depth or a.log_loss or a.log_ppl

    panels = ["loss"]
    if a.log_loss or not user_panels:
        panels.append("log_loss")

    if a.ppl or (not user_panels and "ppl" in available_metrics):
        panels.append("ppl")
        if a.log_ppl:
            panels.append("log_ppl")

    if a.lr or (not user_panels and "lr" in available_metrics):
        panels.append("lr")

    if a.grad or (not user_panels and "grad_norm" in available_metrics):
        panels.append("grad_norm")

    if a.depth or (not user_panels and "depth" in available_metrics):
        panels.append("depth")

    if a.all:
        for m in sorted(
            available_metrics - {"loss", "ppl", "lr", "grad_norm", "depth"}
        ):
            if m not in panels:
                panels.append(m)
    return panels


def _draw_run_series(
    ax,
    metric_data: dict,
    metric_key: str,
    is_log: bool,
    color: str,
    label: str | None,
    smooth: float,
    gray: bool,
) -> None:
    """Plot one run's raw + EMA-smoothed series on an axis."""
    steps = np.array(sorted(metric_data.keys()))
    values = np.array([metric_data[s] for s in steps])

    # Filter non-finite (and non-positive for log scales)
    mask = np.isfinite(values)
    if is_log:
        mask &= values > 0
    steps, values = steps[mask], values[mask]
    if len(steps) == 0:
        return

    # Raw data (faint)
    ax.plot(steps, values, color="gray" if gray else color, alpha=0.15, linewidth=0.7)

    # Smoothed data (bold)
    smooth_alpha = smooth if len(steps) > 100 else 0.2
    ax.plot(steps, _ema(values, smooth_alpha), color=color, linewidth=2.0, label=label)

    # Markers for sparse data
    if metric_key == "ppl" or len(steps) < 50:
        ax.scatter(steps, values, color=color, s=20, alpha=0.5)


def _draw_panel(
    ax,
    panel_name: str,
    runs: OrderedDict,
    a: argparse.Namespace,
    first: bool,
    last: bool,
) -> None:
    """Render a single metric panel across all runs."""
    metric_key = panel_name.replace("log_", "")
    is_log = panel_name.startswith("log_") or panel_name == "grad_norm"

    for j, (label, r_data) in enumerate(runs.items()):
        metric_data = r_data.get(metric_key, {})
        if metric_data:
            _draw_run_series(
                ax,
                metric_data,
                metric_key,
                is_log,
                f"C{j % 10}",
                label if first else None,
                a.smooth,
                gray=len(runs) > 1,
            )

    ylabel = panel_name.replace("_", " ").title()
    if ylabel == "Depth":
        ylabel = "MoR Depth"
    ax.set_ylabel(ylabel)

    ax.grid(True, which="both", alpha=0.2)
    if is_log:
        ax.set_yscale("log")

    if panel_name == "grad_norm":
        ax.axhline(
            y=1.0,
            color="green",
            linestyle="--",
            alpha=0.3,
            label="Healthy Baseline" if first else None,
        )
        ax.axhline(y=10.0, color="orange", linestyle="--", alpha=0.3)

    if first:
        ax.set_title(a.title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", frameon=True, fontsize="small")
    if last:
        ax.set_xlabel("Step")


def main() -> None:
    a = _parse_args()

    runs, available_metrics = _aggregate_runs(a.runs)
    if not runs:
        print("No data found in provided files.")
        return

    panels = _select_panels(a, available_metrics)
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)
    for i, panel_name in enumerate(panels):
        _draw_panel(axes[i][0], panel_name, runs, a, first=i == 0, last=i == n - 1)

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"Saved dashboard to {a.out} ({len(runs)} runs, {n} panels)")


if __name__ == "__main__":
    main()
