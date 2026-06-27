"""Reproduce slot_dplr_run1_40m diagnostic plots that the loss dashboard can't show.

Two figures:

1. Binding-probe trajectory (``*_probe_traj.png``)
   The run's sidecar ``*_probe_step{N}.json`` files record the compositional
   binding battery (all_slots / two_plus / held_slot / held_class) at a few
   steps. Plotted against the native-100K "wall" reference (all_slots == 0.0)
   so the >0.1 "clearing the wall" threshold is visible.

2. Slot-prior-bias dynamics (``*_slot_bias.png``)
   ``slot_prior_bias`` is the lane's learnable per-head x per-slot additive bias
   on the write-route logits (added BEFORE the softmax over slots). It is the
   prior over *which slot a token gets written to*, independent of content. A
   large magnitude => the lane has developed a strong prior preference among
   slots (slot specialization); a flat bias => slots stay interchangeable.

   It is also the parameter holding the global ``w|max|`` in almost every step
   of the watch log ("... w|max| 9.4 b6.lane.slot_prior_bias"). These panels
   show that story from the real checkpoint tensors:
     A. max|slot_prior_bias| per block vs step (which block specializes hardest)
     B. JSONL w_max vs step, coloured by the block that owns the argmax
     C. per-slot head-mean bias of the dominant block (which slots win/lose)
     D. write-route selection entropy per block (collapse vs spread; max=log2 8)

Usage:
  python -m research.tools.plot_slot_dplr_diagnostics \
      --run slot_dplr_run1_40m --reports research/reports
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_STEP_RE = re.compile(r"step(\d+)")
_BLOCK_RE = re.compile(r"blocks\.(\d+)\.")


def _block_of(name: str) -> int:
    """Return the block index in a param name, or -1 if there is none."""
    m = _BLOCK_RE.search(name)
    return int(m.group(1)) if m else -1


def _probe_points(reports: Path, run: str) -> list[dict]:
    """Parse the ``*_probe_step{N}.json`` text sidecars into metric dicts."""
    points: list[dict] = []
    for path in sorted(glob.glob(str(reports / f"{run}_probe_step*.json"))):
        text = Path(path).read_text()
        rec: dict[str, float] = {}
        for key in ("step", "all_slots", "two_plus", "held_slot", "held_class"):
            m = re.search(rf"{key}=([0-9.]+)", text)
            if m:
                rec[key] = float(m.group(1))
        if "step" in rec:
            points.append(rec)
    return sorted(points, key=lambda r: r["step"])


def _plot_probe_traj(points: list[dict], run: str, out: Path) -> None:
    metrics = ["all_slots", "two_plus", "held_slot", "held_class"]
    colors = {
        "all_slots": "#d62728",
        "two_plus": "#1f77b4",
        "held_slot": "#2ca02c",
        "held_class": "#9467bd",
    }
    steps = [p["step"] for p in points]

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in metrics:
        ys = [p.get(m, np.nan) for p in points]
        ax.plot(steps, ys, "o-", lw=2, ms=9, color=colors[m], label=m)
        for x, y in zip(steps, ys):
            if np.isfinite(y):
                ax.annotate(
                    f"{y:.3f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )
    ax.axhline(
        0.0, color="k", ls="--", lw=1.2, label="native-100K wall (all_slots=0.0)"
    )
    ax.axhline(
        0.1, color="gray", ls=":", lw=1.2, label="clearing-the-wall threshold (~0.1)"
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("probe accuracy")
    ax.set_title(f"{run} — compositional binding probe trajectory")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out} ({len(points)} probe points)")


def _load_slot_bias(reports: Path, run: str):
    """Extract per-block slot_prior_bias (heads x slots) from every checkpoint."""

    def _step_of(path: str) -> int:
        m = _STEP_RE.search(path)
        if m is None:
            raise ValueError(f"no step token in checkpoint name: {path}")
        return int(m.group(1))

    ckpts = sorted(
        glob.glob(str(reports / "native_adaptive_hydra_ckpts" / f"{run}_*step*.pt")),
        key=_step_of,
    )
    steps: list[int] = []
    per_block: dict[int, list[np.ndarray]] = {}
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        sd = ck["model_state_dict"]
        keys = sorted(k for k in sd if k.endswith("lane.slot_prior_bias"))
        if not keys:
            continue
        steps.append(int(ck["step"]))
        for k in keys:
            per_block.setdefault(_block_of(k), []).append(sd[k].float().numpy())
    # per_block[blk] -> array (n_steps, heads, slots)
    arrs = {b: np.stack(v) for b, v in per_block.items()}
    return np.asarray(steps), arrs


def _jsonl_wmax(reports: Path, run: str):
    paths = glob.glob(str(reports / f"{run}_*.jsonl"))
    steps, wmax, blocks = [], [], []
    for path in paths:
        for line in Path(path).open():
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("event") != "step" or o.get("w_max") is None:
                continue
            steps.append(o["step"])
            wmax.append(o["w_max"])
            blocks.append(_block_of(o.get("w_max_param") or ""))
    order = np.argsort(steps)
    return (
        np.asarray(steps)[order],
        np.asarray(wmax)[order],
        np.asarray(blocks)[order],
    )


def _entropy_bits(prior: np.ndarray) -> np.ndarray:
    """Selection entropy (bits) of softmax over slots, per step.

    prior: (n_steps, heads, slots). Softmax over slots, average over heads.
    """
    z = prior - prior.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    ent = -(p * np.log2(np.clip(p, 1e-12, 1.0))).sum(axis=-1)  # (steps, heads)
    return ent.mean(axis=1)


def _block_label(b: int, dom: int) -> str:
    return f"b{b}" + (" (dom)" if b == dom else "")


def _panel_maxbias(ax, steps, arrs, blocks, dom, cmap) -> None:
    for b in blocks:
        ax.plot(
            steps,
            np.abs(arrs[b]).reshape(len(steps), -1).max(axis=1),
            color=cmap(b % 10),
            lw=1.8,
            label=_block_label(b, dom),
        )
    ax.set_title("A. max |slot_prior_bias| per block")
    ax.set_xlabel("step")
    ax.set_ylabel("max |bias|")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)


def _panel_wmax(fig, ax, jl) -> None:
    js, jw, jb = jl
    sc = ax.scatter(js, jw, c=jb, cmap="tab10", s=6, vmin=0, vmax=9)
    ax.set_title("B. global w|max| (JSONL), coloured by owning block")
    ax.set_xlabel("step")
    ax.set_ylabel("w|max|")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax).set_label("block holding argmax")


def _panel_perslot(ax, steps, arrs, dom, n_slots, cmap) -> None:
    headmean = arrs[dom].mean(axis=1)  # (steps, slots)
    for sidx in range(n_slots):
        ax.plot(
            steps,
            headmean[:, sidx],
            color=cmap(sidx % 10),
            lw=1.8,
            label=f"slot {sidx}",
        )
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title(f"C. b{dom} per-slot bias (head-mean) — which slots win")
    ax.set_xlabel("step")
    ax.set_ylabel("bias")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)


def _panel_entropy(ax, steps, arrs, blocks, dom, n_slots, cmap) -> None:
    for b in blocks:
        ax.plot(
            steps,
            _entropy_bits(arrs[b]),
            color=cmap(b % 10),
            lw=1.8,
            label=_block_label(b, dom),
        )
    ax.axhline(
        np.log2(n_slots), color="k", ls="--", lw=1.0, label=f"uniform = log2({n_slots})"
    )
    ax.set_title("D. write-route selection entropy (low = specialized)")
    ax.set_xlabel("step")
    ax.set_ylabel("entropy (bits)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)


def _plot_slot_bias(steps, arrs, jl, run: str, out: Path) -> None:
    blocks = sorted(arrs)
    n_slots = next(iter(arrs.values())).shape[-1]
    dom = max(blocks, key=lambda b: np.abs(arrs[b]).max())
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f"{run} — slot_prior_bias dynamics (lane write-route prior; "
        f"dominant block = b{dom})",
        fontsize=13,
    )
    _panel_maxbias(axes[0, 0], steps, arrs, blocks, dom, cmap)
    _panel_wmax(fig, axes[0, 1], jl)
    _panel_perslot(axes[1, 0], steps, arrs, dom, n_slots, cmap)
    _panel_entropy(axes[1, 1], steps, arrs, blocks, dom, n_slots, cmap)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out} ({len(steps)} checkpoints, dominant block b{dom})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="slot_dplr_run1_40m")
    ap.add_argument("--reports", type=Path, default=Path("research/reports"))
    a = ap.parse_args()

    points = _probe_points(a.reports, a.run)
    if points:
        _plot_probe_traj(points, a.run, a.reports / f"{a.run}_probe_traj.png")
    else:
        print("no probe sidecars found")

    steps, arrs = _load_slot_bias(a.reports, a.run)
    if arrs:
        jl = _jsonl_wmax(a.reports, a.run)
        _plot_slot_bias(steps, arrs, jl, a.run, a.reports / f"{a.run}_slot_bias.png")
    else:
        print("no slot_prior_bias checkpoints found")


if __name__ == "__main__":
    main()
