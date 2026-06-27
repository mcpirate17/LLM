"""Probe branch gates and weighted outputs for native_recip_slot checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from research.defaults import PROJECT_ROOT, VOCAB_SIZE
from research.tools._scaling_lanes import NativeAdaptiveReciprocalSlotDeltaLane
from research.tools.native_gate_floor_utils import (
    DEFAULT_NATIVE_GATE_FLOORS,
    parse_float_csv,
)
from research.tools.native_adaptive_hydra_train import (
    _make_loader,
    _prepare_batch,
)
from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm


def _rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt().item())


def _patch_lane(
    lane: NativeAdaptiveReciprocalSlotDeltaLane,
    *,
    block_idx: int,
    native_gate_floor: float,
    records: list[dict[str, float | int]],
) -> None:
    def patched_forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_weights = torch.softmax(self.gate(x), dim=-1)
        floor = min(max(float(native_gate_floor), 0.0), 1.0)
        weights = raw_weights.clone()
        weights[..., 0] = floor + (1.0 - floor) * raw_weights[..., 0]
        weights[..., 1:] = (1.0 - floor) * raw_weights[..., 1:]
        if x.is_cuda:
            with torch.autocast(device_type=x.device.type, enabled=False):
                native_raw = self.native(x.float()).to(x.dtype)
        else:
            native_raw = self.native(x)
        reciprocal_raw = self.reciprocal(x)
        slot_raw = self.slot(x)

        native = self._rms_normalize_branch(native_raw)
        reciprocal = self._rms_normalize_branch(reciprocal_raw)
        slot = self._rms_normalize_branch(slot_raw)

        weighted_native = weights[..., 0:1] * native
        weighted_reciprocal = weights[..., 1:2] * reciprocal
        weighted_slot = weights[..., 2:3] * slot
        total = weighted_native + weighted_reciprocal + weighted_slot

        records.append(
            {
                "block": block_idx,
                "native_gate_floor": floor,
                "raw_native_gate_mean": float(
                    raw_weights[..., 0].detach().float().mean()
                ),
                "raw_reciprocal_gate_mean": float(
                    raw_weights[..., 1].detach().float().mean()
                ),
                "raw_slot_gate_mean": float(
                    raw_weights[..., 2].detach().float().mean()
                ),
                "native_gate_mean": float(weights[..., 0].detach().float().mean()),
                "reciprocal_gate_mean": float(
                    weights[..., 1].detach().float().mean()
                ),
                "slot_gate_mean": float(weights[..., 2].detach().float().mean()),
                "native_gate_min": float(weights[..., 0].detach().float().min()),
                "native_raw_rms": _rms(native_raw),
                "reciprocal_raw_rms": _rms(reciprocal_raw),
                "slot_raw_rms": _rms(slot_raw),
                "native_weighted_rms": _rms(weighted_native),
                "reciprocal_weighted_rms": _rms(weighted_reciprocal),
                "slot_weighted_rms": _rms(weighted_slot),
                "total_rms": _rms(total),
            }
        )
        return total

    lane.forward = MethodType(patched_forward, lane)


def _mean_rows(rows: list[dict[str, float | int]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in rows[0]:
        if key == "block":
            continue
        out[key] = sum(float(row[key]) for row in rows) / len(rows)
    return out


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    records: list[dict[str, float | int]] = []
    factory = _build_lane_factory("native_adaptive_reciprocal_slot_delta")
    model = _build_tinylm(
        factory,
        dim=args.dim,
        n_blocks=args.n_blocks,
        vocab_size=args.vocab_size,
        max_seq_len=max(args.seq_len, 1024),
        use_ffn=True,
    ).to(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    lanes = [
        module
        for module in model.modules()
        if isinstance(module, NativeAdaptiveReciprocalSlotDeltaLane)
    ]
    native_gate_floors = args.native_gate_floors
    if len(native_gate_floors) != len(lanes):
        raise ValueError(
            f"--native-gate-floors has {len(native_gate_floors)} values but "
            f"the model has {len(lanes)} native/reciprocal/slot lanes"
        )
    for block_idx, lane in enumerate(lanes):
        lane.native_gate_floor = float(native_gate_floors[block_idx])
        _patch_lane(
            lane,
            block_idx=block_idx,
            native_gate_floor=float(native_gate_floors[block_idx]),
            records=records,
        )

    loader_args = argparse.Namespace(
        hydra_root=args.hydra_root,
        tokenizer=args.tokenizer,
        dataset=args.dataset,
        batch=args.batch,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        steps=args.batches,
        require_sources=False,
    )
    loader = _make_loader(loader_args, dataset=args.dataset, seed=args.seed)
    losses: list[float] = []
    autocast_enabled = str(args.device).startswith("cuda")
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
    ):
        for _ in range(args.batches):
            batch = next(loader)
            ids, labels = _prepare_batch(
                batch, vocab_size=args.vocab_size, device=args.device
            )
            logits = model(ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
            losses.append(float(loss.detach().float().item()))
    if hasattr(loader, "close"):
        loader.close()

    if not records:
        raise RuntimeError("No native_adaptive_reciprocal_slot_delta lane records found")

    by_block: dict[str, dict[str, float]] = {}
    for block in sorted({int(row["block"]) for row in records}):
        rows = [row for row in records if int(row["block"]) == block]
        by_block[str(block)] = _mean_rows(rows)

    loss_mean = sum(losses) / len(losses)
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "gate_floor": NativeAdaptiveReciprocalSlotDeltaLane.GATE_FLOOR,
        "native_gate_floors": list(native_gate_floors),
        "batches": args.batches,
        "loss_mean": loss_mean,
        "ppl": math.exp(loss_mean),
        "overall": _mean_rows(records),
        "by_block": by_block,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, args.out_plot)
    return summary


def _plot(summary: dict[str, Any], out_plot: Path) -> None:
    by_block = summary["by_block"]
    blocks = sorted(int(block) for block in by_block)
    native_gate_floors = summary.get("native_gate_floors")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for branch in ("native", "reciprocal", "slot"):
        axes[0].plot(
            blocks,
            [by_block[str(block)][f"{branch}_gate_mean"] for block in blocks],
            marker="o",
            label=branch,
        )
    if isinstance(native_gate_floors, list) and len(native_gate_floors) == len(blocks):
        axes[0].plot(
            blocks,
            [float(value) for value in native_gate_floors],
            color="black",
            linestyle="--",
            linewidth=1,
            label="native floor",
        )
    else:
        axes[0].axhline(
            float(summary["gate_floor"]),
            color="black",
            linestyle="--",
            linewidth=1,
            label="floor",
        )
    axes[0].set_ylabel("gate mean")
    axes[0].set_title(
        f"Branch gate means at checkpoint step {summary['checkpoint_step']}"
    )
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    for branch in ("native", "reciprocal", "slot"):
        axes[1].plot(
            blocks,
            [by_block[str(block)][f"{branch}_weighted_rms"] for block in blocks],
            marker="o",
            label=f"{branch} weighted RMS",
        )
    axes[1].set_xlabel("block")
    axes[1].set_ylabel("weighted output RMS")
    axes[1].set_title("Normalized branch contribution RMS")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-plot", type=Path, required=True)
    parser.add_argument("--dataset", default="codex_ffw60_chat30_pleias10_local")
    parser.add_argument("--dim", type=int, default=640)
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--hydra-root", type=Path, default=PROJECT_ROOT / "HYDRA")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--native-gate-floors",
        type=parse_float_csv,
        default=DEFAULT_NATIVE_GATE_FLOORS,
        help="Comma-separated native branch effective floor per block.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_probe(args)
    print(f"Saved report to: {args.out_json}")
    print(f"Saved plot to: {args.out_plot}")
    print(
        "overall gate means: "
        f"native={summary['overall']['native_gate_mean']:.4f}, "
        f"reciprocal={summary['overall']['reciprocal_gate_mean']:.4f}, "
        f"slot={summary['overall']['slot_gate_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
