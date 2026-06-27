"""CPU synthetic gate probe for native reciprocal slot checkpoints.

This is a zero-training monitor: it feeds small synthetic inline, induction,
binding, and associative-recall prompts through saved checkpoints and records
raw/effective branch gates by task and difficulty. It is meant to diagnose
whether the branch router responds to capability-shaped inputs while the main
CUDA training run continues separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from research.defaults import VOCAB_SIZE
from research.tools._scaling_lanes import NativeAdaptiveReciprocalSlotDeltaLane
from research.tools.native_gate_floor_utils import (
    DEFAULT_NATIVE_GATE_FLOORS,
    parse_float_csv,
)
from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm


TOKEN_LO = 100
TOKEN_HI = 900


@dataclass(frozen=True)
class ProbeBatch:
    task: str
    difficulty: int
    ids: torch.Tensor
    labels: torch.Tensor


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"step(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt().item())


def _patch_lane(
    lane: NativeAdaptiveReciprocalSlotDeltaLane,
    *,
    block_idx: int,
    records: list[dict[str, float | int | str]],
    task: str,
    difficulty: int,
) -> None:
    def patched_forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_weights = torch.softmax(self.gate(x), dim=-1)
        native_floor = float(getattr(self, "native_gate_floor", self.GATE_FLOOR))
        native_floor = min(max(native_floor, 0.0), 1.0)
        weights = raw_weights.clone()
        weights[..., 0] = native_floor + (1.0 - native_floor) * raw_weights[..., 0]
        weights[..., 1:] = (1.0 - native_floor) * raw_weights[..., 1:]
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
                "task": task,
                "difficulty": difficulty,
                "block": block_idx,
                "raw_native_gate": float(raw_weights[..., 0].float().mean()),
                "raw_reciprocal_gate": float(raw_weights[..., 1].float().mean()),
                "raw_slot_gate": float(raw_weights[..., 2].float().mean()),
                "effective_native_gate": float(weights[..., 0].float().mean()),
                "effective_reciprocal_gate": float(weights[..., 1].float().mean()),
                "effective_slot_gate": float(weights[..., 2].float().mean()),
                "native_weighted_rms": _rms(weighted_native),
                "reciprocal_weighted_rms": _rms(weighted_reciprocal),
                "slot_weighted_rms": _rms(weighted_slot),
                "total_rms": _rms(total),
            }
        )
        return total

    lane.forward = MethodType(patched_forward, lane)


def _labels_for_last(ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    labels = torch.full_like(ids, -100)
    labels[:, -1] = targets
    return labels


def _make_inline(batch: int, difficulty: int, device: torch.device) -> ProbeBatch:
    seq_len = 24 + difficulty * 16
    starts = torch.arange(batch, device=device).unsqueeze(1) * 7 + 101
    pattern = (starts + torch.arange(seq_len, device=device).unsqueeze(0)) % 251
    ids = pattern + TOKEN_LO
    targets = (ids[:, -1] + 1 - TOKEN_LO) % 251 + TOKEN_LO
    return ProbeBatch("inline", difficulty, ids.long(), _labels_for_last(ids, targets))


def _make_induction(batch: int, difficulty: int, device: torch.device) -> ProbeBatch:
    n_pairs = 2 + difficulty
    filler = 700 + difficulty
    seq_len = 4 * n_pairs + 3 + difficulty * 8
    ids = torch.full((batch, seq_len), filler, dtype=torch.long, device=device)
    targets = torch.empty(batch, dtype=torch.long, device=device)
    for b in range(batch):
        base = TOKEN_LO + 20 * difficulty + b * 3
        keys = torch.arange(base, base + n_pairs, device=device)
        vals = keys + 100
        pos = 0
        for k, v in zip(keys, vals):
            ids[b, pos] = k
            ids[b, pos + 1] = v
            pos += 2
        pos += difficulty * 4
        for k, v in zip(keys, vals):
            ids[b, pos] = k
            ids[b, pos + 1] = v
            pos += 2
        query = (b + difficulty) % n_pairs
        ids[b, -2] = keys[query]
        ids[b, -1] = 801
        targets[b] = vals[query]
    return ProbeBatch("induction", difficulty, ids, _labels_for_last(ids, targets))


def _make_binding(batch: int, difficulty: int, device: torch.device) -> ProbeBatch:
    n_entities = 2 + difficulty * 2
    seq_len = 4 * n_entities + 4 + difficulty * 8
    ids = torch.full((batch, seq_len), 750 + difficulty, dtype=torch.long, device=device)
    targets = torch.empty(batch, dtype=torch.long, device=device)
    attr = 880 + difficulty
    for b in range(batch):
        entities = torch.arange(120 + b * 5, 120 + b * 5 + n_entities, device=device)
        values = entities + 180
        pos = 0
        for entity, value in zip(entities, values):
            ids[b, pos] = entity
            ids[b, pos + 1] = attr
            ids[b, pos + 2] = value
            ids[b, pos + 3] = 760
            pos += 4
        query = (difficulty + b) % n_entities
        ids[b, -3] = entities[query]
        ids[b, -2] = attr
        ids[b, -1] = 802
        targets[b] = values[query]
    return ProbeBatch("binding", difficulty, ids, _labels_for_last(ids, targets))


def _make_ar(batch: int, difficulty: int, device: torch.device) -> ProbeBatch:
    n_pairs = 3 + difficulty * 3
    seq_len = 3 * n_pairs + 4 + difficulty * 8
    ids = torch.full((batch, seq_len), 770 + difficulty, dtype=torch.long, device=device)
    targets = torch.empty(batch, dtype=torch.long, device=device)
    for b in range(batch):
        keys = torch.arange(150 + b * 9, 150 + b * 9 + n_pairs * 2, device=device)
        keys = keys.reshape(n_pairs, 2)
        values = torch.arange(400 + b * 9, 400 + b * 9 + n_pairs, device=device)
        pos = 0
        for idx in range(n_pairs):
            ids[b, pos] = keys[idx, 0]
            ids[b, pos + 1] = keys[idx, 1]
            ids[b, pos + 2] = values[idx]
            pos += 3
        query = (b + difficulty) % n_pairs
        ids[b, -3] = keys[query, 0]
        ids[b, -2] = keys[query, 1]
        ids[b, -1] = 803
        targets[b] = values[query]
    return ProbeBatch("ar", difficulty, ids, _labels_for_last(ids, targets))


def _make_batches(batch: int, difficulties: int, device: torch.device) -> list[ProbeBatch]:
    out: list[ProbeBatch] = []
    for difficulty in range(1, difficulties + 1):
        out.extend(
            [
                _make_inline(batch, difficulty, device),
                _make_induction(batch, difficulty, device),
                _make_binding(batch, difficulty, device),
                _make_ar(batch, difficulty, device),
            ]
        )
    return out


def _mean_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (float, int)) and key not in {"block", "difficulty"}
    ]
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def _probe_checkpoint(args: argparse.Namespace, checkpoint: Path) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    factory = _build_lane_factory("native_adaptive_reciprocal_slot_delta")
    model = _build_tinylm(
        factory,
        dim=args.dim,
        n_blocks=args.n_blocks,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        use_ffn=True,
    ).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    lanes = [
        module
        for module in model.modules()
        if isinstance(module, NativeAdaptiveReciprocalSlotDeltaLane)
    ]
    if len(args.native_gate_floors) != len(lanes):
        raise ValueError(
            f"--native-gate-floors has {len(args.native_gate_floors)} values but "
            f"the model has {len(lanes)} native/reciprocal/slot lanes"
        )
    for block_idx, lane in enumerate(lanes):
        lane.native_gate_floor = float(args.native_gate_floors[block_idx])

    all_rows: list[dict[str, Any]] = []
    step = int(payload.get("step", _checkpoint_step(checkpoint)))
    for batch in _make_batches(args.batch, args.difficulties, device):
        records: list[dict[str, float | int | str]] = []
        for block_idx, lane in enumerate(lanes):
            _patch_lane(
                lane,
                block_idx=block_idx,
                records=records,
                task=batch.task,
                difficulty=batch.difficulty,
            )
        with torch.no_grad():
            logits = model(batch.ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch.labels.reshape(-1),
                ignore_index=-100,
            )
            targets = batch.labels[:, -1]
            preds = logits[:, -1].argmax(dim=-1)
            acc = float(preds.eq(targets).float().mean())
        for row in records:
            row_out = dict(row)
            row_out.update(
                {
                    "checkpoint": str(checkpoint),
                    "step": step,
                    "loss": float(loss),
                    "ppl": float(math.exp(min(float(loss), 20.0))),
                    "acc": acc,
                    "native_gate_floor": float(
                        getattr(
                            lanes[int(row["block"])],
                            "native_gate_floor",
                            NativeAdaptiveReciprocalSlotDeltaLane.GATE_FLOOR,
                        )
                    ),
                }
            )
            all_rows.append(row_out)
    return all_rows


def _write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    fieldnames = list(rows[0].keys())
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    groups = sorted({(int(r["step"]), str(r["task"]), int(r["difficulty"])) for r in rows})
    for step, task, difficulty in groups:
        group_rows = [
            r
            for r in rows
            if int(r["step"]) == step
            and str(r["task"]) == task
            and int(r["difficulty"]) == difficulty
        ]
        means = _mean_rows(group_rows)
        summary_rows.append(
            {
                "step": step,
                "task": task,
                "difficulty": difficulty,
                **means,
                "dominant_raw": max(
                    ("native", "reciprocal", "slot"),
                    key=lambda name: means[f"raw_{name}_gate"],
                ),
                "dominant_effective": max(
                    ("native", "reciprocal", "slot"),
                    key=lambda name: means[f"effective_{name}_gate"],
                ),
            }
        )
    args.out_summary.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("research/reports/native_recip_slot_synthetic_gate_probe.jsonl"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("research/reports/native_recip_slot_synthetic_gate_probe.csv"),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("research/reports/native_recip_slot_synthetic_gate_probe_summary.json"),
    )
    parser.add_argument("--dim", type=int, default=640)
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--difficulties", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--native-gate-floors",
        type=parse_float_csv,
        default=DEFAULT_NATIVE_GATE_FLOORS,
        help="Comma-separated native branch effective floor per block.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for checkpoint in sorted(args.checkpoints, key=_checkpoint_step):
        checkpoint_rows = _probe_checkpoint(args, checkpoint)
        rows.extend(checkpoint_rows)
        step = int(checkpoint_rows[0]["step"])
        overall = _mean_rows(checkpoint_rows)
        print(
            f"step {step}: raw native={overall['raw_native_gate']:.4f} "
            f"effective native={overall['effective_native_gate']:.4f}"
        )
    _write_outputs(rows, args)
    print(f"wrote {len(rows)} rows to {args.out_jsonl}")
    print(f"wrote summary to {args.out_summary}")


if __name__ == "__main__":
    main()
