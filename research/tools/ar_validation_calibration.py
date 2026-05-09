#!/usr/bin/env python
"""Calibration harness for the AR Validation champion probe.

The harness sweeps ``ARValidationConfig`` difficulty knobs against compact
reference models and writes read-only artifacts under
``research/runtime/ar_validation_calibration/``.  It does not mutate the notebook DB.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from research.eval.ar_validation import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_EVERY,
    DEFAULT_HELD_PAIRS,
    DEFAULT_KEY_TOKENS,
    DEFAULT_LR,
    DEFAULT_PAIRS_PER_EXAMPLE,
    DEFAULT_TRAIN_PAIRS,
    DEFAULT_TRAIN_STEPS,
    DEFAULT_VALUE_CLASSES,
    DEFAULT_VALUE_TOKENS,
    INTEGER_AR_VALIDATION_METRIC_VERSION,
    ARValidationConfig,
    run_ar_validation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "research/runtime/ar_validation_calibration"
CALIBRATION_PROTOCOL_VERSION = "ar_validation_calibration_v1"
SELECTED_CONFIG_NAME = "easy25_v2_default"
SWEEP_FIELDS = (
    "n_key_tokens",
    "n_value_tokens",
    "n_value_classes",
    "n_train_pairs",
    "n_held_pairs",
    "pairs_per_example",
    "train_steps",
    "lr",
)


class TinyCausalAttentionLM(nn.Module):
    """Small causal-attention sanity target for calibration sweeps."""

    def __init__(
        self,
        vocab_size: int,
        *,
        max_seq_len: int,
        dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed = nn.Embedding(self.vocab_size, int(dim))
        self.pos = nn.Embedding(int(max_seq_len), int(dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(dim),
            nhead=int(n_heads),
            dim_feedforward=int(dim) * 4,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.norm = nn.LayerNorm(int(dim))
        self.head = nn.Linear(int(dim), self.vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = int(input_ids.shape[1])
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.embed(input_ids) + self.pos(pos).unsqueeze(0)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.head(self.norm(self.layers(x, mask=mask)))


class NoContextEmbeddingLM(nn.Module):
    """Embedding-only baseline; answer logits cannot inspect prior context."""

    def __init__(self, vocab_size: int, *, dim: int = 64) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed = nn.Embedding(self.vocab_size, int(dim))
        self.head = nn.Linear(int(dim), self.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))


class BagOfTokensLM(nn.Module):
    """Diagnostic baseline that can see token frequency but not token order."""

    def __init__(self, vocab_size: int, *, dim: int = 64) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed = nn.Embedding(self.vocab_size, int(dim))
        self.head = nn.Linear(int(dim), self.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pooled = self.embed(input_ids).mean(dim=1, keepdim=True)
        return self.head(pooled.expand(-1, input_ids.shape[1], -1))


def selected_ar_validation_config(**overrides: Any) -> ARValidationConfig:
    """Return the selected v2 default, with optional run-local overrides."""
    cfg = ARValidationConfig(
        protocol="integer_v2",
        n_key_tokens=DEFAULT_KEY_TOKENS,
        n_value_tokens=DEFAULT_VALUE_TOKENS,
        n_value_classes=DEFAULT_VALUE_CLASSES,
        n_train_pairs=DEFAULT_TRAIN_PAIRS,
        n_held_pairs=DEFAULT_HELD_PAIRS,
        pairs_per_example=DEFAULT_PAIRS_PER_EXAMPLE,
        train_steps=DEFAULT_TRAIN_STEPS,
        eval_every=DEFAULT_EVAL_EVERY,
        batch_size=DEFAULT_BATCH_SIZE,
        lr=DEFAULT_LR,
        episodic_values=True,
    )
    return replace(cfg, **overrides) if overrides else cfg


def default_sweep_configs(
    *, quick: bool = False
) -> list[tuple[str, ARValidationConfig]]:
    """Difficulty sweep covering the requested AR Validation knobs."""
    if quick:
        steps = 600
        eval_every = 200
        n_eval = 256
    else:
        steps = DEFAULT_TRAIN_STEPS
        eval_every = 500
        n_eval = 512
    base = selected_ar_validation_config(
        train_steps=steps,
        eval_every=eval_every,
        n_eval=n_eval,
        timeout_s=900.0,
    )
    return [
        (
            "easy_low_values",
            replace(
                base,
                n_key_tokens=512,
                n_value_tokens=64,
                n_value_classes=8,
                n_train_pairs=128,
                n_held_pairs=32,
                pairs_per_example=8,
            ),
        ),
        (SELECTED_CONFIG_NAME, base),
        (
            "hard_more_pairs",
            replace(
                base,
                n_key_tokens=2048,
                n_value_tokens=128,
                n_value_classes=16,
                n_train_pairs=512,
                n_held_pairs=128,
                pairs_per_example=20,
                lr=7.5e-4,
            ),
        ),
    ]


def _required_vocab_size(cfg: ARValidationConfig) -> int:
    return int(cfg.vocab_lo) + int(cfg.n_key_tokens) + int(cfg.n_value_tokens) + 2


def _seq_len(cfg: ARValidationConfig) -> int:
    return 3 * int(cfg.pairs_per_example) + 4


def build_calibration_model(
    family: str,
    cfg: ARValidationConfig,
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
) -> nn.Module:
    vocab_size = _required_vocab_size(cfg)
    if family == "attention":
        return TinyCausalAttentionLM(
            vocab_size,
            max_seq_len=_seq_len(cfg),
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )
    if family == "no_context":
        return NoContextEmbeddingLM(vocab_size, dim=dim)
    if family == "bag":
        return BagOfTokensLM(vocab_size, dim=dim)
    raise ValueError(f"unknown calibration model family: {family}")


def _result_row(
    *,
    config_name: str,
    family: str,
    cfg: ARValidationConfig,
    result: Any,
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        "config_name": config_name,
        "model_family": family,
        "protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "metric_version": result.metric_version,
        "config": {field: getattr(cfg, field) for field in SWEEP_FIELDS},
        "seed": int(cfg.seed),
        "episodic_values": bool(cfg.episodic_values),
        "value_token_chance": round(1.0 / max(float(cfg.n_value_tokens), 1.0), 6),
        "value_class_chance": round(1.0 / max(float(cfg.n_value_classes), 1.0), 6),
        "value_set_chance": round(1.0 / max(float(cfg.pairs_per_example), 1.0), 6),
        "total_token_span": int(cfg.n_key_tokens) + int(cfg.n_value_tokens),
        "seq_len": _seq_len(cfg),
        "status": result.status,
        "held_pair_acc": result.held_pair_acc,
        "held_class_acc": result.held_class_acc,
        "final_acc": result.final_acc,
        "steps_to_floor": result.steps_to_floor,
        "score": result.score,
        "steps_trained": result.steps_trained,
        "elapsed_ms": result.elapsed_ms,
        "wall_seconds": round(float(wall_seconds), 3),
        "learning_curve": result.learning_curve,
        "error": result.error,
    }


def run_calibration_grid(
    configs: list[tuple[str, ARValidationConfig]],
    *,
    families: tuple[str, ...] = ("attention", "no_context"),
    device: str = "cuda",
    dim: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_name, cfg in configs:
        for family in families:
            torch.manual_seed(int(cfg.seed))
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(cfg.seed))
            model = build_calibration_model(
                family,
                cfg,
                dim=int(dim),
                n_layers=int(n_layers),
                n_heads=int(n_heads),
            )
            t0 = time.perf_counter()
            try:
                result = run_ar_validation(model, cfg=cfg, device=device)
            finally:
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            rows.append(
                _result_row(
                    config_name=config_name,
                    family=family,
                    cfg=cfg,
                    result=result,
                    wall_seconds=time.perf_counter() - t0,
                )
            )
    return rows


def select_calibrated_setting(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick a config where attention rises and no-context stays near chance."""
    by_config: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        by_config.setdefault(str(row["config_name"]), {})[str(row["model_family"])] = (
            row
        )

    candidates: list[dict[str, Any]] = []
    for config_name, family_rows in by_config.items():
        attention = family_rows.get("attention")
        no_context = family_rows.get("no_context")
        if attention is None or no_context is None:
            continue
        chance = float(attention["value_token_chance"])
        attn_acc = float(attention["held_pair_acc"])
        noctx_acc = float(no_context["held_pair_acc"])
        if attn_acc < max(0.05, chance * 5.0):
            continue
        if noctx_acc > max(0.035, chance * 3.0):
            continue
        gap = attn_acc - noctx_acc
        if gap < max(0.04, chance * 3.0):
            continue
        candidates.append(
            {
                "config_name": config_name,
                "attention_held_pair_acc": attn_acc,
                "no_context_held_pair_acc": noctx_acc,
                "gap": round(gap, 6),
                "chance": chance,
                "attention_score": attention["score"],
                "config": attention["config"],
            }
        )

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["config_name"] == SELECTED_CONFIG_NAME,
            row["gap"],
            row["attention_held_pair_acc"],
        ),
    )


def build_report(
    rows: list[dict[str, Any]], *, args: argparse.Namespace
) -> dict[str, Any]:
    selected = select_calibrated_setting(rows)
    return {
        "protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "metric_version": INTEGER_AR_VALIDATION_METRIC_VERSION,
        "created_unix": round(time.time(), 3),
        "device": args.device,
        "model_dim": int(args.dim),
        "model_layers": int(args.layers),
        "model_heads": int(args.heads),
        "families": list(args.families),
        "sweep_fields": list(SWEEP_FIELDS),
        "selected": selected,
        "rows": rows,
    }


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"ar_validation_calibration_{stamp}.json"
    md_path = out_dir / f"ar_validation_calibration_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        f"# AR Validation Calibration - {stamp}",
        "",
        f"- Protocol: `{report['protocol_version']}`",
        f"- Metric: `{report['metric_version']}`",
        f"- Device: `{report['device']}`",
        f"- Selected: `{(report.get('selected') or {}).get('config_name')}`",
        "",
        "| config | family | held pair | held class | chance | score | floor | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        floor = "" if row["steps_to_floor"] is None else str(row["steps_to_floor"])
        lines.append(
            "| {config_name} | {model_family} | {held_pair_acc:.4f} | "
            "{held_class_acc:.4f} | {value_token_chance:.4f} | {score:.4f} | "
            f"{floor} | {row['status']} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _parse_override(value: str) -> tuple[str, int | float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("override must look like FIELD=VALUE")
    key, raw = value.split("=", 1)
    if key not in set(SWEEP_FIELDS) | {"seed", "eval_every", "batch_size", "n_eval"}:
        raise argparse.ArgumentTypeError(f"unsupported override field: {key}")
    try:
        parsed: int | float
        parsed = float(raw) if key == "lr" else int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid value for {key}: {raw}") from exc
    return key, parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a short smoke sweep.")
    parser.add_argument(
        "--override",
        action="append",
        type=_parse_override,
        default=[],
        help="Override selected-config field as FIELD=VALUE. Can be repeated.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["attention", "no_context"],
        choices=["attention", "no_context", "bag"],
    )
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    configs = default_sweep_configs(quick=bool(args.quick))
    if args.override:
        overrides = dict(args.override)
        cfg = replace(selected_ar_validation_config(), **overrides)
        configs = [(SELECTED_CONFIG_NAME, cfg)]
    rows = run_calibration_grid(
        configs,
        families=tuple(args.families),
        device=str(args.device),
        dim=int(args.dim),
        n_layers=int(args.layers),
        n_heads=int(args.heads),
    )
    report = build_report(rows, args=args)
    json_path, md_path = write_report(report, args.out_dir)
    selected = report.get("selected")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if selected is None:
        print("no calibrated setting met attention-gap/no-context criteria")
        return 2
    print(
        "selected {config_name}: attention={attention_held_pair_acc:.4f} "
        "no_context={no_context_held_pair_acc:.4f} gap={gap:.4f} "
        "chance={chance:.4f}".format(**selected)
    )
    if not math.isfinite(float(selected["gap"])):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
