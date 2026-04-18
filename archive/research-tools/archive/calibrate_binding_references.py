#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.eval.binding_curriculum import curriculum_binding_range_profile
from research.eval.binding_range import binding_range_profile
from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph
from research.synthesis.reference_architectures import build_reference


def _build_local_conv_graph(model_dim: int) -> ComputationGraph:
    g = ComputationGraph(model_dim)
    inp = g.add_input()
    conv = g.add_op("conv_only", [inp])
    norm = g.add_op("rmsnorm", [conv])
    proj = g.add_op("linear_proj", [norm])
    out = g.add_op("add", [inp, proj])
    g.set_output(out)
    g.metadata = {
        "architecture": "local_conv",
        "reference_name": "Local Conv",
        "description": "Minimal local-only conv residual block",
    }
    return g


def _build_model(
    name: str, *, model_dim: int, n_layers: int, vocab_size: int, seq_len: int
):
    if name == "local_conv":
        layers = [_build_local_conv_graph(model_dim) for _ in range(n_layers)]
    elif name in {"gpt2", "mamba", "rwkv"}:
        layers = [build_reference(name, model_dim) for _ in range(n_layers)]
    else:
        raise KeyError(name)
    return compile_model(layers, vocab_size=vocab_size, max_seq_len=seq_len)


def run_panel(
    *,
    device: str,
    model_dim: int,
    n_layers: int,
    seq_len: int,
    eval_examples: int,
    adapted_steps: int,
) -> list[dict]:
    panel = ["local_conv", "gpt2", "mamba", "rwkv"]
    rows: list[dict] = []
    for name in panel:
        model = _build_model(
            name,
            model_dim=model_dim,
            n_layers=n_layers,
            vocab_size=256,
            seq_len=seq_len,
        ).to(device)
        zero = binding_range_profile(
            model,
            distances=(2, 4, 8, 16, 32, 64),
            n_eval=eval_examples,
            seq_len=seq_len,
            batch_size=32,
            device=device,
            seed=42,
        )
        curriculum = curriculum_binding_range_profile(
            model,
            distances=(4, 8, 16, 32),
            n_train_steps=max(800, adapted_steps),
            n_eval=eval_examples,
            train_seq_len=min(seq_len, 128),
            eval_seq_len=seq_len,
            train_batch_size=16,
            eval_batch_size=32,
            device=device,
            seed=42,
        )
        rows.append(
            {
                "name": name,
                "zero_shot_binding_auc": zero.auc,
                "zero_shot_binding_status": zero.status,
                "zero_shot_distance_accuracies": zero.distance_accuracies,
                "curriculum_binding_auc": curriculum.auc,
                "curriculum_binding_status": curriculum.status,
                "curriculum_distance_accuracies": curriculum.distance_accuracies,
                "curriculum_steps": curriculum.train_steps,
                "curriculum_protocol_version": curriculum.protocol_version,
            }
        )
        del model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _markdown(
    rows: list[dict],
    *,
    model_dim: int,
    n_layers: int,
    seq_len: int,
    eval_examples: int,
    adapted_steps: int,
) -> str:
    lines = [
        "# Binding Reference Calibration",
        "",
        f"- `model_dim={model_dim}`",
        f"- `n_layers={n_layers}`",
        f"- `seq_len={seq_len}`",
        f"- `eval_examples={eval_examples}`",
        f"- `curriculum_steps={max(800, adapted_steps)}`",
        "",
        "| Model | Zero-shot binding AUC | Curriculum binding AUC | Zero status | Curriculum status |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {row['zero_shot_binding_auc']:.4f} | {row['curriculum_binding_auc']:.4f} | `{row['zero_shot_binding_status']}` | `{row['curriculum_binding_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Distances",
            "",
        ]
    )
    for row in rows:
        lines.append(f"### `{row['name']}`")
        lines.append(
            f"- zero-shot: `{json.dumps(row['zero_shot_distance_accuracies'], sort_keys=True)}`"
        )
        lines.append(
            f"- curriculum: `{json.dumps(row['curriculum_distance_accuracies'], sort_keys=True)}`"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate zero-shot and adapted binding against reference architectures."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--adapted-steps", type=int, default=300)
    parser.add_argument("--out-dir", default="research/docs")
    args = parser.parse_args()

    rows = run_panel(
        device=args.device,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
        eval_examples=args.eval_examples,
        adapted_steps=args.adapted_steps,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "binding_reference_calibration_2026-04-08"
    stem.with_suffix(".json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    stem.with_suffix(".md").write_text(
        _markdown(
            rows,
            model_dim=args.model_dim,
            n_layers=args.n_layers,
            seq_len=args.seq_len,
            eval_examples=args.eval_examples,
            adapted_steps=args.adapted_steps,
        )
    )
    for row in rows:
        print(
            f"{row['name']}: zero={row['zero_shot_binding_auc']:.4f} curriculum={row['curriculum_binding_auc']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
