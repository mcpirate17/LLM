#!/usr/bin/env python3
"""CPU survivability sweep for templates.

Runs cheap compile/forward/backward checks across templates, seeds, and dims
without touching the notebook pipeline. This is intended as a first-pass filter
before low-budget backfill.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class SurvivabilityRow:
    template_name: str
    seed: int
    model_dim: int
    n_layers: int
    vocab_size: int
    seq_len: int
    batch_size: int
    compile_ok: int
    forward_ok: int
    backward_ok: int
    n_params: int
    forward_ms: float | None
    backward_ms: float | None
    grad_norm: float | None
    error_stage: str | None
    error_type: str | None
    error_message: str | None


def _build_graphs(template_name: str, n_layers: int, dim: int, seed: int):
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.templates import apply_template

    rng = random.Random(seed)
    graphs = []
    for _ in range(n_layers):
        graph = ComputationGraph(model_dim=dim)
        inp = graph.add_input()
        out = apply_template(graph, inp, rng, template_name=template_name)
        graph.set_output(out)
        graphs.append(graph)
    return graphs


def _grad_norm(model: torch.nn.Module) -> float | None:
    total = 0.0
    seen = False
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        seen = True
        total += float(torch.sum(grad.detach().float() ** 2).item())
    if not seen:
        return None
    return total ** 0.5


def run_case(
    *,
    template_name: str,
    seed: int,
    model_dim: int,
    n_layers: int,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
) -> SurvivabilityRow:
    from research.synthesis.compiler import compile_model

    torch.manual_seed(seed)
    row = SurvivabilityRow(
        template_name=template_name,
        seed=seed,
        model_dim=model_dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=batch_size,
        compile_ok=0,
        forward_ok=0,
        backward_ok=0,
        n_params=0,
        forward_ms=None,
        backward_ms=None,
        grad_norm=None,
        error_stage=None,
        error_type=None,
        error_message=None,
    )
    try:
        graphs = _build_graphs(template_name, n_layers, model_dim, seed)
        model = compile_model(graphs, vocab_size=vocab_size, max_seq_len=seq_len)
        row.compile_ok = 1
        row.n_params = sum(int(p.numel()) for p in model.parameters())
        device = torch.device("cpu")
        model = model.to(device)

        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        t0 = time.perf_counter()
        logits = model(x)
        row.forward_ms = (time.perf_counter() - t0) * 1000.0
        if logits.shape[-1] != vocab_size or not torch.isfinite(logits).all():
            raise RuntimeError(f"invalid_forward shape={tuple(logits.shape)}")
        row.forward_ok = 1

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            x[:, 1:].reshape(-1),
        )
        t1 = time.perf_counter()
        loss.backward()
        row.backward_ms = (time.perf_counter() - t1) * 1000.0
        row.grad_norm = _grad_norm(model)
        if row.grad_norm is None:
            raise RuntimeError("missing_gradients")
        if not torch.isfinite(torch.tensor(row.grad_norm)):
            raise RuntimeError("non_finite_grad_norm")
        row.backward_ok = 1
        return row
    except Exception as exc:  # pragma: no cover - operational path
        row.error_type = type(exc).__name__
        row.error_message = str(exc)[:500]
        if row.backward_ok:
            row.error_stage = None
        elif row.forward_ok:
            row.error_stage = "backward"
        elif row.compile_ok:
            row.error_stage = "forward"
        else:
            row.error_stage = "compile"
        return row


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU survivability sweep for templates")
    parser.add_argument("--templates", nargs="+", required=True)
    parser.add_argument("--dims", default="64,128")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    dims = _parse_int_list(args.dims)
    seeds = _parse_int_list(args.seeds)
    rows: list[SurvivabilityRow] = []
    for template_name in args.templates:
        for model_dim in dims:
            for seed in seeds:
                row = run_case(
                    template_name=template_name,
                    seed=seed,
                    model_dim=model_dim,
                    n_layers=int(args.n_layers),
                    vocab_size=int(args.vocab_size),
                    seq_len=int(args.seq_len),
                    batch_size=int(args.batch_size),
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "template": row.template_name,
                            "seed": row.seed,
                            "dim": row.model_dim,
                            "compile_ok": row.compile_ok,
                            "forward_ok": row.forward_ok,
                            "backward_ok": row.backward_ok,
                            "error_stage": row.error_stage,
                            "error_type": row.error_type,
                            "n_params": row.n_params,
                            "forward_ms": row.forward_ms,
                            "backward_ms": row.backward_ms,
                            "grad_norm": row.grad_norm,
                        }
                    )
                )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    print("\nSummary")
    for template_name in args.templates:
        template_rows = [row for row in rows if row.template_name == template_name]
        total = max(len(template_rows), 1)
        compile_rate = sum(row.compile_ok for row in template_rows) / total
        forward_rate = sum(row.forward_ok for row in template_rows) / total
        backward_rate = sum(row.backward_ok for row in template_rows) / total
        good_rows = [row for row in template_rows if row.backward_ok]
        mean_forward = (
            sum(float(row.forward_ms or 0.0) for row in good_rows) / len(good_rows)
            if good_rows
            else None
        )
        mean_backward = (
            sum(float(row.backward_ms or 0.0) for row in good_rows) / len(good_rows)
            if good_rows
            else None
        )
        common_error = None
        error_counts: dict[tuple[str | None, str | None], int] = {}
        for row in template_rows:
            if row.backward_ok:
                continue
            key = (row.error_stage, row.error_type)
            error_counts[key] = error_counts.get(key, 0) + 1
        if error_counts:
            common_error = max(error_counts.items(), key=lambda item: item[1])[0]
        print(
            json.dumps(
                {
                    "template": template_name,
                    "n": total,
                    "compile_rate": round(compile_rate, 4),
                    "forward_rate": round(forward_rate, 4),
                    "backward_rate": round(backward_rate, 4),
                    "mean_forward_ms": round(mean_forward, 3)
                    if mean_forward is not None
                    else None,
                    "mean_backward_ms": round(mean_backward, 3)
                    if mean_backward is not None
                    else None,
                    "common_error": common_error,
                }
            )
        )


if __name__ == "__main__":
    main()
