"""Discriminating binding/AR scale-ladder for a loss-monster graph champion.

ACTION 1 intermediate (de-risks the 40M frontier run). ``recursive_depth_router`` scored a
saturated 1.0 on the nano AR-gate (in_dist + held_class) AND clears nano_bind's no-go, but
held-out *pair* generalization is 0 — the recency/positional-shortcut signature flagged in
``cross_axis_architecture_matrix_2026-06-07``. Before spending the big run we want to know:
does the binding survive when the corpus is scaled so it CANNOT be memorized, and does the
rule generalize to held-out nouns?

This reuses ``scale_ladder``'s scalable binding corpus + discriminating metric verbatim
(``_make_binding_corpus`` + ``score_model_on_corpus``: in-dist bound recall, held-out rule
generalization, memorization guard, chance-calibrated). The only difference vs the baseline
ladder is the model: a loss-monster graph compiles as a FULL model (its own norm/FFN), not a
position-mixer lane, so we build it via ``compile_model_native_first`` and scale by LAYER
count (the graph's hidden dim is baked in) — more layers → more params → the corpus auto-grows
(tokens_per_param), keeping memorization impossible at the larger rungs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median

import torch

from research.scientist.native_runner import compile_model_native_first
from research.synthesis.serializer import graph_from_json
from research.tools.loss_monster_screen import (
    _OUT_DIR,
    _RUNS_DB,
    select_family_champions,
)
from research.tools.scale_ladder import _make_binding_corpus, score_model_on_corpus


def _hidden_dim(model: torch.nn.Module, vocab_size: int) -> int:
    """Read the graph model's hidden dim from its token-embedding weight."""
    for _, p in model.named_parameters():
        if p.dim() == 2 and p.shape[0] == vocab_size:
            return int(p.shape[1])
    return 0


def _build_graph_model(graph_json: str, n_layers: int, vocab_size: int, device: str):
    return compile_model_native_first(
        [graph_from_json(graph_json)] * n_layers,
        vocab_size=vocab_size,
        max_seq_len=8,
    ).to(device)


def _score_rung(graph_json: str, n_layers: int, seed: int, args) -> dict:
    """One (n_layers, seed) cell: size corpus by params, fresh-init, train, grade recall."""
    vocab_size = len(_make_binding_corpus(n_sentences=1, seed=0)["vocab"])
    torch.manual_seed(seed)
    probe = _build_graph_model(graph_json, n_layers, vocab_size, args.device)
    params = sum(p.numel() for p in probe.parameters())
    n_sentences = max(400, (args.tokens_per_param * params) // 4)
    corpus = _make_binding_corpus(n_sentences=n_sentences, seed=seed)
    torch.manual_seed(seed)
    model = _build_graph_model(graph_json, n_layers, vocab_size, args.device)
    res = score_model_on_corpus(
        model,
        corpus,
        params=params,
        dim=_hidden_dim(model, vocab_size),
        device=args.device,
    )
    res["n_layers"] = n_layers
    res["seed"] = seed
    return res


def _run_layer_rung(graph_json: str, n_layers: int, args) -> dict:
    cells = []
    for seed in args.seeds:
        t0 = time.time()
        cell = _score_rung(graph_json, n_layers, seed, args)
        cell["seconds"] = round(time.time() - t0, 1)
        cells.append(cell)
        print(
            f"  L{n_layers} seed{seed} dim={cell['dim']} params={cell['params']:>9,} "
            f"sents={cell['n_sentences']:>9,} bound_recall={cell['in_dist_bound_recall']:.3f} "
            f"held_out_adj={cell['held_out_is_adj']:.3f} loss={cell['final_loss']:.2f}"
            f"{' [MEMORIZED]' if cell['memorized'] else ''} ({cell['seconds']}s)",
            flush=True,
        )
    return {
        "n_layers": n_layers,
        "dim": cells[0]["dim"],
        "params": cells[0]["params"],
        "n_sentences": cells[0]["n_sentences"],
        "median_bound_recall": round(
            median(c["in_dist_bound_recall"] for c in cells), 3
        ),
        "median_held_out_adj": round(median(c["held_out_is_adj"] for c in cells), 3),
        "median_final_loss": round(median(c["final_loss"] for c in cells), 3),
        "memorized_any": any(c["memorized"] for c in cells),
        "cells": cells,
    }


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="recursive_depth_router")
    ap.add_argument("--n-layers", type=int, nargs="*", default=[2, 4, 6])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--tokens-per-param", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "binding_scale_ladder_graph.json"))
    return ap


def main() -> int:
    args = _build_parser().parse_args()
    champ = next(
        (c for c in select_family_champions(_RUNS_DB) if c.family == args.family), None
    )
    if champ is None:
        print(f"No family champion for {args.family!r}")
        return 1

    chance = _make_binding_corpus(n_sentences=1, seed=0)["chance"]
    print(
        f"Binding scale-ladder (graph champion) for {args.family} "
        f"(loss_ratio={champ.screening_loss_ratio:.3f})\n"
        f"chance bound-recall ~ {chance}; held_out_adj near 1.0 = rule generalizes, "
        f"memorized flag = in-dist high but held-out ~chance\n"
        f"n_layers={args.n_layers} seeds={args.seeds} tpp={args.tokens_per_param} "
        f"device={args.device}\n",
        flush=True,
    )

    rungs = [_run_layer_rung(champ.graph_json, nl, args) for nl in args.n_layers]

    print("\nSummary (n_layers -> median bound_recall / held_out_adj):")
    for r in rungs:
        flag = " [MEMORIZED]" if r["memorized_any"] else ""
        print(
            f"  L{r['n_layers']} (params={r['params']:,}) "
            f"bound_recall={r['median_bound_recall']:.3f} "
            f"held_out_adj={r['median_held_out_adj']:.3f}{flag}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "config": vars(args),
                "family": args.family,
                "loss_ratio": champ.screening_loss_ratio,
                "chance": chance,
                "rungs": rungs,
            },
            indent=2,
        )
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
