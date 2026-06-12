#!/usr/bin/env python
"""Nano grade: reciprocal_semiring_attention vs its two parents vs softmax.

Builds a fixed-seed N-layer model per attention op, micro-trains on REAL
WikiText (``screening_wikitext_eval``) for a real-data perplexity signal, and
runs the induction/binding screening probes (read as a dict — the bug in
``eval_templates`` that silently nulled them). Same seed/data order per op so
the only variable is the mixer.

Usage:
    python -m research.tools.grade_reciprocal_semiring_nano \
        [--layers 2] [--dim 256] [--wikitext-steps 400] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from research.eval.binding_pipeline import run_screening_binding_probes
from research.eval.wikitext_eval import screening_wikitext_eval
from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph
from research.synthesis.templates import apply_template

VOCAB = 100277

OPS = {
    "reciprocal_semiring (composed)": "reciprocal_semiring_attention_block",
    "reciprocal_rank (parent A)": "reciprocal_rank_attention_block",
    "learnable_semiring (parent B)": "learnable_semiring_attention_block",
    "softmax (gpt2_reference)": "gpt2_reference",
}


def _build(template: str, layers: int, dim: int, seed: int) -> torch.nn.Module:
    rng = random.Random(seed)
    graphs = []
    for _ in range(layers):
        g = ComputationGraph(model_dim=dim)
        inp = g.add_input()
        g.set_output(apply_template(g, inp, rng, template_name=template))
        graphs.append(g)
    return compile_model(graphs, vocab_size=VOCAB, max_seq_len=256)


def _grade(template: str, layers: int, dim: int, seed: int, wt_steps: int, device: str):
    torch.manual_seed(seed)
    model = _build(template, layers, dim, seed).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    wt = screening_wikitext_eval(
        model, vocab_size=VOCAB, device=device, n_train_steps=wt_steps
    )
    probes = run_screening_binding_probes(model, device=device)
    return {
        "template": template,
        "n_params_m": round(n_params / 1e6, 2),
        "wikitext_ppl": wt.get("wikitext_perplexity"),
        "wikitext_score": wt.get("wikitext_score"),
        "induction_auc": probes.get("induction_screening_auc"),
        "binding_auc": probes.get("binding_screening_auc"),
        "binding_composite": probes.get("binding_screening_composite"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--wikitext-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("research/reports/reciprocal_semiring_nano_grade.json"),
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for label, tpl in OPS.items():
        t0 = time.perf_counter()
        r = _grade(tpl, args.layers, args.dim, args.seed, args.wikitext_steps, device)
        r["label"] = label
        r["elapsed_s"] = round(time.perf_counter() - t0, 1)
        rows.append(r)
        print(
            f"{label:32s} ppl={r['wikitext_ppl']!s:>9.9} "
            f"ind={r['induction_auc']!s:>7.7} bind={r['binding_auc']!s:>7.7} "
            f"({r['elapsed_s']}s)"
        )

    args.out.write_text(json.dumps(rows, indent=2, default=str))
    print(
        f"\nseed={args.seed} layers={args.layers} dim={args.dim} "
        f"wt_steps={args.wikitext_steps} device={device}"
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
