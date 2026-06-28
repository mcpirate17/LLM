#!/usr/bin/env python3
"""Fixed-seed scoring reference for the regression gate.

Drives the REAL gMQAR scoring path (research/eval/gmqar.score_model_gmqar — the AR metric
of record) with a fully seeded model on a fixed difficulty grid, and emits the result as
canonical JSON. Same code in → byte-identical JSON out. If a fix perturbs the scoring
substrate (or quietly reconverges a mechanism on a softmax-shaped path), these numbers
move and gate.run_reference rejects the fix.

Usage:
    python -m reference_score --out <path> [--seed 0] [--vocab 256]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from research.eval.gmqar import default_grid, score_model_gmqar


class _SeededModel(nn.Module):
    """Deterministic, fixed-weight token->logits map. Not trained — we are pinning the
    SCORING math, not model quality. Seeded weights make every cell reproducible."""

    def __init__(self, vocab: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


def reference_payload(seed: int, vocab: int) -> dict:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = _SeededModel(vocab, seed)
    grid = default_grid(vocab_size=vocab)
    res = score_model_gmqar(model, grid=grid, vocab_size=vocab, device="cpu")
    return {
        "seed": seed,
        "vocab": vocab,
        "audc": res.audc,
        "d50": res.d50,
        "chance": res.chance,
        "cells": res.cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=256)
    args = ap.parse_args()
    payload = reference_payload(args.seed, args.vocab)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"reference scoring written: {out}  (audc={payload['audc']} d50={payload['d50']})"
    )


if __name__ == "__main__":
    main()
