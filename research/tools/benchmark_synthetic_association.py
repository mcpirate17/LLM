"""Benchmark the nano synthetic association probe against baselines/references.

This tool is read-only: it does not touch the notebook DB.  Use it to tune
``synthetic_association_score`` before any scoring/dashboard wiring.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Callable, Dict, Iterable, List

import torch

from research.eval.synthetic_association_eval import synthetic_association_score


class NounRelationLearner(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_dim: int = 48) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.relation_emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(
            self.token_emb(input_ids[:, 0]) + self.relation_emb(input_ids[:, 1])
        )
        pred = self.out(hidden)
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            device=input_ids.device,
        )
        logits[:, 1, :] = pred
        return logits


class RelationOnlyLearner(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.relation_emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pred = self.out(torch.tanh(self.relation_emb(input_ids[:, 1])))
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            device=input_ids.device,
        )
        logits[:, 1, :] = pred
        return logits


def _reference_factory(
    arch_key: str,
    *,
    d_model: int,
    vocab_size: int = 128,
) -> Callable[[], torch.nn.Module]:
    def _build() -> torch.nn.Module:
        from research.synthesis.compiler import compile_model
        from research.synthesis.reference_architectures import build_reference

        graph = build_reference(arch_key, d_model=d_model)
        return compile_model([graph], vocab_size=vocab_size, max_seq_len=8)

    return _build


def _run_one(
    name: str,
    factory: Callable[[], torch.nn.Module],
    *,
    active_vocab_size: int,
    n_train_steps: int,
    eval_repeats: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    timeout_s: float,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = synthetic_association_score(
        factory(),
        active_vocab_size=active_vocab_size,
        n_train_steps=n_train_steps,
        eval_repeats=eval_repeats,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
        timeout_s=timeout_s,
    )
    payload = result.to_dict()
    payload["name"] = name
    payload["seed"] = seed
    payload["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return payload


def _summarize(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["name"]), int(row["synthetic_association_train_steps"]))
        grouped.setdefault(key, []).append(row)

    out = []
    for (name, steps), values in sorted(grouped.items()):
        scores = [float(v["synthetic_association_score"]) for v in values]
        verbs = [float(v["synthetic_association_verb_acc"]) for v in values]
        adjs = [float(v["synthetic_association_adjective_acc"]) for v in values]
        out.append(
            {
                "name": name,
                "steps": steps,
                "n": len(values),
                "score_mean": round(statistics.mean(scores), 4),
                "score_sd": round(statistics.pstdev(scores), 4),
                "verb_mean": round(statistics.mean(verbs), 4),
                "adjective_mean": round(statistics.mean(adjs), 4),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-vocab-size", type=int, default=40)
    parser.add_argument("--steps", type=int, nargs="+", default=[15, 20, 25, 30])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--eval-repeats", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--include-references", action="store_true")
    parser.add_argument("--reference-d-model", type=int, default=64)
    args = parser.parse_args()

    factories: dict[str, Callable[[], torch.nn.Module]] = {
        "noun_relation": lambda: NounRelationLearner(),
        "relation_only": lambda: RelationOnlyLearner(),
    }
    if args.include_references:
        for arch in ("gpt2", "mamba", "rwkv"):
            factories[f"ref_{arch}"] = _reference_factory(
                arch,
                d_model=args.reference_d_model,
                vocab_size=max(128, args.active_vocab_size),
            )

    rows: list[dict[str, Any]] = []
    for name, factory in factories.items():
        for steps in args.steps:
            for seed in args.seeds:
                rows.append(
                    _run_one(
                        name,
                        factory,
                        active_vocab_size=args.active_vocab_size,
                        n_train_steps=steps,
                        eval_repeats=args.eval_repeats,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        device=args.device,
                        seed=seed,
                        timeout_s=args.timeout_s,
                    )
                )

    print(json.dumps({"summary": _summarize(rows), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
