"""Fixed-seed parity snapshot of the eval probe family.

Run before and after refactoring research/eval probes; outputs must be
byte-identical JSON. Tiny CPU configs — runtime is seconds, the point is
bit-identical numbers, not meaningful scores.

Usage:
    python -m research.tools.eval_probe_parity_snapshot (from repo root) reports/probe_parity_before.json
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 1024, dim: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(input_ids))


def _as_jsonable(result: Any) -> Any:
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return {k: _as_jsonable(v) for k, v in dataclasses.asdict(result).items()}
    if isinstance(result, dict):
        return {
            str(k): _as_jsonable(v)
            for k, v in sorted(result.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(result, (list, tuple)):
        return [_as_jsonable(v) for v in result]
    if isinstance(result, float):
        return round(result, 10)
    return result


def _strip_timing(node: Any) -> Any:
    """Drop nondeterministic wall-clock fields (elapsed_ms etc.)."""
    if isinstance(node, dict):
        return {
            k: _strip_timing(v)
            for k, v in node.items()
            if "elapsed" not in k and not k.endswith("_ms") and k != "timeout_s"
        }
    if isinstance(node, list):
        return [_strip_timing(v) for v in node]
    return node


def _fresh_model() -> _TinyLM:
    torch.manual_seed(0)
    return _TinyLM()


def _probe_calls() -> dict[str, Callable[[], Any]]:
    from research.eval.binding_curriculum import curriculum_binding_range_profile
    from research.eval.binding_intermediate_probe import run_binding_intermediate
    from research.eval.binding_range import binding_range_profile
    from research.eval.induction_intermediate_probe import run_induction_intermediate
    from research.eval.induction_probe import induction_score
    from research.eval.long_range_ar import long_range_ar_score
    from research.eval import induction_validation_probe

    return {
        "binding_range": lambda: binding_range_profile(
            _fresh_model(),
            distances=(2, 4),
            n_eval=32,
            seq_len=64,
            batch_size=16,
            device="cpu",
            seed=11,
        ),
        "induction_score": lambda: induction_score(
            _fresh_model(),
            gaps=(4, 8),
            n_train_steps=5,
            n_eval=32,
            batch_size=16,
            device="cpu",
            seed=11,
        ),
        "binding_intermediate": lambda: run_binding_intermediate(
            _fresh_model(),
            seeds=(11, 23, 47),
            distances=(2, 4),
            n_train_steps=4,
            n_eval=32,
            train_seq_len=64,
            eval_seq_len=64,
            train_batch_size=8,
            eval_batch_size=16,
            device="cpu",
        ),
        "induction_intermediate": lambda: run_induction_intermediate(
            _fresh_model(),
            seeds=(11, 23, 47),
            gaps=(4, 8),
            n_train_steps=4,
            n_eval=32,
            batch_size=16,
            device="cpu",
        ),
        # The public champion entry only accepts 2K/5K/10K budgets; for a
        # parity snapshot the internal median runner with a tiny budget is
        # the same code path.
        "induction_validation": lambda: (
            induction_validation_probe._run_induction_validation_median(
                _fresh_model(),
                seeds=(11, 23, 47),
                gaps=(4, 8),
                n_train_steps=4,
                n_eval=16,
                batch_size=8,
                lr=1e-3,
                device="cpu",
                timeout_s=600.0,
            )
        ),
        "long_range_ar": lambda: long_range_ar_score(
            _fresh_model(),
            seq_lens=(64, 128),
            n_train_steps=5,
            n_eval=32,
            batch_size=8,
            device="cpu",
        ),
        "binding_curriculum": lambda: curriculum_binding_range_profile(
            _fresh_model(),
            distances=(2, 4),
            n_train_steps=4,
            n_eval=32,
            train_seq_len=64,
            eval_seq_len=64,
            device="cpu",
            seed=11,
        ),
    }


def main(out_path: str) -> None:
    torch.use_deterministic_algorithms(True)
    snapshot = {name: _as_jsonable(call()) for name, call in _probe_calls().items()}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_strip_timing(snapshot), indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
