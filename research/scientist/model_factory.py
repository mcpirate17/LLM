"""
Model Factory — unified model construction for the runner pipeline.

Replaces 4 near-identical closure definitions (_make_model_t, _make_model_ood,
_make_model_lc, _make_model_lc2) that were duplicated across execution.py
and continuous.py.
"""

from __future__ import annotations

import json
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


def make_model_factory(
    model_source: str,
    graph_json_str: str,
    arch_spec_json_str: Optional[str],
    n_layers: int,
    model_dim: int,
    vocab_size: int,
    max_seq_len: int,
) -> Callable[[], "nn.Module"]:
    """Return a zero-arg callable that builds a fresh model each time.

    This is the single source of truth for model construction in the runner
    pipeline. Callers pass the returned factory to evaluation routines that
    need to instantiate fresh models (OOD checks, sensitivity, long-context).

    Args:
        model_source: "morphological_box" or "grammar".
        graph_json_str: Serialized ComputationGraph JSON.
        arch_spec_json_str: Serialized ArchSpec JSON (morphological_box only).
        n_layers: Number of layers to replicate.
        model_dim: Model hidden dimension.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length for the model.
    """
    if model_source == "morphological_box" and arch_spec_json_str:
        def _factory():
            from ..morphological_box import ArchSpec
            from ..arch_builder import build_model, BuildConfig
            spec = ArchSpec(**json.loads(arch_spec_json_str))
            bc = BuildConfig(
                dim=model_dim,
                n_layers=n_layers,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
            )
            return build_model(spec, bc)
    else:
        def _factory():
            from ..synthesis.serializer import graph_from_json
            from ..synthesis.compiler import compile_model
            g = graph_from_json(graph_json_str)
            return compile_model(
                [g] * n_layers,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
            )
    return _factory
