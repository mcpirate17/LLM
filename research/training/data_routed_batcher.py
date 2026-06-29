"""Non-invasive data-route wrapper around a token batcher.

Wraps any batcher exposing ``sample_batch(...) -> [B, L] | None`` (notably
``CorpusTokenBatcher``) and applies a ``DataRouteSpec`` position transform to
every sampled batch. The underlying batcher — including its native zero-copy hot
path — is left completely untouched; this is a thin decorator so the
data-pipeline search axis can be turned on per candidate without forking the
perf-critical loader.

See ``research/synthesis/data_pipeline_grammar.py`` for the genotype and
``tasks/loss_monster_scaffolding_plan.md`` Workstream D for the plan.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import torch

from research.synthesis.data_pipeline_grammar import DataRouteSpec, apply_data_route


class _Batcher(Protocol):
    def sample_batch(self, *args: Any, **kwargs: Any) -> Optional[torch.Tensor]: ...


class DataRoutedBatcher:
    """Decorator applying a ``DataRouteSpec`` to each sampled token batch."""

    def __init__(self, inner: _Batcher, spec: DataRouteSpec) -> None:
        if spec.pack != "contiguous":
            raise NotImplementedError(
                f"data pack {spec.pack!r} must be selected at corpus-window time; "
                "DataRoutedBatcher only applies post-sample order/fold transforms"
            )
        self.inner = inner
        self.spec = spec

    def sample_batch(self, *args: Any, **kwargs: Any) -> Optional[torch.Tensor]:
        batch = self.inner.sample_batch(*args, **kwargs)
        if batch is None:
            return None
        return apply_data_route(batch, self.spec)

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (ready, vocab_size, ...) to the wrapped batcher
        # without re-implementing its surface. ``inner`` is set in __init__, so
        # this only fires for attributes the wrapper does not define itself.
        return getattr(self.inner, name)


def maybe_route_batcher(inner: _Batcher, spec: DataRouteSpec | None) -> _Batcher:
    """Return ``inner`` unchanged for an identity/None route, else wrap it."""
    if spec is None or spec.is_identity:
        return inner
    return DataRoutedBatcher(inner, spec)
