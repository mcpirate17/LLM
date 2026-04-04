from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def disable_native_probe_dispatch(model, *, device: str) -> Iterator[None]:
    """Temporarily bypass native subgraph/chain dispatch for CUDA probe runs.

    The Rust scheduler path currently bridges torch tensors through CPU numpy
    buffers for inference dispatch. That is fine for some CPU-heavy workloads,
    but it is counterproductive for small CUDA probe batches where the regular
    PyTorch/Triton path can stay entirely on device.
    """
    if not str(device).startswith("cuda"):
        yield
        return

    patched = []
    for module in model.modules():
        updates = {}
        if hasattr(module, "_subgraph_dispatcher"):
            updates["_subgraph_dispatcher"] = getattr(module, "_subgraph_dispatcher")
            setattr(module, "_subgraph_dispatcher", None)
        if hasattr(module, "_native_chain_segment_slots"):
            updates["_native_chain_segment_slots"] = getattr(
                module, "_native_chain_segment_slots"
            )
            setattr(module, "_native_chain_segment_slots", ())
        if hasattr(module, "_has_native_chain_slots"):
            updates["_has_native_chain_slots"] = getattr(
                module, "_has_native_chain_slots"
            )
            setattr(module, "_has_native_chain_slots", False)
        if hasattr(module, "_cached_native_wrapper"):
            updates["_cached_native_wrapper"] = getattr(
                module, "_cached_native_wrapper"
            )
            setattr(module, "_cached_native_wrapper", None)
        if updates:
            patched.append((module, updates))

    try:
        yield
    finally:
        for module, updates in reversed(patched):
            for attr, value in updates.items():
                setattr(module, attr, value)
