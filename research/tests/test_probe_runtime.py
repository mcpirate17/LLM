from __future__ import annotations

import torch

from research.eval._probe_runtime import disable_native_probe_dispatch


class _DummyProbeModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._subgraph_dispatcher = object()
        self._native_chain_segment_slots = ("slot",)
        self._has_native_chain_slots = True
        self._cached_native_wrapper = object()
        self._native_forward_wrapper = object()
        self._native_chain_segments = ("segment",)


def test_disable_native_probe_dispatch_patches_cpu_modules():
    module = _DummyProbeModule()

    with disable_native_probe_dispatch(module, device="cpu"):
        assert module._subgraph_dispatcher is None
        assert module._native_chain_segment_slots == ()
        assert module._has_native_chain_slots is False
        assert module._cached_native_wrapper is None
        assert module._native_forward_wrapper is None
        assert module._native_chain_segments == ()

    assert module._subgraph_dispatcher is not None
    assert module._native_chain_segment_slots == ("slot",)
    assert module._has_native_chain_slots is True
    assert module._cached_native_wrapper is not None
    assert module._native_forward_wrapper is not None
    assert module._native_chain_segments == ("segment",)


def test_disable_native_probe_dispatch_honors_env_override(monkeypatch):
    module = _DummyProbeModule()
    monkeypatch.setenv("ARIA_ALLOW_SLOW_NATIVE_CUDA_PROBES", "1")

    with disable_native_probe_dispatch(module, device="cpu"):
        assert module._subgraph_dispatcher is not None
        assert module._native_chain_segment_slots == ("slot",)
        assert module._has_native_chain_slots is True
        assert module._cached_native_wrapper is not None
        assert module._native_forward_wrapper is not None
        assert module._native_chain_segments == ("segment",)

    monkeypatch.delenv("ARIA_ALLOW_SLOW_NATIVE_CUDA_PROBES", raising=False)
