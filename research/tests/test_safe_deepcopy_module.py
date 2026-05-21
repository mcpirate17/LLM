"""Regression test for safe_deepcopy_module.

The 2026-05-20 120M head-to-head run exposed that probes which deepcopy the
trained model (binding_intermediate, binding_multislot, induction_validation,
ar_validation, etc.) record ``copy_failed: Only Tensors created explicitly by
the user (graph leaves) support the deepcopy protocol`` when the model caches a
non-leaf tensor as a module attribute. ``weight_norm`` and synthesis-graph op
caches both trigger this. ``safe_deepcopy_module`` materializes inference
tensors and detaches non-leaf attribute caches in place before deepcopying.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from research.eval._probe_utils import (
    _detach_non_leaf_attrs_,
    _materialize_non_inference_,
    safe_deepcopy_module,
)


class _ModuleWithNonLeafCache(nn.Module):
    """Caches a computed (non-leaf) tensor as a module attribute on forward.

    Mirrors ``nn.utils.weight_norm`` (replaces ``weight`` with a non-leaf
    ``weight_g * weight_v`` tensor) and synthesis-graph op caches that store
    intermediate results on the module.
    """

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.wg = nn.Parameter(torch.randn(dim))
        self.wv = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.cached_weight = self.wg * self.wv  # non-leaf: has grad_fn
        return x * self.cached_weight


class _ModuleWithInferenceBuffer(nn.Module):
    """Lazy-inits a buffer during forward; tainted if forward runs under inference_mode."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self._dim = dim
        self._built = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self.register_buffer("cache", torch.zeros(self._dim, device=x.device))
            self._built = True
        return self.linear(x) + self.cache


def _make_non_leaf_tainted_model() -> _ModuleWithNonLeafCache:
    model = _ModuleWithNonLeafCache(dim=8)
    _ = model(torch.randn(4, 8))
    return model


def _make_inference_tainted_model() -> _ModuleWithInferenceBuffer:
    model = _ModuleWithInferenceBuffer(dim=8)
    with torch.inference_mode():
        _ = model(torch.randn(4, 8))
    return model


def test_plain_deepcopy_fails_on_non_leaf_cached_attr():
    model = _make_non_leaf_tainted_model()
    assert not model.cached_weight.is_leaf
    with pytest.raises(RuntimeError, match="graph leaves"):
        copy.deepcopy(model)


def test_safe_deepcopy_module_succeeds_on_non_leaf_cached_attr():
    model = _make_non_leaf_tainted_model()
    copied = safe_deepcopy_module(model)
    assert isinstance(copied, _ModuleWithNonLeafCache)
    # Source attr is now detached (idempotent cleanup).
    assert model.cached_weight.is_leaf
    assert copied.cached_weight.is_leaf
    out_orig = model(torch.randn(2, 8))
    out_copy = copied(torch.randn(2, 8))
    assert out_orig.shape == out_copy.shape == (2, 8)


def test_safe_deepcopy_module_succeeds_on_inference_tainted_model():
    model = _make_inference_tainted_model()
    copied = safe_deepcopy_module(model)
    assert isinstance(copied, _ModuleWithInferenceBuffer)
    assert not copied.cache.is_inference()
    assert not model.cache.is_inference()


def test_safe_deepcopy_module_isolates_storage():
    model = _make_non_leaf_tainted_model()
    copied = safe_deepcopy_module(model)
    expected_wg = model.wg.detach().clone()
    copied.wg.data.add_(1.0)
    assert torch.allclose(model.wg.detach(), expected_wg)


def test_safe_deepcopy_module_works_on_clean_model():
    model = nn.Linear(8, 8)
    copied = safe_deepcopy_module(model)
    assert torch.allclose(copied.weight, model.weight)
    assert copied.weight.data_ptr() != model.weight.data_ptr()


def test_detach_helper_is_idempotent():
    model = _make_non_leaf_tainted_model()
    _detach_non_leaf_attrs_(model)
    snapshot = model.cached_weight.detach().clone()
    _detach_non_leaf_attrs_(model)
    assert torch.allclose(model.cached_weight, snapshot)
    assert model.cached_weight.is_leaf


def test_materialize_helper_remains_for_pure_inference_case():
    model = _make_inference_tainted_model()
    _materialize_non_inference_(model)
    assert not model.cache.is_inference()
