"""Smoke tests for the mixer_fingerprint eval paths.

These exercise the code paths that the 2026-05-23 ensemble_top_ar_4way run
hung in: `_cheap_evals`, `_mid_tier_evals`, `_expensive_core_evals`,
`_expensive_enrichment_evals`. Confirms:

  1. The function bodies parse and import without ar_curriculum (per the
     2026-05-22 removal).
  2. The ar_validation call uses the v3-stable protocol with multi-seed.
  3. The eval call-graph is reachable on a tiny CPU TinyLM (no GPU
     required for the structural pieces).

Slow functional probes (induction_intermediate, ar_validation v3 3-seed,
binding_*) are CUDA-only and skipped here — they're covered by their own
per-probe smoke tests.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from research.tools import mixer_fingerprint as mf


# ----------------------------------------------------------------------------
# Structural tests — no GPU, no training
# ----------------------------------------------------------------------------


def test_expensive_core_evals_no_longer_imports_ar_curriculum() -> None:
    """The 2026-05-22 removal: ar_curriculum is no longer wired into _expensive_core_evals.
    Docstring may mention the removal for context — only flag actual imports / calls."""
    src = inspect.getsource(mf._expensive_core_evals)
    assert "from research.eval.ar_curriculum_probe import" not in src, (
        "ar_curriculum_probe import not removed"
    )
    assert "ar_curriculum_probe(" not in src, "ar_curriculum_probe still being called"
    assert "ARCurriculumConfig" not in src, "ARCurriculumConfig still referenced"
    # The three remaining probes:
    for expected in ("induction_intermediate", "ar_legacy", "binding_v2"):
        assert expected in src, f"core eval missing {expected!r}"


def test_expensive_enrichment_evals_uses_v3_stable_ar_validation() -> None:
    """The 2026-05-22 swap: ar_validation now uses v3 stable protocol + 3 seeds."""
    src = inspect.getsource(mf._expensive_enrichment_evals)
    assert "STABLE_AR_VALIDATION_PROTOCOL" in src, (
        "ar_validation not pinned to v3 stable protocol"
    )
    assert "seed_count=3" in src, "ar_validation seed_count != 3"
    assert "auto_size_budget=True" in src, "ar_validation missing auto_size_budget"
    assert "deterministic_episode_bank=True" in src, (
        "ar_validation missing deterministic_episode_bank"
    )


def test_mid_tier_evals_calls_run_screening_binding_probes() -> None:
    """The mid_tier event invokes run_screening_binding_probes (induction + binding screen + curriculum)."""
    src = inspect.getsource(mf._mid_tier_evals)
    assert "run_screening_binding_probes" in src
    assert "wikitext_ppl" in src


def test_cheap_evals_signature_is_stable() -> None:
    """Cheap eval contract: 6 keyword-only args + probe_dim default."""
    sig = inspect.signature(mf._cheap_evals)
    required = {"model", "factory", "val_batches", "device", "seed", "amp", "amp_dtype"}
    assert required <= set(sig.parameters)
    assert sig.parameters["probe_dim"].default == 32


def test_resolve_lane_factories_interleaved_pattern_parses() -> None:
    """Interleaved + pattern: returns (stateful_model_factory, probe_factory)."""
    model_f, probe_f = mf._resolve_lane_factories("interleaved", "conv:2,three_lane:1")
    # Stateful — each call yields the next lane in the pattern
    block0 = model_f(64)
    block1 = model_f(64)
    block2 = model_f(64)  # third call exhausts the 3-lane pattern
    assert isinstance(block0, nn.Module)
    assert isinstance(block1, nn.Module)
    assert isinstance(block2, nn.Module)
    # Fourth call should raise (pattern exhausted)
    with pytest.raises(RuntimeError):
        model_f(64)


def test_scheduler_keyed_by_completed_steps_for_restartable_resume() -> None:
    """_WarmupCosineSchedule.lr_at(step) is a pure function — same step yields
    same LR regardless of prior calls. This is what makes --resume work."""
    p = torch.nn.Parameter(torch.tensor([0.0]))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = mf._WarmupCosineSchedule(
        opt, learning_rate=3e-4, min_lr=1e-5, warmup_steps=100, total_steps=1000
    )

    # Pure function: same input → same output
    assert abs(sched.lr_at(200) - sched.lr_at(200)) < 1e-12

    # Fresh scheduler with same hyperparams yields the same LR at the same step
    p2 = torch.nn.Parameter(torch.tensor([0.0]))
    opt2 = torch.optim.SGD([p2], lr=1.0)
    sched2 = mf._WarmupCosineSchedule(
        opt2, learning_rate=3e-4, min_lr=1e-5, warmup_steps=100, total_steps=1000
    )
    assert abs(sched.lr_at(200) - sched2.lr_at(200)) < 1e-12, "not step-keyed"

    # Warmup region (step 0..99) is monotone increasing
    assert sched.lr_at(0) < sched.lr_at(50) < sched.lr_at(99)
    # Cosine region (step 100..total_steps) is monotone decreasing
    assert sched.lr_at(100) > sched.lr_at(500) > sched.lr_at(999)
    # At total_steps, LR hits min_lr
    assert abs(sched.lr_at(1000) - 1e-5) < 1e-7


# ----------------------------------------------------------------------------
# Functional smoke — tiny CPU model
# ----------------------------------------------------------------------------


def _make_tiny_model() -> nn.Module:
    """Tiny softmax_attention model, dim=64, n_blocks=2, CPU-friendly."""
    from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm

    factory = _build_lane_factory("softmax_attention")
    return _build_tinylm(
        factory, dim=64, n_blocks=2, vocab_size=100_277, max_seq_len=256
    )


def test_tiny_model_builds_and_forward_works() -> None:
    """Smallest possible end-to-end: build + forward + loss."""
    model = _make_tiny_model()
    ids = torch.randint(0, 100_277, (2, 32))
    logits = model(ids)
    assert logits.shape == (2, 32, 100_277)
    # No NaN/Inf in fresh forward
    assert torch.isfinite(logits).all()


def test_train_loss_wrapper_returns_scalar() -> None:
    """The training wrapper used by _maybe_compile_training_model."""
    model = _make_tiny_model()
    wrapper = mf._TrainLossWrapper(model)
    ids = torch.randint(0, 100_277, (2, 32))
    loss = wrapper(ids)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


# ----------------------------------------------------------------------------
# Deepcopy + probe-wrapper safety
# ----------------------------------------------------------------------------


class _NonLeafCacheModule(nn.Module):
    """Mimics the synthesis-graph / weight_norm failure mode: stores a
    *computed* (non-leaf) tensor as a Python attribute on the module."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        # Non-leaf cache (has grad_fn)
        self.cached = self.weight * 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


def test_safe_deepcopy_module_handles_non_leaf_cached_tensor() -> None:
    """Raw deepcopy fails on a module with non-leaf cached attribute;
    safe_deepcopy_module survives it. This is the bug that polluted
    program_results with all-zero probe rows."""
    import copy

    from research.eval._probe_utils import safe_deepcopy_module

    src = _NonLeafCacheModule()
    # Confirm the failure mode: raw deepcopy raises
    with pytest.raises(RuntimeError, match="graph leaves"):
        copy.deepcopy(src)
    # safe_deepcopy_module survives it
    copied = safe_deepcopy_module(src)
    assert isinstance(copied, _NonLeafCacheModule)
    # Copy has its own parameters (not a reference)
    assert copied.weight.data_ptr() != src.weight.data_ptr()
    assert torch.allclose(copied.weight, src.weight)


def test_safe_deepcopy_module_is_idempotent() -> None:
    """Cleaning a module multiple times is a no-op (post-clean state stable)."""
    from research.eval._probe_utils import safe_deepcopy_module

    src = _NonLeafCacheModule()
    copy_a = safe_deepcopy_module(src)
    copy_b = safe_deepcopy_module(copy_a)  # second pass on already-clean module
    assert torch.allclose(copy_a.weight, copy_b.weight)


def test_try_probe_captures_exceptions_in_output_dict() -> None:
    """_try_probe must never raise; failures land in the output dict with a status string.
    This is the wrapper around every expensive probe call."""
    out: dict = {}
    mf._try_probe(out, "good_probe", lambda: {"score": 0.5})
    assert out["good_probe"] == {"score": 0.5}
    assert "_t_good_probe" in out

    def _boom() -> dict:
        raise RuntimeError("simulated failure")

    mf._try_probe(out, "bad_probe", _boom)
    assert out["bad_probe"]["status"] == "error"
    assert "simulated failure" in out["bad_probe"]["error"]
    assert "_t_bad_probe" in out


def test_adjacent_token_merge_lane_factory_builds_and_forwards() -> None:
    """The new consensus-lever lane: factory returns a Module that maps
    [B, T, dim] -> [B, T, dim] with finite outputs (binding specialist built
    on the adjacent_token_merge primitive)."""
    from research.tools.scaling_blimp_study import _build_lane_factory

    lane = _build_lane_factory("adjacent_token_merge_lane")(dim=128)
    assert isinstance(lane, nn.Module)
    x = torch.randn(2, 32, 128)
    out = lane(x)
    assert out.shape == (2, 32, 128)
    assert torch.isfinite(out).all()


def test_parse_pattern_basic() -> None:
    """conv:5,three_lane:5,ensemble_top_ar_2way:2 should parse to 3 (name, count) tuples."""
    parsed = mf._parse_pattern("conv:5,three_lane:5,ensemble_top_ar_2way:2")
    assert parsed == [("conv", 5), ("three_lane", 5), ("ensemble_top_ar_2way", 2)]


def test_scheduled_seq_len_growing_at_warmup_returns_max() -> None:
    """At step == warmup_steps, growing schedule reaches max_seq_len."""
    seq = mf._scheduled_seq_len(
        schedule="growing",
        step=2000,
        max_seq_len=256,
        initial_seq_len=16,
        warmup_steps=2000,
    )
    assert seq == 256


def test_scheduled_seq_len_growing_past_warmup_stays_max() -> None:
    """Past warmup, growing schedule sits at max_seq_len."""
    seq = mf._scheduled_seq_len(
        schedule="growing",
        step=5000,
        max_seq_len=256,
        initial_seq_len=16,
        warmup_steps=2000,
    )
    assert seq == 256


def test_compact_exception_message_truncates_dynamo_noise() -> None:
    """Dynamo error messages include verbose suffixes; _compact_exception_message strips them."""

    class _FakeDynamoError(Exception):
        pass

    exc = _FakeDynamoError(
        "the actual error message here. Set TORCHDYNAMO_VERBOSE=1 for more. "
        "For even more developer context see X."
    )
    msg = mf._compact_exception_message(exc)
    assert "actual error message" in msg
    assert "TORCHDYNAMO_VERBOSE" not in msg
    assert "developer context" not in msg
