"""Tests for the live ID Collapse snapshot hook in _train_with_program.

Regression for Task #24 — early-stops, NaN aborts, and checkpoint resumes
were leaving fp_id_collapse_rate NULL because:

* probe_ids were stashed only on ``step == 0`` (broken on resume), and
* the late snapshot was only attempted at ``n_steps - 1`` (skipped if
  the loop broke out via early-stop or inflight gate).

The fix stashes probe_ids on the first iteration encountered and
re-attempts the late snapshot post-loop with whatever step we last
completed.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

import torch
import torch.nn as nn

from research.scientist.runner import ExperimentRunner, RunConfig


pytestmark = pytest.mark.unit


class _TinyModel(nn.Module):
    """Mini SynthesizedModel-shaped stub.

    capture_hidden_state_snapshot looks for ``model.embed`` (the token
    embedding) and one of ``_fingerprint_pre_logits_from_embed`` /
    ``_fingerprint_forward_from_embed`` so it can capture the
    pre-LM-head representation. Provide both so the test exercises the
    "happy path" of the snapshot machinery — anything weaker would let
    the snapshot silently no-op and mask regressions.
    """

    def __init__(self, vocab_size: int = 64, d_model: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.body = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _fingerprint_forward_from_embed(self, embed):
        return self.body(embed)

    def forward(self, input_ids):
        e = self.embed(input_ids)
        h = self._fingerprint_forward_from_embed(e)
        return self.head(self.norm(h))


class _Curriculum:
    @staticmethod
    def get_seq_len(_step, _total):
        return 8


class _Loss:
    @staticmethod
    def compute(logits, target):
        return torch.nn.functional.cross_entropy(logits, target)


class _Optimizer:
    @staticmethod
    def create(params):
        return torch.optim.SGD(params, lr=1e-3)


class _Program:
    init_scheme = "default"
    init_scale = 0.02
    n_steps = 60
    batch_size = 2
    max_grad_norm = 1.0
    curriculum = _Curriculum()
    loss = _Loss()
    optimizer = _Optimizer()


def _runner(tmp_dir: str) -> ExperimentRunner:
    return ExperimentRunner(os.path.join(tmp_dir, "id_collapse_hook.db"))


def test_id_collapse_snapshots_populated_on_full_run(monkeypatch):
    """Full ``n_steps`` training run captures both early and late snapshots
    at the planned 20% and (n_steps-1) marks.

    Random data on a tiny model never improves loss, so both inflight
    gates and early-stop would otherwise short-circuit the loop. Disable
    them here so the test isolates the snapshot-at-planned-step path
    from the post-loop fallback path (covered separately).
    """
    monkeypatch.setattr(
        "research.scientist.runner.execution_training_program.check_inflight_health",
        lambda **_kwargs: None,
    )
    tmp = tempfile.mkdtemp()
    runner = _runner(tmp)
    try:
        model = _TinyModel()
        config = RunConfig(
            vocab_size=64,
            max_seq_len=16,
            data_mode="random",
            early_stop_min_steps=10_000,
            early_stop_patience=10_000,
        )
        runner._train_with_program(
            model, _Program(), config, torch.device("cpu"), seed=11
        )
        assert runner._id_collapse_early_snap is not None
        assert runner._id_collapse_late_snap is not None
        # Early at 20% of n_steps=60 → step 12; late at n_steps-1 → 59.
        assert runner._id_collapse_early_snap.step == 12
        assert runner._id_collapse_late_snap.step == 59
    finally:
        runner.close()


def test_id_collapse_late_snapshot_captured_on_early_loop_exit():
    """When training breaks out before ``n_steps - 1`` (early-stop or
    inflight gate), the late snapshot is still captured at the last
    completed step. Without the fallback, fp_id_collapse_rate would
    stay NULL on every short-circuited run.
    """
    tmp = tempfile.mkdtemp()
    runner = _runner(tmp)
    try:
        model = _TinyModel()
        # Force the loop to break out far short of n_steps-1 by making
        # both early_stop and inflight gates aggressive. Random data on
        # a tiny model triggers one or the other within ~20 steps.
        config = RunConfig(
            vocab_size=64,
            max_seq_len=16,
            data_mode="random",
            early_stop_min_steps=20,
            early_stop_patience=1,
            early_stop_min_delta=1e6,
        )
        runner._train_with_program(
            model, _Program(), config, torch.device("cpu"), seed=21
        )
        # The actual exit path (early_stop vs inflight) is not the
        # contract under test — only that both snapshots end up populated.
        assert runner._id_collapse_early_snap is not None
        assert runner._id_collapse_late_snap is not None
        assert runner._id_collapse_late_snap.step < 59  # well short of planned late_at
        assert (
            runner._id_collapse_late_snap.step
            > runner._id_collapse_early_snap.step
        )
    finally:
        runner.close()


def test_id_collapse_probe_ids_stashed_on_resume_past_early_target():
    """Checkpoint resume past _id_collapse_early_at still stashes probe_ids
    and captures both snapshots — early snap retargets to the resume step.
    """
    tmp = tempfile.mkdtemp()
    runner = _runner(tmp)
    try:
        model = _TinyModel()
        config = RunConfig(vocab_size=64, max_seq_len=16, data_mode="random")
        # Simulate a checkpoint resume past the original early target
        # (20% of 60 = 12) by patching _train_restore_checkpoint to return
        # step_start=30 with the standard zeroed progress fields.
        from research.scientist.runner._helpers_gate import InflightState

        original_restore = runner._train_restore_checkpoint

        def _fake_restore(model_, optimizer_, dev_, result_):
            state = original_restore(model_, optimizer_, dev_, result_)
            state["step_start"] = 30
            state["initial_loss"] = 4.0
            state["final_loss"] = 4.0
            state["min_loss"] = 4.0
            state["inflight_state"] = InflightState()
            state["es_best_loss"] = 4.0
            state["es_steps_since_improve"] = 0
            return state

        with patch.object(
            runner, "_train_restore_checkpoint", side_effect=_fake_restore
        ):
            runner._train_with_program(
                model, _Program(), config, torch.device("cpu"), seed=31
            )

        assert runner._id_collapse_probe_ids is not None
        assert runner._id_collapse_early_snap is not None
        assert runner._id_collapse_late_snap is not None
        # Early snap retargeted to resume step 30.
        assert runner._id_collapse_early_snap.step == 30
        assert runner._id_collapse_late_snap.step == 59
    finally:
        runner.close()
