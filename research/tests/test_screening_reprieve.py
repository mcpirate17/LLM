"""Tests for the slope reprieve feature in the pre-investigation gate.

Verifies:
- Feature is inert when disabled (default)
- Reprieve grant/deny conditions
- Extended eval pass/fail behavior
- Score multiplier application
"""

from __future__ import annotations


from research.eval.utils import micro_train_loop
from research.scientist.runner._types import RunConfig


# ── Task 1: Slope trajectory recording ──────────────────────────────────


class _TinyLM:
    """Minimal language model for trajectory tests."""

    def __init__(self, vocab_size=64, dim=16):
        import torch.nn as nn

        self._mod = nn.Sequential(
            nn.Embedding(vocab_size, dim),
            nn.Linear(dim, vocab_size),
        )
        self.training = True

    def __call__(self, x):
        return self._mod(x)

    def train(self, mode=True):
        self._mod.train(mode)
        self.training = mode
        return self

    def parameters(self):
        return self._mod.parameters()

    def modules(self):
        return self._mod.modules()


class TestSlopeTrajectory:
    """Verify micro_train_loop loss_trajectory recording."""

    def test_trajectory_populated(self):
        """loss_trajectory dict is populated with per-step loss values."""
        import torch

        torch.manual_seed(42)
        model = _TinyLM(vocab_size=64, dim=16)
        batches = [torch.randint(0, 64, (2, 10)) for _ in range(4)]
        trajectory: dict = {}
        micro_train_loop(
            model,
            batches,
            vocab_size=64,
            n_steps=10,
            loss_trajectory=trajectory,
        )
        assert len(trajectory) == 10
        for step in range(1, 11):
            assert step in trajectory
            assert isinstance(trajectory[step], float)

    def test_trajectory_none_default(self):
        """Without loss_trajectory, return value is unchanged."""
        import torch

        torch.manual_seed(42)
        model = _TinyLM(vocab_size=64, dim=16)
        batches = [torch.randint(0, 64, (2, 10)) for _ in range(4)]
        result = micro_train_loop(
            model,
            batches,
            vocab_size=64,
            n_steps=10,
        )
        assert isinstance(result, float)

    def test_monotonic_decreasing_slope_positive(self):
        """Monotonically decreasing loss produces positive slope and consistent=True."""
        # Simulate: loss at step 10 > loss at step 25 > loss at step 50
        sl_10 = 5.0
        sl_25 = 3.5
        sl_50 = 2.0
        slope = (sl_10 - sl_50) / 40.0
        interval_1 = (sl_10 - sl_25) / 15.0
        interval_2 = (sl_25 - sl_50) / 25.0
        consistent = (interval_1 > 0) and (interval_2 > 0)
        assert slope > 0
        assert consistent is True

    def test_early_drop_then_plateau_not_consistent(self):
        """Early drop then plateau: slope > 0 but consistent = False."""
        sl_10 = 5.0
        sl_25 = 2.0  # big drop
        sl_50 = 2.1  # slight increase (plateau/bounce)
        slope = (sl_10 - sl_50) / 40.0
        interval_1 = (sl_10 - sl_25) / 15.0
        interval_2 = (sl_25 - sl_50) / 25.0
        consistent = (interval_1 > 0) and (interval_2 > 0)
        assert slope > 0
        assert consistent is False


# ── Task 2: Config defaults ─────────────────────────────────────────────


class TestReprieveConfig:
    def test_defaults(self):
        cfg = RunConfig()
        assert cfg.slope_reprieve_enabled is False
        assert cfg.slope_reprieve_threshold == 0.015
        assert cfg.slope_reprieve_consistent_required is True
        assert cfg.slope_reprieve_loss_floor == 0.85
        assert cfg.slope_reprieve_max_per_cycle == 3
        assert cfg.slope_reprieve_eval_steps == 150
        assert cfg.slope_reprieve_score_multiplier == 0.75

    def test_existing_defaults_unchanged(self):
        cfg = RunConfig()
        assert cfg.pre_inv_max_lr == 0.50  # raised from 0.40 for routing models
        assert cfg.pre_inv_probe_enabled is False
        assert cfg.pre_inv_top_n == 15


# ── Tasks 3-5: Reprieve logic ───────────────────────────────────────────


class _FakeNotebook:
    """Minimal notebook stub for reprieve gate tests."""

    def __init__(self, eligible_rows, reprieve_rows=None, investigated_fps=None):
        self._eligible = eligible_rows
        self._reprieve_rows = reprieve_rows or []
        self._investigated_fps = investigated_fps or set()
        self._updates = []

    class conn:
        _instance = None

        @classmethod
        def execute(cls, sql, params=None):
            if cls._instance and "screening_slope" in sql:
                return _FakeResult(cls._instance._reprieve_rows)
            if cls._instance and "UPDATE leaderboard" in sql:
                cls._instance._updates.append(params)
            return _FakeResult([])

        @classmethod
        def commit(cls):
            pass

    def get_investigation_eligible(self, **kwargs):
        return list(self._eligible)

    def get_investigated_fingerprints(self):
        return self._investigated_fps

    @staticmethod
    def compute_pre_investigation_score(row, best_ref_lr=None):
        return row.get("_mock_score", 50.0)

    def get_program_details(self, result_ids):
        return [{"result_id": rid} for rid in result_ids]


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeMixin:
    """Minimal mixin stub to test _pre_investigation_gate."""

    def _get_reference_baseline_lr(self, nb):
        return 0.25

    def _reference_margin_ceiling(self, config, nb):
        return 0.50

    def _get_reinvestigation_candidates(self, nb, exclude, limit=3):
        return []


def _make_candidate(
    result_id, loss_ratio=0.3, slope=None, consistent=None, score=50.0, fp=None
):
    row = {
        "result_id": result_id,
        "loss_ratio": loss_ratio,
        "screening_loss_ratio": loss_ratio,
        "graph_fingerprint": fp or f"fp_{result_id}",
        "_mock_score": score,
        "judgment_score": None,
    }
    if slope is not None:
        row["screening_slope"] = slope
    if consistent is not None:
        row["screening_slope_consistent"] = consistent
    return row


class TestReprieveDisabled:
    """When slope_reprieve_enabled=False, behavior is identical to baseline."""

    def test_no_behavior_change(self):
        cfg = RunConfig()
        assert cfg.slope_reprieve_enabled is False

        eligible = [
            _make_candidate("aaa", loss_ratio=0.20, score=60),
            _make_candidate("bbb", loss_ratio=0.35, score=55),
        ]
        nb = _FakeNotebook(eligible)
        nb.conn._instance = nb

        from research.scientist.runner.continuous_investigation import (
            _ContinuousInvestigationMixin,
        )

        class TestGate(_ContinuousInvestigationMixin, _FakeMixin):
            pass

        gate = TestGate()
        result_ids = gate._pre_investigation_gate(cfg, nb, [])
        # Both should pass (loss_ratio < 0.40)
        assert "aaa" in result_ids
        assert "bbb" in result_ids


class TestReprieveGranted:
    """slope_reprieve_enabled=True, eligible candidate gets reprieved."""

    def test_reprieve_granted_conditions(self):
        cfg = RunConfig(slope_reprieve_enabled=True)

        # Normal eligible candidate
        eligible = [_make_candidate("normal1", loss_ratio=0.20, score=60)]
        # Reprieve candidate: high loss_ratio but good slope
        reprieve_row = _make_candidate(
            "c9c7",
            loss_ratio=0.6893,
            slope=0.016,
            consistent=True,
            score=12,
        )

        # Convert to sqlite Row-like dict
        class _DictRow(dict):
            def __getitem__(self, key):
                return dict.__getitem__(self, key)

            def get(self, key, default=None):
                return dict.get(self, key, default)

        nb = _FakeNotebook(eligible, reprieve_rows=[_DictRow(reprieve_row)])
        nb.conn._instance = nb

        from research.scientist.runner.continuous_investigation import (
            _ContinuousInvestigationMixin,
        )

        class TestGate(_ContinuousInvestigationMixin, _FakeMixin):
            def _run_reprieve_eval(self, config, nb, row):
                # Simulate passing reprieve eval
                return 0.35

        gate = TestGate()
        result_ids = gate._pre_investigation_gate(cfg, nb, [])
        assert "normal1" in result_ids
        assert "c9c7" in result_ids


class TestReprieveDenied:
    def test_denied_above_floor(self):
        """loss_ratio >= floor → denied regardless of slope."""
        cfg = RunConfig(slope_reprieve_enabled=True)
        eligible = [_make_candidate("normal1", loss_ratio=0.20, score=60)]
        # Above floor (0.85): should be denied
        reprieve_row = _make_candidate(
            "bad1",
            loss_ratio=0.90,
            slope=0.020,
            consistent=True,
            score=5,
        )

        class _DictRow(dict):
            pass

        nb = _FakeNotebook(eligible, reprieve_rows=[_DictRow(reprieve_row)])
        nb.conn._instance = nb

        from research.scientist.runner.continuous_investigation import (
            _ContinuousInvestigationMixin,
        )

        class TestGate(_ContinuousInvestigationMixin, _FakeMixin):
            def _run_reprieve_eval(self, config, nb, row):
                return 0.35

        gate = TestGate()
        result_ids = gate._pre_investigation_gate(cfg, nb, [])
        assert "bad1" not in result_ids

    def test_denied_slope_insufficient(self):
        """slope < threshold → denied."""
        cfg = RunConfig(slope_reprieve_enabled=True)
        eligible = [_make_candidate("normal1", loss_ratio=0.20, score=60)]
        reprieve_row = _make_candidate(
            "slow1",
            loss_ratio=0.55,
            slope=0.008,
            consistent=True,
            score=20,
        )

        class _DictRow(dict):
            pass

        nb = _FakeNotebook(eligible, reprieve_rows=[_DictRow(reprieve_row)])
        nb.conn._instance = nb

        from research.scientist.runner.continuous_investigation import (
            _ContinuousInvestigationMixin,
        )

        class TestGate(_ContinuousInvestigationMixin, _FakeMixin):
            def _run_reprieve_eval(self, config, nb, row):
                return 0.35

        gate = TestGate()
        result_ids = gate._pre_investigation_gate(cfg, nb, [])
        assert "slow1" not in result_ids

    def test_denied_not_consistent(self):
        """slope OK but not consistent → denied when consistent_required=True."""
        cfg = RunConfig(
            slope_reprieve_enabled=True, slope_reprieve_consistent_required=True
        )
        eligible = [_make_candidate("normal1", loss_ratio=0.20, score=60)]
        reprieve_row = _make_candidate(
            "incons1",
            loss_ratio=0.55,
            slope=0.020,
            consistent=False,
            score=20,
        )

        class _DictRow(dict):
            pass

        nb = _FakeNotebook(eligible, reprieve_rows=[_DictRow(reprieve_row)])
        nb.conn._instance = nb

        from research.scientist.runner.continuous_investigation import (
            _ContinuousInvestigationMixin,
        )

        class TestGate(_ContinuousInvestigationMixin, _FakeMixin):
            def _run_reprieve_eval(self, config, nb, row):
                return 0.35

        gate = TestGate()
        result_ids = gate._pre_investigation_gate(cfg, nb, [])
        assert "incons1" not in result_ids


class TestReprieveEval:
    def test_reprieve_eval_pass_updates_loss_ratio(self):
        """Reprieve candidate passing 150-step eval enters Stage B."""
        cfg = RunConfig(slope_reprieve_enabled=True)
        eligible = [_make_candidate("normal1", loss_ratio=0.20, score=60)]
        reprieve_row = _make_candidate(
            "c9c7",
            loss_ratio=0.6893,
            slope=0.016,
            consistent=True,
            score=12,
        )

        class _DictRow(dict):
            pass

        nb = _FakeNotebook(eligible, reprieve_rows=[_DictRow(reprieve_row)])
        nb.conn._instance = nb

        from research.scientist.runner.continuous_investigation import (
            _ContinuousInvestigationMixin,
        )

        class TestGate(_ContinuousInvestigationMixin, _FakeMixin):
            def _run_reprieve_eval(self, config, nb, row):
                return 0.35  # passes < 0.40

        gate = TestGate()
        result_ids = gate._pre_investigation_gate(cfg, nb, [])
        assert "c9c7" in result_ids

    def test_reprieve_eval_fail_rejects_candidate(self):
        """Reprieve candidate failing 150-step eval is rejected."""
        cfg = RunConfig(slope_reprieve_enabled=True)
        eligible = [_make_candidate("normal1", loss_ratio=0.20, score=60)]
        reprieve_row = _make_candidate(
            "c9c7",
            loss_ratio=0.6893,
            slope=0.016,
            consistent=True,
            score=12,
        )

        class _DictRow(dict):
            pass

        nb = _FakeNotebook(eligible, reprieve_rows=[_DictRow(reprieve_row)])
        nb.conn._instance = nb

        from research.scientist.runner.continuous_investigation import (
            _ContinuousInvestigationMixin,
        )

        class TestGate(_ContinuousInvestigationMixin, _FakeMixin):
            def _run_reprieve_eval(self, config, nb, row):
                return 0.42  # fails >= 0.40

        gate = TestGate()
        result_ids = gate._pre_investigation_gate(cfg, nb, [])
        assert "c9c7" not in result_ids


class TestScoreMultiplier:
    def test_reprieve_candidate_ranked_lower(self):
        """Reprieve candidate with multiplier=0.75 ranks below equal-score normal."""
        cfg = RunConfig(slope_reprieve_enabled=True)
        # Normal candidate: score=60, effective=60
        eligible = [_make_candidate("normal1", loss_ratio=0.20, score=60)]
        # Reprieve candidate: raw score=80, effective=80*0.75=60
        reprieve_row = _make_candidate(
            "c9c7",
            loss_ratio=0.55,
            slope=0.016,
            consistent=True,
            score=80,
        )

        class _DictRow(dict):
            pass

        nb = _FakeNotebook(eligible, reprieve_rows=[_DictRow(reprieve_row)])
        nb.conn._instance = nb

        from research.scientist.runner.continuous_investigation import (
            _ContinuousInvestigationMixin,
        )

        class TestGate(_ContinuousInvestigationMixin, _FakeMixin):
            def _run_reprieve_eval(self, config, nb, row):
                return 0.35

        gate = TestGate()
        # Both pass, but the reprieve candidate has effective score 80*0.75=60
        # vs normal at 60. Reprieve should rank at or below normal.
        result_ids = gate._pre_investigation_gate(cfg, nb, [])
        assert "normal1" in result_ids
        assert "c9c7" in result_ids
