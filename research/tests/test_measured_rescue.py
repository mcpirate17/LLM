"""Unit tests for the label-free measured-descriptor rescue of GBM-dropped candidates.

Covers the safety invariants that make this safe to leave wired:
- default OFF (no env) → ``_partition_prescreener_candidates`` behaves byte-identically;
- additive + affirmative-only (rescue only re-admits measurably-capable graphs, never fail-open);
- bounded by ``max_rescue`` and ``probe_budget``;
- fail-safe (a broken extractor yields zero rescues, never raises into the gate).
"""

from __future__ import annotations

import json

import pytest

from research.scientist.runner import execution_experiment_phase3 as phase3
from research.scientist.runner import measured_rescue_observability as mro
from research.scientist.runner import screening_measured_rescue as smr

pytestmark = pytest.mark.unit


class _FakeGraph:
    def __init__(self, fp: str) -> None:
        self._fp = fp

    def fingerprint(self) -> str:
        return self._fp


class _FakeNB:
    """Records ``record_program_result`` calls without touching SQLite."""

    def __init__(self) -> None:
        self.recorded: list[dict] = []

    def record_program_result(self, **kwargs) -> None:
        self.recorded.append(kwargs)


def _scored_row(p_pass: float, fp: str, reach):
    """Build one ``scored`` tuple; reach is embedded in graph_dict for the fake extractor."""
    graph = _FakeGraph(fp)
    graph_dict = {"_reach": reach, "fp": fp}
    # (planning, p_pass, p_ind, pred_auc, rank_composite, graph, graph_dict)
    return (
        0.1,
        p_pass,
        0.0,
        0.0,
        phase3._RANK_COMPOSITE_USABLE_CUTOFF + 1.0,
        graph,
        graph_dict,
    )


class _FakeMDX:
    """Stand-in extractor: reads the planted ``_reach`` (None → unmeasurable)."""

    def __init__(self, device=None, n_seeds: int = 1) -> None:
        self.probed = 0

    def descriptors(self, graph_json: str):
        self.probed += 1
        reach = json.loads(graph_json).get("_reach")
        if reach is None:
            return None
        return {"long_range_reach": float(reach), "content_dependence": 0.1}


@pytest.fixture()
def patch_extractor(monkeypatch):
    """Patch the lazily-imported extractor; return the instance to inspect probe count."""
    instances: list[_FakeMDX] = []

    import research.tools.measured_descriptors as md

    def _factory(device=None, n_seeds: int = 1):
        inst = _FakeMDX(device=device, n_seeds=n_seeds)
        instances.append(inst)
        return inst

    monkeypatch.setattr(md, "MeasuredDescriptorExtractor", _factory)
    return instances


# ── measured_rescue_config ───────────────────────────────────────────────


def test_config_default_off(monkeypatch):
    monkeypatch.delenv("ARIA_MEASURED_RESCUE", raising=False)
    assert smr.measured_rescue_config() is None


def test_config_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("ARIA_MEASURED_RESCUE", "1")
    monkeypatch.setenv("ARIA_MEASURED_RESCUE_TAU", "0.05")
    monkeypatch.setenv("ARIA_MEASURED_RESCUE_MAX", "3")
    monkeypatch.setenv("ARIA_MEASURED_RESCUE_PROBE_BUDGET", "9")
    cfg = smr.measured_rescue_config(device="cpu")
    assert cfg is not None
    assert (cfg.tau, cfg.max_rescue, cfg.probe_budget, cfg.device) == (
        0.05,
        3,
        9,
        "cpu",
    )


# ── rescue_skipped_candidates ────────────────────────────────────────────


def test_rescue_affirmative_only(patch_extractor):
    cfg = smr.MeasuredRescueConfig(
        tau=0.01, max_rescue=10, probe_budget=10, device="cpu"
    )
    skipped = [
        (_FakeGraph("capable"), {"_reach": 0.5}, {"predicted_p_s1": 0.1}),  # rescued
        (
            _FakeGraph("incapable"),
            {"_reach": 0.0},
            {"predicted_p_s1": 0.1},
        ),  # below tau
        (_FakeGraph("unmeasurable"), {"_reach": None}, {"predicted_p_s1": 0.1}),  # None
    ]
    rescued, records = smr.rescue_skipped_candidates(skipped, cfg)
    assert [g.fingerprint() for g in rescued] == ["capable"]
    assert records[0]["graph_fingerprint"] == "capable"
    assert records[0]["measured_long_range_reach"] == 0.5
    assert records[0]["rescue_reason"] == "measured_long_range_reach_ge_tau"
    assert records[0]["measured_probe_passed"] is True


def test_rescue_respects_max(patch_extractor):
    cfg = smr.MeasuredRescueConfig(
        tau=0.01, max_rescue=2, probe_budget=10, device="cpu"
    )
    skipped = [(_FakeGraph(f"g{i}"), {"_reach": 0.9}, {}) for i in range(5)]
    rescued, _ = smr.rescue_skipped_candidates(skipped, cfg)
    assert len(rescued) == 2


def test_rescue_respects_probe_budget(patch_extractor):
    cfg = smr.MeasuredRescueConfig(
        tau=0.01, max_rescue=10, probe_budget=2, device="cpu"
    )
    skipped = [(_FakeGraph(f"g{i}"), {"_reach": 0.9}, {}) for i in range(5)]
    rescued, _ = smr.rescue_skipped_candidates(skipped, cfg)
    assert len(rescued) == 2  # only 2 probed before budget exhausted
    assert patch_extractor[0].probed == 2


def test_rescue_failsafe_on_extractor_error(monkeypatch):
    import research.tools.measured_descriptors as md

    def _boom(device=None, n_seeds: int = 1):
        raise RuntimeError("no GPU")

    monkeypatch.setattr(md, "MeasuredDescriptorExtractor", _boom)
    cfg = smr.MeasuredRescueConfig(tau=0.01, max_rescue=5, probe_budget=5, device="cpu")
    rescued, records = smr.rescue_skipped_candidates(
        [(_FakeGraph("g"), {"_reach": 0.9}, {})], cfg
    )
    assert rescued == [] and records == []


# ── _partition_prescreener_candidates integration ────────────────────────


def test_partition_default_off_is_unchanged():
    """rescue_cfg=None → every sub-floor candidate recorded as predictor_skip, no rescues."""
    nb = _FakeNB()
    scored = [
        _scored_row(0.9, "keep", 0.5),  # >= floor → kept
        _scored_row(0.1, "drop1", 0.5),  # < floor → skip
        _scored_row(0.2, "drop2", 0.0),  # < floor → skip
    ]
    kept, skipped, records = phase3._partition_prescreener_candidates(
        nb, scored, exp_id="e", p_pass_floor=0.5, floor_source="test", rescue_cfg=None
    )
    assert [g.fingerprint() for g, _ in kept] == ["keep"]
    assert skipped == 2
    assert records == []
    assert {r["status"] for r in nb.recorded} == {"predictor_skip"}
    assert len(nb.recorded) == 2


def test_partition_rescues_and_does_not_record_skip(monkeypatch):
    """rescue_cfg set → rescued graph joins kept (sentinel rank), is NOT recorded as a skip."""
    nb = _FakeNB()
    drop_capable = _scored_row(0.1, "drop_capable", 0.9)
    drop_incapable = _scored_row(0.1, "drop_incapable", 0.0)
    scored = [_scored_row(0.9, "keep", 0.5), drop_capable, drop_incapable]

    def _fake_rescue(would_skip, cfg):
        # rescue exactly the "drop_capable" graph
        for g, _gd, _m in would_skip:
            if g.fingerprint() == "drop_capable":
                return [g], [
                    {
                        "graph_fingerprint": "drop_capable",
                        "measured_long_range_reach": 0.9,
                    }
                ]
        return [], []

    monkeypatch.setattr(phase3, "rescue_skipped_candidates", _fake_rescue)
    cfg = smr.MeasuredRescueConfig(
        tau=0.01, max_rescue=8, probe_budget=64, device="cpu"
    )
    kept, skipped, records = phase3._partition_prescreener_candidates(
        nb, scored, exp_id="e", p_pass_floor=0.5, floor_source="test", rescue_cfg=cfg
    )
    kept_fps = [g.fingerprint() for g, _ in kept]
    assert "keep" in kept_fps and "drop_capable" in kept_fps
    # rescued graph rides the explore tail at the sentinel rank
    rescued_rank = dict((g.fingerprint(), r) for g, r in kept)["drop_capable"]
    assert rescued_rank == phase3._RANK_COMPOSITE_USABLE_CUTOFF
    assert skipped == 1  # only the incapable one
    assert len(records) == 1
    recorded_fps = [r["graph"].fingerprint() for r in nb.recorded]
    assert recorded_fps == ["drop_incapable"]  # capable one not recorded as skip


def test_rescue_record_includes_predictor_and_floor_evidence(patch_extractor):
    cfg = smr.MeasuredRescueConfig(tau=0.01, max_rescue=1, probe_budget=1, device="cpu")
    skipped = [
        (
            _FakeGraph("capable"),
            {"_reach": 0.9},
            {
                "predicted_p_s1": 0.1234567,
                "predicted_induction_screening_auc": 0.2345678,
                "predicted_p_induction_learner": 0.3456789,
                "predictor_planning_score": 0.4567891,
                "screening_ensemble_p_pass_floor": 0.8581175411857843,
                "screening_ensemble_p_pass_floor_source": "test_floor",
                "predicted_rank_composite": 0.777,
            },
        )
    ]

    _rescued, records = smr.rescue_skipped_candidates(skipped, cfg)

    assert records == [
        {
            "graph_fingerprint": "capable",
            "rescue_reason": "measured_long_range_reach_ge_tau",
            "measured_probe_passed": True,
            "structural_induction_signal": "long_range_reach",
            "measured_long_range_reach": 0.9,
            "measured_content_dependence": 0.1,
            "predicted_p_s1": 0.123457,
            "predicted_induction_screening_auc": 0.234568,
            "predicted_p_induction_learner": 0.345679,
            "predictor_planning_score": 0.456789,
            "screening_ensemble_p_pass_floor": 0.8581175411857843,
            "screening_ensemble_p_pass_floor_source": "test_floor",
            "predicted_rank_composite": 0.777,
        }
    ]


def test_measured_rescue_records_persist_and_track_downstream_outcome():
    results = {"funnel_counts": {}}
    mro.initialize_measured_rescue_records(
        results,
        [
            {
                "graph_fingerprint": "rescued_fp",
                "measured_long_range_reach": 0.25,
                "predicted_p_s1": 0.1,
            }
        ],
        experiment_id="exp",
        tau=0.01,
        max_rescue=8,
        probe_budget=64,
    )

    mro.mark_measured_rescue_screening(results, "rescued_fp", index=3)
    mro.mark_measured_rescue_stage0_attempted(results, "rescued_fp")
    mro.mark_measured_rescue_stage0_result(
        results,
        "rescued_fp",
        stage0_passed=True,
        stage05_passed=True,
        stability_score=0.97,
    )
    mro.mark_measured_rescue_rapid_result(results, "rescued_fp", passed=True)
    mro.mark_measured_rescue_stage1_queued(results, "rescued_fp")
    mro.mark_measured_rescue_stage1_result(
        results,
        "rescued_fp",
        completed=True,
        passed=True,
        result_id="rid",
        loss_ratio=0.52,
        final_loss=2.1,
    )

    record = results["measured_rescue_records"][0]
    assert record["experiment_id"] == "exp"
    assert record["reached_screening"] is True
    assert record["screening_index"] == 3
    assert record["stage0_passed"] is True
    assert record["stage05_passed"] is True
    assert record["reached_rapid_screening"] is True
    assert record["rapid_screening_passed"] is True
    assert record["stage1_queued"] is True
    assert record["stage1_completed"] is True
    assert record["stage1_passed"] is True
    assert record["result_id"] == "rid"
    assert record["loss_ratio"] == 0.52
    assert record["final_loss"] == 2.1


def test_measured_rescue_metrics_for_persisted_program_row():
    results = {"funnel_counts": {}}
    mro.initialize_measured_rescue_records(
        results,
        [
            {
                "graph_fingerprint": "rescued_fp",
                "rescue_reason": "measured_long_range_reach_ge_tau",
                "measured_long_range_reach": 0.25,
                "predicted_p_s1": 0.1,
            }
        ],
        experiment_id="exp",
    )

    metrics = mro.measured_rescue_metrics_for_fingerprint(results, "rescued_fp")

    assert metrics["measured_rescue_candidate"] == 1
    assert metrics["measured_rescue_reason"] == "measured_long_range_reach_ge_tau"
    assert metrics["measured_long_range_reach"] == 0.25
    assert metrics["predicted_p_s1"] == 0.1
