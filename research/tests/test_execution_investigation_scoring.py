from __future__ import annotations

from research.scientist.runner.execution_investigation_scoring import (
    build_investigation_entry,
    classify_investigation_failures,
    summarize_investigation_program_runs,
)
from research.scientist.thresholds import (
    INVESTIGATION_BRITTLE_OVERRIDE_LR,
    INVESTIGATION_EARLY_PASS_LR,
)


def test_classify_investigation_failures_separates_infra_from_real_failures():
    infra_failures, real_failures = classify_investigation_failures(
        [
            {"passed": False, "error": "CUDA out of memory"},
            {"passed": False, "error": "device-side assert triggered"},
            {"passed": False, "error": "loss plateaued"},
            {"passed": True, "error": None},
        ]
    )

    assert infra_failures == 2
    assert real_failures == 1


def test_summarize_investigation_program_runs_builds_candidate_summary():
    summary = summarize_investigation_program_runs(
        tp_results=[
            {"training_program": "tp_a", "passed": True, "loss_ratio": 0.42},
            {
                "training_program": "tp_b",
                "passed": False,
                "loss_ratio": 0.65,
                "error": "loss plateaued",
            },
            {"training_program": "tp_c", "passed": True, "loss_ratio": 0.31},
        ],
        screening_lr=0.55,
        investigation_max_loss_ratio_multiplier=10.0,
        loss_multiplier_fn=lambda screening_lr, best_lr: best_lr / screening_lr,
    )

    assert summary.n_passed == 2
    assert summary.robustness == 2 / 3
    assert summary.best_tp is not None
    assert summary.best_tp["training_program"] == "tp_c"
    assert summary.best_lr == 0.31
    assert summary.brittle_risk is False
    assert summary.investigation_passed_early == (
        summary.best_lr < INVESTIGATION_EARLY_PASS_LR
    )
    assert summary.training_errors == ["loss plateaued"]


def test_summarize_investigation_program_runs_allows_brittle_override_for_strong_loss():
    best_lr = INVESTIGATION_BRITTLE_OVERRIDE_LR / 2.0
    summary = summarize_investigation_program_runs(
        tp_results=[
            {"training_program": "tp_a", "passed": True, "loss_ratio": best_lr},
        ],
        screening_lr=max(best_lr / 100.0, 1e-6),
        investigation_max_loss_ratio_multiplier=2.0,
        loss_multiplier_fn=lambda screening_lr, best_lr: best_lr / screening_lr,
    )

    assert summary.brittle_risk is True
    assert summary.investigation_passed_early is True


def test_build_investigation_entry_shapes_result_payload():
    summary = summarize_investigation_program_runs(
        tp_results=[
            {
                "training_program": "tp_x",
                "passed": True,
                "loss_ratio": INVESTIGATION_EARLY_PASS_LR / 2.0,
            }
        ],
        screening_lr=0.6,
        investigation_max_loss_ratio_multiplier=10.0,
        loss_multiplier_fn=lambda screening_lr, best_lr: best_lr / screening_lr,
    )

    config = type(
        "Cfg",
        (),
        {
            "data_mode": "corpus",
            "hf_dataset": "tinystories",
            "corpus_path": "",
        },
    )()
    entry = build_investigation_entry(
        source_result_id="rid_123",
        config=config,
        source={
            "loss_ratio": 0.6,
            "baseline_loss_ratio": 0.5,
            "novelty_confidence": 0.8,
        },
        tp_sched={"scheduling_avg_ms": 1.5, "scheduling_max_ms": 3.0},
        n_programs_tested=1,
        fingerprint_incomplete=False,
        summary=summary,
    )

    assert entry["result_id"] == "rid_123"
    assert entry["data_mode"] == "corpus"
    assert entry["data_source"] == "tinystories"
    assert entry["best_training_program"] == "tp_x"
    assert entry["n_programs_tested"] == 1
    assert entry["training_program_scheduling_avg_ms"] == 1.5
