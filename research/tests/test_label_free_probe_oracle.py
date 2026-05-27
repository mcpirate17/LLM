from research.tools.cpu_screening_cascade import (
    MechProfile,
    Scored,
    _probe_oracle_ar_gate_passes,
    _probe_oracle_downstream_gate_passes,
    _select,
)
from research.tools.label_free_probe_oracle import (
    DEFAULT_RANK_AXES,
    probe_any_axis_gate,
    probe_axis_gate,
    probe_axis_score,
)


def test_probe_axis_score_uses_ar_and_nano_thresholds():
    score, details = probe_axis_score(
        {
            "ar_gate": 0.72,
            "nano_induction_nearest": 0.625,
            "stage1_passed": 1.0,
        },
        {
            "ar_gate": 0.9,
            "nano_induction_nearest": 0.5,
            "stage1_passed": 1.0,
        },
    )

    assert score == 1.25
    assert set(details) == {"ar_gate", "nano_induction_nearest"}
    assert details["nano_induction_nearest"]["ratio"] == 1.25
    assert "stage1_passed" not in details


def test_probe_axis_score_can_rank_without_ar_gate():
    score, details = probe_axis_score(
        {
            "ar_gate": 1.0,
            "nano_induction_nearest": 0.3,
            "induction": 0.21,
            "ar_curriculum": 0.2,
        },
        {
            "ar_gate": 0.9,
            "nano_induction_nearest": 0.5,
            "induction": 0.35,
            "ar_curriculum": 0.5,
        },
        axes=DEFAULT_RANK_AXES,
    )

    assert score == 0.6
    assert "ar_gate" not in details


def test_probe_axis_gate_is_ar_only_no_go_decision():
    gate = probe_axis_gate(
        {"ar_gate": 0.89, "nano_induction_nearest": 0.8},
        {"ar_gate": 0.9, "nano_induction_nearest": 0.5},
    )

    assert gate["passed"] is False
    assert gate["axis"] == "ar_gate"
    assert gate["ratio"] < 1.0


def test_probe_any_axis_gate_requires_downstream_threshold_pass():
    fail = probe_any_axis_gate(
        {
            "nano_induction_nearest": 0.49,
            "induction": 0.34,
            "ar_curriculum": 0.49,
        },
        {
            "nano_induction_nearest": 0.5,
            "induction": 0.35,
            "ar_curriculum": 0.5,
        },
    )
    passed = probe_any_axis_gate(
        {
            "nano_induction_nearest": 0.49,
            "induction": 0.36,
            "ar_curriculum": 0.2,
        },
        {
            "nano_induction_nearest": 0.5,
            "induction": 0.35,
            "ar_curriculum": 0.5,
        },
    )

    assert fail["passed"] is False
    assert fail["best_axis"] == "nano_induction_nearest"
    assert passed["passed"] is True
    assert passed["passed_axes"] == ["induction"]


def _profile(mech: float, novelty: float = 0.0) -> MechProfile:
    return MechProfile(
        n_mix=1,
        mixer_depth=1,
        sum_mem=0.0,
        n_global=0,
        alg_div=1,
        n_novel_mix=0,
        mech_score=mech,
        novelty=novelty,
        lit_family="test",
        lit_model="test",
        lit_match_type="novel",
    )


def test_cpu_cascade_exploit_selection_prefers_non_ar_rank_score():
    high_mech_low_rank = Scored(
        "high-mech",
        ["softmax_attention"],
        _profile(99.0),
        {"score": 1.0},
        {"nodes": {}},
        {
            "label_free_probe_score": 10.0,
            "label_free_probe_rank_score": 0.1,
            "label_free_probe_gate_pass": True,
        },
    )
    low_mech_high_rank = Scored(
        "high-rank",
        ["softmax_attention"],
        _profile(1.0),
        {"score": 1.0},
        {"nodes": {}},
        {
            "label_free_probe_score": 1.0,
            "label_free_probe_rank_score": 2.0,
            "label_free_probe_gate_pass": True,
        },
    )

    selected = _select([high_mech_low_rank, low_mech_high_rank], 1, 0)

    assert [s.fingerprint for s in selected] == ["high-rank"]


def test_cpu_cascade_ar_gate_pass_helper_uses_gate_field():
    assert _probe_oracle_ar_gate_passes({"label_free_probe_gate_pass": True}) is True
    assert _probe_oracle_ar_gate_passes({"label_free_probe_gate_pass": False}) is False


def test_cpu_cascade_downstream_gate_helper_blocks_below_threshold_models():
    assert (
        _probe_oracle_downstream_gate_passes(
            {
                "label_free_probe_rank_axes": {
                    "nano_induction_nearest": {"ratio": 0.8},
                    "induction": {"ratio": 0.4},
                    "ar_curriculum": {"ratio": 0.99},
                }
            }
        )
        is False
    )
    assert (
        _probe_oracle_downstream_gate_passes(
            {"label_free_probe_rank_axes": {"induction": {"ratio": 1.01}}}
        )
        is True
    )
