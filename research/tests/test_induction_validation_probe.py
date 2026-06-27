import json
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_induction_validation_selects_default_and_extended_protocol(monkeypatch):
    from research.eval import induction_validation_probe as probe

    calls = []

    def _fake_run(model, **kwargs):
        calls.append(kwargs)
        return probe.InductionValidationResult(
            auc=0.7,
            max_gap_acc=0.8,
            gap_accuracies={4: 0.8, 8: 0.6},
            gap_accuracy_cv=0.1429,
            steps_trained=kwargs["n_train_steps"],
        )

    monkeypatch.setattr(probe, "_run_induction_validation_median", _fake_run)

    default = probe.run_induction_validation_champion(object(), device="cpu")
    extended = probe.run_induction_validation_champion(
        object(),
        device="cpu",
        extended_budget=True,
    )
    explicit = probe.run_induction_validation_champion(
        object(),
        device="cpu",
        n_train_steps=10_000,
    )

    assert calls[0]["n_train_steps"] == 2_000
    assert default.steps_trained == 2_000
    assert default.protocol_version == "induction_validation_full_counterfactual_2k"
    assert calls[1]["n_train_steps"] == 5_000
    assert extended.steps_trained == 5_000
    assert extended.protocol_version == "induction_validation_full_counterfactual_5k"
    assert calls[2]["n_train_steps"] == 10_000
    assert explicit.protocol_version == "induction_validation_full_counterfactual_10k"


def test_induction_validation_rejects_unversioned_budget():
    from research.eval.induction_validation_probe import (
        select_induction_validation_budget,
    )

    with pytest.raises(ValueError, match="explicit champion budgets"):
        select_induction_validation_budget(n_train_steps=7_500)


def test_induction_validation_result_to_dict_uses_separate_fields():
    from research.eval.induction_validation_probe import InductionValidationResult

    result = InductionValidationResult(
        auc=0.91,
        max_gap_acc=0.95,
        gap_accuracies={4: 0.95, 8: 0.87},
        gap_accuracy_cv=0.044,
        steps_trained=2_000,
        protocol_version="induction_validation_full_counterfactual_2k",
    )

    data = result.to_dict()

    assert data["induction_validation_auc"] == pytest.approx(0.91)
    assert data["induction_validation_gap_accuracy_cv"] == pytest.approx(0.044)
    assert (
        data["induction_validation_protocol_version"]
        == "induction_validation_full_counterfactual_2k"
    )
    assert not any(key.startswith("induction_intermediate") for key in data)


def test_induction_validation_counterfactual_batches_change_target_binding():
    import torch

    from research.eval import induction_validation_probe as probe

    generator = torch.Generator(device="cpu").manual_seed(123)
    normal, targets, cf_inputs, cf_targets = (
        probe._generate_counterfactual_binding_batches(
            2,
            3,
            4,
            pairs_per_example=4,
            device="cpu",
            generator=generator,
        )
    )

    assert normal.shape == cf_inputs.shape
    assert targets.shape == cf_targets.shape == (2, 3)
    assert torch.all(targets != cf_targets)
    assert torch.equal(normal[:, :, -1], cf_inputs[:, :, -1])
    assert torch.any(normal != cf_inputs)


def test_investigation_probe_helper_keeps_induction_validation_separate(monkeypatch):
    from research.scientist.runner._helpers_benchmark import (
        _run_investigation_v2_probes,
    )

    captured = {}
    induction_intermediate_result = SimpleNamespace(
        auc=0.12,
        max_gap_acc=0.34,
        gap_accuracies={4: 0.2},
        steps_trained=500,
        status="ok",
        elapsed_ms=123.0,
        protocol_version="induction_intermediate_test",
    )
    induction_validation_result = SimpleNamespace(
        auc=0.91,
        max_gap_acc=0.95,
        gap_accuracy_cv=0.044,
        gap_accuracies={4: 0.95, 8: 0.87},
        steps_trained=10_000,
        status="ok",
        elapsed_ms=456.0,
        protocol_version="induction_validation_full_counterfactual_10k",
    )
    binding_result = SimpleNamespace(
        auc=0.56,
        max_distance_acc=0.78,
        distance_accuracies={4: 0.7},
        train_steps=2400,
        status="ok",
        elapsed_ms=789.0,
        protocol_version="binding-test",
    )

    def _run_v3(model, **kwargs):
        captured.update(kwargs)
        return induction_validation_result

    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_intermediate_probe",
        SimpleNamespace(
            run_induction_intermediate=lambda model, device: (
                induction_intermediate_result
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_validation_probe",
        SimpleNamespace(run_induction_validation_champion=_run_v3),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_intermediate_probe",
        SimpleNamespace(run_binding_intermediate=lambda model, device: binding_result),
    )

    updates = _run_investigation_v2_probes(
        object(),
        "cpu",
        run_induction_validation=True,
        induction_validation_extended_budget=True,
    )

    assert captured["extended_budget"] is True
    assert updates["induction_intermediate_auc"] == pytest.approx(0.12)
    assert (
        updates["induction_intermediate_protocol_version"]
        == "induction_intermediate_test"
    )
    assert updates["induction_validation_auc"] == pytest.approx(0.91)
    assert updates["induction_validation_gap_accuracy_cv"] == pytest.approx(0.044)
    assert (
        updates["induction_validation_protocol_version"]
        == "induction_validation_full_counterfactual_10k"
    )
    assert json.loads(updates["induction_validation_gap_accuracies_json"]) == {
        "4": 0.95,
        "8": 0.87,
    }


def test_champion_snapshot_writes_v3_without_induction_intermediate_overwrite(
    monkeypatch,
):
    from research.scientist.runner.execution_champion_confirmation import (
        ChampionConfirmationEvaluator,
    )

    induction_validation_result = SimpleNamespace(
        to_dict=lambda: {
            "induction_validation_auc": 0.88,
            "induction_validation_max_gap_acc": 0.9,
            "induction_validation_gap_accuracy_cv": 0.05,
            "induction_validation_gap_accuracies": {4: 0.9, 8: 0.86},
            "induction_validation_steps_trained": 2_000,
            "induction_validation_status": "ok",
            "induction_validation_elapsed_ms": 321.0,
            "induction_validation_protocol_version": "induction_validation_full_counterfactual_2k",
        }
    )
    binding_result = SimpleNamespace(
        to_dict=lambda: {
            "binding_intermediate_auc": 0.19,
            "binding_intermediate_max_distance_acc": 0.33,
            "binding_intermediate_distance_accuracies": {8: 0.4},
            "binding_intermediate_train_steps": 2400,
            "binding_intermediate_status": "ok",
            "binding_intermediate_elapsed_ms": 456.7,
            "binding_intermediate_protocol_version": "binding_intermediate_test",
        }
    )

    captured = {}

    def _run_v3(model, **kwargs):
        captured["induction_device"] = kwargs.get("device")
        return induction_validation_result

    def _run_binding(model, device):
        captured["binding_device"] = device
        return binding_result

    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_validation_probe",
        SimpleNamespace(run_induction_validation_champion=_run_v3),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_intermediate_probe",
        SimpleNamespace(run_binding_intermediate=_run_binding),
    )

    snapshot = {}
    ChampionConfirmationEvaluator(object())._scale_up_run_investigation_v2_snapshot(
        object(),
        "cuda",
        snapshot,
        SimpleNamespace(champion_induction_validation_extended_budget=False),
    )

    assert captured == {"induction_device": "cuda", "binding_device": "cuda"}
    assert snapshot["induction_validation_auc"] == pytest.approx(0.88)
    assert (
        snapshot["induction_validation_protocol_version"]
        == "induction_validation_full_counterfactual_2k"
    )
    assert json.loads(snapshot["induction_validation_gap_accuracies_json"]) == {
        "4": 0.9,
        "8": 0.86,
    }
    assert "induction_intermediate_auc" not in snapshot
    assert snapshot["binding_intermediate_auc"] == pytest.approx(0.19)


def test_champion_milestone_evals_skip_cpu_without_escape_hatch():
    import torch

    from research.scientist.runner.execution_champion_confirmation import (
        ChampionConfirmationEvaluator,
    )

    program_metrics = {}
    ChampionConfirmationEvaluator(object())._scale_up_champion_milestone_evals(
        exp_id="exp",
        source_result_id="result-123456",
        prog_idx=0,
        graph=object(),
        config=SimpleNamespace(mode="confirmation", scale_up_steps=40_000),
        dev=torch.device("cpu"),
        dev_str="cpu",
        s1_passed=True,
        program_metrics=program_metrics,
    )

    payload = json.loads(program_metrics["external_benchmarks_json"])
    assert payload["champion_confirmation_milestones"][0]["status"] == (
        "missing_accelerator"
    )
