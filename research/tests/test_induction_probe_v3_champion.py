import json
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_induction_v3_selects_default_and_extended_protocol(monkeypatch):
    from research.eval import induction_probe_v3_champion as probe

    calls = []

    def _fake_run(model, **kwargs):
        calls.append(kwargs)
        return probe.InductionV3Result(
            auc=0.7,
            max_gap_acc=0.8,
            gap_accuracies={4: 0.8, 8: 0.6},
            gap_accuracy_cv=0.1429,
            steps_trained=kwargs["n_train_steps"],
        )

    monkeypatch.setattr(probe, "_run_induction_v3_median", _fake_run)

    default = probe.run_induction_v3_champion(object(), device="cpu")
    extended = probe.run_induction_v3_champion(
        object(),
        device="cpu",
        extended_budget=True,
    )
    explicit = probe.run_induction_v3_champion(
        object(),
        device="cpu",
        n_train_steps=10_000,
    )

    assert calls[0]["n_train_steps"] == 5_000
    assert default.steps_trained == 5_000
    assert default.protocol_version == "induction_v3_head_counterfactual_5k"
    assert calls[1]["n_train_steps"] == 10_000
    assert extended.steps_trained == 10_000
    assert extended.protocol_version == "induction_v3_head_counterfactual_10k"
    assert calls[2]["n_train_steps"] == 10_000
    assert explicit.protocol_version == "induction_v3_head_counterfactual_10k"


def test_induction_v3_rejects_unversioned_budget():
    from research.eval.induction_probe_v3_champion import select_induction_v3_budget

    with pytest.raises(ValueError, match="explicit champion budgets"):
        select_induction_v3_budget(n_train_steps=7_500)


def test_induction_v3_result_to_dict_uses_separate_fields():
    from research.eval.induction_probe_v3_champion import InductionV3Result

    result = InductionV3Result(
        auc=0.91,
        max_gap_acc=0.95,
        gap_accuracies={4: 0.95, 8: 0.87},
        gap_accuracy_cv=0.044,
        steps_trained=5_000,
        protocol_version="induction_v3_head_counterfactual_5k",
    )

    data = result.to_dict()

    assert data["induction_v3_auc"] == pytest.approx(0.91)
    assert data["induction_v3_gap_accuracy_cv"] == pytest.approx(0.044)
    assert (
        data["induction_v3_protocol_version"] == "induction_v3_head_counterfactual_5k"
    )
    assert not any(key.startswith("induction_v2_investigation") for key in data)


def test_induction_v3_counterfactual_batches_change_target_binding():
    import torch

    from research.eval import induction_probe_v3_champion as probe

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


def test_induction_v3_freezes_backbone_and_infers_readout_dim():
    import torch.nn as nn

    from research.eval import induction_probe_v3_champion as probe

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_dim = 4
            self.proj = nn.Linear(4, 4)

    model = TinyBackbone()
    probe._freeze_backbone(model)

    assert probe._infer_model_dim(model) == 4
    assert not any(param.requires_grad for param in model.parameters())


def test_investigation_probe_helper_keeps_induction_v3_separate(monkeypatch):
    from research.scientist.runner._helpers_benchmark import (
        _run_investigation_v2_probes,
    )

    captured = {}
    induction_v2_result = SimpleNamespace(
        auc=0.12,
        max_gap_acc=0.34,
        gap_accuracies={4: 0.2},
        steps_trained=500,
        status="ok",
        elapsed_ms=123.0,
        protocol_version="induction_v2_test",
    )
    induction_v3_result = SimpleNamespace(
        auc=0.91,
        max_gap_acc=0.95,
        gap_accuracy_cv=0.044,
        gap_accuracies={4: 0.95, 8: 0.87},
        steps_trained=10_000,
        status="ok",
        elapsed_ms=456.0,
        protocol_version="induction_v3_head_counterfactual_10k",
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
        return induction_v3_result

    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_probe_v2_investigation",
        SimpleNamespace(
            run_induction_v2_investigation=lambda model, device: induction_v2_result
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_probe_v3_champion",
        SimpleNamespace(run_induction_v3_champion=_run_v3),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_probe_v2_investigation",
        SimpleNamespace(
            run_binding_v2_investigation=lambda model, device: binding_result
        ),
    )

    updates = _run_investigation_v2_probes(
        object(),
        "cpu",
        run_induction_v3=True,
        induction_v3_extended_budget=True,
    )

    assert captured["extended_budget"] is True
    assert updates["induction_v2_investigation_auc"] == pytest.approx(0.12)
    assert updates["induction_v2_investigation_protocol_version"] == "induction_v2_test"
    assert updates["induction_v3_auc"] == pytest.approx(0.91)
    assert updates["induction_v3_gap_accuracy_cv"] == pytest.approx(0.044)
    assert (
        updates["induction_v3_protocol_version"]
        == "induction_v3_head_counterfactual_10k"
    )
    assert json.loads(updates["induction_v3_gap_accuracies_json"]) == {
        "4": 0.95,
        "8": 0.87,
    }


def test_champion_snapshot_writes_v3_without_induction_v2_overwrite(monkeypatch):
    from research.scientist.runner.execution_champion_confirmation import (
        ChampionConfirmationEvaluator,
    )

    induction_v3_result = SimpleNamespace(
        to_dict=lambda: {
            "induction_v3_auc": 0.88,
            "induction_v3_max_gap_acc": 0.9,
            "induction_v3_gap_accuracy_cv": 0.05,
            "induction_v3_gap_accuracies": {4: 0.9, 8: 0.86},
            "induction_v3_steps_trained": 5_000,
            "induction_v3_status": "ok",
            "induction_v3_elapsed_ms": 321.0,
            "induction_v3_protocol_version": "induction_v3_head_counterfactual_5k",
        }
    )
    binding_result = SimpleNamespace(
        to_dict=lambda: {
            "binding_v2_investigation_auc": 0.19,
            "binding_v2_investigation_max_distance_acc": 0.33,
            "binding_v2_investigation_distance_accuracies": {8: 0.4},
            "binding_v2_investigation_train_steps": 2400,
            "binding_v2_investigation_status": "ok",
            "binding_v2_investigation_elapsed_ms": 456.7,
            "binding_v2_investigation_protocol_version": "binding_v2_test",
        }
    )

    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_probe_v3_champion",
        SimpleNamespace(
            run_induction_v3_champion=lambda model, **kwargs: induction_v3_result
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_probe_v2_investigation",
        SimpleNamespace(
            run_binding_v2_investigation=lambda model, device: binding_result
        ),
    )

    snapshot = {}
    ChampionConfirmationEvaluator(object())._scale_up_run_investigation_v2_snapshot(
        object(),
        "cpu",
        snapshot,
        SimpleNamespace(champion_induction_v3_extended_budget=False),
    )

    assert snapshot["induction_v3_auc"] == pytest.approx(0.88)
    assert (
        snapshot["induction_v3_protocol_version"]
        == "induction_v3_head_counterfactual_5k"
    )
    assert json.loads(snapshot["induction_v3_gap_accuracies_json"]) == {
        "4": 0.9,
        "8": 0.86,
    }
    assert "induction_v2_investigation_auc" not in snapshot
    assert snapshot["binding_v2_investigation_auc"] == pytest.approx(0.19)
