import pytest

from research.eval.pruning import run_dense_vs_structured_sparse_ablation
from research.synthesis.kernels import validate_numerical_stability

pytestmark = pytest.mark.unit


def test_dense_vs_structured_sparse_ablation_reports_accuracy_and_speed():
    report = run_dense_vs_structured_sparse_ablation(
        model_dim=64,
        vocab_size=512,
        seq_len=24,
        batch_size=2,
        steps=6,
    )
    rows = report.get("rows", [])
    assert rows, "Expected ablation rows."

    dense = next(
        (r for r in rows if r.get("label") == "dense" and r.get("passed")), None
    )
    assert dense is not None, "Dense baseline must run."

    sparse_rows = [
        r for r in rows if r.get("label") in {"nm_2_4", "block_16"} and r.get("passed")
    ]
    assert sparse_rows, "At least one sparse variant should run."

    for row in sparse_rows:
        assert row.get("avg_step_ms", 0.0) > 0.0
        assert row.get("final_loss") is not None
        # Accuracy guardrail: sparse should remain within a broad degradation bound.
        ratio = float(row.get("loss_ratio_vs_dense", 1.0))
        assert ratio < 20.0
        # Speed telemetry must be surfaced even if speedup is <1 on CPU fallback.
        assert "speedup_vs_dense" in row


def test_kernel_numerical_stability_report_structure():
    report = validate_numerical_stability()
    assert "available" in report
    assert "checked" in report
    if not report.get("available"):
        assert report.get("reason") == "cuda_required_for_triton_validation"
        return

    checked = report.get("checked", [])
    assert checked, "CUDA run should produce kernel stability rows."
    assert "all_passed" in report
