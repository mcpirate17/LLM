from research.scientist.analytics import ExperimentAnalytics
from research.scientist.notebook import LabNotebook
from research.scientist.runner.selection import _SelectionMixin
from unittest.mock import patch


def _save_scaffold_rows(
    nb: LabNotebook,
    *,
    run_id: str,
    op_name: str,
    count: int,
    family: str = "gpt2_attn",
) -> None:
    nb.save_scaffold_profile_run(run_id=run_id, config={"stage1_steps": 8})
    for idx in range(count):
        nb.save_scaffold_profile_result(
            run_id=run_id,
            family=family,
            case_name=f"{family}:{op_name}:{idx}",
            status="ok",
            metrics={
                "sandbox_passed": True,
                "passed": True,
                "loss_ratio": 0.42,
                "validation_loss_ratio": 0.39,
                "throughput_tok_s": 4200.0,
            },
            graph_json='{"metadata":{"scaffold_family":"gpt2_attn"}}',
            graph_fingerprint=f"fp-{op_name}-{idx}",
            op_a=op_name,
        )


def test_selection_lookup_includes_scaffold_only_op(tmp_path):
    nb = LabNotebook(tmp_path / "lab_notebook.db")
    try:
        _save_scaffold_rows(
            nb,
            run_id="run-scaffold-only",
            op_name="linear_attention",
            count=4,
        )
        lookup = _SelectionMixin()._op_success_lookup(nb)
        assert "linear_attention" in lookup
        assert lookup["linear_attention"] > 0.5
    finally:
        nb.close()


def test_compute_op_weights_blends_scaffold_evidence(tmp_path):
    nb = LabNotebook(tmp_path / "lab_notebook.db")
    try:
        _save_scaffold_rows(
            nb,
            run_id="run-strong-linear",
            op_name="linear_attention",
            count=20,
        )
        analytics = ExperimentAnalytics(nb)
        deduped_rows = [
            {
                "graph_json": '{"nodes":{"0":{"op_name":"linear_attention"}}}',
                "stage0_any_passed": 1,
                "stage1_any_passed": 0,
                "latest_timestamp": 1.0,
            }
            for _ in range(5)
        ] + [
            {
                "graph_json": '{"nodes":{"0":{"op_name":"diff_attention"}}}',
                "stage0_any_passed": 1,
                "stage1_any_passed": 1,
                "latest_timestamp": 1.0,
            }
            for _ in range(5)
        ]
        with patch.object(
            ExperimentAnalytics, "_deduped_graph_rows", return_value=deduped_rows
        ):
            weights = analytics.compute_op_weights(min_used=5)
        assert weights["linear_attention"] > 0.1
        assert weights["diff_attention"] > weights["linear_attention"]
    finally:
        nb.close()
