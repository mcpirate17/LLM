from research.scientist.notebook import LabNotebook
from research.tools.backfill_templates import (
    _NON_ROUTING_TEMPLATES,
    _phase_settings,
    _scaffold_guided_priors,
)


def test_phase_settings_differentiate_isolation_and_stack():
    isolation = _phase_settings("isolation")
    stack = _phase_settings("stack")
    assert isolation["composition_depth"] == 1
    assert stack["composition_depth"] == 2
    assert stack["n_layers"] > isolation["n_layers"]
    assert stack["stage1_steps"] > isolation["stage1_steps"]


def test_scaffold_guided_priors_build_weights_from_scaffold_stats(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    try:
        nb.save_scaffold_profile_run(run_id="run1", config={"stage1_steps": 8})
        for idx in range(4):
            nb.save_scaffold_profile_result(
                run_id="run1",
                family="gpt2_attn",
                case_name=f"gpt2_attn:linear_attention:{idx}",
                status="ok",
                metrics={
                    "sandbox_passed": True,
                    "passed": True,
                    "loss_ratio": 0.4,
                    "validation_loss_ratio": 0.38,
                    "throughput_tok_s": 3500.0,
                },
                graph_json='{"metadata":{"scaffold_family":"gpt2_attn"}}',
                graph_fingerprint=f"fp{idx}",
                op_a="linear_attention",
            )
        op_weights, category_weights = _scaffold_guided_priors(str(db_path))
        assert op_weights["linear_attention"] > 1.0
        assert "mixing" in category_weights or "sequence" in category_weights
    finally:
        nb.close()


def test_non_routing_template_whitelist_covers_attention_ablations():
    assert "attn_softmax_normalized_matmul" in _NON_ROUTING_TEMPLATES
    assert "attn_linear_no_matmul_ffn" in _NON_ROUTING_TEMPLATES
    assert "attn_linear_matmul_sparse_tail" in _NON_ROUTING_TEMPLATES
    assert "attn_linear_matmul_router_sidecar" not in _NON_ROUTING_TEMPLATES
