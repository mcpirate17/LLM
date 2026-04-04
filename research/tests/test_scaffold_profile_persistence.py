from research.scientist.notebook import LabNotebook


def test_scaffold_profile_run_and_result_persist(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    try:
        nb.save_scaffold_profile_run(
            run_id="run123",
            config={"stage1_steps": 8},
            device="cuda",
            metadata={"families": ["gpt2_attn"]},
        )
        result_id = nb.save_scaffold_profile_result(
            run_id="run123",
            family="gpt2_attn",
            case_name="gpt2_attn:linear_attention",
            status="ok",
            metrics={
                "compile_time_ms": 12.5,
                "sandbox_passed": True,
                "passed": False,
                "loss_ratio": 0.52,
                "validation_loss_ratio": 0.49,
                "throughput_tok_s": 1234.5,
            },
            graph_json='{"metadata":{"scaffold_family":"gpt2_attn"}}',
            graph_fingerprint="fp123",
            op_a="linear_attention",
        )
        assert result_id
        rows = nb.list_scaffold_profile_results(run_id="run123", limit=10)
        assert len(rows) == 1
        row = rows[0]
        assert row["run_id"] == "run123"
        assert row["family"] == "gpt2_attn"
        assert row["case_name"] == "gpt2_attn:linear_attention"
        assert row["graph_fingerprint"] == "fp123"
        assert row["metrics"]["loss_ratio"] == 0.52
        stats = nb.get_scaffold_component_stats(min_support=1)
        assert "linear_attention" in stats
        assert stats["linear_attention"]["support"] == 1
        assert stats["linear_attention"]["families"]["gpt2_attn"] == 1
    finally:
        nb.close()
