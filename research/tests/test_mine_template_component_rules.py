from research.tools.mine_template_component_rules import mine_template_component_rules


def test_mine_template_component_rules_samples_named_template() -> None:
    report = mine_template_component_rules(
        template_names=("residual_block",),
        seeds_per_template=1,
        model_dim=32,
        min_window_ops=4,
    )

    assert report["templates_requested"] == 1
    assert report["templates_sampled"] == 1
    assert report["template_summaries"][0]["template"] == "residual_block"
    assert report["template_summaries"][0]["max_ops"] >= 1


def test_mine_template_component_rules_records_unknown_template_failure() -> None:
    report = mine_template_component_rules(
        template_names=("missing_template",),
        seeds_per_template=1,
    )

    assert report["templates_sampled"] == 0
    assert report["failures"][0]["error"] == "unknown_template"
