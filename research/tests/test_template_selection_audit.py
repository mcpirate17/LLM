from research.synthesis.templates import DEFAULT_TEMPLATE_WEIGHTS, TEMPLATES


RETIRED_TEMPLATE_NAMES = {
    "attn_reciprocal_gated",
    "attn_softmax_router_sidecar",
    "multiscale_difficulty_router_blocksparse_attn_ssm",
    "multiscale_difficulty_router_easy_attn_ssm",
}


def test_all_registered_templates_have_default_weights() -> None:
    missing = sorted(set(TEMPLATES) - set(DEFAULT_TEMPLATE_WEIGHTS))
    assert missing == []


def test_only_retired_templates_are_zero_weight() -> None:
    zero_weight = {
        name
        for name, weight in DEFAULT_TEMPLATE_WEIGHTS.items()
        if float(weight) == 0.0
    }
    assert zero_weight == set()
    assert RETIRED_TEMPLATE_NAMES.isdisjoint(TEMPLATES)


def test_all_non_retired_templates_are_selectable_by_default_weight() -> None:
    non_selectable = sorted(
        name
        for name, weight in DEFAULT_TEMPLATE_WEIGHTS.items()
        if name in TEMPLATES
        and name not in RETIRED_TEMPLATE_NAMES
        and float(weight) <= 0.0
    )
    assert non_selectable == []
