from __future__ import annotations

from research.tools.meta_surrogate_analysis import GraphRow
from research.tools.meta_surrogate_screening_proposal import (
    _template_policy_rows,
    escape_hatches,
)


def test_escape_hatches_capture_strong_metrics_and_mixer_presence():
    row = GraphRow(
        result_id="r1",
        template_name="salvage_tpl",
        failure_op="",
        wikitext_perplexity=42.0,
        tinystories_score=0.61,
        controlled_lang_s05_sa_score=0.98,
        motif_count=4,
        non_norm_motif_count=2,
        norm_motif_count=2,
        norm_dominance=0.5,
        has_attention_motif=1,
        has_ssm_motif=0,
        has_conv_motif=0,
        has_recurrent_motif=0,
        has_routing_motif=1,
        has_compression_motif=1,
        has_effective_positional_mixer=1,
        mixer_after_compression=1,
        motif_thinness_score=0.5,
        frequency_collapse_risk=0.55,
    )

    assert escape_hatches(row, {"salvage_tpl"}) == [
        "strong_wikitext",
        "strong_tinystories",
        "strong_controlled_sa",
        "effective_positional_mixer",
        "routing_present",
        "explicit_mixer_family",
        "salvageable_template_family",
    ]


def test_template_policy_rows_bucket_actions():
    rows = [
        {
            "template_name": "bad",
            "n": 100,
            "nano_bind_rate": 0.4,
            "high_freq_risk_rate": 0.8,
            "effective_pos_mixer_rate": 0.05,
            "ppl_lt_50": 3,
        },
        {
            "template_name": "rescue",
            "n": 100,
            "nano_bind_rate": 0.15,
            "high_freq_risk_rate": 0.4,
            "effective_pos_mixer_rate": 0.2,
            "ppl_lt_50": 1,
        },
        {
            "template_name": "mine",
            "n": 100,
            "nano_bind_rate": 0.0,
            "high_freq_risk_rate": 0.1,
            "effective_pos_mixer_rate": 0.8,
            "ppl_lt_50": 0,
        },
    ]

    actions = {
        row["template_name"]: row["action"] for row in _template_policy_rows(rows)
    }

    assert actions == {
        "bad": "downweight_or_constrain",
        "rescue": "rescue_by_slot_fill",
        "mine": "mine",
    }
