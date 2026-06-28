"""Shared stage-1 program-result kwargs builder for notebook/API tests.

Several suites (test_discoveries_api, test_program_result_merge_patch, ...) need
a representative stage-1 metrics payload to seed `record_program_result`. They
all wrapped the same s1 dict + extra block around
`program_result_kwargs_from_s1`; this is the single source of truth so the
fixture can't drift between files.
"""

from __future__ import annotations

from typing import Any

from research.scientist.runner._helpers import program_result_kwargs_from_s1


def stage1_kwargs(
    *,
    loss_ratio: float = 0.42,
    novelty_score: float = 0.66,
    model_source: str = "graph_synthesis",
    **extra: Any,
) -> dict[str, Any]:
    """Return `record_program_result` kwargs for a passing stage-1 sample.

    Extra keyword args are merged into the metrics `extra` block, so callers can
    override or add fields (e.g. ``final_loss=1.8``, ``cka_source="deferred"``).
    """
    return program_result_kwargs_from_s1(
        {
            "passed": True,
            "final_loss": 4.5,
            "loss_ratio": loss_ratio,
            "wikitext_perplexity": 150.0,
            "wikitext_score": 0.55,
            "screening_wikitext_metric_version": "unit_test_wikitext_v1",
            "hellaswag_acc": 0.31,
            "hellaswag_status": "ran",
            "blimp_overall_accuracy": 0.55,
            "blimp_status": "ran",
            "induction_screening_auc": 0.21,
            "binding_screening_auc": 0.18,
            "binding_screening_composite": 0.12,
            "ar_legacy_auc": 0.06,
        },
        model_source=model_source,
        extra={
            "stage1_passed": True,
            "novelty_score": novelty_score,
            "data_mode": "random",
            "tokenizer_mode": "byte",
            "vocab_size": 256,
            **extra,
        },
    )
