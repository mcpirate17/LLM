"""O3 contract pins: NM op proposal weight actually reaches the grammar blocks.

The discovery registry wired the NM-C / NM-F ops into proposable template blocks,
but `_build_compaction_block` / `_build_nmf_block` picked the concrete op with a
uniform `rng.choice(pool)` — so a strong S1 performer (e.g. `cdma_slot_binding`)
and a weak one (e.g. `padic_lowprec_mix`) were drawn with identical probability
forever, regardless of measured success. Their per-op `op_stats` weight (the same
dict `resolve_step` consumes, stashed on `graph.metadata["_op_weights"]`) never
lifted their odds. That decoupling was the next funnel bottleneck flagged in
`research/notes/nm_verification_split_plan_2026-07-02.md` (O3).

These pins assert the coupling is live and cannot silently regress to uniform:
`weighted_op_choice` biases toward higher-weighted ops, and both mixer blocks go
through it — a favored op must dominate an equally-eligible disfavored one across
many builds. A regression to `rng.choice` collapses that gap and fails here (same
silent-drift class as the gates-5/6/8 incident).
"""

from __future__ import annotations

import random

import pytest

from research.synthesis._template_helpers import weighted_op_choice
from research.synthesis._templates_compaction import COMPACTION_OPS
from research.synthesis._templates_nmf import NMF_OPS
from research.synthesis.graph import ComputationGraph
from research.synthesis.templates import TEMPLATES

_HI = 4.5  # op_stats weight clamp ceiling (grammar_support._build_db_op_weights)
_LO = 0.25  # op_stats weight clamp floor


# ── unit: the shared weighted-choice mechanism ──────────────────────────────


def _graph_with_weights(weights: dict[str, float] | None) -> ComputationGraph:
    graph = ComputationGraph(model_dim=64)
    if weights is not None:
        graph.metadata["_op_weights"] = weights
    return graph


def test_weighted_choice_singleton_returns_element() -> None:
    graph = _graph_with_weights({"a": 4.5})
    assert weighted_op_choice(graph, random.Random(0), ("only_op",)) == "only_op"


def test_weighted_choice_uniform_without_weights() -> None:
    """No `_op_weights` attached (cold start / tests) → uniform fallback, every
    op reachable. This is the no-regression guarantee for pre-feedback runs."""
    graph = _graph_with_weights(None)
    pool = COMPACTION_OPS
    rng = random.Random(0)
    counts = {op: 0 for op in pool}
    for _ in range(4000):
        counts[weighted_op_choice(graph, rng, pool)] += 1
    expected = 4000 / len(pool)
    assert all(c > 0 for c in counts.values())  # nothing starved
    # roughly balanced: no op wildly over/under its uniform share
    assert max(counts.values()) < 2.0 * expected
    assert min(counts.values()) > 0.4 * expected


def test_weighted_choice_biases_toward_high_weight() -> None:
    favored, disfavored = COMPACTION_OPS[0], COMPACTION_OPS[1]
    weights = {op: _LO for op in COMPACTION_OPS}
    weights[favored] = _HI
    graph = _graph_with_weights(weights)
    rng = random.Random(0)
    counts = {favored: 0, disfavored: 0}
    for _ in range(4000):
        pick = weighted_op_choice(graph, rng, COMPACTION_OPS)
        if pick in counts:
            counts[pick] += 1
    # 4.5 vs 0.25 = 18x weight → favored must dominate by a wide margin.
    assert counts[favored] > 10 * counts[disfavored]


def test_weighted_choice_skips_zero_weight_when_others_positive() -> None:
    weights = {op: 1.0 for op in COMPACTION_OPS}
    zeroed = COMPACTION_OPS[3]
    weights[zeroed] = 0.0
    graph = _graph_with_weights(weights)
    rng = random.Random(1)
    picks = {weighted_op_choice(graph, rng, COMPACTION_OPS) for _ in range(2000)}
    assert zeroed not in picks


def test_weighted_choice_all_zero_falls_back_uniform() -> None:
    """Degenerate all-zero weights must not raise (rng.choices rejects a
    zero-sum weight vector) — fall back to uniform instead of crashing."""
    weights = {op: 0.0 for op in COMPACTION_OPS}
    graph = _graph_with_weights(weights)
    rng = random.Random(2)
    picks = {weighted_op_choice(graph, rng, COMPACTION_OPS) for _ in range(2000)}
    assert len(picks) > 1  # not starved to a single op


# ── contract: the template blocks consult the weights ───────────────────────


def _op_multiset_over_builds(
    template_name: str,
    weights: dict[str, float] | None,
    n_builds: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seed in range(n_builds):
        graph = ComputationGraph(model_dim=64)
        if weights is not None:
            graph.metadata["_op_weights"] = weights
        inp = graph.add_input()
        out = TEMPLATES[template_name](graph, inp, random.Random(seed), None)
        graph.set_output(out)
        for node in graph.nodes.values():
            counts[node.op_name] = counts.get(node.op_name, 0) + 1
    return counts


@pytest.mark.parametrize(
    "template_name, pool, favored, disfavored",
    [
        (
            "compaction_mixer_block",
            COMPACTION_OPS,
            "token_merge_mix",
            "recurrent_depth_refine",
        ),
        ("nmf_mixer_block", NMF_OPS, "cdma_slot_binding", "nilpotent_lie_scan"),
    ],
)
def test_block_selection_follows_op_weights(
    template_name: str, pool: tuple[str, ...], favored: str, disfavored: str
) -> None:
    """A heavily-weighted op must appear in far more built graphs than an
    equally-eligible disfavored one. If the block regresses to uniform
    `rng.choice`, both land near 1/pool and this gap collapses."""
    assert favored in pool and disfavored in pool
    weights = {op: _LO for op in pool}
    weights[favored] = _HI
    counts = _op_multiset_over_builds(template_name, weights, n_builds=300)
    fav, dis = counts.get(favored, 0), counts.get(disfavored, 0)
    assert fav > 3 * max(dis, 1), (
        f"{template_name}: {favored}={fav} vs {disfavored}={dis} — op weights "
        f"not reaching block selection (regressed to uniform?)"
    )


@pytest.mark.parametrize("template_name", ["compaction_mixer_block", "nmf_mixer_block"])
def test_block_still_builds_without_weights(template_name: str) -> None:
    """No-regression: with no `_op_weights` the blocks still build valid graphs
    (uniform fallback), mirroring the cold-start / registry-wiring contract."""
    counts = _op_multiset_over_builds(template_name, None, n_builds=8)
    assert counts, f"{template_name} produced no ops without weights"
