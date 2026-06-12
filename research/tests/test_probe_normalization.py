"""Unit tests for the canonical probe-metric registry + pure stats helpers."""

from __future__ import annotations

import math

import pytest

from research.scientist.probe_normalization import (
    FAMILY_AR,
    FAMILY_BINDING,
    FAMILY_INDUCTION,
    FAMILY_LOSS,
    PROBE_METRICS,
    PROBE_METRICS_BY_COLUMN,
    agglomerative_clusters,
    aligned_pairs,
    aligned_triples,
    metrics_for_family,
    metrics_for_tier,
    oriented_values,
    partial_spearman,
    rankdata,
    safe_float,
    spearman,
    spearman_ci,
    table_columns,
    template_family,
    variance_decomposition,
)

pytestmark = pytest.mark.unit


def test_registry_covers_induction_binding_ar_loss():
    families = {pm.family for pm in PROBE_METRICS}
    assert FAMILY_INDUCTION in families
    assert FAMILY_BINDING in families
    assert FAMILY_AR in families
    assert FAMILY_LOSS in families


def test_registry_ar_cascade_columns_present():
    ar_cols = {pm.column for pm in metrics_for_family(FAMILY_AR)}
    # Members of the coalesce chain in
    # research/meta_analysis/ar_binding_overlay.py::_AR_EXPR.
    assert "ar_gate_score" in ar_cols
    assert "ar_curriculum_auc_pair_final" in ar_cols
    assert "ar_intermediate_auc" in ar_cols
    assert "ar_validation_rank_score" in ar_cols
    assert "ar_legacy_auc" in ar_cols


def test_registry_columns_unique():
    cols = [pm.column for pm in PROBE_METRICS]
    assert len(cols) == len(set(cols))
    # Lookup map is consistent.
    for pm in PROBE_METRICS:
        assert PROBE_METRICS_BY_COLUMN[pm.column] is pm


def test_safe_float_filters_nan_and_strings():
    assert safe_float(1.5) == 1.5
    assert safe_float("2.0") == 2.0
    assert safe_float(None) is None
    assert safe_float(float("nan")) is None
    assert safe_float("not-a-number") is None


def test_template_family_buckets():
    assert template_family("softmax_attention_block") == "attention"
    assert template_family("token_merge_stack") == "token_merge"
    assert template_family("mamba_block_v2") == "ssm_recurrent"
    assert template_family("retrieval_augmented_layer") == "retrieval"
    assert template_family("conditional_compute_x") == "conditional_compute"
    assert template_family(None) == "other"


def test_rankdata_ties_average_rank():
    # [3, 1, 4, 1, 5, 9, 2, 6]; the two 1s share ranks (1+2)/2 = 1.5.
    ranks = rankdata([3, 1, 4, 1, 5, 9, 2, 6])
    assert ranks == [4.0, 1.5, 5.0, 1.5, 6.0, 8.0, 3.0, 7.0]


def test_spearman_monotonic_increasing_is_one():
    rho = spearman([1, 2, 3, 4], [10, 20, 30, 40])
    assert rho == pytest.approx(1.0)


def test_spearman_returns_none_when_too_short():
    assert spearman([1.0], [2.0]) is None


def test_spearman_ci_contains_point_estimate():
    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    ys = [1, 2, 3, 4, 5, 6, 7, 8]
    out = spearman_ci(xs, ys, n_bootstrap=100, seed=1)
    assert out is not None
    rho, lo, hi = out
    assert rho == pytest.approx(1.0)
    assert lo <= rho <= hi


def test_partial_spearman_collapses_correlation_via_confound():
    # x and y both depend on z; conditional on z they should be near-independent.
    z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x = [v + 0.0 for v in z]
    y = [v + 0.0 for v in z]
    direct = spearman(x, y)
    partial = partial_spearman(x, y, z)
    assert direct == pytest.approx(1.0)
    # Both variables are exactly z; partial should be NaN / undefined.
    assert partial is None or math.isnan(partial) or abs(partial) < 1.0


def test_partial_spearman_two_variables_independent_of_z():
    z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # rho(x,z) = 1
    y = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]  # rho(y,z) = -1
    # Partial of x vs y controlling for z is undefined (perfect collinearity).
    assert partial_spearman(x, y, z) is None


def test_oriented_values_flips_lower_is_better():
    pm = PROBE_METRICS_BY_COLUMN["wikitext_perplexity"]
    vals = oriented_values(pm, [100.0, 200.0, 50.0])
    # lower_is_better → sign flipped, then None / NaN dropped.
    assert vals == [-100.0, -200.0, -50.0]


def test_oriented_values_keeps_higher_is_better():
    pm = PROBE_METRICS_BY_COLUMN["induction_screening_auc"]
    vals = oriented_values(pm, [0.5, None, 0.8])
    assert vals == [0.5, 0.8]


def test_aligned_pairs_drops_missing():
    rows = [
        {"a": 1.0, "b": 2.0},
        {"a": None, "b": 3.0},
        {"a": 4.0, "b": None},
        {"a": 5.0, "b": 6.0},
    ]
    xs, ys = aligned_pairs(rows, "a", "b")
    assert xs == [1.0, 5.0]
    assert ys == [2.0, 6.0]


def test_aligned_triples_drops_any_missing():
    rows = [
        {"a": 1.0, "b": 2.0, "c": 3.0},
        {"a": None, "b": 4.0, "c": 5.0},
        {"a": 6.0, "b": 7.0, "c": 8.0},
    ]
    xs, ys, zs = aligned_triples(rows, "a", "b", "c")
    assert xs == [1.0, 6.0]
    assert ys == [2.0, 7.0]
    assert zs == [3.0, 8.0]


def test_variance_decomposition_separates_signal_from_noise():
    # Two families, large between-family difference and small within-family noise
    # → signal_ratio close to 1.
    rows = [
        {"family": "A", "x": 0.10},
        {"family": "A", "x": 0.11},
        {"family": "A", "x": 0.09},
        {"family": "B", "x": 0.90},
        {"family": "B", "x": 0.91},
        {"family": "B", "x": 0.89},
    ]
    decomp = variance_decomposition(rows, "x")
    assert decomp is not None
    assert decomp["signal_ratio"] > 0.95


def test_variance_decomposition_noise_dominated():
    # Same mean across families, large within-family variance → signal_ratio≈0.
    rows = [
        {"family": "A", "x": 0.10},
        {"family": "A", "x": 0.50},
        {"family": "A", "x": 0.90},
        {"family": "B", "x": 0.10},
        {"family": "B", "x": 0.50},
        {"family": "B", "x": 0.90},
    ]
    decomp = variance_decomposition(rows, "x")
    assert decomp is not None
    assert decomp["signal_ratio"] < 0.05


def test_agglomerative_clusters_groups_correlated_columns():
    # A & B perfectly correlated; C independent.
    rho = {
        ("A", "B"): 0.99,
        ("A", "C"): 0.05,
        ("B", "C"): 0.06,
    }
    clusters = agglomerative_clusters(["A", "B", "C"], rho, distance_threshold=0.2)
    flat = sorted([sorted(g) for g in clusters], key=lambda g: (-len(g), g[0]))
    assert ["A", "B"] in flat
    assert ["C"] in flat


def test_metrics_for_tier_returns_only_matching():
    screening = metrics_for_tier("screening")
    for pm in screening:
        assert pm.tier == "screening"


def test_table_columns_reads_schema(tmp_path):
    import sqlite3

    db = tmp_path / "t.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE foo (a TEXT, b REAL, c INTEGER)")
    try:
        assert table_columns(conn, "foo") == {"a", "b", "c"}
    finally:
        conn.close()


def test_table_columns_rejects_unsafe_identifier():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(ValueError):
            table_columns(conn, "foo); DROP TABLE bar;--")
        with pytest.raises(ValueError):
            table_columns(conn, "")
    finally:
        conn.close()
