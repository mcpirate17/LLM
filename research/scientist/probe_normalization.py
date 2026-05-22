"""Canonical probe-metric registry and pure statistical helpers.

This module is the single source of truth for:

* which leaderboard / program_results columns count as "probe outputs" and
  what each one measures (probe family, tier, direction)
* the small set of rank-based statistics used to compare those probes
  (rank-data, Spearman ρ + bootstrap CI, partial Spearman controlling
  for a third variable)

The helpers are deliberately stdlib + numpy only — they intentionally do
not pull scipy so they can be exercised from any module in the repo
without import-time cost. They are lifted (verbatim semantics) from
``research/tools/real_lm_quickcheck_audit.py`` and
``research/tools/ar_probe_comparison.py`` so the multiple ad-hoc
implementations in the codebase collapse to one canonical pair.

The registry extends ``research.scientist.probe_metric_names.
PROBE_METRIC_RENAMES``: every metric in the registry already appears in
that rename table (or is a derived quantity such as
``neg_log_wikitext_ppl``) — this module classifies them, it does not
introduce new column names.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

# ── Probe family taxonomy ────────────────────────────────────────────

# The four "capability tier" families plus the loss / understanding
# probes that the composite scorer treats as separate signals. Names
# match the keys already used in ``research/scoring_config.yaml`` and
# in ``leaderboard_scoring/components.py``.
FAMILY_LOSS = "loss"
FAMILY_INDUCTION = "induction"
FAMILY_BINDING = "binding"
FAMILY_AR = "associative_recall"
FAMILY_UNDERSTANDING = "understanding"
FAMILY_LANGUAGE_CONTROL = "language_control"
FAMILY_TRAJECTORY = "trajectory"
FAMILY_LONG_CTX = "long_context"

# Tier names match ``leaderboard.tier`` values seen in the schema.
TIER_SCREENING = "screening"
TIER_INVESTIGATION = "investigation"
TIER_VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class ProbeMetric:
    """One canonical probe output column.

    ``column`` is the literal ``program_results`` / ``leaderboard``
    column name. ``lower_is_better`` controls how the value is oriented
    before correlation — ``True`` for perplexity / collapse rates,
    ``False`` for accuracy-like quantities. ``expected_range`` is
    advisory (used by the audit to flag clearly broken rows) and ``None``
    means unbounded.
    """

    column: str
    family: str
    tier: str
    lower_is_better: bool
    expected_range: tuple[float, float] | None = None
    description: str = ""


# ── Canonical registry ──────────────────────────────────────────────
#
# Source columns picked from:
#   * the COALESCE chains in ``research/meta_analysis/ar_binding_overlay.py``
#     (_AR_EXPR, _BINDING_EXPR) — these are the columns the system
#     already treats as interchangeable family-members
#   * the kwargs read by ``leaderboard_scoring/components.py`` /
#     ``v10.py`` / ``v14.py``
#   * the kwargs read by ``leaderboard_scoring/kwargs.py::_PR_SELECT_COLS``
#
# When the same logical capability has multiple columns (e.g. AR has
# five), each gets its own ProbeMetric — orientation and family let the
# audit code group them.

PROBE_METRICS: tuple[ProbeMetric, ...] = (
    # ── Loss / perplexity ──
    ProbeMetric(
        "wikitext_perplexity",
        FAMILY_LOSS,
        TIER_SCREENING,
        lower_is_better=True,
        expected_range=(1.0, 1e6),
        description="WikiText micro-train perplexity at screening tier",
    ),
    ProbeMetric(
        "wikitext_score",
        FAMILY_LOSS,
        TIER_SCREENING,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
        description="WikiText derived score (higher = better)",
    ),
    ProbeMetric(
        "final_loss",
        FAMILY_LOSS,
        TIER_SCREENING,
        lower_is_better=True,
        description="Final stage-1 training loss",
    ),
    ProbeMetric(
        "champion_floor_loss",
        FAMILY_LOSS,
        TIER_VALIDATION,
        lower_is_better=True,
        description="Champion-tier 500-step plateau median loss",
    ),
    ProbeMetric(
        "champion_floor_ppl",
        FAMILY_LOSS,
        TIER_VALIDATION,
        lower_is_better=True,
        description="Champion-tier plateau perplexity",
    ),
    # ── Induction ──
    ProbeMetric(
        "induction_screening_auc",
        FAMILY_INDUCTION,
        TIER_SCREENING,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
        description="Mean per-gap accuracy on induction-head probe",
    ),
    ProbeMetric(
        "induction_intermediate_auc",
        FAMILY_INDUCTION,
        TIER_INVESTIGATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "induction_validation_auc",
        FAMILY_INDUCTION,
        TIER_VALIDATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    # ── Binding ──
    ProbeMetric(
        "binding_screening_auc",
        FAMILY_BINDING,
        TIER_SCREENING,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "binding_screening_composite",
        FAMILY_BINDING,
        TIER_SCREENING,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
        description="0.4*ar_gate + 0.3*induction + 0.3*binding (legacy blend)",
    ),
    ProbeMetric(
        "binding_curriculum_auc",
        FAMILY_BINDING,
        TIER_SCREENING,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "binding_intermediate_auc",
        FAMILY_BINDING,
        TIER_INVESTIGATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "binding_multislot_auc",
        FAMILY_BINDING,
        TIER_INVESTIGATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    # ── Associative recall cascade (see meta_analysis._AR_EXPR) ──
    ProbeMetric(
        "ar_gate_score",
        FAMILY_AR,
        TIER_SCREENING,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
        description="AR-gate in-distribution exact-match accuracy",
    ),
    ProbeMetric(
        "ar_curriculum_auc_pair_final",
        FAMILY_AR,
        TIER_INVESTIGATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "ar_intermediate_auc",
        FAMILY_AR,
        TIER_INVESTIGATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "ar_validation_rank_score",
        FAMILY_AR,
        TIER_VALIDATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "ar_legacy_auc",
        FAMILY_AR,
        TIER_SCREENING,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
        description="Legacy AR probe (deprecated; kept for back-compat)",
    ),
    # ── Understanding tier ──
    ProbeMetric(
        "blimp_overall_accuracy",
        FAMILY_UNDERSTANDING,
        TIER_VALIDATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "hellaswag_acc",
        FAMILY_UNDERSTANDING,
        TIER_VALIDATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "tinystories_score",
        FAMILY_UNDERSTANDING,
        TIER_VALIDATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "cross_task_score",
        FAMILY_UNDERSTANDING,
        TIER_VALIDATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "diagnostic_score",
        FAMILY_UNDERSTANDING,
        TIER_VALIDATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "fp_hierarchy_fitness",
        FAMILY_UNDERSTANDING,
        TIER_VALIDATION,
        lower_is_better=False,
    ),
    # ── Trajectory / Gemini signals ──
    ProbeMetric(
        "fp_jacobian_erf_density",
        FAMILY_TRAJECTORY,
        TIER_SCREENING,
        lower_is_better=False,
    ),
    ProbeMetric(
        "fp_id_collapse_rate",
        FAMILY_TRAJECTORY,
        TIER_SCREENING,
        lower_is_better=True,
    ),
    ProbeMetric(
        "fp_jacobian_erf_decay_slope",
        FAMILY_TRAJECTORY,
        TIER_SCREENING,
        lower_is_better=True,
    ),
    ProbeMetric(
        "fp_logit_margin_velocity",
        FAMILY_TRAJECTORY,
        TIER_SCREENING,
        lower_is_better=False,
    ),
    # ── Language control ladder (v14) ──
    ProbeMetric(
        "language_control_s05_sentence_assoc_score",
        FAMILY_LANGUAGE_CONTROL,
        TIER_SCREENING,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "language_control_s10_sentence_assoc_score",
        FAMILY_LANGUAGE_CONTROL,
        TIER_INVESTIGATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    ProbeMetric(
        "language_control_investigation_sentence_assoc_score",
        FAMILY_LANGUAGE_CONTROL,
        TIER_INVESTIGATION,
        lower_is_better=False,
        expected_range=(0.0, 1.0),
    ),
    # ── Long-context retrieval ──
    ProbeMetric(
        "robustness_long_ctx_passkey_score",
        FAMILY_LONG_CTX,
        TIER_INVESTIGATION,
        lower_is_better=False,
    ),
    ProbeMetric(
        "robustness_long_ctx_multi_hop_score",
        FAMILY_LONG_CTX,
        TIER_INVESTIGATION,
        lower_is_better=False,
    ),
)


PROBE_METRICS_BY_COLUMN: Mapping[str, ProbeMetric] = {
    pm.column: pm for pm in PROBE_METRICS
}


def metrics_for_family(family: str) -> tuple[ProbeMetric, ...]:
    return tuple(pm for pm in PROBE_METRICS if pm.family == family)


def metrics_for_tier(tier: str) -> tuple[ProbeMetric, ...]:
    return tuple(pm for pm in PROBE_METRICS if pm.tier == tier)


# ── Pure stat helpers — stdlib only ───────────────────────────────────


def safe_float(value: Any) -> float | None:
    """Coerce to a finite float or return None.

    Identical semantics to ``real_lm_quickcheck_audit._safe_float``.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def template_family(template: str | None) -> str:
    """Bucket a template name into an architecture family.

    Same buckets as ``real_lm_quickcheck_audit._family``. Centralized so
    the audit and the simulator share one definition.
    """
    t = (template or "").lower()
    if "token_merge" in t:
        return "token_merge"
    if "retrieval" in t:
        return "retrieval"
    if "ssm" in t or "mamba" in t or "rwkv" in t or "recurrent" in t:
        return "ssm_recurrent"
    if "attn" in t or "attention" in t:
        return "attention"
    if "conditional" in t:
        return "conditional_compute"
    return "other"


def rankdata(values: Sequence[float]) -> list[float]:
    """Average-rank ties (Spearman convention).

    Identical to ``real_lm_quickcheck_audit._rankdata``.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 0 or den_y <= 0:
        return None
    return num / (den_x * den_y)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation. Returns None if N<3 or zero variance."""
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return _pearson(rankdata(list(xs)), rankdata(list(ys)))


def spearman_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float] | None:
    """Bootstrap percentile CI for Spearman ρ.

    Returns ``(rho, lo, hi)`` or None when N<3. Resamples (x_i, y_i)
    pairs with replacement ``n_bootstrap`` times and reports the central
    ``confidence`` percentile band on the bootstrap distribution.
    """
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    rho = spearman(xs, ys)
    if rho is None:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    indices = list(range(n))
    for _ in range(n_bootstrap):
        idx = [rng.choice(indices) for _ in range(n)]
        sx = [xs[i] for i in idx]
        sy = [ys[i] for i in idx]
        s = spearman(sx, sy)
        if s is not None:
            samples.append(s)
    if not samples:
        return (rho, float("nan"), float("nan"))
    samples.sort()
    lo_q = (1.0 - confidence) / 2.0
    hi_q = 1.0 - lo_q
    lo = samples[max(0, int(math.floor(lo_q * len(samples))))]
    hi = samples[min(len(samples) - 1, int(math.ceil(hi_q * len(samples))) - 1)]
    return (rho, lo, hi)


def partial_spearman(
    xs: Sequence[float],
    ys: Sequence[float],
    zs: Sequence[float],
) -> float | None:
    """Spearman partial correlation of x and y controlling for z.

    Formula: rho_xy.z = (rho_xy - rho_xz * rho_yz) /
    sqrt((1 - rho_xz^2)(1 - rho_yz^2)).
    """
    if len(xs) < 4 or not (len(xs) == len(ys) == len(zs)):
        return None
    r_xy = spearman(xs, ys)
    r_xz = spearman(xs, zs)
    r_yz = spearman(ys, zs)
    if r_xy is None or r_xz is None or r_yz is None:
        return None
    den = math.sqrt(max(0.0, (1.0 - r_xz**2) * (1.0 - r_yz**2)))
    if den <= 0:
        return None
    return (r_xy - r_xz * r_yz) / den


def oriented_values(pm: ProbeMetric, values: Iterable[float | None]) -> list[float]:
    """Return finite values flipped so higher = better.

    Applies negation for ``lower_is_better`` metrics. Drops None / NaN.
    """
    out: list[float] = []
    sign = -1.0 if pm.lower_is_better else 1.0
    for v in values:
        f = safe_float(v)
        if f is None:
            continue
        out.append(sign * f)
    return out


def aligned_pairs(
    rows: Sequence[Mapping[str, Any]],
    col_a: str,
    col_b: str,
) -> tuple[list[float], list[float]]:
    """Return paired finite values for two columns across rows.

    Rows missing either column are dropped (pairwise deletion).
    """
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        av = safe_float(row.get(col_a))
        bv = safe_float(row.get(col_b))
        if av is None or bv is None:
            continue
        xs.append(av)
        ys.append(bv)
    return xs, ys


def aligned_triples(
    rows: Sequence[Mapping[str, Any]],
    col_a: str,
    col_b: str,
    col_c: str,
) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for row in rows:
        av = safe_float(row.get(col_a))
        bv = safe_float(row.get(col_b))
        cv = safe_float(row.get(col_c))
        if av is None or bv is None or cv is None:
            continue
        xs.append(av)
        ys.append(bv)
        zs.append(cv)
    return xs, ys, zs


def variance_decomposition(
    rows: Sequence[Mapping[str, Any]],
    column: str,
    group_key: str = "family",
) -> dict[str, float] | None:
    """Decompose a column's variance into between- vs within-group.

    Returns dict with ``total``, ``between``, ``within``, ``signal_ratio``
    (between/total). ``signal_ratio`` < ~0.2 at screening tier means the
    probe is noise-dominated for ranking architecture families.
    """
    by_group: dict[str, list[float]] = {}
    for row in rows:
        v = safe_float(row.get(column))
        if v is None:
            continue
        g = str(row.get(group_key) or "other")
        by_group.setdefault(g, []).append(v)
    all_vals = [v for vs in by_group.values() for v in vs]
    if len(all_vals) < 2:
        return None
    grand_mean = sum(all_vals) / len(all_vals)
    total = sum((v - grand_mean) ** 2 for v in all_vals)
    between = 0.0
    within = 0.0
    for vs in by_group.values():
        if not vs:
            continue
        gm = sum(vs) / len(vs)
        between += len(vs) * (gm - grand_mean) ** 2
        within += sum((v - gm) ** 2 for v in vs)
    if total <= 0:
        return {
            "total": 0.0,
            "between": 0.0,
            "within": 0.0,
            "signal_ratio": 0.0,
        }
    return {
        "total": total,
        "between": between,
        "within": within,
        "signal_ratio": between / total,
    }


def agglomerative_clusters(
    columns: Sequence[str],
    rho_matrix: Mapping[tuple[str, str], float | None],
    *,
    distance_threshold: float = 0.2,
) -> list[list[str]]:
    """Cluster columns by ``1 - |rho|`` with single-link agglomeration.

    Two columns merge when their distance is below ``distance_threshold``.
    Missing rho values are treated as distance 1.0 (no link).
    """

    def dist(a: str, b: str) -> float:
        r = rho_matrix.get((a, b)) or rho_matrix.get((b, a))
        if r is None:
            return 1.0
        return 1.0 - abs(r)

    parent = {c: c for c in columns}

    def find(c: str) -> str:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cols = list(columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if dist(a, b) < distance_threshold:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(find(c), []).append(c)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g[0]))
