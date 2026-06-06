"""Scope existing primitives + templates into the fab catalog.

Reads ``op_property_catalog`` and ``template_property_catalog`` from
``research/meta_analysis.db``, classifies each entity into one of
``lane`` / ``routing`` / ``compression``, and surfaces:

1. The full inventory the fab uses to anchor new proposals on existing
   patterns and to pick test-bed substrates for new components.
2. The under-explored regions — rare property values that have low
   eval counts.
3. The "novel but underperforming" candidates for goal (b) — existing
   ops/templates with rare math axes whose realized pass rates trail
   the cohort (e.g. ``tropical_attention``, ``padic_gate``,
   ``clifford_attention``).

Read-only; no DB mutation; no coupling to the synthesis runtime.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_META_DB = _REPO / "research" / "meta_analysis.db"

CATEGORY_LANE = "lane"
CATEGORY_ROUTING = "routing"
CATEGORY_COMPRESSION = "compression"
ALL_CATEGORIES = (CATEGORY_LANE, CATEGORY_ROUTING, CATEGORY_COMPRESSION)

_NOVEL_ALGEBRAIC_SPACES = frozenset(
    {"tropical", "clifford", "padic", "spiking", "complex", "hyperbolic"}
)


@dataclass(frozen=True, slots=True)
class ComponentRecord:
    name: str
    kind: str  # "op" or "template"
    category: str | None
    is_multilane: bool
    declared_properties: dict[str, Any]
    performance: dict[str, Any]
    notes: tuple[str, ...] = field(default_factory=tuple)


def _connect_ro(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"meta_analysis.db not found at {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_op_rows(db_path: Path | str = DEFAULT_META_DB) -> list[dict[str, Any]]:
    conn = _connect_ro(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM op_property_catalog").fetchall()
    finally:
        conn.close()
    return [{k: r[k] for k in r.keys()} for r in rows]


def load_template_rows(db_path: Path | str = DEFAULT_META_DB) -> list[dict[str, Any]]:
    conn = _connect_ro(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM template_property_catalog").fetchall()
    finally:
        conn.close()
    return [{k: r[k] for k in r.keys()} for r in rows]


def _name_matches(name: str, tokens: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(tok in lowered for tok in tokens)


_ROUTING_NAME_TOKENS = (
    "router",
    "route_",
    "_route",
    "moe_",
    "topk_gate",
    "top_k_gate",
    "lane_mixer",
    "blend",
    "n_way_sparse",
    "compute_budget",
    "expert",
    "hybrid_sparse",
    "split2",
    "split3",
    "gather_topk",
)
_COMPRESSION_NAME_TOKENS = (
    "compress",
    "bottleneck",
    "latent_attention",
    "tied_proj",
    "shared_basis",
    "low_rank",
    "proj_down",
    "adjacent_token_merge",
)


_LANE_CATEGORIES_SINGLE_INPUT = frozenset(
    {
        "mixing",
        "math_space",
        "functional",
        "parameterized",
        "elementwise_unary",
        "sequence",
        "frequency",
        "linear_algebra",
    }
)


def classify_op_row(row: dict[str, Any]) -> str | None:
    """Classify a single ``op_property_catalog`` row into a fab category.

    Reduce / structural / elementwise_binary stay unclassified — they are
    "wiring" (concat, split, residual add, cumsum) rather than a fab-slot
    candidate.
    """
    op_name = str(row.get("op_name") or "")
    op_category = str(row.get("op_category") or "")
    n_inputs = int(row.get("op_n_inputs") or 1)

    if _name_matches(op_name, _ROUTING_NAME_TOKENS):
        return CATEGORY_ROUTING
    if _name_matches(op_name, _COMPRESSION_NAME_TOKENS):
        return CATEGORY_COMPRESSION

    sparsity = str(row.get("op_activation_sparsity_pattern") or "")
    if sparsity == "top_k" and op_category in ("parameterized", "reduce"):
        return CATEGORY_ROUTING

    if op_category in _LANE_CATEGORIES_SINGLE_INPUT and n_inputs == 1:
        return CATEGORY_LANE
    return None


def classify_template_row(row: dict[str, Any]) -> str | None:
    """Classify a single ``template_property_catalog`` row into a fab category."""
    name = str(row.get("template_name") or "")
    family = str(row.get("template_family") or "")
    has_routing = int(row.get("template_has_routing") or 0)
    has_parallel = int(row.get("template_has_parallel_paths") or 0)
    has_moe = int(row.get("template_has_moe") or 0)
    has_compression = int(row.get("template_has_compression") or 0)
    compression_intensity = float(row.get("template_compression_intensity") or 0.0)
    routing_intensity = float(row.get("template_routing_intensity") or 0.0)

    routing_structural = (
        has_routing
        or has_moe
        or has_parallel
        or family in ("routing", "moe")
        or routing_intensity >= 0.3
    )
    if routing_structural:
        return CATEGORY_ROUTING
    if has_compression or family == "compression" or compression_intensity >= 0.3:
        return CATEGORY_COMPRESSION
    if _name_matches(name, _ROUTING_NAME_TOKENS):
        return CATEGORY_ROUTING
    if _name_matches(name, _COMPRESSION_NAME_TOKENS):
        return CATEGORY_COMPRESSION
    return CATEGORY_LANE


def _template_is_multilane(row: dict[str, Any]) -> bool:
    if int(row.get("template_has_parallel_paths") or 0):
        return True
    if int(row.get("template_est_parallel_paths") or 0) >= 2:
        return True
    return False


def _op_record(row: dict[str, Any]) -> ComponentRecord:
    name = str(row.get("op_name") or "")
    category = classify_op_row(row)
    declared = {
        key: row.get(key)
        for key in (
            "op_category",
            "op_algebraic_space",
            "op_spectral_preferred_basis",
            "op_dynamical_memory_length_class",
            "op_dynamical_has_state",
            "op_activation_sparsity_pattern",
            "op_geometric_receptive_field",
            "op_n_inputs",
            "op_is_parameterized",
            "op_is_stateless",
            "op_composition_residual_safe",
            "op_composition_parallel_safe",
        )
    }
    eval_count = int(row.get("eval_count") or 0)
    s1_pass = int(row.get("s1_pass_count") or 0)
    perf = {
        "eval_count": eval_count,
        "s1_pass_count": s1_pass,
        "pass_rate": (s1_pass / eval_count) if eval_count else 0.0,
        "mean_loss": row.get("mean_loss"),
        "min_loss": row.get("min_loss"),
        "mean_novelty": row.get("mean_novelty"),
    }
    return ComponentRecord(
        name=name,
        kind="op",
        category=category,
        is_multilane=False,
        declared_properties=declared,
        performance=perf,
    )


def _template_record(row: dict[str, Any]) -> ComponentRecord:
    name = str(row.get("template_name") or "")
    category = classify_template_row(row)
    declared = {
        key: row.get(key)
        for key in (
            "template_family",
            "template_topology",
            "template_receptive_field",
            "template_preferred_basis",
            "template_has_attention",
            "template_has_ssm",
            "template_has_conv",
            "template_has_routing",
            "template_has_compression",
            "template_has_moe",
            "template_has_parallel_paths",
            "template_has_state",
            "template_est_parallel_paths",
            "template_est_branch_factor",
            "template_routing_intensity",
            "template_compression_intensity",
            "template_memory_intensity",
            "template_novelty_prior",
            "template_trainability_prior",
            "slot_count",
        )
    }
    perf = {
        "observed_count": int(row.get("observed_count") or 0),
    }
    return ComponentRecord(
        name=name,
        kind="template",
        category=category,
        is_multilane=_template_is_multilane(row),
        declared_properties=declared,
        performance=perf,
    )


def select_underperforming_novel(
    records: Sequence[ComponentRecord],
    *,
    min_evals: int = 30,
    pass_rate_ceiling: float = 0.35,
) -> list[ComponentRecord]:
    """Pull the goal-(b) targets: novel-axis ops that underperform under current use.

    A record qualifies when:
      - its declared `op_algebraic_space` is a non-euclidean / novel family
      - it has at least ``min_evals`` evals (so the pass rate is meaningful)
      - its empirical pass rate is at or below ``pass_rate_ceiling``
    """
    out: list[ComponentRecord] = []
    for record in records:
        if record.kind != "op":
            continue
        space = record.declared_properties.get("op_algebraic_space")
        if space not in _NOVEL_ALGEBRAIC_SPACES:
            continue
        perf = record.performance
        evals = int(perf.get("eval_count") or 0)
        if evals < min_evals:
            continue
        if float(perf.get("pass_rate") or 0.0) <= pass_rate_ceiling:
            out.append(record)
    out.sort(
        key=lambda r: (
            -int(r.performance.get("eval_count") or 0),
            float(r.performance.get("pass_rate") or 0.0),
        )
    )
    return out


def scope_all(db_path: Path | str = DEFAULT_META_DB) -> dict[str, Any]:
    """Build the full intake report.

    Returns a dict with per-category buckets, an inventory count summary,
    a multilane-routing-template index, and a list of goal-(b) targets.
    """
    op_rows = load_op_rows(db_path)
    template_rows = load_template_rows(db_path)

    op_records = [_op_record(r) for r in op_rows]
    template_records = [_template_record(r) for r in template_rows]
    all_records = op_records + template_records

    by_category: dict[str, list[ComponentRecord]] = {c: [] for c in ALL_CATEGORIES}
    by_category["unclassified"] = []
    for record in all_records:
        bucket = record.category or "unclassified"
        by_category.setdefault(bucket, []).append(record)

    multilane_templates = [
        r for r in template_records if r.is_multilane and r.category == CATEGORY_ROUTING
    ]
    multilane_templates.sort(
        key=lambda r: int(r.performance.get("observed_count") or 0), reverse=True
    )

    underperforming = select_underperforming_novel(op_records)

    return {
        "schema_version": "fab_intake_v1",
        "source_db": str(db_path),
        "totals": {
            "ops": len(op_records),
            "templates": len(template_records),
            "by_category": {c: len(by_category[c]) for c in by_category},
        },
        "ops": [_record_to_json(r) for r in op_records],
        "templates": [_record_to_json(r) for r in template_records],
        "multilane_routing_templates": [
            _record_to_json(r) for r in multilane_templates
        ],
        "underperforming_novel_ops": [_record_to_json(r) for r in underperforming],
    }


def _record_to_json(record: ComponentRecord) -> dict[str, Any]:
    return {
        "name": record.name,
        "kind": record.kind,
        "category": record.category,
        "is_multilane": record.is_multilane,
        "declared_properties": dict(record.declared_properties),
        "performance": dict(record.performance),
        "notes": list(record.notes),
    }
