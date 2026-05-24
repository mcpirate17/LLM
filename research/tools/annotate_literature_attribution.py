"""Annotate runs.db components with literature attribution.

Non-destructive, idempotent. Records, for every op / template / slot / graph,
which published architecture family it matches, the external model name, a
verified reference URL, and a match_type (exact | family | partial | novel).
The goal is honest provenance: never imply the project invented a mechanism
that has a published precedent, while clearly flagging the genuinely novel ones.

Storage model:
  * ``literature_attribution`` — canonical table, one row per (entity_type, entity_key).
  * inline columns on ``op_stats`` / ``template_stats`` / ``slot_stats`` / ``graphs``
    (``lit_family``, ``lit_model``, ``lit_match_type``, ``lit_ref``) for cheap joins
    and dashboard display.

Graphs (~22K) are bulk-classified by their dominant sequence-mixer op (or, when
no mixer is present, by their routing / MoE / recursion / dense-FFN structure).

runs.db uses DELETE journal mode and is guarded by a writer-lock held by the
live dashboard. This tool REFUSES to run while that process is alive unless
``--force`` is given, per the aria-db WAL-hygiene rule.

Usage:
    python -m research.tools.annotate_literature_attribution \
        --mapping research/reports/lit_attr/attribution_mapping.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO / "runs.db"
DEFAULT_MAPPING = Path(__file__).with_name("literature_attribution_mapping.json")
CREATED_BY = "literature_attribution_pass_2026-05-24"

# ── Graph family classifier ─────────────────────────────────────────────
# Dominant sequence-mixer ops, in priority order. The first match defines the
# graph's family. Keys here must exist in mapping["graph_families"].
_MIXER_PRIORITY = (
    "softmax_attention",
    "diff_attention",
    "mla_attention",
    "latent_attention_compressor",
    "graph_attention",
    "local_window_attn",
    "strided_attention",
    "difficulty_routed_attention",
    "gated_progressive_attention",
    "selective_scan",
    "state_space",
    "gated_delta",
    "rwkv_time_mixing",
    "linear_attention",
    "gated_linear_attention",
    "long_conv_hyena",
    "mixture_of_recursions",
    "associative_memory",
    "tropical_attention",
    "tropical_matmul",
    "tropical_softmax",
    "clifford_attention",
    "geometric_product",
    "ultrametric_attention",
    "stdp_attention",
    "integral_kernel",
    "spectral_filter",
    "chebyshev_spectral_mix",
    "fixed_point_iter",
    "multi_head_mix",
    "conv1d_seq",
    "matmul",
)
_MOE_OPS = frozenset(
    {
        "moe_topk",
        "moe_2expert",
        "relu_gated_moe",
        "sparse_bottleneck_moe",
        "hetero_moe",
        "arch_router",
        "compute_budget_router",
        "tropical_moe",
        "n_way_sparse_router",
        "compression_mixture_experts",
    }
)
_RECURSION_OPS = frozenset(
    {
        "adaptive_recursion",
        "mixed_recursion_gate",
        "route_recursion",
        "depth_weighted_proj",
        "mixture_of_recursions",
        "early_exit",
        "confidence_token_gate",
        "cheap_verify_blend",
        "speculative",
    }
)
_CONDITIONAL_OPS = frozenset(
    {
        "token_class_proj",
        "token_type_classifier",
        "token_entropy",
        "entropy_score",
    }
)
_TERNARY_OPS = frozenset({"ternary_projection", "sign_ste"})
# Token-level mixing that is not a classic "attention/SSM" mixer but still moves
# information across sequence positions (so NOT dense-FFN-degenerate).
_TOKEN_MERGE_OPS = frozenset({"adjacent_token_merge", "token_merge"})
_RETRIEVAL_OPS = frozenset({"gather_topk"})


def classify_graph_family(op_names: set[str]) -> str:
    """Return the family key for a graph given its set of op names.

    Empty op set => structure was reaped/compacted from the DB (the graph was a
    real evaluated model, we just can't see its structure here). It must NOT be
    bucketed as dense-FFN.
    """
    if not op_names:
        return "_reaped_no_structure"
    for mixer in _MIXER_PRIORITY:
        if mixer in op_names:
            return mixer
    if op_names & _TOKEN_MERGE_OPS:
        return "token_merge"
    if op_names & _RETRIEVAL_OPS:
        return "gather_topk"
    if op_names & _MOE_OPS:
        return "_moe"
    if op_names & _RECURSION_OPS:
        return "_recursion"
    if op_names & _CONDITIONAL_OPS:
        return "_conditional_compute"
    if op_names & _TERNARY_OPS:
        return "_ternary_dense"
    return "_dense_ffn"


# ── Schema ──────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS literature_attribution (
    entity_type         TEXT NOT NULL,   -- op | template | slot | graph | graph_family
    entity_key          TEXT NOT NULL,
    family_label        TEXT,
    external_model_name TEXT,
    match_type          TEXT NOT NULL,   -- exact | family | partial | novel
    citation            TEXT,
    reference_url       TEXT,
    notes               TEXT,
    confidence          REAL,
    created_ts          REAL NOT NULL,
    created_by          TEXT,
    PRIMARY KEY (entity_type, entity_key)
)
"""

_INLINE_COLS = ("lit_family", "lit_model", "lit_match_type", "lit_ref")


def _add_columns(con: sqlite3.Connection, table: str) -> None:
    existing = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}  # nosec B608  # nosemgrep: python-sql-string-formatting
    for col in _INLINE_COLS:
        if col not in existing:
            # table/col are internal constants, never user input
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")  # nosec B608  # nosemgrep: python-sql-string-formatting


def _writer_lock_alive(db_path: Path) -> int | None:
    lock = db_path.with_name(db_path.name + ".writer-lock")
    if not lock.exists():
        return None
    try:
        pid = int(lock.read_text().strip())
    except (ValueError, OSError):
        return None
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None  # stale lock, holder is gone


# ── Population ──────────────────────────────────────────────────────────


def _upsert(con: sqlite3.Connection, etype: str, row: dict, ts: float) -> None:
    con.execute(
        """INSERT INTO literature_attribution
           (entity_type, entity_key, family_label, external_model_name,
            match_type, citation, reference_url, notes, confidence,
            created_ts, created_by)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(entity_type, entity_key) DO UPDATE SET
            family_label=excluded.family_label,
            external_model_name=excluded.external_model_name,
            match_type=excluded.match_type,
            citation=excluded.citation,
            reference_url=excluded.reference_url,
            notes=excluded.notes,
            confidence=excluded.confidence,
            created_ts=excluded.created_ts,
            created_by=excluded.created_by""",
        (
            etype,
            row["entity_key"],
            row.get("family_label"),
            row.get("external_model_name"),
            row["match_type"],
            row.get("citation"),
            row.get("reference_url"),
            row.get("notes"),
            row.get("confidence"),
            ts,
            CREATED_BY,
        ),
    )


def _annotate_stats_table(con, table, key_col, rows, ts) -> int:
    """Write literature_attribution rows + inline cols for op/template/slot."""
    etype = {"op_stats": "op", "template_stats": "template", "slot_stats": "slot"}[
        table
    ]
    valid_keys = {r[0] for r in con.execute(f"SELECT {key_col} FROM {table}")}  # nosec B608  # nosemgrep: python-sql-string-formatting
    n = 0
    for row in rows:
        _upsert(con, etype, row, ts)
        key = row["entity_key"]
        if key in valid_keys:
            ref = row.get("reference_url") or row.get("citation")
            con.execute(  # nosemgrep: python-sql-string-formatting
                f"UPDATE {table} SET lit_family=?, lit_model=?, "  # nosec B608
                f"lit_match_type=?, lit_ref=? WHERE {key_col}=?",
                (
                    row.get("family_label"),
                    row.get("external_model_name"),
                    row["match_type"],
                    ref,
                    key,
                ),
            )
        n += 1
    return n


def _annotate_slots_inherit(con, template_rows, ts) -> int:
    """Slots inherit their parent template's family attribution.

    Slot classes (norm_wrap, ssm_core, moe_core, …) are functional roles, not
    architectures, so a slot's literature provenance is that of the template it
    belongs to. We record it explicitly per slot for queryability.
    """
    tpl_attr = {r["entity_key"]: r for r in template_rows}
    n = 0
    for slot_key, tpl, slot_classes in con.execute(
        "SELECT slot_key, template_name, slot_classes FROM slot_stats"
    ).fetchall():
        attr = tpl_attr.get(tpl)
        if attr is None:
            continue
        ref = attr.get("reference_url") or attr.get("citation")
        note = f"Inherits family from template '{tpl}'. Slot role(s): {slot_classes}."
        _upsert(
            con,
            "slot",
            {
                "entity_key": slot_key,
                "family_label": attr.get("family_label"),
                "external_model_name": attr.get("external_model_name"),
                "match_type": attr["match_type"],
                "citation": attr.get("citation"),
                "reference_url": attr.get("reference_url"),
                "notes": note,
            },
            ts,
        )
        con.execute(
            "UPDATE slot_stats SET lit_family=?, lit_model=?, lit_match_type=?, "
            "lit_ref=? WHERE slot_key=?",
            (
                attr.get("family_label"),
                attr.get("external_model_name"),
                attr["match_type"],
                ref,
                slot_key,
            ),
        )
        n += 1
    return n


def _best_structure_ops(con) -> dict:
    """fp -> op-name set from the richest available program_results.graph_json.

    The `graphs` table stores a placeholder ``{}`` for ~half the rows (structure
    reaped to save space), but `program_results` often still holds the real
    evaluated structure. Prefer the row with the most nodes per fingerprint.
    """
    best: dict[str, tuple[int, set]] = {}
    for fp, gj in con.execute(
        "SELECT graph_fingerprint, graph_json FROM program_results "
        "WHERE graph_json IS NOT NULL AND graph_json NOT IN ('', '{}')"
    ):
        try:
            nodes = (json.loads(gj).get("nodes")) or {}
        except (ValueError, TypeError):
            continue
        if len(nodes) > best.get(fp, (-1, None))[0]:
            best[fp] = (len(nodes), {n.get("op_name") for n in nodes.values()})
    return {fp: ops for fp, (_, ops) in best.items()}


def _annotate_graphs(con, families: dict, ts: float) -> dict:
    """Bulk-classify every graph by mixer family and write inline cols.

    The per-family attribution rows themselves are stored once under
    entity_type='graph_family'. Individual graphs get inline columns only,
    to avoid 22K rows in literature_attribution. Structure is taken from
    program_results (real) in preference to the graphs-table placeholder.
    """
    for key, fam in families.items():
        _upsert(con, "graph_family", {"entity_key": key, **fam}, ts)

    best = _best_structure_ops(con)
    counts: dict[str, int] = {}
    updates: list[tuple] = []
    skipped_bad = 0
    cur = con.execute("SELECT graph_fingerprint, graph_json FROM graphs")
    for fp, gj in cur.fetchall():
        ops = best.get(fp)
        if ops is None:  # not evaluated / no real structure in program_results
            try:
                ops = {
                    n.get("op_name")
                    for n in (json.loads(gj).get("nodes") or {}).values()
                }
            except (ValueError, TypeError):
                skipped_bad += 1
                continue
        key = classify_graph_family(ops)
        fam = families.get(key)
        if fam is None:
            continue
        counts[key] = counts.get(key, 0) + 1
        ref = fam.get("reference_url") or fam.get("citation")
        updates.append(
            (
                fam.get("family_label"),
                fam.get("external_model_name"),
                fam["match_type"],
                ref,
                fp,
            )
        )
    con.executemany(
        "UPDATE graphs SET lit_family=?, lit_model=?, lit_match_type=?, "
        "lit_ref=? WHERE graph_fingerprint=?",
        updates,
    )
    counts["_skipped_malformed_json"] = skipped_bad
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Write even if the dashboard writer-lock is held (UNSAFE).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pid = _writer_lock_alive(args.db)
    if pid is not None and not args.force:
        raise SystemExit(
            f"ABORT: runs.db writer-lock held by live PID {pid} (dashboard). "
            f"Stop it first (pkill -f 'research --mode=dashboard') or pass --force."
        )

    mapping = json.loads(args.mapping.read_text())
    ts = time.time()
    con = sqlite3.connect(args.db, timeout=30.0)
    try:
        con.execute("BEGIN")
        con.execute(_CREATE_TABLE)
        for tbl in ("op_stats", "template_stats", "slot_stats", "graphs"):
            _add_columns(con, tbl)
        n_ops = _annotate_stats_table(
            con, "op_stats", "op_name", mapping.get("ops", []), ts
        )
        template_rows = mapping.get("templates", [])
        n_tpl = _annotate_stats_table(
            con, "template_stats", "template_name", template_rows, ts
        )
        n_slot = _annotate_slots_inherit(con, template_rows, ts)
        gcounts = _annotate_graphs(con, mapping.get("graph_families", {}), ts)
        if args.dry_run:
            con.execute("ROLLBACK")
            print("DRY RUN — rolled back.")
        else:
            con.execute("COMMIT")
        print(f"ops={n_ops} templates={n_tpl} slots={n_slot}")
        print("graph family counts:")
        for k, v in sorted(gcounts.items(), key=lambda kv: -kv[1]):
            print(f"  {k:32s}{v}")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
