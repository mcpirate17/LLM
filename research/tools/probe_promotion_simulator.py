"""Dry-run alternative scoring weights / promotion gates against history.

Companion to ``research.tools.probe_divergence_audit``. Loads a
``weight_refit_proposal_<ts>.yaml`` produced by that audit, then for
every historical leaderboard row recomputes the composite under both
the current and proposed configs using
``research.scientist.leaderboard_scoring.compute_composite``. Reports:

* Promotion / demotion delta against the current
  ``breakthrough_gates.capability_floor`` and an arbitrary proposed
  capability floor.
* Reference-architecture sanity: do the GPT-2, Mamba, RWKV, and
  retrieval-augmented rows still rank in the canonical order under the
  proposal? Spearman ρ ≥ 0.9 vs current order is the gate.

The simulator writes nothing to ``research/runs.db``. It writes a
JSON + markdown report under
``research/reports/probe_normalization/promotion_sim_<ts>.{json,md}``.

Uses the existing read-only DB connection pattern from
``research.tools._db_maintenance.connect_readonly`` and the existing
``prefetch_program_results`` / ``build_score_kwargs_from_prefetch``
helpers so the kwargs construction stays in lockstep with production.
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from research.defaults import RUNS_DB
from research.scientist.probe_normalization import (
    safe_float,
    spearman,
    template_family,
)

DEFAULT_OUT_DIR = Path("research/reports/probe_normalization")
DEFAULT_SCORING_YAML = Path("research/scoring_config.yaml")

# Canonical rank order for the reference architectures, anchored by the
# capability-tier ordering already encoded in
# ``research/synthesis/reference_architectures.py``: full attention >
# retrieval > Mamba > RWKV on associative-recall-heavy tasks at nano scale.
CANONICAL_REFERENCE_ORDER: tuple[str, ...] = (
    "gpt2",
    "retrieval_augmented",
    "mamba",
    "rwkv",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge_proposal(
    base_cfg: Mapping[str, Any], proposal: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a deep copy of ``base_cfg`` with proposal ``base:`` overrides applied."""
    merged = copy.deepcopy(dict(base_cfg))
    base = dict(merged.get("base") or {})
    overrides = dict((proposal.get("base") or {}))
    base.update(overrides)
    merged["base"] = base
    return merged


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _select_leaderboard_rows(
    conn: sqlite3.Connection, *, tiers: Sequence[str], limit: int | None
) -> list[dict[str, Any]]:
    lb_cols = _table_columns(conn, "leaderboard")
    needed = [
        "result_id",
        "tier",
        "composite_score",
        "is_reference",
        "reference_name",
        "template_name",
    ]
    cols = [c for c in needed if c in lb_cols]
    placeholders = ",".join("?" for _ in tiers)
    sql = f"SELECT {', '.join(cols)} FROM leaderboard WHERE tier IN ({placeholders})"
    params: list[Any] = list(tiers)
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _score_rows(
    *,
    db_path: Path,
    config_payload: Mapping[str, Any],
    tiers: Sequence[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Compute composite_score for every selected leaderboard row under config_payload.

    Patches the loaded scoring config in-place inside the
    ``research.scientist.leaderboard_scoring`` module by reloading its
    YAML state, scoring rows, then restoring.
    """
    # Imported lazily so test fixtures can stub the underlying YAML loader
    # without forcing the production module to load during simple unit tests.
    from research.scientist import scoring_config as _scfg
    from research.scientist.leaderboard_scoring import (
        build_score_kwargs_from_prefetch,
        compute_composite,
        prefetch_program_results,
    )

    # Patch the YAML payload in memory for this scoring run.
    original_path = _scfg._CONFIG_PATH
    tmp_yaml = original_path.with_suffix(".sim.yaml")
    try:
        tmp_yaml.write_text(yaml.safe_dump(dict(config_payload)), encoding="utf-8")
        _scfg._CONFIG_PATH = tmp_yaml
        _scfg.reload_scoring_config()

        conn = _connect_readonly(db_path)
        try:
            rows = _select_leaderboard_rows(conn, tiers=tiers, limit=limit)
            result_ids = [str(r["result_id"]) for r in rows]
            pr_cache = prefetch_program_results(conn, result_ids)
        finally:
            conn.close()

        out: list[dict[str, Any]] = []
        for row in rows:
            rid = str(row["result_id"])
            pr = pr_cache.get(rid)
            if not pr:
                continue
            try:
                kwargs = build_score_kwargs_from_prefetch(row, pr)
                score = float(compute_composite(**kwargs))
            except Exception as exc:  # noqa: BLE001 - simulator must keep going
                out.append(
                    {
                        **row,
                        "scored": False,
                        "error": str(exc),
                    }
                )
                continue
            out.append({**row, "scored": True, "score": score})
        return out
    finally:
        _scfg._CONFIG_PATH = original_path
        _scfg.reload_scoring_config()
        if tmp_yaml.exists():
            tmp_yaml.unlink()


# ── Aggregation ──────────────────────────────────────────────────────


def _join_scores(
    current: Sequence[Mapping[str, Any]],
    proposed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cur_by_id = {str(r["result_id"]): r for r in current}
    out = []
    for p in proposed:
        rid = str(p["result_id"])
        c = cur_by_id.get(rid)
        if not c:
            continue
        out.append(
            {
                "result_id": rid,
                "tier": p.get("tier"),
                "template_name": p.get("template_name") or c.get("template_name"),
                "is_reference": bool(p.get("is_reference") or c.get("is_reference")),
                "reference_name": p.get("reference_name") or c.get("reference_name"),
                "current_score": safe_float(c.get("score")),
                "proposed_score": safe_float(p.get("score")),
                "current_scored": bool(c.get("scored")),
                "proposed_scored": bool(p.get("scored")),
            }
        )
    return out


def _promotion_delta(
    joined: Sequence[Mapping[str, Any]],
    *,
    current_floor: float,
    proposed_floor: float,
) -> dict[str, Any]:
    promoted: list[str] = []
    demoted: list[str] = []
    unchanged_passing = 0
    unchanged_failing = 0
    for r in joined:
        cs = safe_float(r["current_score"])
        ps = safe_float(r["proposed_score"])
        cur_pass = cs is not None and cs >= current_floor
        prop_pass = ps is not None and ps >= proposed_floor
        if cur_pass and not prop_pass:
            demoted.append(str(r["result_id"]))
        elif prop_pass and not cur_pass:
            promoted.append(str(r["result_id"]))
        elif cur_pass and prop_pass:
            unchanged_passing += 1
        else:
            unchanged_failing += 1
    return {
        "current_floor": current_floor,
        "proposed_floor": proposed_floor,
        "promoted_n": len(promoted),
        "demoted_n": len(demoted),
        "promoted_ids": promoted[:50],
        "demoted_ids": demoted[:50],
        "unchanged_passing": unchanged_passing,
        "unchanged_failing": unchanged_failing,
    }


def _reference_rank_check(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute Spearman ρ between current and proposed rank order on refs.

    A proposal that fails this check would reshuffle GPT-2/Mamba/RWKV
    away from their canonical capability order — strong evidence the
    refit is overfit to non-reference rows.
    """
    refs: dict[str, dict[str, float]] = {}
    for r in joined:
        if not r.get("is_reference"):
            continue
        name = (r.get("reference_name") or "").lower()
        if not name:
            name = template_family(r.get("template_name"))
        if not name:
            continue
        # Take the max score per reference (multiple rows possible).
        cs = safe_float(r["current_score"])
        ps = safe_float(r["proposed_score"])
        slot = refs.setdefault(
            name, {"current": float("-inf"), "proposed": float("-inf")}
        )
        if cs is not None and cs > slot["current"]:
            slot["current"] = cs
        if ps is not None and ps > slot["proposed"]:
            slot["proposed"] = ps
    names = sorted(refs.keys())
    cur_vec = [refs[n]["current"] for n in names if refs[n]["current"] > float("-inf")]
    prop_vec = [
        refs[n]["proposed"] for n in names if refs[n]["proposed"] > float("-inf")
    ]
    rho = spearman(cur_vec, prop_vec) if len(cur_vec) == len(prop_vec) else None
    return {
        "references_seen": names,
        "current_scores": {n: refs[n]["current"] for n in names},
        "proposed_scores": {n: refs[n]["proposed"] for n in names},
        "spearman_current_vs_proposed": rho,
        "pass": rho is not None and rho >= 0.9,
    }


def _summary_stats(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cur = [safe_float(r["current_score"]) for r in joined]
    prop = [safe_float(r["proposed_score"]) for r in joined]
    cur = [v for v in cur if v is not None]
    prop = [v for v in prop if v is not None]
    if not cur or not prop:
        return {"n": 0}
    paired = [
        (safe_float(r["current_score"]), safe_float(r["proposed_score"]))
        for r in joined
        if safe_float(r["current_score"]) is not None
        and safe_float(r["proposed_score"]) is not None
    ]
    rho = spearman([p[0] for p in paired], [p[1] for p in paired])
    return {
        "n": len(joined),
        "n_scored_both": len(paired),
        "current_mean": sum(cur) / len(cur),
        "proposed_mean": sum(prop) / len(prop),
        "current_max": max(cur),
        "proposed_max": max(prop),
        "rank_correlation": rho,
    }


# ── Report ───────────────────────────────────────────────────────────


def build_report(
    *,
    db_path: Path,
    scoring_yaml_path: Path,
    proposal_path: Path,
    tiers: Sequence[str],
    current_floor: float,
    proposed_floor: float | None,
    limit: int | None,
) -> dict[str, Any]:
    base_cfg = _load_yaml(scoring_yaml_path)
    proposal = _load_yaml(proposal_path)
    merged_cfg = _merge_proposal(base_cfg, proposal)

    current_scores = _score_rows(
        db_path=db_path, config_payload=base_cfg, tiers=tiers, limit=limit
    )
    proposed_scores = _score_rows(
        db_path=db_path, config_payload=merged_cfg, tiers=tiers, limit=limit
    )
    joined = _join_scores(current_scores, proposed_scores)
    summary = _summary_stats(joined)
    floor = (
        float(proposed_floor) if proposed_floor is not None else float(current_floor)
    )
    promo = _promotion_delta(
        joined, current_floor=float(current_floor), proposed_floor=floor
    )
    ref = _reference_rank_check(joined)
    return {
        "proposal_path": str(proposal_path),
        "scoring_config_path": str(scoring_yaml_path),
        "tiers": list(tiers),
        "current_floor": float(current_floor),
        "proposed_floor": floor,
        "summary": summary,
        "promotion_delta": promo,
        "reference_rank_check": ref,
        "rows": joined,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    promo = report["promotion_delta"]
    ref = report["reference_rank_check"]
    lines = [
        "# Probe Promotion Simulator",
        "",
        f"Proposal: `{report['proposal_path']}`",
        f"Tiers: {', '.join(report['tiers'])}",
        f"Current floor: {report['current_floor']:.2f} → Proposed floor: "
        f"{report['proposed_floor']:.2f}",
        "",
        "## Summary",
        "",
        f"- Rows compared: {summary.get('n', 0)}",
        f"- Rows scored under both configs: {summary.get('n_scored_both', 0)}",
        f"- Current mean composite: {_fmt(summary.get('current_mean'))}",
        f"- Proposed mean composite: {_fmt(summary.get('proposed_mean'))}",
        f"- Spearman ρ (current vs proposed scores): {_fmt(summary.get('rank_correlation'))}",
        "",
        "## Promotion / demotion delta",
        "",
        f"- Promoted under proposal: {promo['promoted_n']}",
        f"- Demoted under proposal: {promo['demoted_n']}",
        f"- Unchanged passing: {promo['unchanged_passing']}",
        f"- Unchanged failing: {promo['unchanged_failing']}",
        "",
        "## Reference architecture sanity",
        "",
        f"- Spearman ρ (current vs proposed reference order): "
        f"{_fmt(ref['spearman_current_vs_proposed'])}",
        f"- Pass (ρ ≥ 0.9): {ref['pass']}",
        f"- References seen: {', '.join(ref['references_seen']) or '(none)'}",
    ]
    if ref["references_seen"]:
        lines.extend(
            [
                "",
                "| reference | current score | proposed score |",
                "|---|---:|---:|",
            ]
        )
        for name in ref["references_seen"]:
            cs = ref["current_scores"].get(name)
            ps = ref["proposed_scores"].get(name)
            lines.append(f"| {name} | {_fmt(cs, 2)} | {_fmt(ps, 2)} |")
    return "\n".join(lines) + "\n"


def _fmt(value: Any, digits: int = 3) -> str:
    val = safe_float(value)
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(RUNS_DB))
    parser.add_argument("--scoring-yaml", type=Path, default=DEFAULT_SCORING_YAML)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--tiers",
        type=str,
        default="screening,investigation,validation",
    )
    parser.add_argument(
        "--current-floor",
        type=float,
        default=450.0,
        help="Match breakthrough_gates.composite_floor in scoring_config.yaml.",
    )
    parser.add_argument(
        "--proposed-floor",
        type=float,
        default=None,
        help="If unset, reuses --current-floor.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    report = build_report(
        db_path=args.db,
        scoring_yaml_path=args.scoring_yaml,
        proposal_path=args.proposal,
        tiers=tiers,
        current_floor=args.current_floor,
        proposed_floor=args.proposed_floor,
        limit=args.limit,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    json_path = args.out / f"promotion_sim_{ts}.json"
    md_path = args.out / f"promotion_sim_{ts}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
