#!/usr/bin/env python
"""Rank-order remaining un-backfilled archs by predicted-curriculum-value.

Consumes the trends report from ``ar_curriculum_trends`` (correlations between
upstream features and ar_curriculum_auc_pair_final) and produces a JSONL of
result_ids ordered so the most informative archs go first.

Priority score per arch (higher = run sooner):

  score =   alpha * normalized(strong_upstream_signal_predicted_auc)
          + beta  * diversity_template_bonus
          + gamma * suspicious_profile_bonus
          + delta * random_jitter

Then we deduplicate, cap, and write JSONL.

Output:
  research/runtime/ar_curriculum_experiment/priority_<run_id>.jsonl
  research/runtime/ar_curriculum_experiment/priority_<run_id>.md
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.scientist.notebook import LabNotebook
from research.tools.ar_curriculum_trends import UPSTREAM_FEATURES, _detect_motifs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / "research/runtime/ar_curriculum_experiment"
DEFAULT_DB = REPO_ROOT / "research/runs.db"


def _load_correlations(trends_json: Path) -> dict[str, float]:
    payload = json.loads(trends_json.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for c in payload.get("correlations", []):
        out[c["feature"]] = float(c.get("spearman_vs_auc") or 0.0)
    return out


def _load_template_means(trends_json: Path) -> dict[str, float]:
    payload = json.loads(trends_json.read_text(encoding="utf-8"))
    return {t["template"]: float(t["mean_auc"]) for t in payload.get("templates", [])}


def _load_done_fingerprints(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT graph_fingerprint FROM program_results_compat "
        "WHERE ar_curriculum_auc_pair_final IS NOT NULL"
    ).fetchall()
    return {str(r[0]) for r in rows if r[0]}


def _fetch_remaining(
    conn: sqlite3.Connection, tiers: tuple[str, ...]
) -> list[dict[str, Any]]:
    feature_cols = ", ".join(f"pr.{f}" for f in UPSTREAM_FEATURES)
    placeholders = ",".join("?" for _ in tiers)
    sql = f"""
        SELECT
            pr.result_id,
            pr.graph_fingerprint,
            pr.graph_json,
            l.tier,
            l.composite_score,
            json_extract(pr.graph_json, '$.metadata.templates_used') AS templates_used,
            {feature_cols}
        FROM program_results_compat pr
        JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE pr.ar_curriculum_auc_pair_final IS NULL
          AND l.tier IN ({placeholders})
          AND pr.graph_json IS NOT NULL
        ORDER BY l.composite_score DESC
    """
    rows = conn.execute(sql, tiers).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        d = (
            dict(row)
            if hasattr(row, "keys")
            else {k: row[i] for i, k in enumerate(row.keys())}
        )
        try:
            d["templates_list"] = (
                json.loads(d["templates_used"]) if d.get("templates_used") else []
            )
        except (TypeError, json.JSONDecodeError):
            d["templates_list"] = []
        d["motifs"] = _detect_motifs(d.get("graph_json"))
        out.append(d)
    return out


def _predicted_auc_score(arch: dict[str, Any], correlations: dict[str, float]) -> float:
    """Linear weighted sum of normalized upstream features × |spearman corr|.

    Weight is the absolute correlation; sign flips if correlation is negative
    (high feature value → low predicted curriculum AUC).
    """
    score = 0.0
    weight_sum = 0.0
    for feat in UPSTREAM_FEATURES:
        rho = correlations.get(feat, 0.0)
        if abs(rho) < 0.05:
            continue
        v = arch.get(feat)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        # We don't have access to the cohort min/max here for normalization;
        # use a heuristic by feature type. Most features are bounded [0, 1]
        # except wikitext_perplexity (lower is better) and decay_slope (signed).
        if feat == "wikitext_perplexity":
            normalized = max(0.0, min(1.0, 1.0 - (fv - 5.0) / 200.0))
        elif feat == "fp_jacobian_erf_decay_slope":
            normalized = max(0.0, min(1.0, (fv + 1.0) / 2.0))
        else:
            normalized = max(0.0, min(1.0, fv))
        contribution = normalized if rho > 0 else (1.0 - normalized)
        score += abs(rho) * contribution
        weight_sum += abs(rho)
    return score / weight_sum if weight_sum > 0 else 0.0


def _template_bonus(arch: dict[str, Any], template_means: dict[str, float]) -> float:
    """Return mean AUC across this arch's templates seen in the trends."""
    seen = [
        template_means[t] for t in arch.get("templates_list", []) if t in template_means
    ]
    if not seen:
        return 0.0
    return sum(seen) / len(seen)


def _suspicious_bonus(arch: dict[str, Any]) -> float:
    """Reward archs with conflicting upstream signals — diagnostic value.

    e.g. high induction AUC but low binding AUC, or high ar_legacy but
    low binding screening. These are mostly mid-tier mismatches.
    """
    ind = float(arch.get("induction_screening_auc") or 0)
    bnd = float(arch.get("binding_screening_auc") or 0)
    ar_leg = float(arch.get("ar_legacy_auc") or 0)
    spread = max(ind, bnd, ar_leg) - min(ind, bnd, ar_leg)
    return float(min(1.0, spread))


def assign_priority(
    archs: list[dict[str, Any]],
    *,
    correlations: dict[str, float],
    template_means: dict[str, float],
    alpha: float = 1.0,
    beta: float = 0.6,
    gamma: float = 0.3,
    delta: float = 0.05,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    scored = []
    for arch in archs:
        pred = _predicted_auc_score(arch, correlations)
        tpl_b = _template_bonus(arch, template_means)
        susp = _suspicious_bonus(arch)
        jitter = rng.random()
        priority = alpha * pred + beta * tpl_b + gamma * susp + delta * jitter
        scored.append(
            {
                "result_id": arch["result_id"],
                "graph_fingerprint": arch["graph_fingerprint"],
                "tier": arch["tier"],
                "composite_score": float(arch.get("composite_score") or 0),
                "templates": arch.get("templates_list", [])[:3],
                "motifs": arch.get("motifs", []),
                "priority": round(priority, 4),
                "predicted_auc_score": round(pred, 4),
                "template_bonus": round(tpl_b, 4),
                "suspicious_bonus": round(susp, 4),
            }
        )
    scored.sort(key=lambda d: d["priority"], reverse=True)
    return scored


def write_report(
    priority: list[dict[str, Any]],
    correlations: dict[str, float],
    out_dir: Path,
    run_id: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"priority_{run_id}.jsonl"
    md_path = out_dir / f"priority_{run_id}.md"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for entry in priority:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    lines: list[str] = [
        f"# AR curriculum priority list — {run_id}",
        "",
        f"n_remaining = {len(priority)}",
        "",
        "## Used correlations (>|0.05|)",
        "",
        "| feature | spearman vs AUC |",
        "|---|---:|",
    ]
    for feat in UPSTREAM_FEATURES:
        rho = correlations.get(feat, 0.0)
        if abs(rho) >= 0.05:
            lines.append(f"| {feat} | {rho:+.3f} |")

    lines += [
        "",
        "## Top 30 by priority",
        "",
        "| rank | fp | tier | composite | priority | pred_auc | tpl_bonus | susp_bonus | motifs |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, p in enumerate(priority[:30], 1):
        motifs = ",".join(p["motifs"]) if p["motifs"] else "—"
        lines.append(
            f"| {i} | {p['graph_fingerprint'][:12]} | {p['tier']} | "
            f"{p['composite_score']:.0f} | {p['priority']:.3f} | "
            f"{p['predicted_auc_score']:.3f} | {p['template_bonus']:.3f} | "
            f"{p['suspicious_bonus']:.3f} | {motifs} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path, md_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument(
        "--trends-json",
        type=Path,
        required=True,
        help="Trends report JSON (output of ar_curriculum_trends).",
    )
    p.add_argument(
        "--tiers",
        default="validation,investigation",
        help="Tiers to draw from for the priority pass.",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.6)
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tiers = tuple(t.strip() for t in str(args.tiers).split(",") if t.strip())
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    correlations = _load_correlations(args.trends_json)
    template_means = _load_template_means(args.trends_json)
    logger.info(
        "Loaded %d correlations and %d template means from %s",
        len(correlations),
        len(template_means),
        args.trends_json.name,
    )

    nb = LabNotebook(str(args.db), read_only=True)
    archs = _fetch_remaining(nb.conn, tiers)
    nb.close()
    logger.info("Found %d remaining archs across tiers=%s", len(archs), tiers)
    if not archs:
        return 0

    priority = assign_priority(
        archs,
        correlations=correlations,
        template_means=template_means,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        seed=int(args.seed),
    )
    jsonl_path, md_path = write_report(priority, correlations, RUNTIME_ROOT, run_id)
    logger.info("Wrote %s (%d entries)", jsonl_path, len(priority))
    logger.info("Wrote %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
