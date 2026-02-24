"""Pure statistical analysis of experiment history.

All functions take a LabNotebook and return structured results.
No LLM calls — only numpy/scipy.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .digest import (
    ArchitectureFamily,
    ConfigEffect,
    ConvergenceProfile,
    EfficiencyProfile,
    HypothesisOutcome,
    OpSynergy,
)

logger = logging.getLogger(__name__)


def _safe_float(val, default: float = 0.0) -> float:
    """Convert a DB value to float, handling bytes/blobs/None gracefully."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, bytes):
        # Some values stored as raw binary — skip
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# Input/output pseudo-ops to exclude from analysis
_SKIP_OPS = {"input", "output", ""}


def _extract_ops_from_graph(graph: dict) -> set:
    """Extract op names from a graph JSON dict.

    Handles both dict-keyed nodes (``{id: {op_name: ...}}``) and
    list-format nodes (``[{op_type: ...}, ...]``).
    """
    ops: set = set()
    nodes = graph.get("nodes", {})
    if isinstance(nodes, dict):
        for node in nodes.values():
            if isinstance(node, dict):
                op = node.get("op_name") or node.get("op_type") or node.get("op") or ""
            elif isinstance(node, str):
                op = node
            else:
                continue
            ops.add(op)
    elif isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict):
                op = node.get("op_name") or node.get("op_type") or node.get("op") or ""
            elif isinstance(node, str):
                op = node
            else:
                continue
            ops.add(op)
    return ops - _SKIP_OPS


# ---------------------------------------------------------------------------
# Training curve analysis
# ---------------------------------------------------------------------------

def analyze_training_curves(nb, limit: int = 500) -> List[ConvergenceProfile]:
    """Cluster training curves into convergence categories.

    Joins training_curves with program_results to get S1 labels.
    Returns one ConvergenceProfile per category.
    """
    try:
        rows = nb.conn.execute(
            """
            SELECT tc.result_id, tc.step, tc.loss,
                   pr.stage1_passed, pr.loss_ratio
            FROM training_curves tc
            JOIN program_results pr ON pr.result_id = tc.result_id
            WHERE tc.loss IS NOT NULL
            ORDER BY tc.result_id, tc.step
            LIMIT ?
            """,
            (limit * 50,),  # ~50 steps per curve × limit curves
        ).fetchall()
    except Exception as e:
        logger.warning("Failed to query training curves: %s", e)
        return []

    if not rows:
        return []

    # Group by result_id
    curves: Dict[str, dict] = {}
    for r in rows:
        rid = r[0]
        if rid not in curves:
            curves[rid] = {
                "steps": [], "losses": [],
                "s1_passed": bool(r[3]), "loss_ratio": r[4],
            }
        curves[rid]["steps"].append(r[1])
        curves[rid]["losses"].append(r[2])

    # Limit to most recent curves
    curve_list = list(curves.values())[-limit:]

    # Compute per-curve features
    categorized: Dict[str, list] = defaultdict(list)
    for c in curve_list:
        losses = np.array(c["losses"], dtype=np.float64)
        if len(losses) < 3:
            continue

        # Convergence speed: steps to reach 90% of total improvement
        total_improvement = losses[0] - losses[-1]
        if total_improvement > 0:
            threshold = losses[0] - 0.9 * total_improvement
            speed_steps = np.searchsorted(-losses, -threshold)
            convergence_speed = float(speed_steps) / len(losses)
        else:
            convergence_speed = 1.0  # never converged

        # Variance of second half (stability)
        half = len(losses) // 2
        variance = float(np.std(losses[half:])) if half > 0 else 0.0

        # Monotonicity: fraction of steps where loss decreases
        diffs = np.diff(losses)
        monotonicity = float(np.mean(diffs < 0)) if len(diffs) > 0 else 0.0

        # Classify
        if total_improvement <= 0 or losses[-1] > losses[0]:
            category = "divergent"
        elif convergence_speed < 0.3 and monotonicity > 0.6:
            category = "fast_converge"
        elif variance > 0.1 * abs(losses[-1]) and monotonicity < 0.4:
            category = "plateau"
        else:
            category = "slow_converge"

        categorized[category].append({
            "final_loss": float(losses[-1]),
            "convergence_speed": convergence_speed,
            "variance": variance,
            "monotonicity": monotonicity,
            "s1_passed": c["s1_passed"],
        })

    profiles = []
    for cat, entries in categorized.items():
        n = len(entries)
        s1_count = sum(1 for e in entries if e["s1_passed"])
        profiles.append(ConvergenceProfile(
            category=cat,
            count=n,
            avg_final_loss=float(np.mean([e["final_loss"] for e in entries])),
            avg_convergence_speed=float(np.mean([e["convergence_speed"] for e in entries])),
            avg_variance=float(np.mean([e["variance"] for e in entries])),
            avg_monotonicity=float(np.mean([e["monotonicity"] for e in entries])),
            s1_pass_rate=s1_count / n if n > 0 else 0.0,
        ))

    return profiles


# ---------------------------------------------------------------------------
# Architecture family clustering
# ---------------------------------------------------------------------------

def cluster_architecture_families(nb, min_cluster_size: int = 3) -> List[ArchitectureFamily]:
    """Cluster S1-passing architectures by op-set Jaccard similarity.

    Uses scipy agglomerative clustering on Jaccard distance matrix.
    """
    try:
        rows = nb.conn.execute(
            """
            SELECT pr.result_id, pr.graph_json, pr.graph_fingerprint,
                   pr.stage1_passed, pr.novelty_score, pr.loss_ratio
            FROM program_results pr
            WHERE pr.stage1_passed = 1
              AND pr.graph_json IS NOT NULL
            ORDER BY pr.timestamp DESC
            LIMIT 500
            """
        ).fetchall()
    except Exception as e:
        logger.warning("Failed to query S1 programs: %s", e)
        return []

    if len(rows) < min_cluster_size:
        return []

    # Extract op-sets
    programs = []
    all_ops: set = set()
    for r in rows:
        try:
            graph = json.loads(r[1])
            ops = _extract_ops_from_graph(graph)
            if ops:
                programs.append({
                    "result_id": r[0],
                    "fingerprint": str(r[2] or ""),
                    "ops": ops,
                    "novelty": _safe_float(r[4]),
                    "loss_ratio": _safe_float(r[5]),
                })
                all_ops.update(ops)
        except (json.JSONDecodeError, TypeError):
            continue

    if len(programs) < min_cluster_size:
        return []

    # Build Jaccard distance matrix
    n = len(programs)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            intersection = len(programs[i]["ops"] & programs[j]["ops"])
            union = len(programs[i]["ops"] | programs[j]["ops"])
            dist = 1.0 - (intersection / union if union > 0 else 0.0)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Agglomerative clustering
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        condensed = squareform(dist_matrix)
        linkage_matrix = linkage(condensed, method="average")
        # Cut at distance threshold 0.6 (40% Jaccard overlap)
        labels = fcluster(linkage_matrix, t=0.6, criterion="distance")
    except ImportError:
        # Fallback: no scipy — skip clustering
        logger.info("scipy not available, skipping architecture clustering")
        return []

    # Build families
    family_members: Dict[int, list] = defaultdict(list)
    for idx, label in enumerate(labels):
        family_members[int(label)].append(programs[idx])

    families = []
    for fid, members in sorted(family_members.items()):
        if len(members) < min_cluster_size:
            continue

        # Representative ops = ops shared by >50% of members
        op_counts: Counter = Counter()
        for m in members:
            for op in m["ops"]:
                op_counts[op] += 1
        threshold = len(members) * 0.5
        rep_ops = sorted(op for op, cnt in op_counts.items() if cnt >= threshold)

        families.append(ArchitectureFamily(
            family_id=fid,
            representative_ops=rep_ops[:10],
            n_members=len(members),
            s1_rate=1.0,  # all members are S1 by query
            avg_novelty=float(np.mean([m["novelty"] for m in members])),
            avg_loss_ratio=float(np.mean([m["loss_ratio"] for m in members])),
            example_fingerprints=[m["fingerprint"][:16] for m in members[:3]],
        ))

    # Sort by member count desc
    families.sort(key=lambda f: f.n_members, reverse=True)
    return families[:10]


# ---------------------------------------------------------------------------
# Config parameter effects
# ---------------------------------------------------------------------------

def analyze_config_effects(nb) -> List[ConfigEffect]:
    """Spearman correlation between config parameters and experiment outcomes."""
    try:
        rows = nb.conn.execute(
            """
            SELECT config_json, n_stage1_passed, best_loss_ratio,
                   n_programs_generated
            FROM experiments
            WHERE status = 'completed'
              AND config_json IS NOT NULL
              AND n_programs_generated > 0
            ORDER BY timestamp DESC
            LIMIT 200
            """
        ).fetchall()
    except Exception as e:
        logger.warning("Failed to query experiment configs: %s", e)
        return []

    if len(rows) < 10:
        return []

    # Parse configs
    params_of_interest = [
        "max_depth", "max_ops", "model_dim", "residual_prob",
        "math_space_weight", "n_programs", "structured_sparsity_bias",
    ]
    targets = {
        "s1_count": [],
        "best_loss_ratio": [],
    }
    param_values: Dict[str, list] = {p: [] for p in params_of_interest}

    for r in rows:
        try:
            config = json.loads(r[0])
        except (json.JSONDecodeError, TypeError):
            continue

        s1 = r[1] or 0
        loss = r[2]
        n_prog = r[3] or 1

        targets["s1_count"].append(s1 / max(n_prog, 1))  # normalize by batch size
        targets["best_loss_ratio"].append(loss if loss is not None else 1.0)

        for p in params_of_interest:
            val = config.get(p)
            param_values[p].append(float(val) if val is not None else np.nan)

    try:
        from scipy.stats import spearmanr
    except ImportError:
        logger.info("scipy not available, skipping config effect analysis")
        return []

    effects = []
    for param in params_of_interest:
        vals = np.array(param_values[param], dtype=np.float64)
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < 10:
            continue

        for target_name, target_vals in targets.items():
            tv = np.array(target_vals, dtype=np.float64)
            v = vals[valid_mask]
            t = tv[valid_mask]

            rho, p_value = spearmanr(v, t)
            if np.isnan(rho):
                continue

            direction = "neutral"
            if p_value < 0.05:
                direction = "positive" if rho > 0 else "negative"

            effects.append(ConfigEffect(
                param_name=param,
                target=target_name,
                rho=float(rho),
                p_value=float(p_value),
                direction=direction,
                n_samples=int(valid_mask.sum()),
            ))

    # Sort by p-value
    effects.sort(key=lambda e: e.p_value)
    return effects


# ---------------------------------------------------------------------------
# Op synergies
# ---------------------------------------------------------------------------

def analyze_op_synergies(nb, min_co_occurrences: int = 5) -> List[OpSynergy]:
    """Op-pair co-occurrence lift in S1 survivors vs all programs."""
    try:
        s1_rows = nb.conn.execute(
            """
            SELECT graph_json FROM program_results
            WHERE stage1_passed = 1 AND graph_json IS NOT NULL
            ORDER BY timestamp DESC LIMIT 300
            """
        ).fetchall()
        all_rows = nb.conn.execute(
            """
            SELECT graph_json, stage1_passed FROM program_results
            WHERE graph_json IS NOT NULL
            ORDER BY timestamp DESC LIMIT 1000
            """
        ).fetchall()
    except Exception as e:
        logger.warning("Failed to query op synergies: %s", e)
        return []

    def extract_ops(graph_json_str: str) -> set:
        try:
            graph = json.loads(graph_json_str)
            return _extract_ops_from_graph(graph)
        except (json.JSONDecodeError, TypeError):
            return set()

    # Count op-pair frequencies in S1 vs all
    def count_pairs(rows, json_idx: int = 0) -> Tuple[Counter, int]:
        pair_counts: Counter = Counter()
        n = 0
        for r in rows:
            ops = sorted(extract_ops(r[json_idx]))
            if len(ops) < 2:
                continue
            n += 1
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    pair_counts[(ops[i], ops[j])] += 1
        return pair_counts, n

    s1_pairs, n_s1 = count_pairs(s1_rows)
    all_pairs, n_all = count_pairs(all_rows)

    if n_s1 < 5 or n_all < 10:
        return []

    synergies = []
    for pair, s1_count in s1_pairs.items():
        if s1_count < min_co_occurrences:
            continue
        all_count = all_pairs.get(pair, 0)
        if all_count < min_co_occurrences:
            continue

        # Lift: P(pair | S1) / P(pair | all)
        s1_rate = s1_count / n_s1
        all_rate = all_count / n_all
        lift = s1_rate / all_rate if all_rate > 0 else 1.0

        label = "neutral"
        if lift > 1.5:
            label = "synergistic"
        elif lift < 0.5:
            label = "anti_synergistic"

        synergies.append(OpSynergy(
            op_a=pair[0],
            op_b=pair[1],
            lift=float(lift),
            co_occurrences=s1_count,
            label=label,
        ))

    # Also find anti-synergistic: pairs common overall but rare in S1
    for pair, all_count in all_pairs.items():
        if all_count < min_co_occurrences:
            continue
        s1_count = s1_pairs.get(pair, 0)
        all_rate = all_count / n_all
        s1_rate = s1_count / n_s1 if n_s1 > 0 else 0.0
        lift = s1_rate / all_rate if all_rate > 0 else 1.0

        if lift < 0.5 and pair not in s1_pairs:
            synergies.append(OpSynergy(
                op_a=pair[0],
                op_b=pair[1],
                lift=float(lift),
                co_occurrences=s1_count,
                label="anti_synergistic",
            ))

    # Deduplicate and sort
    seen = set()
    unique = []
    for s in synergies:
        key = (s.op_a, s.op_b)
        if key not in seen:
            seen.add(key)
            unique.append(s)

    # Sort: synergistic first (high lift), then anti (low lift)
    unique.sort(key=lambda s: -abs(s.lift - 1.0))
    return unique[:20]


# ---------------------------------------------------------------------------
# Hypothesis closure
# ---------------------------------------------------------------------------

def close_hypotheses(nb) -> List[HypothesisOutcome]:
    """Match experiment hypotheses with outcomes."""
    try:
        rows = nb.conn.execute(
            """
            SELECT e.experiment_id, e.hypothesis, e.n_stage1_passed,
                   e.n_programs_generated, e.best_loss_ratio, e.status
            FROM experiments e
            WHERE e.hypothesis IS NOT NULL
              AND e.hypothesis != ''
              AND e.status = 'completed'
            ORDER BY e.timestamp DESC
            LIMIT 50
            """
        ).fetchall()
    except Exception as e:
        logger.warning("Failed to query hypotheses: %s", e)
        return []

    outcomes = []
    for r in rows:
        exp_id = r[0]
        hypothesis = r[1]
        s1 = r[2] or 0
        n_prog = r[3] or 0
        loss = r[4]

        # Simple heuristic: if S1 > 0, consider partially confirmed
        if s1 > 0 and loss is not None and loss < 0.5:
            outcome = "confirmed"
            evidence = f"{s1} S1 survivors, loss_ratio={loss:.4f}"
        elif s1 > 0:
            outcome = "inconclusive"
            evidence = f"{s1} S1 survivors but loss_ratio={loss:.4f}" if loss else f"{s1} S1 survivors"
        else:
            outcome = "refuted"
            evidence = f"0/{n_prog} programs passed S1"

        outcomes.append(HypothesisOutcome(
            hypothesis=hypothesis[:200],
            experiment_id=exp_id,
            outcome=outcome,
            evidence=evidence,
            s1_count=s1,
        ))

    return outcomes


# ---------------------------------------------------------------------------
# Efficiency profiling
# ---------------------------------------------------------------------------

def analyze_efficiency_profiles(nb, families: List[ArchitectureFamily]) -> List[EfficiencyProfile]:
    """Compute per-family FLOP and parameter efficiency, identify Pareto-optimal.

    Uses S1 survivors with param_count and loss_ratio data.
    """
    if not families:
        return []

    try:
        rows = nb.conn.execute(
            """
            SELECT pr.result_id, pr.graph_json, pr.loss_ratio,
                   pr.param_count, pr.graph_n_params_estimate,
                   pr.novelty_score
            FROM program_results pr
            WHERE pr.stage1_passed = 1
              AND pr.graph_json IS NOT NULL
              AND pr.loss_ratio IS NOT NULL
            ORDER BY pr.timestamp DESC
            LIMIT 500
            """
        ).fetchall()
    except Exception as e:
        logger.warning("Failed to query efficiency data: %s", e)
        return []

    if not rows:
        return []

    # Map result → (ops, loss_ratio, params)
    program_data = []
    for r in rows:
        try:
            graph = json.loads(r[1])
            ops = _extract_ops_from_graph(graph)
            loss_ratio = _safe_float(r[2], 1.0)
            params = _safe_float(r[3]) or _safe_float(r[4])
            if not ops or params <= 0:
                continue
            program_data.append({
                "ops": ops,
                "loss_ratio": loss_ratio,
                "params": params,
            })
        except (json.JSONDecodeError, TypeError):
            continue

    if not program_data:
        return []

    # Assign programs to families by op-set overlap
    profiles = []
    family_metrics: Dict[int, list] = defaultdict(list)

    for prog in program_data:
        best_fam = None
        best_overlap = 0
        for fam in families:
            rep_ops = set(fam.representative_ops)
            if not rep_ops:
                continue
            overlap = len(prog["ops"] & rep_ops) / len(rep_ops)
            if overlap > best_overlap:
                best_overlap = overlap
                best_fam = fam.family_id
        if best_fam is not None and best_overlap >= 0.3:
            family_metrics[best_fam].append(prog)

    for fam in families:
        members = family_metrics.get(fam.family_id, [])
        if not members:
            continue
        avg_params = float(np.mean([m["params"] for m in members]))
        avg_loss = float(np.mean([m["loss_ratio"] for m in members]))
        # Approximate flops_per_token as 2 * params (standard estimate)
        avg_fpt = avg_params * 2.0
        mega = max(1.0, avg_params / 1e6)
        loss_per_mp = (1.0 - avg_loss) / mega if avg_loss < 1.0 else 0.0

        profiles.append(EfficiencyProfile(
            family_id=fam.family_id,
            avg_flops_per_token=avg_fpt,
            avg_params=avg_params,
            loss_per_megaparam=loss_per_mp,
            pareto_optimal=False,
        ))

    # Identify Pareto-optimal: no other family dominates on BOTH loss and params
    for i, pi in enumerate(profiles):
        dominated = False
        for j, pj in enumerate(profiles):
            if i == j:
                continue
            # pj dominates pi if it has lower loss AND fewer params
            pi_loss = 1.0 - pi.loss_per_megaparam  # lower is worse
            pj_loss = 1.0 - pj.loss_per_megaparam
            if pj.avg_params <= pi.avg_params and pj_loss <= pi_loss and (
                pj.avg_params < pi.avg_params or pj_loss < pi_loss
            ):
                dominated = True
                break
        if not dominated:
            pi.pareto_optimal = True

    profiles.sort(key=lambda p: p.loss_per_megaparam, reverse=True)
    return profiles[:10]


# ---------------------------------------------------------------------------
# Full analysis runner
# ---------------------------------------------------------------------------

def run_full_analysis(nb) -> dict:
    """Run all analysis functions and return a summary dict.

    Returns a dict suitable for building an ExperimentDigest.
    """
    convergence = analyze_training_curves(nb)
    families = cluster_architecture_families(nb)
    config_effects = analyze_config_effects(nb)
    synergies = analyze_op_synergies(nb)
    hypotheses = close_hypotheses(nb)
    efficiency = analyze_efficiency_profiles(nb, families)

    # Count experiments analyzed
    try:
        n_exp = nb.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status = 'completed'"
        ).fetchone()[0]
    except Exception:
        n_exp = 0

    # Count curves analyzed
    n_curves = sum(p.count for p in convergence)

    return {
        "convergence_profiles": convergence,
        "architecture_families": families,
        "config_effects": config_effects,
        "op_synergies": synergies,
        "hypothesis_outcomes": hypotheses,
        "efficiency_profiles": efficiency,
        "n_experiments_analyzed": n_exp,
        "n_curves_analyzed": n_curves,
    }
