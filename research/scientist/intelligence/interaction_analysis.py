"""Op interaction analysis: pairwise success/loss matrices from experiment + profiling data.

Generates component×component interaction heatmaps showing which op pairs
succeed or fail together, grounded in both experiment history and profiling data.

Usage:
    python -m research.scientist.intelligence.interaction_analysis --category math
    python -m research.scientist.intelligence.interaction_analysis --output heatmap
    python -m research.scientist.intelligence.interaction_analysis --output json
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .ml_corpus import load_deduped_graph_training_rows

logger = logging.getLogger(__name__)

_DEFAULT_NOTEBOOK_DB = Path(__file__).parents[2] / "lab_notebook.db"
_DEFAULT_PROFILING_DB = (
    Path(__file__).parents[2] / "profiling" / "component_profiles.db"
)


@dataclass(slots=True)
class PairStats:
    """Accumulated statistics for one (op_a, op_b) pair."""

    n_graphs: int = 0
    n_s0_pass: int = 0
    n_s1_pass: int = 0
    loss_sum: float = 0.0
    loss_count: int = 0
    # From profiling DB (if available)
    profiling_stable: Optional[bool] = None
    profiling_lipschitz: Optional[float] = None
    profiling_stability_delta: Optional[float] = None

    @property
    def s0_rate(self) -> float:
        return self.n_s0_pass / max(self.n_graphs, 1)

    @property
    def s1_rate(self) -> float:
        return self.n_s1_pass / max(self.n_graphs, 1)

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / max(self.loss_count, 1)


@dataclass(slots=True)
class InteractionMatrix:
    """Pairwise interaction data for all ops."""

    ops: List[str]  # ordered op names
    op_to_idx: Dict[str, int]
    pair_stats: Dict[Tuple[str, str], PairStats]
    op_categories: Dict[str, str]  # op_name → category

    def s0_matrix(self) -> np.ndarray:
        """NxN matrix of S0 pass rates. NaN where no observations."""
        n = len(self.ops)
        m = np.full((n, n), np.nan)
        for (a, b), ps in self.pair_stats.items():
            if a in self.op_to_idx and b in self.op_to_idx and ps.n_graphs >= 1:
                m[self.op_to_idx[a], self.op_to_idx[b]] = ps.s0_rate
        return m

    def s1_matrix(self) -> np.ndarray:
        """NxN matrix of S1 pass rates. NaN where no observations."""
        n = len(self.ops)
        m = np.full((n, n), np.nan)
        for (a, b), ps in self.pair_stats.items():
            if a in self.op_to_idx and b in self.op_to_idx and ps.n_graphs >= 1:
                m[self.op_to_idx[a], self.op_to_idx[b]] = ps.s1_rate
        return m

    def loss_matrix(self) -> np.ndarray:
        """NxN matrix of mean loss ratios (S1-passing only). NaN where no data."""
        n = len(self.ops)
        m = np.full((n, n), np.nan)
        for (a, b), ps in self.pair_stats.items():
            if a in self.op_to_idx and b in self.op_to_idx and ps.loss_count >= 1:
                m[self.op_to_idx[a], self.op_to_idx[b]] = ps.mean_loss
        return m

    def obs_matrix(self) -> np.ndarray:
        """NxN matrix of observation counts."""
        n = len(self.ops)
        m = np.zeros((n, n), dtype=np.int32)
        for (a, b), ps in self.pair_stats.items():
            if a in self.op_to_idx and b in self.op_to_idx:
                m[self.op_to_idx[a], self.op_to_idx[b]] = ps.n_graphs
        return m

    def filter_category(self, category: str) -> "InteractionMatrix":
        """Return a sub-matrix containing only ops from the given category."""
        filtered_ops = sorted(
            op
            for op, cat in self.op_categories.items()
            if cat == category and op in self.op_to_idx
        )
        new_idx = {op: i for i, op in enumerate(filtered_ops)}
        filtered_pairs = {
            (a, b): ps
            for (a, b), ps in self.pair_stats.items()
            if a in new_idx and b in new_idx
        }
        return InteractionMatrix(
            ops=filtered_ops,
            op_to_idx=new_idx,
            pair_stats=filtered_pairs,
            op_categories={op: self.op_categories[op] for op in filtered_ops},
        )

    def category_rollup(self) -> "InteractionMatrix":
        """Roll up to category×category level."""
        cats = sorted(set(self.op_categories.values()))
        cat_idx = {c: i for i, c in enumerate(cats)}
        cat_pairs: Dict[Tuple[str, str], PairStats] = {}

        for (a, b), ps in self.pair_stats.items():
            cat_a = self.op_categories.get(a, "unknown")
            cat_b = self.op_categories.get(b, "unknown")
            key = (cat_a, cat_b)
            if key not in cat_pairs:
                cat_pairs[key] = PairStats()
            cp = cat_pairs[key]
            cp.n_graphs += ps.n_graphs
            cp.n_s0_pass += ps.n_s0_pass
            cp.n_s1_pass += ps.n_s1_pass
            cp.loss_sum += ps.loss_sum
            cp.loss_count += ps.loss_count

        return InteractionMatrix(
            ops=cats,
            op_to_idx=cat_idx,
            pair_stats=cat_pairs,
            op_categories={c: c for c in cats},
        )

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary."""
        n_pairs = len(self.pair_stats)
        n_observed = sum(1 for ps in self.pair_stats.values() if ps.n_graphs > 0)
        n_s1_pairs = sum(1 for ps in self.pair_stats.values() if ps.n_s1_pass > 0)
        return {
            "n_ops": len(self.ops),
            "n_pairs_total": n_pairs,
            "n_pairs_observed": n_observed,
            "n_pairs_with_s1_pass": n_s1_pairs,
            "coverage": n_observed / max(len(self.ops) ** 2, 1),
        }


def _load_op_categories(notebook_db: Path, profiling_db: Path) -> Dict[str, str]:
    """Load op → category mapping from profiling DB and primitives registry."""
    categories: Dict[str, str] = {}

    # From profiling DB
    if profiling_db.exists():
        try:
            conn = sqlite3.connect(str(profiling_db), timeout=5)
            rows = conn.execute(
                "SELECT op_name, category FROM op_profiles WHERE category IS NOT NULL"
            ).fetchall()
            conn.close()
            for op_name, cat in rows:
                categories[op_name] = cat
        except Exception as e:
            logger.warning("Failed to load categories from profiling DB: %s", e)

    # Fill gaps from primitives registry
    try:
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        for name, prim in PRIMITIVE_REGISTRY.items():
            if name not in categories:
                cat = prim.category
                categories[name] = cat.value if hasattr(cat, "value") else str(cat)
    except ImportError as exc:
        logger.debug(
            "Primitive registry unavailable while loading op categories: %s", exc
        )

    return categories


def _extract_co_occurring_pairs(graph_json: str) -> Set[Tuple[str, str]]:
    """Extract all unique (op_a, op_b) co-occurrence pairs from a graph.

    Returns unordered pairs where a <= b lexicographically to avoid double-counting.
    """
    try:
        g = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
    except (json.JSONDecodeError, TypeError):
        return set()

    nodes = g.get("nodes") or {}
    ops = sorted(
        set(
            n.get("op_name", "")
            for n in nodes.values()
            if n.get("op_name", "") and n.get("op_name") != "input"
        )
    )

    pairs: Set[Tuple[str, str]] = set()
    for i, a in enumerate(ops):
        for b in ops[i:]:  # include self-pairs (a, a)
            pairs.add((a, b))
    return pairs


def build_interaction_matrix(
    notebook_db: Path = _DEFAULT_NOTEBOOK_DB,
    profiling_db: Path = _DEFAULT_PROFILING_DB,
    min_observations: int = 0,
) -> InteractionMatrix:
    """Build the full pairwise interaction matrix from experiment + profiling data.

    Args:
        notebook_db: Path to lab_notebook.db with experiment results.
        profiling_db: Path to component_profiles.db with profiling data.
        min_observations: Minimum observation count to include a pair.

    Returns:
        InteractionMatrix with all observed op pairs.
    """
    categories = _load_op_categories(notebook_db, profiling_db)
    pair_stats: Dict[Tuple[str, str], PairStats] = {}
    all_ops: Set[str] = set()

    # ── Phase 1: Extract from experiment data ──
    if notebook_db.exists():
        try:
            rows = load_deduped_graph_training_rows(notebook_db)

            for row in rows:
                graph_json = row["graph_json"]
                s0 = row["stage0_any_passed"]
                s1 = row["stage1_any_passed"]
                lr = row.get("loss_ratio_best")
                pairs = _extract_co_occurring_pairs(graph_json)
                for a, b in pairs:
                    all_ops.add(a)
                    all_ops.add(b)
                    if (a, b) not in pair_stats:
                        pair_stats[(a, b)] = PairStats()
                    ps = pair_stats[(a, b)]
                    ps.n_graphs += 1
                    if s0:
                        ps.n_s0_pass += 1
                    if s1:
                        ps.n_s1_pass += 1
                    if s1 and lr is not None and math.isfinite(lr):
                        ps.loss_sum += lr
                        ps.loss_count += 1

            logger.info(
                "Loaded %d experiment graphs → %d op pairs", len(rows), len(pair_stats)
            )
        except Exception as e:
            logger.warning("Failed to load experiment data: %s", e)

    # ── Phase 2: Enrich from profiling DB ──
    if profiling_db.exists():
        try:
            conn = sqlite3.connect(str(profiling_db), timeout=5)
            rows = conn.execute(
                """SELECT op_a, op_b, stability_delta, lipschitz_estimate,
                          grad_vanishing, grad_exploding, output_has_nan, grad_has_nan
                   FROM pair_profiles
                   WHERE error IS NULL"""
            ).fetchall()
            conn.close()

            for (
                op_a,
                op_b,
                stab_delta,
                lip,
                grad_van,
                grad_exp,
                out_nan,
                grad_nan,
            ) in rows:
                # Normalize order
                a, b = (op_a, op_b) if op_a <= op_b else (op_b, op_a)
                all_ops.add(a)
                all_ops.add(b)
                if (a, b) not in pair_stats:
                    pair_stats[(a, b)] = PairStats()
                ps = pair_stats[(a, b)]
                stable = not (out_nan or grad_nan or grad_van or grad_exp)
                ps.profiling_stable = stable
                ps.profiling_lipschitz = lip
                ps.profiling_stability_delta = stab_delta

            logger.info("Enriched with %d profiling pairs", len(rows))
        except Exception as e:
            logger.warning("Failed to load profiling data: %s", e)

    # ── Build matrix ──
    if min_observations > 0:
        pair_stats = {
            k: v for k, v in pair_stats.items() if v.n_graphs >= min_observations
        }

    ops_sorted = sorted(all_ops)
    op_to_idx = {op: i for i, op in enumerate(ops_sorted)}

    return InteractionMatrix(
        ops=ops_sorted,
        op_to_idx=op_to_idx,
        pair_stats=pair_stats,
        op_categories=categories,
    )


def render_heatmap(
    matrix: InteractionMatrix,
    metric: str = "s1_rate",
    title: str = "Op Interaction Heatmap",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 12),
) -> Optional[Path]:
    """Render a heatmap of the interaction matrix.

    Args:
        matrix: InteractionMatrix to visualize.
        metric: One of 's0_rate', 's1_rate', 'loss', 'obs'.
        title: Plot title.
        output_path: If provided, save to this path. Otherwise show interactively.
        figsize: Figure size.

    Returns:
        Path to saved file, or None if shown interactively.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for heatmap rendering")
        return None

    if metric == "s0_rate":
        data = matrix.s0_matrix()
        cmap = "RdYlGn"
        vmin, vmax = 0.0, 1.0
        label = "S0 Pass Rate"
    elif metric == "s1_rate":
        data = matrix.s1_matrix()
        cmap = "RdYlGn"
        vmin, vmax = 0.0, 1.0
        label = "S1 Pass Rate"
    elif metric == "loss":
        data = matrix.loss_matrix()
        cmap = "RdYlGn_r"  # reversed: lower loss = green
        vmin, vmax = 0.0, 1.0
        label = "Mean Loss Ratio"
    elif metric == "obs":
        data = matrix.obs_matrix().astype(float)
        data[data == 0] = np.nan
        cmap = "Blues"
        vmin, vmax = None, None
        label = "Observation Count"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    n = len(matrix.ops)
    if n == 0:
        logger.warning("Empty matrix, nothing to render")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrix.ops, rotation=90, fontsize=max(4, 10 - n // 10))
    ax.set_yticklabels(matrix.ops, fontsize=max(4, 10 - n // 10))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Op B")
    ax.set_ylabel("Op A")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(label)

    # Annotate cells for small matrices
    if n <= 20:
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                if np.isfinite(val):
                    text = f"{val:.2f}" if metric != "obs" else f"{int(val)}"
                    color = "black" if 0.3 < val < 0.7 else "white"
                    if metric == "obs":
                        color = "black"
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=max(5, 8 - n // 5),
                        color=color,
                    )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Heatmap saved to %s", output_path)
        return output_path
    else:
        plt.show()
        plt.close(fig)
        return None


def export_json(
    matrix: InteractionMatrix,
    output_path: Path,
) -> Path:
    """Export interaction data as JSON for dashboard consumption."""
    records = []
    for (a, b), ps in sorted(matrix.pair_stats.items()):
        rec = {
            "op_a": a,
            "op_b": b,
            "category_a": matrix.op_categories.get(a, "unknown"),
            "category_b": matrix.op_categories.get(b, "unknown"),
            "n_graphs": ps.n_graphs,
            "s0_rate": round(ps.s0_rate, 4) if ps.n_graphs > 0 else None,
            "s1_rate": round(ps.s1_rate, 4) if ps.n_graphs > 0 else None,
            "mean_loss": round(ps.mean_loss, 4) if ps.loss_count > 0 else None,
            "n_s1_pass": ps.n_s1_pass,
            "profiling_stable": ps.profiling_stable,
            "profiling_lipschitz": round(ps.profiling_lipschitz, 4)
            if ps.profiling_lipschitz is not None
            else None,
        }
        records.append(rec)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "summary": matrix.summary(),
                "pairs": records,
            },
            f,
            indent=2,
        )

    logger.info("Exported %d pair records to %s", len(records), output_path)
    return output_path


def print_top_pairs(
    matrix: InteractionMatrix,
    metric: str = "s1_rate",
    n: int = 20,
    min_obs: int = 5,
) -> None:
    """Print top N op pairs by a given metric."""
    pairs = []
    for (a, b), ps in matrix.pair_stats.items():
        if ps.n_graphs < min_obs:
            continue
        if metric == "s1_rate":
            val = ps.s1_rate
        elif metric == "loss":
            if ps.loss_count == 0:
                continue
            val = -ps.mean_loss  # negate so lower loss = higher rank
        else:
            val = ps.s0_rate
        pairs.append(((a, b), ps, val))

    pairs.sort(key=lambda x: -x[2])

    print(f"\n{'=' * 80}")
    print(f"Top {n} pairs by {metric} (min {min_obs} observations)")
    print(f"{'=' * 80}")
    print(
        f"{'Rank':>4}  {'Op A':<25} {'Op B':<25} {'S0%':>5} {'S1%':>5} {'Loss':>6} {'N':>5}"
    )
    print(f"{'-' * 80}")
    for i, ((a, b), ps, _) in enumerate(pairs[:n], 1):
        loss_str = f"{ps.mean_loss:.3f}" if ps.loss_count > 0 else "  n/a"
        print(
            f"{i:>4}  {a:<25} {b:<25} {ps.s0_rate:>4.0%} {ps.s1_rate:>4.0%} {loss_str:>6} {ps.n_graphs:>5}"
        )


def print_worst_pairs(
    matrix: InteractionMatrix,
    n: int = 20,
    min_obs: int = 5,
) -> None:
    """Print worst N op pairs by S1 pass rate."""
    pairs = []
    for (a, b), ps in matrix.pair_stats.items():
        if ps.n_graphs < min_obs:
            continue
        pairs.append(((a, b), ps, ps.s1_rate))

    pairs.sort(key=lambda x: x[2])

    print(f"\n{'=' * 80}")
    print(f"Worst {n} pairs by S1 rate (min {min_obs} observations)")
    print(f"{'=' * 80}")
    print(
        f"{'Rank':>4}  {'Op A':<25} {'Op B':<25} {'S0%':>5} {'S1%':>5} {'Loss':>6} {'N':>5}"
    )
    print(f"{'-' * 80}")
    for i, ((a, b), ps, _) in enumerate(pairs[:n], 1):
        loss_str = f"{ps.mean_loss:.3f}" if ps.loss_count > 0 else "  n/a"
        print(
            f"{i:>4}  {a:<25} {b:<25} {ps.s0_rate:>4.0%} {ps.s1_rate:>4.0%} {loss_str:>6} {ps.n_graphs:>5}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Op interaction analysis and heatmap generation"
    )
    parser.add_argument("--notebook-db", type=Path, default=_DEFAULT_NOTEBOOK_DB)
    parser.add_argument("--profiling-db", type=Path, default=_DEFAULT_PROFILING_DB)
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter to ops in this category (e.g., math, math_space, routing)",
    )
    parser.add_argument(
        "--rollup", action="store_true", help="Roll up to category×category level"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="s1_rate",
        choices=["s0_rate", "s1_rate", "loss", "obs"],
        help="Metric to display",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="print",
        choices=["print", "heatmap", "json", "all"],
        help="Output format",
    )
    parser.add_argument(
        "--min-obs", type=int, default=3, help="Minimum observations to include"
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parents[2] / "artifacts"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Building interaction matrix...")
    matrix = build_interaction_matrix(
        notebook_db=args.notebook_db,
        profiling_db=args.profiling_db,
        min_observations=args.min_obs,
    )
    logger.info("Matrix: %s", matrix.summary())

    if args.rollup:
        matrix = matrix.category_rollup()
        logger.info("Rolled up to category level: %d categories", len(matrix.ops))

    if args.category:
        matrix = matrix.filter_category(args.category)
        logger.info("Filtered to category '%s': %d ops", args.category, len(matrix.ops))

    if args.output in ("print", "all"):
        print_top_pairs(matrix, metric=args.metric, n=args.top_n, min_obs=args.min_obs)
        print_worst_pairs(matrix, n=args.top_n, min_obs=args.min_obs)
        print(f"\nSummary: {matrix.summary()}")

    if args.output in ("heatmap", "all"):
        cat_suffix = f"_{args.category}" if args.category else ""
        rollup_suffix = "_category" if args.rollup else ""
        fname = f"interaction_{args.metric}{cat_suffix}{rollup_suffix}.png"
        title = f"Op Interaction: {args.metric}"
        if args.category:
            title += f" ({args.category})"
        if args.rollup:
            title += " (category rollup)"
        render_heatmap(
            matrix,
            metric=args.metric,
            title=title,
            output_path=args.output_dir / fname,
        )

    if args.output in ("json", "all"):
        cat_suffix = f"_{args.category}" if args.category else ""
        rollup_suffix = "_category" if args.rollup else ""
        fname = f"interaction_data{cat_suffix}{rollup_suffix}.json"
        export_json(matrix, args.output_dir / fname)


if __name__ == "__main__":
    main()
