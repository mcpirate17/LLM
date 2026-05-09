"""UCB1 Exploration Scheduler for template selection.

Adapts template weights based on observed performance using Upper Confidence
Bound (UCB1). Under-explored templates get a confidence bonus; over-explored
templates with poor results get suppressed.

Usage:
    from research.search.scheduler import ExplorationScheduler

    scheduler = ExplorationScheduler(db_path="research/lab_notebook.db")
    weights = scheduler.step()
    # Pass `weights` as template_weights to GrammarConfig
"""

from __future__ import annotations

import math
import random
import sqlite3
import time
from typing import Dict, Optional

from research.defaults import RUNS_DB


def _fetch_template_stats(db_path: str, query: str) -> list[tuple]:
    with sqlite3.connect(db_path, timeout=5.0) as conn:
        conn.execute("PRAGMA busy_timeout=5000")
        return conn.execute(query).fetchall()


def _normalize_weight_map(weights: Dict[str, float]) -> Dict[str, float]:
    if not weights:
        return weights
    max_w = max(weights.values())
    min_w = min(weights.values())
    span = max_w - min_w if max_w > min_w else 1.0
    for key, value in list(weights.items()):
        weights[key] = 0.5 + 7.5 * (value - min_w) / span
    return weights


class ExplorationScheduler:
    """UCB1-based template exploration scheduler.

    Balances exploitation (templates with good mean loss) against exploration
    (templates with few evaluations). Outputs a weight dict suitable for
    passing to GrammarConfig.template_weights or grammar.generate_layer_graph().
    """

    __slots__ = (
        "_db_path",
        "_exploration_constant",
        "_min_evals",
        "_cache",
        "_cache_expires",
        "_cache_ttl",
    )

    def __init__(
        self,
        db_path: str = RUNS_DB,
        exploration_constant: float = 1.5,
        min_evals: int = 50,
        cache_ttl: float = 120.0,
    ):
        self._db_path = db_path
        self._exploration_constant = exploration_constant
        self._min_evals = min_evals
        self._cache: Optional[Dict[str, float]] = None
        self._cache_expires: float = 0.0
        self._cache_ttl = cache_ttl

    def step(self) -> Dict[str, float]:
        """Compute UCB1-weighted template probabilities.

        Returns dict of template_name → weight. Templates below min_evals
        get maximum exploration bonus.
        """
        now = time.time()
        if self._cache is not None and now < self._cache_expires:
            return self._cache

        try:
            rows = _fetch_template_stats(
                self._db_path,
                "SELECT template_name, eval_count, s1_pass_count, mean_loss "
                "FROM template_stats",
            )
        except Exception:
            return self._cache or {}

        if not rows:
            return {}

        total_evals = sum(r[1] for r in rows)
        log_total = math.log(max(total_evals, 1))

        weights: Dict[str, float] = {}
        for tpl_name, eval_count, s1_count, mean_loss in rows:
            if eval_count < self._min_evals:
                # Under-explored: give maximum UCB bonus to force exploration
                ucb_bonus = self._exploration_constant * math.sqrt(
                    log_total / max(eval_count, 1)
                )
                weights[tpl_name] = 3.0 + ucb_bonus
            else:
                # Enough data: exploit with exploration bonus
                s1_rate = s1_count / max(eval_count, 1)
                # Reward = normalized inverse loss (lower loss → higher reward)
                if mean_loss is not None and math.isfinite(mean_loss) and mean_loss > 0:
                    reward = math.exp(-2.0 * mean_loss) * (1.0 + s1_rate)
                else:
                    reward = s1_rate

                ucb_bonus = self._exploration_constant * math.sqrt(
                    log_total / eval_count
                )
                weights[tpl_name] = reward + ucb_bonus

        self._cache = _normalize_weight_map(weights)
        self._cache_expires = now + self._cache_ttl
        return self._cache


class ThompsonScheduler:
    """Thompson sampling for template selection.

    For each template, maintains Beta(alpha, beta) posterior:
      alpha = s1_pass_count + 1  (successes + prior)
      beta = (eval_count - s1_pass_count) + 1  (failures + prior)

    To select weights: sample theta ~ Beta(alpha, beta) for each template,
    normalize to [0.5, 8.0] range compatible with GrammarConfig.template_weights.

    Advantages over UCB1:
    - Better for non-stationary rewards (templates improve as rules improve)
    - Natural exploration without tuning C parameter
    - Probability matching: explores proportional to probability of being best
    """

    __slots__ = ("_db_path", "_cache", "_cache_expires", "_cache_ttl", "_rng")

    def __init__(
        self,
        db_path: str = RUNS_DB,
        cache_ttl: float = 120.0,
        seed: Optional[int] = None,
    ):
        self._db_path = db_path
        self._cache: Optional[Dict[str, float]] = None
        self._cache_expires: float = 0.0
        self._cache_ttl = cache_ttl
        self._rng = random.Random(seed)

    def sample(self) -> Dict[str, float]:
        """Sample template weights from Beta posteriors.

        Returns dict of template_name → weight in [0.5, 8.0].
        Each call produces different weights (Thompson sampling is stochastic).
        Cache is used for the underlying DB stats, not the samples.
        """
        stats = self._load_stats()
        if not stats:
            return {}

        weights: Dict[str, float] = {}
        for tpl_name, (alpha, beta) in stats.items():
            # Sample from Beta(alpha, beta) using stdlib random
            theta = self._rng.betavariate(alpha, beta)
            weights[tpl_name] = theta

        return _normalize_weight_map(weights)

    def _load_stats(self) -> Dict[str, tuple]:
        """Load Beta parameters from template_stats table.

        Returns dict of template_name → (alpha, beta).
        """
        now = time.time()
        if self._cache is not None and now < self._cache_expires:
            return self._cache

        try:
            rows = _fetch_template_stats(
                self._db_path,
                "SELECT template_name, eval_count, s1_pass_count FROM template_stats",
            )
        except Exception:
            return self._cache or {}

        if not rows:
            return {}

        stats: Dict[str, tuple] = {}
        for tpl_name, eval_count, s1_count in rows:
            alpha = max(s1_count, 0) + 1  # successes + prior
            beta = max(eval_count - s1_count, 0) + 1  # failures + prior
            stats[tpl_name] = (alpha, beta)

        self._cache = stats
        self._cache_expires = now + self._cache_ttl
        return stats
