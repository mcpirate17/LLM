"""Temporal Bayesian tracker for op/template/motif success estimation.

Maintains Beta(α, β) posteriors per entity with:
- Exponential temporal decay (effective sample size shrinks over time)
- Code-fix detection (partial posterior reset when success rate jumps)
- Thompson sampling for exploration
- Soft score floors (never hard-blocks any op)

Usage:
    tracker = TemporalBayesianTracker.from_db(notebook_db_path)
    weights = tracker.op_weights(mode="thompson")
    template_weights = tracker.template_weights(mode="mean")
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .predictor_artifacts import read_json, write_json

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).parents[2] / "lab_notebook.db"

# Temporal decay: effective sample size shrinks by this factor per day
_DECAY_PER_DAY = 0.95
# Code-fix detection: if success rate jumps by more than this in 48h, reset
_FIX_DETECTION_THRESHOLD = 0.3
_FIX_WINDOW_HOURS = 48
# Post-fix: retain this fraction of historical evidence
_FIX_RETAIN_FRACTION = 0.3
# Weight range for grammar integration
_WEIGHT_MIN = 0.1
_WEIGHT_MAX = 8.0
# Prior: weakly informative (slightly pessimistic, ~33% expected success)
_DEFAULT_ALPHA = 1.5
_DEFAULT_BETA = 3.0


@dataclass(slots=True)
class BetaPosterior:
    """Beta distribution posterior for a single entity."""

    alpha: float = _DEFAULT_ALPHA
    beta: float = _DEFAULT_BETA
    last_updated: float = 0.0  # unix timestamp
    n_updates: int = 0
    # For code-fix detection
    prev_success_rate: Optional[float] = None
    prev_check_time: float = 0.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        ab = self.alpha + self.beta
        return self.alpha * self.beta / (ab * ab * (ab + 1))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def effective_n(self) -> float:
        """Effective sample size (α + β - prior)."""
        return self.alpha + self.beta - _DEFAULT_ALPHA - _DEFAULT_BETA

    def sample(self, rng: np.random.RandomState) -> float:
        """Thompson sampling: draw from posterior."""
        return float(rng.beta(max(self.alpha, 0.01), max(self.beta, 0.01)))

    def weight(
        self, mode: str = "mean", rng: Optional[np.random.RandomState] = None
    ) -> float:
        """Compute weight for grammar integration.

        Args:
            mode: 'mean' for posterior mean, 'thompson' for Thompson sampling,
                  'ucb' for upper confidence bound.
            rng: Required for 'thompson' mode.

        Returns:
            Weight in [_WEIGHT_MIN, _WEIGHT_MAX].
        """
        if mode == "mean":
            raw = self.mean
        elif mode == "thompson":
            if rng is None:
                rng = np.random.RandomState()
            raw = self.sample(rng)
        elif mode == "ucb":
            raw = min(1.0, self.mean + 2.0 * self.std)
        else:
            raw = self.mean

        # Map [0, 1] → [_WEIGHT_MIN, _WEIGHT_MAX] with contrast amplification
        # Center at ~0.3 (typical S1 rate), amplify deviations
        centered = (raw - 0.3) * 3.0 + 1.0
        return float(np.clip(centered, _WEIGHT_MIN, _WEIGHT_MAX))


@dataclass
class TemporalBayesianTracker:
    """Hierarchical Bayesian tracker for ops, templates, and motifs."""

    op_posteriors: Dict[str, BetaPosterior] = field(default_factory=dict)
    template_posteriors: Dict[str, BetaPosterior] = field(default_factory=dict)
    motif_posteriors: Dict[str, BetaPosterior] = field(default_factory=dict)
    _rng: np.random.RandomState = field(
        default_factory=lambda: np.random.RandomState(42)
    )
    _last_decay_time: float = 0.0

    def _get_or_create(
        self, store: Dict[str, BetaPosterior], key: str
    ) -> BetaPosterior:
        if key not in store:
            store[key] = BetaPosterior()
        return store[key]

    def update_op(self, op_name: str, success: bool, timestamp: float = 0.0) -> None:
        """Update a single op's posterior with a new observation."""
        p = self._get_or_create(self.op_posteriors, op_name)
        if success:
            p.alpha += 1.0
        else:
            p.beta += 1.0
        p.last_updated = timestamp or time.time()
        p.n_updates += 1

    def update_template(
        self, template_name: str, success: bool, timestamp: float = 0.0
    ) -> None:
        """Update a single template's posterior."""
        p = self._get_or_create(self.template_posteriors, template_name)
        if success:
            p.alpha += 1.0
        else:
            p.beta += 1.0
        p.last_updated = timestamp or time.time()
        p.n_updates += 1

    def update_motif(
        self, motif_name: str, success: bool, timestamp: float = 0.0
    ) -> None:
        """Update a single motif's posterior."""
        p = self._get_or_create(self.motif_posteriors, motif_name)
        if success:
            p.alpha += 1.0
        else:
            p.beta += 1.0
        p.last_updated = timestamp or time.time()
        p.n_updates += 1

    def apply_temporal_decay(self, now: Optional[float] = None) -> int:
        """Apply exponential decay to all posteriors based on elapsed time.

        Returns number of posteriors decayed.
        """
        now = now or time.time()
        if self._last_decay_time == 0.0:
            self._last_decay_time = now
            return 0

        elapsed_days = (now - self._last_decay_time) / 86400.0
        if elapsed_days < 0.5:
            return 0  # Skip if less than 12 hours since last decay

        decay = _DECAY_PER_DAY**elapsed_days
        n_decayed = 0

        for store in (
            self.op_posteriors,
            self.template_posteriors,
            self.motif_posteriors,
        ):
            for p in store.values():
                old_alpha, old_beta = p.alpha, p.beta
                p.alpha = _DEFAULT_ALPHA + decay * (p.alpha - _DEFAULT_ALPHA)
                p.beta = _DEFAULT_BETA + decay * (p.beta - _DEFAULT_BETA)
                if p.alpha != old_alpha or p.beta != old_beta:
                    n_decayed += 1

        self._last_decay_time = now
        logger.info(
            "Applied temporal decay: %.4f over %.1f days, %d posteriors decayed",
            decay,
            elapsed_days,
            n_decayed,
        )
        return n_decayed

    def detect_code_fixes(self, failure_signatures: List[Dict[str, Any]]) -> List[str]:
        """Detect ops/pairs that had recent code fixes based on failure_signature changes.

        A "fix" is detected when a signature's success rate increases by >0.3
        within the fix detection window. Each op is reset AT MOST ONCE per
        detection pass to prevent cascading resets (an op appearing in many
        signatures would otherwise get reset N times, destroying its posterior).

        Args:
            failure_signatures: List of dicts with keys:
                signature, n_failures, n_successes, last_updated

        Returns:
            List of entity names that were reset due to detected fixes.
        """
        now = time.time()
        window_seconds = _FIX_WINDOW_HOURS * 3600
        reset_entities: List[str] = []

        # Aggregate: for each op, find the signature with the largest positive
        # success-rate jump. Only use that single best signal for reset.
        op_best_delta: Dict[str, float] = {}  # op -> max delta seen
        op_best_sig: Dict[str, str] = {}  # op -> signature that caused it
        op_best_rates: Dict[str, tuple] = {}  # op -> (prev_rate, current_rate)

        for sig in failure_signatures:
            signature = sig.get("signature", "")
            n_fail = sig.get("n_failures", 0)
            n_succ = sig.get("n_successes", 0)
            last_updated = sig.get("last_updated", 0)

            total = n_fail + n_succ
            if total < 10:  # need minimum evidence
                continue

            current_rate = n_succ / total
            age = now - last_updated

            if age > window_seconds:
                continue

            parts = signature.split("->")
            for op_name in parts:
                op_name = op_name.strip()
                if not op_name or op_name not in self.op_posteriors:
                    continue

                p = self.op_posteriors[op_name]
                if p.prev_success_rate is None:
                    p.prev_success_rate = current_rate
                    continue

                delta = current_rate - p.prev_success_rate
                if delta > _FIX_DETECTION_THRESHOLD:
                    if op_name not in op_best_delta or delta > op_best_delta[op_name]:
                        op_best_delta[op_name] = delta
                        op_best_sig[op_name] = signature
                        op_best_rates[op_name] = (p.prev_success_rate, current_rate)

        # Apply at most ONE reset per op, using the strongest signal
        for op_name, delta in op_best_delta.items():
            p = self.op_posteriors[op_name]
            prev_rate, current_rate = op_best_rates[op_name]

            old_alpha, old_beta = p.alpha, p.beta
            p.alpha = _DEFAULT_ALPHA + _FIX_RETAIN_FRACTION * (p.alpha - _DEFAULT_ALPHA)
            p.beta = _DEFAULT_BETA + _FIX_RETAIN_FRACTION * (p.beta - _DEFAULT_BETA)
            logger.info(
                "Code fix detected for '%s' (via %s): rate %.2f → %.2f, "
                "posterior reset α=%.1f→%.1f β=%.1f→%.1f",
                op_name,
                op_best_sig[op_name],
                prev_rate,
                current_rate,
                old_alpha,
                p.alpha,
                old_beta,
                p.beta,
            )
            reset_entities.append(op_name)

        # Update prev_success_rate for all ops seen (even if not reset)
        for sig in failure_signatures:
            signature = sig.get("signature", "")
            n_fail = sig.get("n_failures", 0)
            n_succ = sig.get("n_successes", 0)
            total = n_fail + n_succ
            if total == 0:
                continue
            current_rate = n_succ / total
            for op_name in (part.strip() for part in signature.split("->")):
                if op_name in self.op_posteriors:
                    self.op_posteriors[op_name].prev_success_rate = current_rate
                    self.op_posteriors[op_name].prev_check_time = now

        return reset_entities

    def op_weights(self, mode: str = "mean") -> Dict[str, float]:
        """Get op weights for grammar integration.

        Args:
            mode: 'mean', 'thompson', or 'ucb'.

        Returns:
            Dict of op_name → weight in [0.1, 8.0].
        """
        rng = self._rng if mode == "thompson" else None
        return {
            name: p.weight(mode=mode, rng=rng) for name, p in self.op_posteriors.items()
        }

    def template_weights(self, mode: str = "mean") -> Dict[str, float]:
        """Get template weights for grammar integration."""
        rng = self._rng if mode == "thompson" else None
        return {
            name: p.weight(mode=mode, rng=rng)
            for name, p in self.template_posteriors.items()
        }

    def motif_weights(self, mode: str = "mean") -> Dict[str, float]:
        """Get motif weights for grammar integration."""
        rng = self._rng if mode == "thompson" else None
        return {
            name: p.weight(mode=mode, rng=rng)
            for name, p in self.motif_posteriors.items()
        }

    def diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about the tracker state."""

        def _store_stats(store: Dict[str, BetaPosterior]) -> Dict[str, Any]:
            if not store:
                return {"n": 0}
            means = [p.mean for p in store.values()]
            eff_ns = [p.effective_n for p in store.values()]
            return {
                "n": len(store),
                "mean_success_rate": float(np.mean(means)),
                "std_success_rate": float(np.std(means)),
                "mean_effective_n": float(np.mean(eff_ns)),
                "max_effective_n": float(max(eff_ns)),
                "n_low_confidence": sum(1 for n in eff_ns if n < 5),
            }

        return {
            "ops": _store_stats(self.op_posteriors),
            "templates": _store_stats(self.template_posteriors),
            "motifs": _store_stats(self.motif_posteriors),
            "last_decay_time": self._last_decay_time,
        }

    @classmethod
    def from_db(
        cls,
        db_path: Path = _DEFAULT_DB,
        apply_decay: bool = True,
        detect_fixes: bool = True,
    ) -> "TemporalBayesianTracker":
        """Build tracker from historical experiment data in lab_notebook.db.

        Replays all experiment results to build posteriors, then applies
        temporal decay and code-fix detection.
        """
        tracker = cls()
        db_path = Path(db_path)

        if not db_path.exists():
            logger.warning("Database not found: %s", db_path)
            return tracker

        try:
            from ..notebook.shared_conn import get_notebook_conn

            conn = get_notebook_conn(str(db_path))
        except (sqlite3.Error, OSError) as e:
            logger.error("Failed to connect to DB: %s", e)
            return tracker

        # ── Load op success data ──
        try:
            rows = conn.execute(
                "SELECT op_name, s0_pass_count, s1_pass_count, eval_count, last_updated "
                "FROM op_stats"
            ).fetchall()
            for op_name, s0, s1, n_eval, ts in rows:
                p = tracker._get_or_create(tracker.op_posteriors, op_name)
                # s1_pass_count are successes, (eval_count - s1_pass_count) are failures
                p.alpha = _DEFAULT_ALPHA + float(s1 or 0)
                p.beta = _DEFAULT_BETA + float((n_eval or 0) - (s1 or 0))
                p.last_updated = float(ts or 0)
                p.n_updates = int(n_eval or 0)
            logger.info("Loaded %d op posteriors from op_stats", len(rows))
        except Exception as e:
            logger.warning("Failed to load op_stats: %s", e)

        # ── Load template success data ──
        try:
            rows = conn.execute(
                "SELECT template_name, s0_pass_count, s1_pass_count, eval_count, last_updated "
                "FROM template_stats"
            ).fetchall()
            for tpl_name, s0, s1, n_eval, ts in rows:
                p = tracker._get_or_create(tracker.template_posteriors, tpl_name)
                p.alpha = _DEFAULT_ALPHA + float(s1 or 0)
                p.beta = _DEFAULT_BETA + float((n_eval or 0) - (s1 or 0))
                p.last_updated = float(ts or 0)
                p.n_updates = int(n_eval or 0)
            logger.info("Loaded %d template posteriors from template_stats", len(rows))
        except Exception as e:
            logger.warning("Failed to load template_stats: %s", e)

        # ── Load motif success data ──
        try:
            rows = conn.execute(
                "SELECT motif_name, s0_pass_count, s1_pass_count, eval_count, last_updated "
                "FROM motif_stats"
            ).fetchall()
            for motif_name, s0, s1, n_eval, ts in rows:
                p = tracker._get_or_create(tracker.motif_posteriors, motif_name)
                p.alpha = _DEFAULT_ALPHA + float(s1 or 0)
                p.beta = _DEFAULT_BETA + float((n_eval or 0) - (s1 or 0))
                p.last_updated = float(ts or 0)
                p.n_updates = int(n_eval or 0)
            logger.info("Loaded %d motif posteriors from motif_stats", len(rows))
        except Exception as e:
            logger.warning("Failed to load motif_stats: %s", e)

        # ── Code-fix detection ──
        if detect_fixes:
            try:
                rows = conn.execute(
                    "SELECT signature, n_failures, n_successes, error_types, last_updated "
                    "FROM failure_signatures"
                ).fetchall()
                signatures = [
                    {
                        "signature": r[0],
                        "n_failures": r[1],
                        "n_successes": r[2],
                        "error_types": r[3],
                        "last_updated": r[4],
                    }
                    for r in rows
                ]
                resets = tracker.detect_code_fixes(signatures)
                if resets:
                    logger.info("Code-fix resets applied to: %s", resets)
            except Exception as e:
                logger.warning("Failed to detect code fixes: %s", e)

        # ── Apply temporal decay ──
        if apply_decay:
            tracker._last_decay_time = (
                time.time() - 86400
            )  # pretend last decay was 1 day ago
            tracker.apply_temporal_decay()

        return tracker

    def save_state(self, path: Path) -> None:
        """Serialize tracker state to JSON for persistence across sessions."""
        state = {
            "version": 1,
            "timestamp": time.time(),
            "last_decay_time": self._last_decay_time,
            "op_posteriors": {
                name: {
                    "alpha": p.alpha,
                    "beta": p.beta,
                    "last_updated": p.last_updated,
                    "n_updates": p.n_updates,
                    "prev_success_rate": p.prev_success_rate,
                }
                for name, p in self.op_posteriors.items()
            },
            "template_posteriors": {
                name: {
                    "alpha": p.alpha,
                    "beta": p.beta,
                    "last_updated": p.last_updated,
                    "n_updates": p.n_updates,
                }
                for name, p in self.template_posteriors.items()
            },
            "motif_posteriors": {
                name: {
                    "alpha": p.alpha,
                    "beta": p.beta,
                    "last_updated": p.last_updated,
                    "n_updates": p.n_updates,
                }
                for name, p in self.motif_posteriors.items()
            },
        }
        path = Path(path)
        write_json(path, state)
        logger.info("Saved tracker state to %s", path)

    @classmethod
    def load_state(cls, path: Path) -> "TemporalBayesianTracker":
        """Load tracker state from JSON."""
        path = Path(path)
        state = read_json(path)

        tracker = cls()
        tracker._last_decay_time = state.get("last_decay_time", 0.0)

        for name, data in state.get("op_posteriors", {}).items():
            p = BetaPosterior(
                alpha=data["alpha"],
                beta=data["beta"],
                last_updated=data.get("last_updated", 0),
                n_updates=data.get("n_updates", 0),
                prev_success_rate=data.get("prev_success_rate"),
            )
            tracker.op_posteriors[name] = p

        for name, data in state.get("template_posteriors", {}).items():
            p = BetaPosterior(
                alpha=data["alpha"],
                beta=data["beta"],
                last_updated=data.get("last_updated", 0),
                n_updates=data.get("n_updates", 0),
            )
            tracker.template_posteriors[name] = p

        for name, data in state.get("motif_posteriors", {}).items():
            p = BetaPosterior(
                alpha=data["alpha"],
                beta=data["beta"],
                last_updated=data.get("last_updated", 0),
                n_updates=data.get("n_updates", 0),
            )
            tracker.motif_posteriors[name] = p

        logger.info(
            "Loaded tracker: %d ops, %d templates, %d motifs",
            len(tracker.op_posteriors),
            len(tracker.template_posteriors),
            len(tracker.motif_posteriors),
        )
        return tracker
