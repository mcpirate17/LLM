"""External Scaling Law Comparison System.

Compares candidate architectures against GPT-2/Mamba scaling curves to
determine if they achieve genuine 3-5x parameter efficiency improvements.

Two-stage approach:
  1. Published curves (Kaplan et al. 2020) as cheap plausibility filter
  2. Locally-trained reference models on same data for apples-to-apples comparison

Supports multi-scale evaluation at d=256 and d=512 for scaling slope analysis.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


from research.defaults import VOCAB_SIZE
from ._reference_cache import average_finite_reference_runs, open_sqlite_cache
from .reference_training import train_reference_transformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ScalingCurvePoint:
    """A single (param_count, loss) measurement."""

    param_count: int
    loss: float  # cross-entropy (nats)
    dataset: str  # "webtext", "pile", "local", "random"
    source: str  # "kaplan2020", "gu2023", "local_train"


@dataclass(slots=True)
class ScalingCurve:
    """A model family's loss-vs-params scaling behavior.

    Fits power law: L(N) = A * N^(-alpha)
    """

    family: str
    points: List[ScalingCurvePoint] = field(default_factory=list)
    A: float = 0.0
    alpha: float = 0.0
    fit_r2: float = 0.0

    def loss_at_params(self, n_params: int) -> float:
        """Interpolate/extrapolate loss for given param count."""
        if self.A > 0 and self.alpha > 0 and n_params > 0:
            return self.A * (n_params ** (-self.alpha))
        # Fallback: log-log linear interpolation
        return self._interp_log_log(n_params)

    def params_for_loss(self, target_loss: float) -> int:
        """Inverse: how many params needed to achieve target loss."""
        if self.A > 0 and self.alpha > 0 and target_loss > 0:
            # L = A * N^(-alpha)  =>  N = (L / A)^(-1/alpha)
            ratio = target_loss / self.A
            if ratio > 0:
                return max(1, int(ratio ** (-1.0 / self.alpha)))
        # Fallback
        return self._interp_inverse_log_log(target_loss)

    def _interp_log_log(self, n_params: int) -> float:
        """Log-log linear interpolation between points."""
        if not self.points or n_params <= 0:
            return float("inf")
        pts = sorted(self.points, key=lambda p: p.param_count)
        log_n = math.log(n_params)
        log_ns = [math.log(p.param_count) for p in pts]
        log_ls = [math.log(max(p.loss, 1e-10)) for p in pts]
        # Clamp to range
        if log_n <= log_ns[0]:
            return pts[0].loss
        if log_n >= log_ns[-1]:
            # Extrapolate from last two points
            if len(pts) >= 2:
                slope = (log_ls[-1] - log_ls[-2]) / max(log_ns[-1] - log_ns[-2], 1e-10)
                return math.exp(log_ls[-1] + slope * (log_n - log_ns[-1]))
            return pts[-1].loss
        # Linear interpolation in log-log space
        for i in range(len(pts) - 1):
            if log_ns[i] <= log_n <= log_ns[i + 1]:
                t = (log_n - log_ns[i]) / max(log_ns[i + 1] - log_ns[i], 1e-10)
                log_loss = log_ls[i] + t * (log_ls[i + 1] - log_ls[i])
                return math.exp(log_loss)
        return pts[-1].loss

    def _interp_inverse_log_log(self, target_loss: float) -> int:
        """Inverse interpolation: find params for target loss."""
        if not self.points or target_loss <= 0:
            return 1
        pts = sorted(self.points, key=lambda p: p.param_count)
        log_target = math.log(target_loss)
        log_ns = [math.log(p.param_count) for p in pts]
        log_ls = [math.log(max(p.loss, 1e-10)) for p in pts]
        # Losses decrease with params, so log_ls is decreasing
        for i in range(len(pts) - 1):
            if log_ls[i] >= log_target >= log_ls[i + 1]:
                t = (log_target - log_ls[i]) / max(log_ls[i + 1] - log_ls[i], 1e-10)
                log_n = log_ns[i] + t * (log_ns[i + 1] - log_ns[i])
                return max(1, int(math.exp(log_n)))
        # Extrapolate
        if log_target > log_ls[0]:
            return max(1, pts[0].param_count // 2)
        if len(pts) >= 2:
            slope = (log_ns[-1] - log_ns[-2]) / max(log_ls[-1] - log_ls[-2], 1e-10)
            log_n = log_ns[-1] + slope * (log_target - log_ls[-1])
            return max(1, int(math.exp(log_n)))
        return pts[-1].param_count * 2


@dataclass(slots=True)
class FamilyComparison:
    """Comparison result for a single reference family."""

    family: str
    reference_loss_at_candidate_params: float
    reference_params_for_candidate_loss: int
    param_efficiency_ratio: float  # ref_params / candidate_params
    flop_efficiency_ratio: float  # ref_flops / candidate_flops
    curve_source: str  # "published" | "local"

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "reference_loss_at_candidate_params": round(
                self.reference_loss_at_candidate_params, 6
            ),
            "reference_params_for_candidate_loss": self.reference_params_for_candidate_loss,
            "param_efficiency_ratio": round(self.param_efficiency_ratio, 4),
            "flop_efficiency_ratio": round(self.flop_efficiency_ratio, 4),
            "curve_source": self.curve_source,
        }


@dataclass(slots=True)
class ScalingComparisonResult:
    """Full scaling comparison output."""

    family_comparisons: Dict[str, FamilyComparison] = field(default_factory=dict)
    best_param_efficiency: float = 0.0
    best_param_efficiency_family: str = ""
    flop_efficiency: float = 0.0
    flop_gate_passed: bool = False
    scaling_gate_passed: bool = False
    data_quality: str = "random"
    confidence: str = "published_only"
    d512_result: Optional[Dict] = None

    def to_dict(self) -> dict:
        return {
            "family_comparisons": {
                k: v.to_dict() for k, v in self.family_comparisons.items()
            },
            "best_param_efficiency": round(self.best_param_efficiency, 4),
            "best_param_efficiency_family": self.best_param_efficiency_family,
            "flop_efficiency": round(self.flop_efficiency, 4),
            "flop_gate_passed": self.flop_gate_passed,
            "scaling_gate_passed": self.scaling_gate_passed,
            "data_quality": self.data_quality,
            "confidence": self.confidence,
            "d512_result": self.d512_result,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScalingComparisonResult:
        comparisons = {}
        for k, v in d.get("family_comparisons", {}).items():
            comparisons[k] = FamilyComparison(**v)
        return cls(
            family_comparisons=comparisons,
            best_param_efficiency=d.get("best_param_efficiency", 0.0),
            best_param_efficiency_family=d.get("best_param_efficiency_family", ""),
            flop_efficiency=d.get("flop_efficiency", 0.0),
            flop_gate_passed=d.get("flop_gate_passed", False),
            scaling_gate_passed=d.get("scaling_gate_passed", False),
            data_quality=d.get("data_quality", "random"),
            confidence=d.get("confidence", "published_only"),
            d512_result=d.get("d512_result"),
        )


# ---------------------------------------------------------------------------
# Published scaling curves
# ---------------------------------------------------------------------------


def _build_published_curves() -> Dict[str, ScalingCurve]:
    """Hardcoded reference data from published scaling law papers.

    These are dataset-specific (WebText/Pile) so absolute losses don't
    transfer to our training data.  Used as plausibility filters only.

    Sources:
      - Kaplan et al. 2020: "Scaling Laws for Neural Language Models"
        L(N) ≈ 5.3 * N^(-0.076) for transformer LMs on WebText
      - Gu & Dao 2023: "Mamba: Linear-Time Sequence Modeling with
        Selective State Spaces" — roughly 2x param-efficient vs transformer
    """
    gpt2_curve = ScalingCurve(
        family="gpt2",
        A=11.94,  # fit from data points below
        alpha=0.0696,
        fit_r2=0.98,
        points=[
            ScalingCurvePoint(768_000, 4.80, "webtext", "kaplan2020"),
            ScalingCurvePoint(3_000_000, 4.20, "webtext", "kaplan2020"),
            ScalingCurvePoint(6_000_000, 3.95, "webtext", "kaplan2020"),
            ScalingCurvePoint(13_000_000, 3.75, "webtext", "kaplan2020"),
            ScalingCurvePoint(42_000_000, 3.50, "webtext", "kaplan2020"),
            ScalingCurvePoint(117_000_000, 3.29, "webtext", "kaplan2020"),
            ScalingCurvePoint(345_000_000, 3.10, "webtext", "kaplan2020"),
        ],
    )

    # Mamba: approximately 2x parameter efficiency over transformer.
    # Similar scaling exponent, slightly steeper.
    mamba_curve = ScalingCurve(
        family="mamba",
        A=12.0,  # fit from data points below
        alpha=0.0741,
        fit_r2=0.998,
        points=[
            ScalingCurvePoint(3_000_000, 4.00, "pile", "gu2023"),
            ScalingCurvePoint(13_000_000, 3.55, "pile", "gu2023"),
            ScalingCurvePoint(42_000_000, 3.25, "pile", "gu2023"),
            ScalingCurvePoint(130_000_000, 3.00, "pile", "gu2023"),
            ScalingCurvePoint(370_000_000, 2.80, "pile", "gu2023"),
        ],
    )

    return {"gpt2": gpt2_curve, "mamba": mamba_curve}


PUBLISHED_CURVES = _build_published_curves()


# ---------------------------------------------------------------------------
# Power law fitting
# ---------------------------------------------------------------------------


def fit_power_law(
    params: Sequence[int], losses: Sequence[float]
) -> Tuple[float, float, float]:
    """Fit L(N) = A * N^(-alpha) via log-log linear regression.

    Returns (A, alpha, r_squared).
    """
    import numpy as np

    if len(params) < 2 or len(losses) < 2:
        return 0.0, 0.0, 0.0

    log_ns = np.log(np.maximum(params, 1).astype(np.float64))
    log_ls = np.log(np.maximum(losses, 1e-10).astype(np.float64))

    # log(L) = log(A) - alpha * log(N)
    coeffs = np.polynomial.polynomial.polyfit(log_ns, log_ls, 1)
    intercept, slope = coeffs[0], coeffs[1]
    alpha = -slope
    A = math.exp(intercept)

    # R² computed directly
    predicted = intercept + slope * log_ns
    ss_res = float(np.sum((log_ls - predicted) ** 2))
    ss_tot = float(np.sum((log_ls - np.mean(log_ls)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15) if ss_tot > 1e-15 else 0.0

    return A, alpha, r2


# ---------------------------------------------------------------------------
# Graph rescaling for multi-scale evaluation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ScalingReferenceManager
# ---------------------------------------------------------------------------


class ScalingReferenceManager:
    """Trains and caches reference models, computes scaling comparisons.

    Mirrors the TransformerBaseline pattern from eval/baseline.py but
    supports multiple reference families and curve fitting.
    """

    def __init__(self, cache_path: str = "research/scaling_reference_cache.db"):
        self.cache_path, self._conn = open_sqlite_cache(
            cache_path,
            schema_statements=(
                """
                CREATE TABLE IF NOT EXISTS reference_results (
                    config_key TEXT PRIMARY KEY,
                    family TEXT NOT NULL,
                    d_model INTEGER NOT NULL,
                    n_layers INTEGER NOT NULL,
                    param_count INTEGER NOT NULL,
                    final_loss REAL NOT NULL,
                    initial_loss REAL NOT NULL,
                    n_steps INTEGER NOT NULL,
                    seq_len INTEGER NOT NULL,
                    data_tag TEXT NOT NULL,
                    trained_at REAL NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS fitted_curves (
                    curve_key TEXT PRIMARY KEY,
                    family TEXT NOT NULL,
                    d_model INTEGER NOT NULL,
                    data_tag TEXT NOT NULL,
                    A REAL NOT NULL,
                    alpha REAL NOT NULL,
                    fit_r2 REAL NOT NULL,
                    n_points INTEGER NOT NULL,
                    fitted_at REAL NOT NULL
                )
                """,
            ),
            wal=True,
        )
        self._published = PUBLISHED_CURVES

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Cache helpers ──

    def _config_key(
        self,
        family: str,
        d_model: int,
        n_layers: int,
        seq_len: int,
        n_steps: int,
        vocab_size: int,
        data_tag: str,
    ) -> str:
        return (
            f"{family}_{d_model}_{n_layers}_{seq_len}_{n_steps}_{vocab_size}_{data_tag}"
        )

    def _curve_key(
        self, family: str, d_model: int, n_steps: int, seq_len: int, data_tag: str
    ) -> str:
        return f"{family}_{d_model}_{n_steps}_{seq_len}_{data_tag}"

    def _get_cached_loss(self, config_key: str) -> Optional[float]:
        row = self._conn.execute(
            "SELECT final_loss FROM reference_results WHERE config_key = ?",
            (config_key,),
        ).fetchone()
        return row[0] if row else None

    def _save_result(
        self,
        config_key: str,
        family: str,
        d_model: int,
        n_layers: int,
        param_count: int,
        final_loss: float,
        initial_loss: float,
        n_steps: int,
        seq_len: int,
        data_tag: str,
    ):
        self._conn.execute(
            """
            INSERT OR REPLACE INTO reference_results
            (config_key, family, d_model, n_layers, param_count,
             final_loss, initial_loss, n_steps, seq_len, data_tag, trained_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                config_key,
                family,
                d_model,
                n_layers,
                param_count,
                final_loss,
                initial_loss,
                n_steps,
                seq_len,
                data_tag,
                time.time(),
            ),
        )
        self._conn.commit()

    def _save_curve(
        self,
        curve_key: str,
        family: str,
        d_model: int,
        data_tag: str,
        A: float,
        alpha: float,
        fit_r2: float,
        n_points: int,
    ):
        self._conn.execute(
            """
            INSERT OR REPLACE INTO fitted_curves
            (curve_key, family, d_model, data_tag, A, alpha, fit_r2, n_points, fitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                curve_key,
                family,
                d_model,
                data_tag,
                A,
                alpha,
                fit_r2,
                n_points,
                time.time(),
            ),
        )
        self._conn.commit()

    def _get_cached_curve(self, curve_key: str, family: str) -> Optional[ScalingCurve]:
        """Return a cached ScalingCurve if one exists with a valid fit."""
        row = self._conn.execute(
            "SELECT A, alpha, fit_r2, n_points FROM fitted_curves WHERE curve_key = ?",
            (curve_key,),
        ).fetchone()
        if row is None or row[2] <= 0:  # no fit or bad R²
            return None
        A, alpha, fit_r2, n_points = row
        if n_points < 2:
            return None
        logger.debug(
            "Curve cache hit: %s A=%.3f alpha=%.4f R²=%.3f (%d points)",
            curve_key,
            A,
            alpha,
            fit_r2,
            n_points,
        )
        return ScalingCurve(family=family, A=A, alpha=alpha, fit_r2=fit_r2)

    # ── Reference model training ──

    def _train_reference(
        self,
        family: str,
        d_model: int,
        n_layers: int,
        n_steps: int,
        seq_len: int,
        vocab_size: int,
        batch_size: int,
        lr: float,
        device: str,
        data_fn: Optional[Callable] = None,
        data_tag: str = "random",
        n_seeds: int = 3,
        cacheable: bool = False,
    ) -> Tuple[float, int]:
        """Train a reference model and return (final_loss, param_count).

        Averages over n_seeds for stability (same pattern as baseline.py).
        When cacheable=True, results are cached even with real data (data_fn).
        """
        config_key = self._config_key(
            family, d_model, n_layers, seq_len, n_steps, vocab_size, data_tag
        )

        # Check cache — skip only for non-cacheable real data (e.g. hydra)
        if data_fn is None or cacheable:
            cached = self._get_cached_loss(config_key)
            if cached is not None:
                row = self._conn.execute(
                    "SELECT param_count FROM reference_results WHERE config_key = ?",
                    (config_key,),
                ).fetchone()
                return cached, row[0] if row else 0
        avg_loss, losses, param_count = average_finite_reference_runs(
            n_seeds,
            lambda seed: train_reference_transformer(
                d_model=d_model,
                seq_len=seq_len,
                n_steps=n_steps,
                vocab_size=vocab_size,
                batch_size=batch_size,
                lr=lr,
                device=device,
                n_layers=n_layers,
                optimizer_name="adamw",
                weight_decay=0.01,
                seed=seed,
                data_fn=data_fn,
            ),
        )
        param_count = int(param_count or 0)

        if math.isfinite(avg_loss):
            self._save_result(
                config_key,
                family,
                d_model,
                n_layers,
                param_count,
                avg_loss,
                losses[0] if losses else float("inf"),
                n_steps,
                seq_len,
                data_tag,
            )

        return avg_loss, param_count

    # ── Scaling curve construction ──

    def build_local_scaling_curve(
        self,
        family: str,
        d_model: int,
        n_steps: int,
        seq_len: int,
        vocab_size: int,
        batch_size: int,
        lr: float,
        device: str,
        data_fn: Optional[Callable] = None,
        data_tag: str = "random",
        layer_counts: Sequence[int] = (2, 4, 6, 8),
        cacheable: bool = False,
    ) -> ScalingCurve:
        """Train reference at multiple layer counts and fit scaling curve."""
        curve_key = self._curve_key(family, d_model, n_steps, seq_len, data_tag)

        # Check curve cache — skip training entirely if fitted curve exists
        if data_fn is None or cacheable:
            cached_curve = self._get_cached_curve(curve_key, family)
            if cached_curve is not None:
                return cached_curve

        points = []
        param_counts = []
        final_losses = []
        random_chance = math.log(max(vocab_size, 2))

        for n_layers in layer_counts:
            loss, params = self._train_reference(
                family,
                d_model,
                n_layers,
                n_steps,
                seq_len,
                vocab_size,
                batch_size,
                lr,
                device,
                data_fn=data_fn,
                data_tag=data_tag,
                cacheable=cacheable,
            )
            # Include any point where training produced finite loss.
            # On random data, models barely beat ln(vocab) but there IS
            # differentiation between sizes that's meaningful for curve fitting.
            if math.isfinite(loss) and loss < random_chance * 1.05:
                points.append(ScalingCurvePoint(params, loss, "local", "local_train"))
                param_counts.append(params)
                final_losses.append(loss)
                logger.debug(
                    "Reference %s d=%d L=%d: params=%d loss=%.4f",
                    family,
                    d_model,
                    n_layers,
                    params,
                    loss,
                )

        if len(param_counts) < 2:
            logger.warning(
                "Insufficient reference points for %s d=%d (%d/%d learned)",
                family,
                d_model,
                len(param_counts),
                len(layer_counts),
            )
            # Return curve with just points, no fit
            return ScalingCurve(family=family, points=points)

        A, alpha, r2 = fit_power_law(param_counts, final_losses)
        curve = ScalingCurve(family=family, points=points, A=A, alpha=alpha, fit_r2=r2)

        self._save_curve(
            curve_key, family, d_model, data_tag, A, alpha, r2, len(points)
        )

        logger.info(
            "Fitted %s scaling curve d=%d: L(N)=%.3f*N^(-%.4f) R²=%.3f (%d points)",
            family,
            d_model,
            A,
            alpha,
            r2,
            len(points),
        )

        return curve

    # ── Candidate comparison ──

    def compare_candidate(
        self,
        candidate_loss: float,
        candidate_params: int,
        candidate_flops: int,
        d_model: int,
        n_steps: int,
        seq_len: int,
        vocab_size: int = VOCAB_SIZE,
        batch_size: int = 4,
        lr: float = 3e-4,
        device: str = "cuda",
        data_fn: Optional[Callable] = None,
        data_tag: str = "random",
        families: Sequence[str] = ("gpt2",),
        param_efficiency_target: float = 3.0,
        flop_ceiling: float = 2.0,
        cacheable: bool = False,
    ) -> ScalingComparisonResult:
        """Compare candidate against reference scaling curves.

        Args:
            candidate_loss: Best validation loss achieved by candidate.
            candidate_params: Total parameter count of candidate.
            candidate_flops: Forward-pass FLOPs of candidate.
            d_model: Model dimension used for training.
            n_steps: Training steps used.
            families: Reference families to compare against.
            param_efficiency_target: Minimum param efficiency for gate pass.
            flop_ceiling: Maximum allowed FLOP ratio (candidate/reference).

        Returns:
            ScalingComparisonResult with per-family comparisons and gate verdict.
        """
        result = ScalingComparisonResult(
            data_quality=data_tag if data_tag != "random" else "random",
        )

        if not math.isfinite(candidate_loss) or candidate_loss <= 0:
            return result
        if candidate_params <= 0:
            return result

        best_param_eff = 0.0
        best_family = ""
        best_flop_eff = 0.0

        for family_name in families:
            family_name = family_name.strip()
            if not family_name:
                continue

            try:
                comparison = self._compare_single_family(
                    family_name,
                    candidate_loss,
                    candidate_params,
                    candidate_flops,
                    d_model,
                    n_steps,
                    seq_len,
                    vocab_size,
                    batch_size,
                    lr,
                    device,
                    data_fn,
                    data_tag,
                    cacheable=cacheable,
                )
            except Exception as e:
                logger.debug("Family %s comparison failed: %s", family_name, e)
                continue

            if comparison is None:
                continue

            result.family_comparisons[family_name] = comparison

            if comparison.param_efficiency_ratio > best_param_eff:
                best_param_eff = comparison.param_efficiency_ratio
                best_family = family_name
                best_flop_eff = comparison.flop_efficiency_ratio

        result.best_param_efficiency = best_param_eff
        result.best_param_efficiency_family = best_family
        result.flop_efficiency = best_flop_eff
        result.flop_gate_passed = best_flop_eff >= (1.0 / flop_ceiling)
        result.scaling_gate_passed = (
            best_param_eff >= param_efficiency_target and result.flop_gate_passed
        )
        result.confidence = (
            "local_reference"
            if data_fn is not None or data_tag != "random"
            else "random_data"
        )

        return result

    def _compare_single_family(
        self,
        family: str,
        candidate_loss: float,
        candidate_params: int,
        candidate_flops: int,
        d_model: int,
        n_steps: int,
        seq_len: int,
        vocab_size: int,
        batch_size: int,
        lr: float,
        device: str,
        data_fn: Optional[Callable],
        data_tag: str,
        cacheable: bool = False,
    ) -> Optional[FamilyComparison]:
        """Compare candidate against one reference family."""
        # Build local scaling curve (trains references if needed)
        curve = self.build_local_scaling_curve(
            family,
            d_model,
            n_steps,
            seq_len,
            vocab_size,
            batch_size,
            lr,
            device,
            data_fn=data_fn,
            data_tag=data_tag,
            cacheable=cacheable,
        )

        if not curve.points and curve.A <= 0:
            # No usable curve — fall back to published
            if family in self._published:
                curve = self._published[family]
                curve_source = "published"
            else:
                return None
        else:
            curve_source = "local"

        # What loss does reference achieve at candidate's param count?
        ref_loss_at_candidate_params = curve.loss_at_params(candidate_params)

        # How many params does reference need for candidate's loss?
        ref_params_for_loss = curve.params_for_loss(candidate_loss)

        # Parameter efficiency: how many times more params does reference need?
        param_eff = ref_params_for_loss / max(candidate_params, 1)

        # FLOP efficiency: compare FLOPs needed
        # Standard transformer: ~2 * params FLOPs per token (forward pass)
        ref_flops_for_loss = 2 * ref_params_for_loss
        flop_eff = ref_flops_for_loss / max(candidate_flops, 1)

        return FamilyComparison(
            family=family,
            reference_loss_at_candidate_params=ref_loss_at_candidate_params,
            reference_params_for_candidate_loss=ref_params_for_loss,
            param_efficiency_ratio=param_eff,
            flop_efficiency_ratio=flop_eff,
            curve_source=curve_source,
        )
