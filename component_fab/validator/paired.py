"""Paired candidate-vs-anchor comparison for noise-robust promotion (WS-2).

The single-seed ``learned_signal`` (loss-ratio over an *absolute* threshold) in
``in_context.py`` promotes on noise: one lucky seed clears the bar. This module
replaces that with a **paired** comparison — for each seed the candidate AND its
anchor baseline train on the *same* seed/data, and we test whether the mean
per-seed delta (candidate − anchor) is significantly positive (a t-interval that
excludes zero). ``policies/promotion.py`` consumes the CI: a candidate only
promotes when its advantage over the anchor is real, not a seed artifact.

Pure-stat helpers (``paired_delta_ci``) are dependency-free and cheap to test;
the compute helper (``run_paired_probe``) reuses ``short_training_probe`` exactly
as ``in_context.py`` does, so the two stay metric-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any, Callable, Hashable, Sequence

import torch
from scipy.stats import t as _student_t
from torch import nn

from ..harness.probe_block import ProbeResult, short_training_probe
from ..harness.probe_tasks import DEFAULT_PROBE_TASKS, ProbeTask

if TYPE_CHECKING:
    from ..proposer.spec_generator import ProposalSpec

LaneFactory = Callable[[], nn.Module]


def _t_critical(df: int, confidence: float) -> float:
    """Two-sided Student-t critical value — exact, small-n is where it matters."""
    if df < 1:
        raise ValueError(f"degrees of freedom must be >= 1, got {df}")
    return float(_student_t.ppf((1.0 + confidence) / 2.0, df))


@dataclass(frozen=True, slots=True)
class PairedDeltaCI:
    n: int
    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    excludes_zero: bool  # True only when the whole CI is strictly above zero

    def to_metadata(self) -> dict[str, float | int | bool]:
        """Flat keys that ride the grade record into ledger metadata_history."""
        return {
            "paired_delta_n": self.n,
            "paired_delta_mean": round(self.mean, 6),
            "paired_delta_ci_low": round(self.ci_low, 6),
            "paired_delta_ci_high": round(self.ci_high, 6),
            "paired_delta_ci_excludes_zero": self.excludes_zero,
        }


def paired_delta_ci(
    deltas: Sequence[float], *, confidence: float = 0.95
) -> PairedDeltaCI:
    """Two-sided t-interval for the mean of paired deltas.

    With < 2 samples the CI is undefined → returned wide and *not* excluding
    zero, so a candidate can never promote on a single seed.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    n = len(deltas)
    m = mean(deltas) if n else 0.0
    if n < 2:
        return PairedDeltaCI(
            n=n,
            mean=m,
            ci_low=float("-inf"),
            ci_high=float("inf"),
            confidence=confidence,
            excludes_zero=False,
        )
    sd = stdev(deltas)
    half = _t_critical(n - 1, confidence) * sd / (n**0.5)
    low, high = m - half, m + half
    return PairedDeltaCI(
        n=n,
        mean=m,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        excludes_zero=low > 0.0,
    )


def _suite_metric(
    lane: nn.Module,
    tasks: Sequence[ProbeTask],
    *,
    dim: int,
    seq_len: int,
    n_steps: int,
    seed: int,
) -> float:
    """Mean loss-ratio across the task suite for one lane at one seed."""
    ratios: list[float] = []
    for task in tasks:
        result: ProbeResult = short_training_probe(
            lane,
            dim=dim,
            seq_len=seq_len,
            n_steps=n_steps,
            seed=seed,
            target_fn=task.target_fn,
        )
        if result.trained_successfully:
            ratios.append(result.loss_ratio_initial_over_final)
    return mean(ratios) if ratios else 0.0


# Anchor-arm metrics are a pure function of (cache key, seed, task suite,
# dim, seq_len, n_steps) once construction is seeded — the anchor does not
# depend on the candidate, so retraining it per candidate was pure waste
# (50% of paired/transplant compute). Process-lifetime cache, opt-in via
# ``anchor_cache_key``.
_ANCHOR_METRIC_CACHE: dict[tuple, float] = {}


def run_paired_probe(
    candidate_factory: LaneFactory,
    anchor_factory: LaneFactory,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    tasks: Sequence[ProbeTask] = DEFAULT_PROBE_TASKS,
    dim: int = 32,
    seq_len: int = 32,
    n_steps: int = 100,
    confidence: float = 0.95,
    anchor_cache_key: Hashable | None = None,
) -> PairedDeltaCI:
    """Train candidate vs anchor on each shared seed; CI of the per-seed deltas.

    Fresh modules are built per seed (training mutates weights), and
    ``torch.manual_seed(seed)`` is applied BEFORE each construction so both
    arms' init weights are deterministic per seed — previously the anchor's
    init depended on ambient RNG state, an unfair pairing. ``anchor_factory``
    must build a real lane — pass a known-good reference; do not let it raise
    here. Pass ``anchor_cache_key`` (uniquely identifying the anchor
    architecture) to reuse anchor metrics across candidates sharing it.
    """
    if not seeds:
        raise ValueError("run_paired_probe needs at least one seed")
    deltas: list[float] = []
    for seed in seeds:
        torch.manual_seed(seed)
        cand_metric = _suite_metric(
            candidate_factory(),
            tasks,
            dim=dim,
            seq_len=seq_len,
            n_steps=n_steps,
            seed=seed,
        )
        cache_key = (
            (
                anchor_cache_key,
                seed,
                tuple(t.name for t in tasks),
                dim,
                seq_len,
                n_steps,
            )
            if anchor_cache_key is not None
            else None
        )
        if cache_key is not None and cache_key in _ANCHOR_METRIC_CACHE:
            anchor_metric = _ANCHOR_METRIC_CACHE[cache_key]
        else:
            torch.manual_seed(seed)
            anchor_metric = _suite_metric(
                anchor_factory(),
                tasks,
                dim=dim,
                seq_len=seq_len,
                n_steps=n_steps,
                seed=seed,
            )
            if cache_key is not None:
                _ANCHOR_METRIC_CACHE[cache_key] = anchor_metric
        deltas.append(cand_metric - anchor_metric)
    return paired_delta_ci(deltas, confidence=confidence)


def paired_metadata_for_spec(
    spec: "ProposalSpec",
    *,
    seeds: Sequence[int] = (0, 1, 2),
    dim: int = 32,
    seq_len: int = 32,
    n_steps: int = 100,
) -> dict[str, Any]:
    """Grade-record metadata: paired CI of ``spec`` vs its anchor baseline.

    Builds the candidate from ``spec`` and the anchor from
    ``spec.anchor_witness_op`` (catalog axes → generator). When the anchor
    cannot be built — no witness op, not in the catalog, or un-dispatchable
    (it now *raises* rather than silently becoming nn.Linear) — no comparison is
    fabricated: an explicit ``paired_skipped_reason`` is recorded instead and no
    CI keys are emitted, so the promotion guard stays on its legacy-safe path.
    Imports are local to avoid a validator→generator import cycle.
    """
    from ..generator.code_generator import (
        UndispatchableSpecError,
        generate_module,
        generate_module_from_spec,
    )
    from ..improver.axis_variants import anchor_axes_for_op

    op = getattr(spec, "anchor_witness_op", "") or ""
    if not op:
        return {"paired_skipped_reason": "no_anchor_witness_op"}
    anchor = anchor_axes_for_op(op)
    if anchor is None:
        return {"paired_skipped_reason": f"anchor_not_in_catalog:{op}"}
    anchor_axes = dict(anchor.axes)
    try:
        generate_module(anchor_axes, dim=dim)  # probe buildability once
    except UndispatchableSpecError:
        return {"paired_skipped_reason": f"anchor_unbuildable:{op}"}

    ci = run_paired_probe(
        lambda: generate_module_from_spec(spec, dim=dim),
        lambda: generate_module(anchor_axes, dim=dim),
        seeds=seeds,
        dim=dim,
        seq_len=seq_len,
        n_steps=n_steps,
        # Anchor is fully determined by its catalog op — reuse its metrics
        # across every candidate sharing the same anchor this process.
        anchor_cache_key=("paired_anchor", op),
    )
    return {**ci.to_metadata(), "paired_anchor_op": op}
