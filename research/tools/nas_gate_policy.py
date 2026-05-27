"""Explicit, ordered, auditable NAS gate policy.

Replaces the single blended NAS rank with an ordered policy that emits a
per-candidate decision record: which gate failed, the observed value, the
threshold, the head/model version, and a human reason string. It *composes*
the probe-axis primitives in :mod:`research.tools.label_free_probe_oracle`;
it does not retrain models or re-derive thresholds.

Gate order (see ``tasks/nas_pipeline_gate_and_probe_model_todos.md``):

1. ``ar_gate``                         — predictor hard no-go.
2. compile / resource / template       — execution hard no-go.
3. ``nano_induction_nearest``          — operating *bands*, never a sole reject.
4. ``nb0.5`` / ``nb1.0``               — measured signal, recorded only (not folded).
5. ``ar_curriculum``                   — predictor threshold-or-rescue.
6. explore-rescue quota                — published-family / known-good admitted even
                                         when static predictors reject, so blind spots
                                         stay measurable.

Two rejection *kinds* decide whether explore-rescue may override:

* ``predictor`` — a static model said no (ar_gate, ar_curriculum, degenerate nano).
  A known-good / published-family candidate can be rescued past these.
* ``hard``      — execution/health failure (compile, resource, probe error). Never
  overridable; rescuing a graph that will not run measures nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import BaseModel

from research.tools.label_free_probe_oracle import (
    AR_GATE_AXIS,
    _finite_float,
    probe_axis_gate,
)

# Spec defaults, used only when the loaded oracle lacks a trained threshold.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "ar_gate": 0.9,
    "nano_induction_nearest": 0.5,
    "induction": 0.35,
    "ar_curriculum": 0.5,
}

# failure_risk sub-keys that are genuine execution no-gos (vs. soft advisories).
HARD_RISK_KEYS: tuple[str, ...] = ("compile", "resource")


class RejectionKind(str, Enum):
    PREDICTOR = "predictor"  # static model said no — rescue may override
    HARD = "hard"  # execution/health failure — never overridable


class NanoBand(str, Enum):
    STRONG = "strong_positive"  # >= 0.50  → rank up
    WEAK = "weak_retest"  # 0.20–0.50 → neutral, do not gate on it alone
    FRONTIER = "frontier_neutral"  # 0.08–0.20 → frontier-compatible, no signal
    DEGENERATE = (
        "possible_degenerate"  # < 0.08 → contributes to no-go only with other failures
    )
    PROBE_ERROR = (
        "probe_error"  # error/timeout/persistent-zero/invalid → execution no-go
    )


class Stage(str, Enum):
    EXPLOIT = "exploit_rank"  # admitted on its own merits
    RESCUE = "explore_rescue"  # admitted via the known-good blind-spot quota
    REJECTED = "rejected"


class GatePolicyConfig(BaseModel):
    """Tunable thresholds / bands for the gate policy (Pydantic v2)."""

    model_config = {"frozen": True}

    nano_strong: float = 0.50
    nano_weak: float = 0.20
    nano_frontier: float = 0.08
    ar_curriculum_threshold: float = 0.50
    # ar_gate is a hard no-go ONLY when the NAS target is AR. The backtest shows it is blind to
    # the induction family (96.75% false-reject on induction-capable graphs), so an induction- or
    # binding-target pass should set this False and let nano/induction/ar_curriculum gate instead.
    ar_gate_hard: bool = True
    max_failure_risk: float = 0.50  # hard-key risk >= this → execution no-go
    min_template_quality: float = (
        0.0  # template_quality < this → hard no-go (off by default)
    )
    rescue_quota: int = 8  # max known-good rescues per batch; <0 = unlimited
    head_version: str = "pls_partition_oracle"


@dataclass(frozen=True)
class GateRejection:
    gate: str
    value: float | None
    threshold: float | None
    kind: RejectionKind
    head_version: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "value": self.value,
            "threshold": self.threshold,
            "kind": self.kind.value,
            "head_version": self.head_version,
            "reason": self.reason,
        }


@dataclass
class GateDecision:
    fingerprint: str
    accepted: bool
    stage: Stage
    rejections: list[GateRejection] = field(default_factory=list)
    rank_signals: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def first_failed_gate(self) -> str | None:
        return self.rejections[0].gate if self.rejections else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "accepted": self.accepted,
            "stage": self.stage.value,
            "first_failed_gate": self.first_failed_gate,
            "rejections": [r.to_dict() for r in self.rejections],
            "rank_signals": self.rank_signals,
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- #
# Candidate normalization — shortlist rows and published-sanity rows differ.
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    fingerprint: str
    predicted: dict[str, float]  # static NAS predictions per axis
    failure_risk: dict[str, float]
    template_quality: float | None
    known_good: bool  # published-family / literature exact|family / explicit flag
    nb05: float | None
    nb10: float | None
    extra: dict[str, Any] = field(default_factory=dict)


def _coerce_floats(mapping: Mapping[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in (mapping or {}).items():
        f = _finite_float(val)
        if f is not None:
            out[key] = f
    return out


def candidate_from_row(row: Mapping[str, Any]) -> Candidate:
    """Normalize a cascade-shortlist row or published-sanity row into a Candidate."""
    predicted = _coerce_floats(
        row.get("predicted") or row.get("label_free_probe_predictions")
    )
    cheap = row.get("cheap_actual") or {}
    nb05 = (
        _finite_float(cheap.get("nb05"))
        if isinstance(cheap, Mapping)
        else _finite_float(row.get("nb05"))
    )
    nb10 = (
        _finite_float(cheap.get("nb10"))
        if isinstance(cheap, Mapping)
        else _finite_float(row.get("nb10"))
    )
    lit_match = str(row.get("lit_match_type") or "").lower()
    known_good = (
        bool(row.get("known_good"))
        or bool(row.get("published_key"))
        or lit_match in {"exact", "family"}
    )
    return Candidate(
        fingerprint=str(row.get("fingerprint") or row.get("fp") or ""),
        predicted=predicted,
        failure_risk=_coerce_floats(row.get("failure_risk")),
        template_quality=_finite_float(row.get("template_quality")),
        known_good=known_good,
        nb05=nb05,
        nb10=nb10,
        extra={
            k: row.get(k)
            for k in ("mech_score", "novelty", "lit_family", "published_key")
            if k in row
        },
    )


# --------------------------------------------------------------------------- #
# Gate helpers
# --------------------------------------------------------------------------- #
def classify_nano(value: float | None, config: GatePolicyConfig) -> NanoBand:
    """Map a predicted/measured nano_induction_nearest value to its operating band."""
    if value is None:
        return NanoBand.PROBE_ERROR
    if value <= 0.0:
        return NanoBand.PROBE_ERROR
    if value >= config.nano_strong:
        return NanoBand.STRONG
    if value >= config.nano_weak:
        return NanoBand.WEAK
    if value >= config.nano_frontier:
        return NanoBand.FRONTIER
    return NanoBand.DEGENERATE


def _resolve_thresholds(thresholds: Mapping[str, Any] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_THRESHOLDS)
    for axis, val in (thresholds or {}).items():
        f = _finite_float(val)
        if f is not None and f > 0.0:
            resolved[axis] = f
    return resolved


def _gate_ar(
    cand: Candidate,
    thr: Mapping[str, float],
    config: GatePolicyConfig,
    hv: str,
    signals: dict[str, Any],
) -> list[GateRejection]:
    """Gate 1: ar_gate — predictor no-go (hard only when the NAS target is AR)."""
    ar = probe_axis_gate(cand.predicted, thr, axis=AR_GATE_AXIS)
    signals["ar_gate"] = ar.get("predicted")
    signals["_ar_passed"] = bool(ar["passed"])
    if ar["passed"] or not config.ar_gate_hard:
        return []  # advisory when ar_gate_hard is off (induction/binding target)
    return [
        GateRejection(
            AR_GATE_AXIS,
            ar.get("predicted"),
            ar.get("threshold"),
            RejectionKind.PREDICTOR,
            hv,
            "predicted ar_gate below trained threshold",
        )
    ]


def _gate_risk(cand: Candidate, config: GatePolicyConfig) -> list[GateRejection]:
    """Gate 2: compile / resource / template — execution hard no-go."""
    out: list[GateRejection] = []
    for key in HARD_RISK_KEYS:
        risk = cand.failure_risk.get(key)
        if risk is not None and risk >= config.max_failure_risk:
            out.append(
                GateRejection(
                    f"failure_risk.{key}",
                    risk,
                    config.max_failure_risk,
                    RejectionKind.HARD,
                    "rules",
                    f"{key} risk at/above hard limit",
                )
            )
    if (
        cand.template_quality is not None
        and cand.template_quality < config.min_template_quality
    ):
        out.append(
            GateRejection(
                "template_quality",
                cand.template_quality,
                config.min_template_quality,
                RejectionKind.HARD,
                "rules",
                "template quality below floor",
            )
        )
    return out


def _gate_nano(
    cand: Candidate,
    config: GatePolicyConfig,
    hv: str,
    signals: dict[str, Any],
    has_other_failure: bool,
) -> list[GateRejection]:
    """Gate 3: nano_induction_nearest operating bands. Never a sole reject."""
    nano_val = _finite_float(cand.predicted.get("nano_induction_nearest"))
    band = classify_nano(nano_val, config)
    signals["nano"] = nano_val
    signals["nano_band"] = band.value
    signals["nano_rank_up"] = band is NanoBand.STRONG
    if band is NanoBand.PROBE_ERROR and nano_val is not None and nano_val <= 0.0:
        return [
            GateRejection(
                "nano_probe_health",
                nano_val,
                None,
                RejectionKind.HARD,
                hv,
                "nano probe persistent-zero/invalid — execution no-go, not capability",
            )
        ]
    if band is NanoBand.DEGENERATE and has_other_failure:
        return [
            GateRejection(
                "nano_degenerate",
                nano_val,
                config.nano_frontier,
                RejectionKind.PREDICTOR,
                hv,
                "nano < 0.08 alongside another failure",
            )
        ]
    return []


def _gate_ar_curriculum(
    cand: Candidate,
    thr: Mapping[str, float],
    config: GatePolicyConfig,
    hv: str,
    signals: dict[str, Any],
) -> list[GateRejection]:
    """Gate 5: ar_curriculum — predictor threshold-or-rescue."""
    arc = probe_axis_gate(
        cand.predicted,
        {**thr, "ar_curriculum": config.ar_curriculum_threshold},
        axis="ar_curriculum",
    )
    signals["ar_curriculum"] = arc.get("predicted")
    if arc["passed"]:
        return []
    return [
        GateRejection(
            "ar_curriculum",
            arc.get("predicted"),
            arc.get("threshold"),
            RejectionKind.PREDICTOR,
            hv,
            "predicted ar_curriculum below threshold",
        )
    ]


def _decide(
    cand: Candidate, rejections: list[GateRejection], signals: dict[str, Any]
) -> GateDecision:
    """Resolve gate rejections into an admission stage (quota applied by the batch fn)."""
    if not rejections:
        return GateDecision(
            cand.fingerprint, True, Stage.EXPLOIT, [], signals, "all gates passed"
        )
    has_hard = any(r.kind is RejectionKind.HARD for r in rejections)
    reason = "; ".join(r.gate for r in rejections)
    if cand.known_good and not has_hard:
        return GateDecision(
            cand.fingerprint,
            False,
            Stage.RESCUE,
            rejections,
            signals,
            f"known-good blind-spot rescue (predictor rejections: {reason})",
        )
    return GateDecision(
        cand.fingerprint, False, Stage.REJECTED, rejections, signals, reason
    )


def evaluate_candidate(
    cand: Candidate,
    thresholds: Mapping[str, float],
    config: GatePolicyConfig,
) -> GateDecision:
    """Run the ordered gate policy for one candidate (rescue quota applied by the batch fn)."""
    thr = _resolve_thresholds(thresholds)
    hv = config.head_version
    signals: dict[str, Any] = {}
    rejections = _gate_ar(cand, thr, config, hv, signals)
    rejections += _gate_risk(cand, config)
    # nano<0.08 counts toward a no-go only if an ar_gate/risk failure already exists.
    rejections += _gate_nano(
        cand, config, hv, signals, has_other_failure=bool(rejections)
    )
    if cand.nb05 is not None:
        signals["nb05"] = cand.nb05
    if cand.nb10 is not None:
        signals["nb10"] = cand.nb10
    rejections += _gate_ar_curriculum(cand, thr, config, hv, signals)
    signals["cheap_evidence_rank_up"] = bool(
        signals.pop("_ar_passed", False) and cand.nb10 is not None and cand.nb10 >= 0.9
    )
    signals["known_good"] = cand.known_good
    return _decide(cand, rejections, signals)


def _rescue_rank_key(decision: GateDecision) -> tuple[float, float, float]:
    """Order rescue candidates: strong cheap evidence first, then nb1.0, then ar_curriculum."""
    s = decision.rank_signals
    return (
        1.0 if s.get("cheap_evidence_rank_up") else 0.0,
        float(s.get("nb10") or 0.0),
        float(s.get("ar_curriculum") or 0.0),
    )


def evaluate_candidates(
    rows: Iterable[Mapping[str, Any]],
    thresholds: Mapping[str, float] | None = None,
    config: GatePolicyConfig | None = None,
) -> list[GateDecision]:
    """Run the policy over a batch and enforce the explore-rescue quota."""
    config = config or GatePolicyConfig()
    thresholds = thresholds or {}
    decisions = [
        evaluate_candidate(candidate_from_row(r), thresholds, config) for r in rows
    ]

    if config.rescue_quota >= 0:
        rescues = [d for d in decisions if d.stage is Stage.RESCUE]
        if len(rescues) > config.rescue_quota:
            keep = set(
                id(d)
                for d in sorted(rescues, key=_rescue_rank_key, reverse=True)[
                    : config.rescue_quota
                ]
            )
            for d in rescues:
                if id(d) not in keep:
                    d.stage = Stage.REJECTED
                    d.reason = "rescue quota exhausted; " + d.reason
    return decisions


def load_thresholds() -> dict[str, float]:
    """Load trained per-axis thresholds from the persisted oracle; defaults if unavailable."""
    from research.tools.label_free_probe_oracle import LabelFreeProbeOracleScorer

    scorer = LabelFreeProbeOracleScorer.try_load()
    if scorer is None:
        return dict(DEFAULT_THRESHOLDS)
    return _resolve_thresholds(scorer.thresholds)


def summarize(decisions: list[GateDecision]) -> dict[str, Any]:
    """Aggregate decision counts by stage and by first-failed gate."""
    by_stage: dict[str, int] = {}
    by_gate: dict[str, int] = {}
    for d in decisions:
        by_stage[d.stage.value] = by_stage.get(d.stage.value, 0) + 1
        if d.first_failed_gate:
            by_gate[d.first_failed_gate] = by_gate.get(d.first_failed_gate, 0) + 1
    return {
        "n": len(decisions),
        "accepted": sum(1 for d in decisions if d.accepted),
        "by_stage": by_stage,
        "by_first_failed_gate": by_gate,
    }


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read candidate rows from a .jsonl shortlist or a published-sanity summary .json."""
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    if isinstance(obj, dict) and "rows" in obj:
        return list(obj["rows"])
    if isinstance(obj, list):
        return obj
    return [obj]
