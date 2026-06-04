"""Unified proposal-quality scorer for component_fab.

Fuses the evidence sources codex already built — Tier-2 cohort feedback, the
NAS/oracle screen (with PPV/NPV/ROC calibration), ledger composite history, and
novelty — plus curated external research priors into a single calibrated
``QualityScore`` per candidate.

This is the ranking/exploration-budget layer the plan calls for. It does NOT
promote anything: Tier-2 / BLiMP evidence (via ``validator.trust``) remains the
only promotion gate. Pure orchestration over small dicts — no heavy math loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from component_fab.proposer.measured_screen import (  # noqa: F401  (used in _verdict_for)
    REASON_UNSTABLE,
)
from component_fab.proposer.nas_screen import NasScreenResult
from component_fab.proposer.research_priors import (
    PriorAffinity,
    ResearchPrior,
    prior_affinity_for_spec,
)
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.proposer.tier2_feedback import (
    Tier2Feedback,
    WEAK_NARROW_DISTRACTOR_ONLY,
    WEAK_NEAR_SURVIVOR,
)
from component_fab.state.ledger import LedgerEntry
from component_fab.validator.trust import (
    NOVELTY_AXIS_NOVEL,
    NOVELTY_DUPLICATE_AXES,
    NOVELTY_INVENTION,
    NOVELTY_KNOWN_WINNER,
    novelty_evidence_for_spec,
)

BUCKET_EXPLOIT = "exploit"
BUCKET_REPAIR = "repair"
BUCKET_EXPLORATION = "exploration"

DEFAULT_BUDGET_SPLIT: dict[str, float] = {
    BUCKET_EXPLOIT: 0.60,
    BUCKET_REPAIR: 0.25,
    BUCKET_EXPLORATION: 0.15,
}

_NOVELTY_CONFIDENCE = {
    NOVELTY_INVENTION: 0.90,
    NOVELTY_AXIS_NOVEL: 0.70,
    NOVELTY_KNOWN_WINNER: 0.30,
    NOVELTY_DUPLICATE_AXES: 0.10,
}

# Strict verdicts — the real filter. A candidate only PASSES if it has been
# MEASURED to beat the existing baseline; everything else is rejected or unproven.
# Cheap proxies cannot certify "beats baseline" (2026-06-03 audit: every cheap
# signal is ~uncorrelated with Tier-2 success), so this is deliberately honest:
# "merely binds / looks ok" is NOT a pass.
VERDICT_BEATS_BASELINE = "beats_baseline"
VERDICT_LOSES_TO_BASELINE = "loses_to_baseline"
VERDICT_REJECT_UNSTABLE = "reject_unstable"
VERDICT_REJECT_NON_BINDER = "reject_non_binder"
VERDICT_UNPROVEN = "unproven"


@dataclass(frozen=True, slots=True)
class QualityScore:
    proposal_id: str
    name: str
    quality_score: float
    tier2_win_probability: float
    novelty_confidence: float
    risk_score: float
    bucket: str
    verdict: str
    passes_hard_filter: bool
    prior_family: str
    prior_affinity: float
    evidence_reasons: tuple[str, ...]
    why_beats_tier2: str
    repair_signatures: tuple[str, ...]
    has_tier2_evidence: bool

    def to_json(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "name": self.name,
            "quality_score": round(self.quality_score, 4),
            "tier2_win_probability": round(self.tier2_win_probability, 4),
            "novelty_confidence": round(self.novelty_confidence, 4),
            "risk_score": round(self.risk_score, 4),
            "bucket": self.bucket,
            "verdict": self.verdict,
            "passes_hard_filter": self.passes_hard_filter,
            "prior_family": self.prior_family,
            "prior_affinity": round(self.prior_affinity, 4),
            "evidence_reasons": list(self.evidence_reasons),
            "why_beats_tier2": self.why_beats_tier2,
            "repair_signatures": list(self.repair_signatures),
            "has_tier2_evidence": self.has_tier2_evidence,
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _nas_norm(nas: NasScreenResult | None) -> float:
    """Map a NAS screen result to [0, 1] (0.5 = neutral / unavailable)."""

    if nas is None or not nas.available:
        return 0.5
    if not nas.gate_pass:
        return 0.2
    if not nas.downstream_gate_pass:
        return 0.4
    return _clamp(0.5 + 0.2 * nas.rank_score)


def _internal_composite(entry: LedgerEntry | None) -> float:
    if entry is None or not entry.composite_history:
        return 0.0
    return _clamp(max(entry.composite_history))


def _measured_binding(entry: LedgerEntry | None) -> tuple[float, bool]:
    """Cheap binding signal MEASURED on the real module by ``validate_capabilities``.

    Reads the most recent ``nb_max_accuracy`` (and ``can_bind``) persisted in the
    ledger. Unlike the NAS proxy-graph oracle — which on a 2026-06-03 audit of 34
    Tier-2-evidence candidates gate-failed 34/34 (incl. 7/7 Tier-2 winners) and
    was significantly *anti*-correlated with Tier-2 mean_delta (ar_gate r=-0.55)
    — this signal is measured on the actual generated nn.Module and was the only
    cheap signal positively correlated with Tier-2 success (r=+0.20). Returns
    ``(score, present)``; ``present=False`` when the candidate was never graded.
    """

    if entry is None:
        return 0.5, False
    for metadata in reversed(entry.metadata_history):
        nb = metadata.get("nb_max_accuracy")
        if nb is not None:
            score = _clamp(float(nb))
            if metadata.get("can_bind"):
                score = min(1.0, score + 0.05)
            return score, True
    return 0.5, False


def _tier2_win_probability(
    tier2: Tier2Feedback | None,
    *,
    binding: float,
    composite: float,
    affinity: PriorAffinity,
    binds_ok: bool = True,
) -> tuple[float, bool, list[str]]:
    reasons: list[str] = []
    if tier2 is not None:
        base = tier2.pass_count / max(1, tier2.n_tasks)
        if tier2.tier2_passed:
            base = max(base, 0.70)
        if tier2.tier2_passed_niche:
            base = min(1.0, base + 0.10)
        if tier2.mean_delta < 0.0:
            base = min(base, 0.40)
            reasons.append(f"Tier-2 mean_delta negative ({tier2.mean_delta:.4f})")
        else:
            reasons.append(
                f"Tier-2 pass {tier2.pass_count}/{tier2.n_tasks}, "
                f"mean_delta {tier2.mean_delta:.4f}"
            )
        return _clamp(base), True, reasons
    # No Tier-2 evidence: estimate from the MEASURED binding probe (the only
    # cheap signal that tracks Tier-2 success), internal composite, and prior
    # affinity. The NAS proxy is deliberately excluded (anti-predictive, OOD).
    estimate = (
        0.5 * binding
        + 0.3 * composite
        + 0.2 * (affinity.affinity * affinity.confidence)
    )
    if not binds_ok:
        # Measured screen says the operator can't route info backward → it will
        # not bind, so cap the pre-Tier-2 win estimate hard.
        estimate *= 0.3
        reasons.append("measured screen: non-binder — win estimate capped")
    reasons.append(
        "no Tier-2 evidence yet — win-probability estimated from measured "
        "nano-binding probe, internal composite, and research-prior affinity"
    )
    return _clamp(estimate), False, reasons


def _risk_score(
    nas: NasScreenResult | None,
    tier2: Tier2Feedback | None,
    novelty_status: str,
) -> tuple[float, list[str]]:
    risk = 0.10
    reasons: list[str] = []
    if nas is not None and nas.available and not nas.gate_pass:
        # The screen now measures the REAL module's position-Jacobian (not the
        # OOD oracle proxy): gate-fail means long_range_reach below the validated
        # binding threshold — the operator does not route information backward, so
        # it cannot bind. That is a genuine risk (confirmed: an MLP-class module
        # scores ~0). A confirmed Tier-2 pass still overrides it in _bucket_for.
        risk += 0.35
        reasons.append(
            "measured screen: operator does not route info backward (won't bind)"
        )
    if tier2 is not None:
        if WEAK_NARROW_DISTRACTOR_ONLY in tier2.signatures:
            risk += 0.30
            reasons.append("Tier-2 wins are distractor-only (narrow)")
        if tier2.mean_delta < 0.0:
            risk += 0.20
    if novelty_status == NOVELTY_DUPLICATE_AXES:
        risk += 0.20
        reasons.append("axes duplicate an already-seen candidate")
    return _clamp(risk), reasons


def _verdict_for(
    *,
    nas: NasScreenResult | None,
    tier2: Tier2Feedback | None,
) -> str:
    """Strict pass/fail. PASS == measured to beat the baseline; nothing else.

    Order matters: an unstable (NaN) module is rejected before anything else, then
    a measured non-binder, then the downstream Tier-2 verdict. Candidates with no
    Tier-2 evidence are UNPROVEN (must be measured) — explicitly NOT a pass, since
    no cheap signal certifies "beats baseline".
    """

    if nas is not None and nas.available and not nas.gate_pass:
        return (
            VERDICT_REJECT_UNSTABLE
            if nas.reason == REASON_UNSTABLE
            else VERDICT_REJECT_NON_BINDER
        )
    if tier2 is None:
        return VERDICT_UNPROVEN
    beats = tier2.tier2_passed or (
        tier2.mean_delta > 0.0 and WEAK_NARROW_DISTRACTOR_ONLY not in tier2.signatures
    )
    return VERDICT_BEATS_BASELINE if beats else VERDICT_LOSES_TO_BASELINE


def _bucket_for(
    *,
    has_tier2: bool,
    tier2: Tier2Feedback | None,
    win_prob: float,
    composite: float,
    risk: float,
    affinity: PriorAffinity,
    repair_signatures: Sequence[str],
) -> str:
    near_survivor = tier2 is not None and WEAK_NEAR_SURVIVOR in tier2.signatures
    distractor_only = (
        tier2 is not None and WEAK_NARROW_DISTRACTOR_ONLY in tier2.signatures
    )
    # A confirmed downstream Tier-2 pass is the strongest evidence we have; a
    # crude NAS proxy gate (which inflates risk) must never override it. Downstream
    # truth beats the cheap screen, so this wins exploit before any risk veto.
    if tier2 is not None and tier2.tier2_passed:
        return BUCKET_EXPLOIT
    # Known false positives (distractor-only) and high-risk candidates must not
    # consume exploit budget on the strength of a cheap internal composite — they
    # belong in repair regardless of how high that composite is.
    if distractor_only or risk >= 0.4:
        if repair_signatures or near_survivor:
            return BUCKET_REPAIR
        return BUCKET_EXPLORATION
    strong_evidence = (tier2 is not None and tier2.tier2_passed) or composite >= 0.6
    if strong_evidence or (win_prob >= 0.6 and has_tier2):
        return BUCKET_EXPLOIT
    if repair_signatures or near_survivor:
        return BUCKET_REPAIR
    if affinity.affinity >= 0.5:
        return BUCKET_EXPLORATION
    # Low-evidence, low-prior candidates default to exploration (cheap probe).
    return BUCKET_EXPLORATION


def _why_beats_tier2(
    affinity: PriorAffinity,
    repair_signatures: Sequence[str],
    nas_norm: float,
    composite: float,
) -> str:
    parts: list[str] = []
    if affinity.affinity > 0.0 and affinity.validation_tasks:
        parts.append(
            f"targets {', '.join(affinity.validation_tasks)} via "
            f"{affinity.family} (prior affinity {affinity.affinity:.2f})"
        )
    if repair_signatures:
        parts.append(f"repairs {', '.join(repair_signatures)}")
    parts.append(f"NAS screen {nas_norm:.2f}, internal composite {composite:.2f}")
    return "; ".join(parts)


def score_quality(
    spec: ProposalSpec,
    *,
    tier2: Tier2Feedback | None = None,
    nas: NasScreenResult | None = None,
    entry: LedgerEntry | None = None,
    axes_counts: Mapping[str, int] | None = None,
    saved_winner_ids: set[str] | None = None,
    priors: Sequence[ResearchPrior] | None = None,
) -> QualityScore:
    """Fuse all evidence into a calibrated, auditable quality record."""

    affinity = prior_affinity_for_spec(spec, priors)
    nas_norm = _nas_norm(nas)
    composite = _internal_composite(entry)
    binding, binding_measured = _measured_binding(entry)
    novelty = novelty_evidence_for_spec(
        spec, axes_counts=axes_counts, saved_winner_ids=saved_winner_ids
    )
    novelty_confidence = _NOVELTY_CONFIDENCE.get(novelty.status, 0.5)

    binds_ok = nas is None or not nas.available or nas.gate_pass
    win_prob, has_tier2, win_reasons = _tier2_win_probability(
        tier2,
        binding=binding,
        composite=composite,
        affinity=affinity,
        binds_ok=binds_ok,
    )
    risk, risk_reasons = _risk_score(nas, tier2, novelty.status)
    repair_signatures = tuple(tier2.signatures) if tier2 is not None else ()

    # Weights reflect the 2026-06-03 audit: downstream Tier-2 win-probability
    # dominates, the MEASURED binding probe is the trusted cheap signal, and the
    # NAS proxy is excluded from the score (advisory only — see _risk_score).
    quality = (
        0.45 * win_prob
        + 0.20 * binding
        + 0.15 * composite
        + 0.10 * affinity.affinity * affinity.confidence
        + 0.10 * novelty_confidence
    )
    quality *= 1.0 - 0.5 * risk
    quality = _clamp(quality)
    if binding_measured:
        win_reasons.append(f"measured nano-binding probe nb={binding:.3f}")

    bucket = _bucket_for(
        has_tier2=has_tier2,
        tier2=tier2,
        win_prob=win_prob,
        composite=composite,
        risk=risk,
        affinity=affinity,
        repair_signatures=repair_signatures,
    )

    verdict = _verdict_for(nas=nas, tier2=tier2)
    reasons = (
        *win_reasons,
        *risk_reasons,
        *(affinity.reasons if affinity.affinity > 0.0 else ()),
        f"novelty={novelty.status}",
        f"verdict={verdict}",
    )
    return QualityScore(
        proposal_id=spec.proposal_id,
        name=spec.name,
        quality_score=quality,
        tier2_win_probability=win_prob,
        novelty_confidence=novelty_confidence,
        risk_score=risk,
        bucket=bucket,
        verdict=verdict,
        passes_hard_filter=verdict == VERDICT_BEATS_BASELINE,
        prior_family=affinity.family,
        prior_affinity=affinity.affinity,
        evidence_reasons=reasons,
        why_beats_tier2=_why_beats_tier2(
            affinity, repair_signatures, nas_norm, composite
        ),
        repair_signatures=repair_signatures,
        has_tier2_evidence=has_tier2,
    )


def score_specs_quality(
    specs: Sequence[ProposalSpec],
    *,
    tier2_by_id: Mapping[str, Tier2Feedback] | None = None,
    nas_by_id: Mapping[str, NasScreenResult] | None = None,
    entries_by_id: Mapping[str, LedgerEntry] | None = None,
    axes_counts: Mapping[str, int] | None = None,
    saved_winner_ids: set[str] | None = None,
    priors: Sequence[ResearchPrior] | None = None,
) -> dict[str, QualityScore]:
    tier2_by_id = tier2_by_id or {}
    nas_by_id = nas_by_id or {}
    entries_by_id = entries_by_id or {}
    return {
        spec.proposal_id: score_quality(
            spec,
            tier2=tier2_by_id.get(spec.proposal_id),
            nas=nas_by_id.get(spec.proposal_id),
            entry=entries_by_id.get(spec.proposal_id),
            axes_counts=axes_counts,
            saved_winner_ids=saved_winner_ids,
            priors=priors,
        )
        for spec in specs
    }


def allocate_budget_buckets(
    scores: Sequence[QualityScore],
    *,
    total: int,
    split: Mapping[str, float] = DEFAULT_BUDGET_SPLIT,
) -> list[QualityScore]:
    """Order ``scores`` into a graded-this-cycle queue under the budget split.

    Reserves ``split`` fractions of ``total`` for exploit/repair/exploration,
    filling each bucket by descending quality. Unused bucket budget is back-filled
    from the global quality-ranked remainder so we never silently drop coverage
    when a bucket is under-supplied.
    """

    if total <= 0:
        return []
    ranked = sorted(scores, key=lambda s: s.quality_score, reverse=True)
    if total >= len(ranked):
        return ranked

    caps = {bucket: int(total * frac) for bucket, frac in split.items()}
    # Hand any rounding remainder to the exploit bucket.
    caps[BUCKET_EXPLOIT] += total - sum(caps.values())

    chosen: list[QualityScore] = []
    chosen_ids: set[str] = set()
    taken: dict[str, int] = {bucket: 0 for bucket in caps}
    for score in ranked:
        if taken.get(score.bucket, 0) < caps.get(score.bucket, 0):
            chosen.append(score)
            chosen_ids.add(score.proposal_id)
            taken[score.bucket] += 1
    # Back-fill remaining slots from the global ranking.
    for score in ranked:
        if len(chosen) >= total:
            break
        if score.proposal_id not in chosen_ids:
            chosen.append(score)
            chosen_ids.add(score.proposal_id)
    chosen.sort(key=lambda s: s.quality_score, reverse=True)
    return chosen


def bucket_counts(scores: Sequence[QualityScore]) -> dict[str, int]:
    counts: dict[str, int] = {
        BUCKET_EXPLOIT: 0,
        BUCKET_REPAIR: 0,
        BUCKET_EXPLORATION: 0,
    }
    for score in scores:
        counts[score.bucket] = counts.get(score.bucket, 0) + 1
    return counts


def verdict_counts(scores: Sequence[QualityScore]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for score in scores:
        counts[score.verdict] = counts.get(score.verdict, 0) + 1
    return counts


def passing_only(scores: Sequence[QualityScore]) -> list[QualityScore]:
    """The strict filter: only candidates MEASURED to beat the baseline survive."""

    return [s for s in scores if s.passes_hard_filter]
