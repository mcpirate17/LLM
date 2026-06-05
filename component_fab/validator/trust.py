"""Evidence-tier trust decisions for component_fab candidates.

Cheap fab scores are useful for triage, but downstream evidence decides whether
a candidate is trusted. This module keeps that policy deterministic and
JSON-serializable so CLIs, daily reports, and saved-winner tooling can share it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from component_fab.proposer.spec_generator import ProposalSpec, axes_fingerprint
from component_fab.state.ledger import (
    Ledger,
    LedgerEntry,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
)

TRUST_REJECTED = "rejected"
TRUST_SCREENED = "screened"
TRUST_PROMISING = "promising"
TRUST_TRUSTED = "trusted"

NOVELTY_UNKNOWN = "unknown"
NOVELTY_INVENTION = "mechanism_invention"
NOVELTY_AXIS_NOVEL = "axis_novel"
NOVELTY_DUPLICATE_AXES = "duplicate_axes"
NOVELTY_KNOWN_WINNER = "known_saved_winner"

_NICHE_TASKS = ("long_gap_recall", "compositional_binding")


@dataclass(frozen=True, slots=True)
class TrustThresholds:
    """Thresholds for downstream evidence certification."""

    min_seed_count: int = 2
    min_blimp_delta: float = 0.005
    min_tier2_mean_delta: float = 0.0
    max_wikitext_ppl_regression: float = 0.10


@dataclass(frozen=True, slots=True)
class Tier2Evidence:
    present: bool = False
    status: str = "missing"
    pass_count: int = 0
    n_tasks: int = 0
    mean_delta: float = 0.0
    min_delta: float = 0.0
    niche_passed: bool = False
    passed: bool = False
    seed_count: int = 0


@dataclass(frozen=True, slots=True)
class BlimpEvidence:
    present: bool = False
    status: str = "missing"
    blimp_accuracy: float = 0.0
    softmax_baseline: float = 0.0
    delta_vs_softmax: float = 0.0
    wikitext_post_ppl: float = 0.0
    softmax_wikitext_post_ppl: float = 0.0
    ppl_regression_ratio: float = 0.0
    passed: bool = False
    seed_count: int = 0


@dataclass(frozen=True, slots=True)
class NoveltyEvidence:
    status: str = NOVELTY_UNKNOWN
    axes_fingerprint: str = ""
    duplicate_count: int = 0
    mechanism: str = ""


@dataclass(frozen=True, slots=True)
class TrustDecision:
    proposal_id: str
    name: str
    trust_tier: str
    evidence_status: str
    reasons: tuple[str, ...]
    tier2: Tier2Evidence = field(default_factory=Tier2Evidence)
    blimp: BlimpEvidence = field(default_factory=BlimpEvidence)
    novelty: NoveltyEvidence = field(default_factory=NoveltyEvidence)
    max_internal_composite: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _best_softmax_ppl(blimp_summary: Mapping[str, Any] | None) -> float:
    if not blimp_summary:
        return 0.0
    baseline = (blimp_summary.get("baselines") or {}).get("softmax_attention") or {}
    wikitext = baseline.get("wikitext") or {}
    return _float(wikitext.get("post_train_ppl"))


def _extract_tier2_row(
    proposal_id: str, tier2_summary: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    if not tier2_summary:
        return None
    results = tier2_summary.get("results") or {}
    row = results.get(proposal_id)
    return row if isinstance(row, Mapping) else None


def _extract_blimp_row(
    proposal_id: str, blimp_summary: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    if not blimp_summary:
        return None
    results = blimp_summary.get("results") or {}
    row = results.get(proposal_id)
    return row if isinstance(row, Mapping) else None


def tier2_evidence_from_summary(
    proposal_id: str,
    tier2_summary: Mapping[str, Any] | None,
    thresholds: TrustThresholds = TrustThresholds(),
) -> Tier2Evidence:
    row = _extract_tier2_row(proposal_id, tier2_summary)
    if row is None:
        return Tier2Evidence()
    status = str(row.get("status") or "unknown")
    if status != "ok":
        return Tier2Evidence(present=True, status=status)
    per_task = row.get("per_task") or {}
    deltas = [_float(task.get("delta")) for task in per_task.values()]
    niche_passed = all(
        bool((per_task.get(task) or {}).get("beats")) for task in _NICHE_TASKS
    )
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    seed_count = int(row.get("seed_count") or tier2_summary.get("seed_count") or 1)
    passed = bool(row.get("tier2_passed")) and niche_passed
    passed = passed and mean_delta >= thresholds.min_tier2_mean_delta
    return Tier2Evidence(
        present=True,
        status=status,
        pass_count=int(row.get("pass_count") or 0),
        n_tasks=int(row.get("n_tasks") or len(per_task)),
        mean_delta=round(mean_delta, 6),
        min_delta=round(min(deltas), 6) if deltas else 0.0,
        niche_passed=niche_passed,
        passed=passed,
        seed_count=seed_count,
    )


def blimp_evidence_from_summary(
    proposal_id: str,
    blimp_summary: Mapping[str, Any] | None,
    thresholds: TrustThresholds = TrustThresholds(),
) -> BlimpEvidence:
    row = _extract_blimp_row(proposal_id, blimp_summary)
    if row is None:
        return BlimpEvidence()
    status = str(row.get("status") or "unknown")
    if status != "ok":
        return BlimpEvidence(present=True, status=status)
    candidate_ppl = _float(row.get("wikitext_post_ppl"))
    softmax_ppl = _best_softmax_ppl(blimp_summary)
    ppl_regression = 0.0
    if softmax_ppl > 0.0 and candidate_ppl > 0.0:
        ppl_regression = (candidate_ppl - softmax_ppl) / softmax_ppl
    delta = _float(row.get("delta_vs_softmax_blimp"))
    seed_count = int(row.get("seed_count") or blimp_summary.get("seed_count") or 1)
    passed = delta >= thresholds.min_blimp_delta
    passed = passed and ppl_regression <= thresholds.max_wikitext_ppl_regression
    return BlimpEvidence(
        present=True,
        status=status,
        blimp_accuracy=_float(row.get("blimp_overall_accuracy")),
        softmax_baseline=_float(
            blimp_summary.get("softmax_baseline_blimp") if blimp_summary else 0.0
        ),
        delta_vs_softmax=round(delta, 6),
        wikitext_post_ppl=candidate_ppl,
        softmax_wikitext_post_ppl=softmax_ppl,
        ppl_regression_ratio=round(ppl_regression, 6),
        passed=passed,
        seed_count=seed_count,
    )


def novelty_evidence_for_spec(
    spec: ProposalSpec | None,
    *,
    axes_counts: Mapping[str, int] | None = None,
    saved_winner_ids: set[str] | None = None,
) -> NoveltyEvidence:
    if spec is None:
        return NoveltyEvidence()
    axes = dict(spec.math_axes)
    fp = axes_fingerprint(axes)
    duplicate_count = int((axes_counts or {}).get(fp) or 0)
    mechanism = str(axes.get("op_invention_mechanism") or "")
    if saved_winner_ids and spec.proposal_id in saved_winner_ids:
        status = NOVELTY_KNOWN_WINNER
    elif duplicate_count > 1:
        status = NOVELTY_DUPLICATE_AXES
    elif axes.get("op_search_track") == "invention" or mechanism:
        status = NOVELTY_INVENTION
    else:
        status = NOVELTY_AXIS_NOVEL
    return NoveltyEvidence(
        status=status,
        axes_fingerprint=fp,
        duplicate_count=duplicate_count,
        mechanism=mechanism,
    )


def _ledger_reason_flags(entry: LedgerEntry | None) -> tuple[bool, bool, float]:
    if entry is None:
        return False, False, 0.0
    rejected = entry.promotion_status == PROMOTION_REJECTED
    promoted = entry.promotion_status == PROMOTION_PROMOTED
    max_score = max(entry.composite_history or [0.0])
    return rejected, promoted, max_score


def _has_negative_lm_binding(entry: LedgerEntry | None) -> bool:
    if entry is None:
        return False
    for metadata in reversed(entry.metadata_history):
        margin = metadata.get("lm_binding_mean_margin")
        if margin is not None:
            return _float(margin) < 0.0
    return False


def decide_trust(
    proposal_id: str,
    *,
    name: str = "",
    entry: LedgerEntry | None = None,
    spec: ProposalSpec | None = None,
    tier2_summary: Mapping[str, Any] | None = None,
    blimp_summary: Mapping[str, Any] | None = None,
    axes_counts: Mapping[str, int] | None = None,
    saved_winner_ids: set[str] | None = None,
    thresholds: TrustThresholds = TrustThresholds(),
) -> TrustDecision:
    """Classify one proposal into rejected/screened/promising/trusted."""
    tier2 = tier2_evidence_from_summary(proposal_id, tier2_summary, thresholds)
    blimp = blimp_evidence_from_summary(proposal_id, blimp_summary, thresholds)
    novelty = novelty_evidence_for_spec(
        spec,
        axes_counts=axes_counts,
        saved_winner_ids=saved_winner_ids,
    )
    rejected, promoted, max_score = _ledger_reason_flags(entry)
    reasons: list[str] = []
    final_name = name or (entry.name if entry else "") or (spec.name if spec else "")
    if rejected:
        return TrustDecision(
            proposal_id=proposal_id,
            name=final_name,
            trust_tier=TRUST_REJECTED,
            evidence_status="ledger_rejected",
            reasons=("ledger terminal status is rejected",),
            tier2=tier2,
            blimp=blimp,
            novelty=novelty,
            max_internal_composite=max_score,
        )

    enough_seeds = min(tier2.seed_count, blimp.seed_count) >= thresholds.min_seed_count
    if tier2.passed and blimp.passed and enough_seeds:
        reasons.append("Tier-2 and BLiMP evidence pass configured thresholds")
        return TrustDecision(
            proposal_id=proposal_id,
            name=final_name,
            trust_tier=TRUST_TRUSTED,
            evidence_status="sufficient_downstream_evidence",
            reasons=tuple(reasons),
            tier2=tier2,
            blimp=blimp,
            novelty=novelty,
            max_internal_composite=max_score,
        )

    if tier2.passed or blimp.passed:
        if not (tier2.passed and blimp.passed):
            reasons.append(
                "positive downstream evidence is missing the complementary tier"
            )
        elif not enough_seeds:
            reasons.append(
                "downstream evidence is positive but seed count is insufficient"
            )
        else:
            reasons.append("only one downstream tier passed")
        return TrustDecision(
            proposal_id=proposal_id,
            name=final_name,
            trust_tier=TRUST_PROMISING,
            evidence_status="partial_downstream_evidence",
            reasons=tuple(reasons),
            tier2=tier2,
            blimp=blimp,
            novelty=novelty,
            max_internal_composite=max_score,
        )

    if _has_negative_lm_binding(entry):
        reasons.append("ledger records a negative LM-binding margin")
    if promoted:
        reasons.append("internally promoted, but no downstream pass is present")
    elif entry is not None:
        reasons.append("seen in ledger, but not internally promoted")
    else:
        reasons.append("no ledger evidence found")
    return TrustDecision(
        proposal_id=proposal_id,
        name=final_name,
        trust_tier=TRUST_SCREENED,
        evidence_status="downstream_unverified",
        reasons=tuple(reasons),
        tier2=tier2,
        blimp=blimp,
        novelty=novelty,
        max_internal_composite=max_score,
    )


def axes_counts_for_specs(specs: Sequence[ProposalSpec]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in specs:
        fp = axes_fingerprint(spec.math_axes)
        counts[fp] = counts.get(fp, 0) + 1
    return counts


def saved_winner_ids_from_payload(payload: Mapping[str, Any] | None) -> set[str]:
    if not payload:
        return set()
    winners = payload.get("winners") or []
    return {str(row.get("proposal_id")) for row in winners if row.get("proposal_id")}


def build_trust_report(
    proposal_ids: Sequence[str],
    *,
    ledger: Ledger,
    proposals_by_id: Mapping[str, ProposalSpec],
    tier2_summary: Mapping[str, Any] | None = None,
    blimp_summary: Mapping[str, Any] | None = None,
    saved_winners: Mapping[str, Any] | None = None,
    thresholds: TrustThresholds = TrustThresholds(),
) -> dict[str, Any]:
    specs = list(proposals_by_id.values())
    axes_counts = axes_counts_for_specs(specs)
    saved_ids = saved_winner_ids_from_payload(saved_winners)
    decisions = [
        decide_trust(
            proposal_id,
            entry=ledger.entries.get(proposal_id),
            spec=proposals_by_id.get(proposal_id),
            tier2_summary=tier2_summary,
            blimp_summary=blimp_summary,
            axes_counts=axes_counts,
            saved_winner_ids=saved_ids,
            thresholds=thresholds,
        )
        for proposal_id in proposal_ids
    ]
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.trust_tier] = counts.get(decision.trust_tier, 0) + 1
    return {
        "thresholds": asdict(thresholds),
        "counts": counts,
        "decisions": [decision.to_json() for decision in decisions],
    }
