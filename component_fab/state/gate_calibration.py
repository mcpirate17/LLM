"""Gate calibration for the fab validator stack (WS-1).

Pure read-only analyzer. Complements ``failure_attribution.py``: that module
measures *how much* each gate kills (volume + over-eager); this one measures
*whether the gate's signal actually separates good candidates from bad* —
the discriminative power the autonomous loop has been trusting blind.

For every gate that records a continuous signal, we compute the ROC/AUC of
that signal against the eventual outcome label, plus a threshold sweep for
an operating-point recommendation. An AUC near 0.5 means the gate is noise;
an AUC below 0.5 means it is *anti-predictive* (ranks bad candidates above
good ones) — the failure mode this project has hit repeatedly with proxy
gates. Calibrating against the ledger's own recorded verdicts keeps the
analyzer free (no re-grading, no GPU) and faithful to what the gates decided.

Chained-gate handling (canonical order = ``validator/capability.py``):
    smoke -> s05_causality_stability -> erf_density -> nano_bind -> ar_*
A candidate's continuous signal for gate G is only meaningful if it *reached*
G (passed every earlier gate). So each gate's AUC is computed over the
population that reached it. We report two AUCs per gate:
  - ``auc_reached``: over everything that reached the gate (full power, but
    the candidates killed *here* were killed *because* their signal was low,
    which inflates separation).
  - ``auc_passed``:  over only the candidates that PASSED this gate (removes
    that circularity — this is the honest residual predictive power).

Gate signals (recorded in ``metadata`` for every graded candidate):
  - ``erf_density``        -> erf_density gate
  - ``nb_max_accuracy``    -> nano_bind gate
  - ``can_bind`` (binary)  -> ar_binding soft gate
``smoke`` and ``s05_causality_stability`` record no continuous score, so
they get kill stats only (no AUC) — see ``failure_attribution.py``.

The ``op_property_catalog`` corpus named in the original WS-1 spec is a poor
fit: 134/182 evaluated ops do not rebuild through the generator dispatch
(they collapse to the ``nn.Linear`` fallback) and the 48 that do collapse to
~7 distinct module classes. ``survey_op_buildability`` recorded that finding;
it is settled, so the survey is opt-in (``run_buildability_survey=True``) —
it imports torch + the full generator stack and rebuilds up to 182 ops.
Stats are sklearn-backed (``roc_auc_score`` / ``roc_curve``), not hand-rolled.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


from sklearn.metrics import roc_auc_score, roc_curve

from .gates import CANONICAL_GATE_ORDER, eliminated_by, passed, reached
from .ledger import DEFAULT_LEDGER_PATH, write_json_report
from .ledger import read_last_grades_and_statuses as _read_ledger

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = _REPO / "research" / "meta_analysis.db"
DEFAULT_OUTPUT_PATH = _REPO / "component_fab" / "catalog" / "gate_calibration.json"

# Gate -> (recorded metadata signal, "higher is better"). Only gates with a
# continuous/binary recorded score appear here.
GATE_SIGNALS: tuple[tuple[str, str], ...] = (
    ("erf_density", "erf_density"),
    ("nano_bind", "nb_max_accuracy"),
    ("ar_binding", "can_bind"),
)
# Continuous signals that get a threshold sweep (binary ones do not).
SWEEP_SIGNALS: tuple[str, ...] = ("erf_density", "nb_max_accuracy")

# AUC bands for the human-readable verdict, applied to the *passed-population*
# (residual) AUC — the honest one. [0.45, 0.55] is indistinguishable from chance
# at the ledger's sample sizes; below 0.45 is a genuine inversion.
AUC_PREDICTIVE = 0.55
AUC_ANTI = 0.45
# A large reached-minus-passed gap means the gate's apparent power is an artifact
# of it defining the outcome label (it kills the low-signal candidates), not
# independent validity. Flag it so a high reached-AUC is not mistaken for a win.
CIRCULAR_GAP = 0.15


@dataclass(slots=True)
class _Candidate:
    proposal_id: str
    eliminated_by: str  # gate name, or SURVIVED
    signals: dict[str, float]  # erf_density / nb_max_accuracy / can_bind(0|1)
    labels: dict[str, int]  # label_name -> 0|1


@dataclass(slots=True)
class GateAUC:
    gate: str
    signal: str
    n_reached: int
    n_passed: int
    n_pos_reached: int
    n_neg_reached: int
    auc_reached: float | None
    auc_passed: float | None
    verdict: str  # "predictive" | "noise" | "anti_predictive" | "insufficient_n"
    circular_inflation: bool  # high reached-AUC is an artifact of defining the label


@dataclass(slots=True)
class ThresholdPoint:
    threshold: float
    kept_frac: float
    tpr: float
    fpr: float
    precision: float
    youden_j: float


@dataclass(slots=True)
class BuildabilitySurvey:
    total_ops: int
    buildable: int
    fallback_linear: int
    build_errors: int
    distinct_module_classes: int
    by_module_class: dict[str, int]
    skipped_op_names: list[str]


@dataclass(slots=True)
class GateCalibrationReport:
    total_graded: int
    labels: dict[str, dict[str, int]]  # label -> {"pos":, "neg":}
    primary_label: str
    gate_aucs: dict[str, list[GateAUC]]  # label -> [GateAUC]
    threshold_sweeps: dict[str, list[ThresholdPoint]]  # signal -> sweep (primary label)
    buildability: BuildabilitySurvey | None
    findings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# AUC (sklearn ROC-AUC; chained-population helpers live in gates.py)
# --------------------------------------------------------------------------- #
def _auc(scores: list[float], labels: list[int]) -> float | None:
    """Probability a random positive outranks a random negative (ties=0.5).

    Returns ``None`` when either class is empty.
    """
    pos = sum(labels)
    if pos == 0 or pos == len(labels):
        return None
    return float(roc_auc_score(labels, scores))


# --------------------------------------------------------------------------- #
# Ledger ingest
# --------------------------------------------------------------------------- #
def _candidate(pid: str, grade: dict[str, Any], status: str) -> _Candidate:
    meta = grade.get("metadata") or {}
    signals: dict[str, float] = {}
    for key in ("erf_density", "nb_max_accuracy"):
        v = meta.get(key)
        if isinstance(v, (int, float)):
            signals[key] = float(v)
    cb = meta.get("can_bind")
    if cb is not None:
        signals["can_bind"] = 1.0 if cb else 0.0
    labels = {
        "promoted": 1 if status == "promoted" else 0,
        "learned_signal": 1 if grade.get("learned_signal") else 0,
    }
    return _Candidate(
        proposal_id=pid,
        eliminated_by=eliminated_by(grade),
        signals=signals,
        labels=labels,
    )


# --------------------------------------------------------------------------- #
# Core computations
# --------------------------------------------------------------------------- #
def _verdict(primary: float | None, n_pos: int, n_neg: int, min_n: int) -> str:
    """Classify a gate by its honest (residual) AUC.

    ``primary`` is the passed-population AUC when available (it strips the
    circularity of the gate defining its own outcome), else the reached AUC.
    """
    if primary is None or n_pos < min_n or n_neg < min_n:
        return "insufficient_n"
    if primary >= AUC_PREDICTIVE:
        return "predictive"
    if primary >= AUC_ANTI:
        return "noise"
    return "anti_predictive"


def _gate_auc(
    cands: list[_Candidate],
    gate: str,
    signal: str,
    label: str,
    order: tuple[str, ...],
    min_n: int,
) -> GateAUC:
    reached_pop = [
        c
        for c in cands
        if reached(c.eliminated_by, gate, order) and signal in c.signals
    ]
    passed_pop = [c for c in reached_pop if passed(c.eliminated_by, gate, order)]
    r_scores = [c.signals[signal] for c in reached_pop]
    r_labels = [c.labels[label] for c in reached_pop]
    p_scores = [c.signals[signal] for c in passed_pop]
    p_labels = [c.labels[label] for c in passed_pop]
    n_pos = sum(r_labels)
    n_neg = len(r_labels) - n_pos
    auc_r = _auc(r_scores, r_labels)
    auc_p = _auc(p_scores, p_labels)
    # The passed-population AUC is the honest one; fall back to reached when
    # the passed subset has only one class.
    headline = auc_p if auc_p is not None else auc_r
    p_pos = sum(p_labels)
    p_neg = len(p_labels) - p_pos
    verdict = _verdict(
        headline,
        p_pos if auc_p is not None else n_pos,
        p_neg if auc_p is not None else n_neg,
        min_n,
    )
    circular = (
        auc_r is not None and auc_p is not None and (auc_r - auc_p) >= CIRCULAR_GAP
    )
    return GateAUC(
        gate=gate,
        signal=signal,
        n_reached=len(reached_pop),
        n_passed=len(passed_pop),
        n_pos_reached=n_pos,
        n_neg_reached=n_neg,
        auc_reached=auc_r,
        auc_passed=auc_p,
        verdict=verdict,
        circular_inflation=circular,
    )


def _threshold_sweep(
    cands: list[_Candidate],
    gate: str,
    signal: str,
    label: str,
    order: tuple[str, ...],
    n_points: int,
) -> list[ThresholdPoint]:
    """Sweep ``signal >= threshold`` over the population that reached ``gate``.

    Built on a single ``sklearn.metrics.roc_curve`` pass (sort-once) instead
    of rescanning the population per threshold point.
    """
    pop = [
        c
        for c in cands
        if reached(c.eliminated_by, gate, order) and signal in c.signals
    ]
    if not pop:
        return []
    labels = [c.labels[label] for c in pop]
    pos = sum(labels)
    neg = len(pop) - pos
    if pos == 0 or neg == 0:
        return []
    fpr_arr, tpr_arr, roc_thresholds = roc_curve(
        labels, [c.signals[signal] for c in pop], drop_intermediate=False
    )
    # roc_curve thresholds descend and lead with a keep-nothing sentinel; the
    # rest are exactly the unique signal values under the ``score >= thr`` rule.
    rates_by_threshold = {
        float(thr): (float(t), float(f))
        for thr, t, f in zip(roc_thresholds[1:], tpr_arr[1:], fpr_arr[1:])
    }
    scores = sorted(rates_by_threshold)
    # Candidate thresholds: evenly sampled unique signal values.
    if len(scores) <= n_points:
        thresholds = scores
    else:
        step = (len(scores) - 1) / (n_points - 1)
        thresholds = [scores[round(i * step)] for i in range(n_points)]
    points: list[ThresholdPoint] = []
    for thr in thresholds:
        tpr, fpr = rates_by_threshold[thr]
        tp = tpr * pos
        kept = tp + fpr * neg
        points.append(
            ThresholdPoint(
                threshold=round(thr, 6),
                kept_frac=round(kept / len(pop), 4),
                tpr=round(tpr, 4),
                fpr=round(fpr, 4),
                precision=round(tp / kept, 4) if kept else 0.0,
                youden_j=round(tpr - fpr, 4),
            )
        )
    return points


def _gate_for_signal(signal: str) -> str:
    for gate, sig in GATE_SIGNALS:
        if sig == signal:
            return gate
    raise KeyError(signal)


# --------------------------------------------------------------------------- #
# Buildability survey (op_property_catalog corpus — documents the pivot)
# --------------------------------------------------------------------------- #
def survey_op_buildability(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    min_eval_count: int = 3,
    dim: int = 32,
) -> BuildabilitySurvey | None:
    """Count how many evaluated catalog ops rebuild into a real lane.

    Imports torch + the generator lazily so the core analyzer stays
    dependency-light. Returns ``None`` if the DB is absent.
    """
    path = Path(db_path)
    if not path.exists():
        return None
    import sqlite3

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        ops = [
            r[0]
            for r in conn.execute(
                "SELECT op_name FROM op_property_catalog WHERE eval_count >= ? "
                "ORDER BY eval_count DESC",
                (min_eval_count,),
            )
        ]
    finally:
        conn.close()

    from component_fab.generator.code_generator import generate_module_from_spec
    from component_fab.improver.axis_variants import anchor_axes_for_op
    from component_fab.proposer.spec_generator import ProposalSpec, category_from_axes

    by_class: Counter[str] = Counter()
    buildable = 0
    errors = 0
    skipped: list[str] = []
    for op in ops:
        anchor = anchor_axes_for_op(op, db_path=path)
        if anchor is None:
            errors += 1
            skipped.append(op)
            continue
        axes = dict(anchor.axes)
        spec = ProposalSpec(
            proposal_id=f"{op}_buildcheck",
            name=op,
            category=category_from_axes(axes),
            synthesis_kind="novel_hybrid",
            math_axes=axes,
            anchor_witness_op=op,
            anchor_witnesses_all=(op,),
            declared_property_row=axes,
            predicted_lift=anchor.pass_rate,
            rationale="buildability survey",
        )
        try:
            module = generate_module_from_spec(spec, dim=dim)
        except Exception:  # noqa: BLE001 — a build failure is a survey datum
            errors += 1
            skipped.append(op)
            continue
        cls = type(module).__name__
        by_class[cls] += 1
        if cls == "Linear":
            skipped.append(op)
        else:
            buildable += 1
    fallback = by_class.get("Linear", 0)
    distinct = len([c for c in by_class if c != "Linear"])
    return BuildabilitySurvey(
        total_ops=len(ops),
        buildable=buildable,
        fallback_linear=fallback,
        build_errors=errors,
        distinct_module_classes=distinct,
        by_module_class=dict(by_class),
        skipped_op_names=skipped,
    )


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #
def _best_threshold(points: list[ThresholdPoint]) -> ThresholdPoint | None:
    """Operating point maximizing Youden's J (TPR - FPR)."""
    return max(points, key=lambda p: p.youden_j) if points else None


def _build_findings(
    gate_aucs: list[GateAUC],
    sweeps: dict[str, list[ThresholdPoint]],
    primary_label: str,
    survey: BuildabilitySurvey | None,
) -> list[str]:
    findings: list[str] = []
    anti = [g for g in gate_aucs if g.verdict == "anti_predictive"]
    noise = [g for g in gate_aucs if g.verdict == "noise"]
    pred = [g for g in gate_aucs if g.verdict == "predictive"]
    insuff = [g for g in gate_aucs if g.verdict == "insufficient_n"]

    def _resid(g: GateAUC) -> float | None:
        return g.auc_passed if g.auc_passed is not None else g.auc_reached

    def _circ_note(g: GateAUC) -> str:
        if g.circular_inflation:
            return (
                f" Its reached-AUC {g.auc_reached:.3f} is inflated by the gate "
                f"defining the label (it kills the low-signal candidates) — "
                f"residual power is {g.auc_passed:.3f}."
            )
        return ""

    for g in anti:
        findings.append(
            f"ANTI-PREDICTIVE: gate '{g.gate}' signal '{g.signal}' is mildly "
            f"inverted vs {primary_label} among the candidates it passes "
            f"(residual AUC={_resid(g):.3f}, n_passed={g.n_passed}). Do not rank "
            f"on this signal; review whether the hard cut earns its place."
            + _circ_note(g)
        )
    for g in noise:
        findings.append(
            f"NO RESIDUAL POWER: gate '{g.gate}' signal '{g.signal}' is "
            f"indistinguishable from chance vs {primary_label} among survivors "
            f"(residual AUC={_resid(g):.3f}). Fine as a coarse safety cut, but it "
            f"adds nothing as a ranking signal — do not tune the composite on it."
            + _circ_note(g)
        )
    for g in pred:
        rec = ""
        sweep = sweeps.get(g.signal)
        best = _best_threshold(sweep) if sweep else None
        if best is not None:
            rec = (
                f" Recommended operating point: {g.signal} >= {best.threshold:g} "
                f"(TPR={best.tpr:.2f}, FPR={best.fpr:.2f}, J={best.youden_j:.2f})."
            )
        findings.append(
            f"PREDICTIVE: gate '{g.gate}' signal '{g.signal}' retains real "
            f"separation among survivors (residual AUC={_resid(g):.3f}).{rec}"
        )
    for g in insuff:
        findings.append(
            f"INSUFFICIENT-N: gate '{g.gate}' signal '{g.signal}' lacks enough "
            f"per-class samples for a trustworthy AUC "
            f"(n_pos={g.n_pos_reached}, n_neg={g.n_neg_reached})."
        )
    if not anti and not noise and pred:
        findings.append(
            f"All gates with computable AUC retain residual separation vs "
            f"{primary_label} — gate signals calibrated OK at the ledger tier."
        )
    elif not pred:
        findings.append(
            f"THRESHOLD-CHANGE FLAG: no gate signal retains positive residual "
            f"predictive power for {primary_label} once its own filtering effect "
            f"is removed. The gates work as coarse safety cuts but none is a valid "
            f"ranker — the composite should not weight these signals, and "
            f"promotion should not lean on them beyond the hard floor."
        )
    if survey is not None:
        findings.append(
            f"CORPUS NOTE: of {survey.total_ops} evaluated op_property_catalog "
            f"ops, only {survey.buildable} rebuild into a real lane "
            f"({survey.fallback_linear} collapse to nn.Linear, "
            f"{survey.build_errors} errors); they span just "
            f"{survey.distinct_module_classes} distinct module classes. The "
            f"op-catalog corpus is too degenerate for per-op gate calibration — "
            f"this analysis uses the behaviorally-diverse fab ledger instead."
        )
    findings.append(
        "CAVEAT: ledger labels (promoted/learned_signal) are nano-tier verdicts "
        "partly downstream of these same gates; AUC here measures internal "
        "separability, not scale ground truth. A non-circular label requires the "
        "WS-7 fidelity ladder (R1/R2)."
    )
    return findings


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def compute_gate_calibration(
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
    *,
    db_path: Path | str | None = DEFAULT_DB_PATH,
    primary_label: str = "learned_signal",
    min_class_n: int = 10,
    sweep_points: int = 12,
    run_buildability_survey: bool = False,
    gate_order: tuple[str, ...] = CANONICAL_GATE_ORDER,
) -> GateCalibrationReport:
    """Replay the ledger and calibrate each gate's discriminative power."""
    last_grade, last_status = _read_ledger(Path(ledger_path))
    cands = [
        _candidate(pid, grade, last_status.get(pid, "pending"))
        for pid, grade in last_grade.items()
    ]
    label_names = ("learned_signal", "promoted")
    if primary_label not in label_names:
        raise ValueError(f"primary_label must be one of {label_names}")

    labels_summary: dict[str, dict[str, int]] = {}
    for label in label_names:
        pos = sum(c.labels[label] for c in cands)
        labels_summary[label] = {"pos": pos, "neg": len(cands) - pos}

    gate_aucs: dict[str, list[GateAUC]] = {}
    for label in label_names:
        gate_aucs[label] = [
            _gate_auc(cands, gate, signal, label, gate_order, min_class_n)
            for gate, signal in GATE_SIGNALS
        ]

    sweeps: dict[str, list[ThresholdPoint]] = {
        signal: _threshold_sweep(
            cands,
            _gate_for_signal(signal),
            signal,
            primary_label,
            gate_order,
            sweep_points,
        )
        for signal in SWEEP_SIGNALS
    }

    survey: BuildabilitySurvey | None = None
    if run_buildability_survey and db_path is not None:
        survey = survey_op_buildability(db_path)

    findings = _build_findings(gate_aucs[primary_label], sweeps, primary_label, survey)
    return GateCalibrationReport(
        total_graded=len(cands),
        labels=labels_summary,
        primary_label=primary_label,
        gate_aucs=gate_aucs,
        threshold_sweeps=sweeps,
        buildability=survey,
        findings=findings,
    )


def write_gate_calibration(
    report: GateCalibrationReport,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    payload = {
        "total_graded": report.total_graded,
        "primary_label": report.primary_label,
        "labels": report.labels,
        "gate_aucs": {
            label: [asdict(g) for g in rows] for label, rows in report.gate_aucs.items()
        },
        "threshold_sweeps": {
            sig: [asdict(p) for p in pts]
            for sig, pts in report.threshold_sweeps.items()
        },
        "buildability": asdict(report.buildability) if report.buildability else None,
        "findings": list(report.findings),
    }
    return write_json_report(payload, output_path)
