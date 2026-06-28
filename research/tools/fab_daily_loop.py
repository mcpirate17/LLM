"""Daily fab autonomous loop — five-phase orchestrator.

Each invocation runs one full day-cycle:

  A. Bootstrap — log start, snapshot pre-run ledger size.
  R. Refresh learning state — recompute axis_lift.json (consumed live by
     the knob sampler) and failure_attribution.json (gate-health diagnostics)
     from the current ledger, so Phase B trains on fresh self-learned signals.
  B. Autonomous run — subprocess `component_fab.tools.run_autonomous`
     with a wall-clock budget. Loop produces fab promotions.
  C. Tier-2 binding — take top-K promotions, run harder_binding suite.
     Drop those that don't beat the best baseline on >= pass_threshold of 6 tasks.
  D. BLiMP cohort — train each Tier-2 survivor as TinyLM on wikitext +
     BLiMP. Compute delta vs softmax_attention baseline at same budget.
  E. Daily report — write a markdown handoff to
     ``research/reports/fab_daily_<YYYY-MM-DD>.md`` and append the
     summary line to the Obsidian `fab_daily_log` note.

The orchestrator is **resumable**: ledger.jsonl is durable. If Phase B
is killed mid-run (SIGINT, OOM), Phases C-E pick up from whatever
reached promoted status. Cohort outputs land under
``component_fab/catalog/cohort_<phase>_<timestamp>.json`` so they can be
inspected manually.

Usage:

    python -m research.tools.fab_daily_loop --budget-hours 4
    python -m research.tools.fab_daily_loop --budget-hours 4 \
        --tier2-top-k 8 --blimp-top-k 4 --skip-phase-b  # post-process only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from component_fab.state.ledger import Ledger, PROMOTION_PROMOTED

_REPO = Path(__file__).resolve().parents[2]
_CATALOG = _REPO / "component_fab" / "catalog"
_REPORTS = _REPO / "research" / "reports"
_LEDGER_PATH = _CATALOG / "ledger.jsonl"
_DAILY_LOG = Path("/home/tim/Documents/CodexVault/research/fab_daily_log.md")
_TIER2_LABELS_PATH = _REPO / "research" / "data" / "tier2_predictor" / "labels.jsonl"
_TIER2_MIN_LABELS = 60


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return _dt.date.today().isoformat()


def _phase_a_bootstrap(quiet: bool) -> dict[str, Any]:
    """Snapshot pre-run ledger size + timestamp."""
    ledger = Ledger(_LEDGER_PATH, include_rotated=True)
    pre_count = sum(
        1 for e in ledger.all_entries() if e.promotion_status == PROMOTION_PROMOTED
    )
    if not quiet:
        print(f"[A] bootstrap — ledger has {pre_count} promoted at start")
    return {
        "started_iso": _now_iso(),
        "pre_promoted_count": pre_count,
        "total_entries": len(ledger.entries),
    }


def _tier2_label_count() -> int:
    try:
        with _TIER2_LABELS_PATH.open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _phase_refresh_learning_state(quiet: bool) -> dict[str, Any]:
    """Recompute the fab self-learning artifacts from the current ledger.

    Runs BEFORE Phase B so the autonomous loop consumes fresh signals:
      - ``axis_lift.json`` — per-(axis, value) shrunken promotion lift,
        auto-loaded by ``enumerate_adaptive_math_knob_compositions`` to bias
        knob sampling toward axes with empirical lift.
      - ``failure_attribution.json`` — gate kill-rates, over-eager gates, and
        the rejected-but-promising anchor pool. Consumed by the daily report's
        gate-health section (surfaces promotion-blocking gate miscalibration).

    The Tier-2 value predictor is deliberately NOT retrained here: it has no
    live consumer (``quality._tier2_win_probability`` is heuristic) and is
    label-starved (needs >= 60). Its readiness is surfaced so activation can be
    scheduled once promotions start producing Tier-2 labels.
    """
    summary: dict[str, Any] = {}
    try:
        from component_fab.state.axis_lift import compute_axis_lift, write_axis_lift

        axis_report = compute_axis_lift()
        axis_path = write_axis_lift(axis_report)
        knob_rows = sorted(
            axis_report.by_axis.get("math_knob", []),
            key=lambda r: r.lift,
            reverse=True,
        )
        summary["axis_lift"] = {
            "path": str(axis_path),
            "global_pass_rate": axis_report.global_pass_rate,
            "top_knob_lifts": [
                {"value": r.value, "lift": round(r.lift, 3), "n": r.n}
                for r in knob_rows[:5]
            ],
        }
    except Exception as exc:  # noqa: BLE001 — orchestrator must survive refresh errors
        summary["axis_lift"] = {"status": f"failed: {exc}"}

    try:
        from component_fab.state.failure_attribution import (
            compute_failure_attribution,
            write_failure_attribution,
        )

        fa_report = compute_failure_attribution()
        fa_path = write_failure_attribution(fa_report)
        summary["failure_attribution"] = {
            "path": str(fa_path),
            "total_graded": fa_report.total_graded,
            "total_promoted": fa_report.total_promoted,
            "total_rejected": fa_report.total_rejected,
            "over_eager_gates": list(fa_report.over_eager_gates),
            "gate_kill_rates": [
                {
                    "gate": g.gate,
                    "kill_rate": round(g.kill_rate, 3),
                    "killed": g.killed,
                    "reached": g.reached,
                    "over_eager": g.over_eager,
                }
                for g in fa_report.gate_stats
            ],
            "anchor_pool_size": len(fa_report.anchor_pool),
        }
    except Exception as exc:  # noqa: BLE001 — orchestrator must survive refresh errors
        summary["failure_attribution"] = {"status": f"failed: {exc}"}

    n_labels = _tier2_label_count()
    summary["tier2_predictor"] = {
        "labels": n_labels,
        "labels_required": _TIER2_MIN_LABELS,
        "ready": n_labels >= _TIER2_MIN_LABELS,
    }

    if not quiet:
        al = summary.get("axis_lift", {})
        fa = summary.get("failure_attribution", {})
        print(
            "[R] learning-state refresh — "
            f"axis_lift global_pass={al.get('global_pass_rate', 0.0):.3f}; "
            f"over_eager_gates={fa.get('over_eager_gates', [])}; "
            f"tier2_labels={n_labels}/{_TIER2_MIN_LABELS}"
        )
    return summary


def _phase_b_autonomous(
    budget_minutes: int,
    *,
    use_promoted_as_anchors: bool,
    max_cross_pairs: int,
    max_knob_specs: int,
    halt_quiescent: int,
    probe_steps: int,
    quiet: bool,
) -> dict[str, Any]:
    """Subprocess the autonomous fab loop with the given budget."""
    log_path = _REPO / "research" / "reports" / f"fab_daily_phaseB_{_today()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "component_fab.tools.run_autonomous",
        "--time-budget-minutes",
        str(budget_minutes),
        "--max-cross-pairs",
        str(max_cross_pairs),
        "--max-knob-specs",
        str(max_knob_specs),
        "--halt-quiescent",
        str(halt_quiescent),
        "--probe-steps",
        str(probe_steps),
    ]
    if use_promoted_as_anchors:
        cmd.append("--use-promoted-as-anchors")
    if not quiet:
        print(f"[B] autonomous loop — {budget_minutes}m budget; log: {log_path}")
        print("    " + " ".join(cmd))
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    return {
        "returncode": rc.returncode,
        "elapsed_s": round(elapsed, 1),
        "log_path": str(log_path),
    }


def _top_promoted(top_k: int, since_iso: str | None) -> list[tuple[str, float]]:
    """Return up to ``top_k`` highest-scoring promoted proposal_ids."""
    ledger = Ledger(_LEDGER_PATH, include_rotated=True)
    rows: list[tuple[str, float]] = []
    for entry in ledger.all_entries():
        if entry.promotion_status != PROMOTION_PROMOTED:
            continue
        if since_iso and entry.last_seen_iso < since_iso:
            continue
        score = float(max(entry.composite_history or [0.0]))
        rows.append((entry.proposal_id, score))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:top_k]


def _phase_c_tier2(
    top_k: int, since_iso: str | None, seed_count: int, quiet: bool
) -> dict[str, Any]:
    from research.tools.run_tier2_binding_cohort import run_cohort as run_tier2

    top = _top_promoted(top_k, since_iso=since_iso)
    if not top:
        if not quiet:
            print("[C] tier-2 — no promoted ids in this run; skipping")
        return {"survivors": [], "results": {}, "n_evaluated": 0}
    pids = [pid for pid, _ in top]
    if not quiet:
        print(f"[C] tier-2 binding — {len(pids)} candidates")
        for pid, score in top:
            print(f"    {score:.3f}  {pid}")
    try:
        summary = run_tier2(
            pids,
            n_train_steps=200,
            seed_count=max(1, int(seed_count)),
            quiet=quiet,
        )
    except Exception as exc:  # noqa: BLE001 — orchestrator must survive cohort crashes
        if not quiet:
            print(f"    [C] FAILED at cohort level: {exc}")
        return {
            "survivors": [],
            "results": {},
            "n_evaluated": len(pids),
            "status": f"cohort_failed: {exc}",
        }
    out = _CATALOG / f"cohort_tier2_{_today()}.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if not quiet:
        print(f"    {summary['n_survivors']}/{summary['n_evaluated']} survived")
    return summary


def _phase_d_blimp(
    survivors: list[str],
    top_k: int,
    n_train_steps: int,
    seed_count: int,
    quiet: bool,
) -> dict[str, Any]:
    from research.tools.run_blimp_cohort import run_cohort as run_blimp

    if not survivors:
        if not quiet:
            print("[D] BLiMP — no Tier-2 survivors; skipping")
        return {
            "results": {},
            "best_candidate_blimp": 0.0,
            "softmax_baseline_blimp": 0.0,
            "n_beat_softmax": 0,
            "n_evaluated": 0,
        }
    pids = survivors[:top_k]
    if not quiet:
        print(f"[D] BLiMP cohort — {len(pids)} candidates × {n_train_steps} steps each")
    try:
        summary = run_blimp(
            pids,
            n_train_steps=n_train_steps,
            seed_count=max(1, int(seed_count)),
            quiet=quiet,
        )
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            print(f"    [D] FAILED at cohort level: {exc}")
        return {
            "results": {},
            "best_candidate_blimp": 0.0,
            "softmax_baseline_blimp": 0.0,
            "n_beat_softmax": 0,
            "n_evaluated": len(pids),
            "status": f"cohort_failed: {exc}",
        }
    out = _CATALOG / f"cohort_blimp_{_today()}.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if not quiet:
        print(
            f"    best blimp = {summary['best_candidate_blimp']:.4f} "
            f"(softmax baseline = {summary['softmax_baseline_blimp']:.4f})"
        )
    return summary


def _format_refresh_section(phase_refresh: dict[str, Any] | None) -> str:
    """Render the pre-run learning-state refresh into the daily report."""
    if not phase_refresh:
        return "_(learning-state refresh skipped)_"
    al = phase_refresh.get("axis_lift", {})
    fa = phase_refresh.get("failure_attribution", {})
    t2 = phase_refresh.get("tier2_predictor", {})
    lines: list[str] = []
    if "top_knob_lifts" in al:
        movers = (
            ", ".join(
                f"{r['value']}={r['lift']:.2f}(n{r['n']})" for r in al["top_knob_lifts"]
            )
            or "none"
        )
        lines.append(
            f"- **axis-lift** (auto-consumed by knob sampler): "
            f"global_pass={al.get('global_pass_rate', 0.0):.3f}; top knob lifts: {movers}"
        )
    else:
        lines.append(f"- **axis-lift**: {al.get('status', 'unavailable')}")
    if "gate_kill_rates" in fa:
        gates = "; ".join(
            f"{g['gate']} {g['kill_rate']:.0%} ({g['killed']}/{g['reached']})"
            + ("⚠" if g["over_eager"] else "")
            for g in fa["gate_kill_rates"]
            if g["reached"]
        )
        lines.append(
            f"- **failure-attribution** (gate health): {fa.get('total_promoted', 0)} promoted / "
            f"{fa.get('total_rejected', 0)} rejected of {fa.get('total_graded', 0)} graded"
        )
        lines.append(f"  - gate kill-rates: {gates or 'none'}")
        oe = fa.get("over_eager_gates") or []
        if oe:
            lines.append(
                f"  - ⚠ over-eager gates (possible promotion blockers): {', '.join(oe)}"
            )
        lines.append(
            f"  - rejected-but-promising anchor pool: {fa.get('anchor_pool_size', 0)}"
        )
    else:
        lines.append(f"- **failure-attribution**: {fa.get('status', 'unavailable')}")
    ready = "READY" if t2.get("ready") else "not ready"
    lines.append(
        f"- **tier-2 value predictor**: {t2.get('labels', 0)}/"
        f"{t2.get('labels_required', _TIER2_MIN_LABELS)} labels ({ready}; not retrained "
        "— no live consumer + label-starved)"
    )
    return "\n".join(lines)


def _render_daily_markdown(
    date: str,
    phase_a: dict[str, Any],
    phase_b: dict[str, Any],
    phase_c: dict[str, Any],
    phase_d: dict[str, Any],
    phase_refresh: dict[str, Any] | None,
) -> str:
    """Build the full daily-report markdown body."""
    survivors = phase_c.get("survivors", [])
    best_blimp = phase_d.get("best_candidate_blimp", 0.0)
    softmax_baseline = phase_d.get("softmax_baseline_blimp", 0.0)
    n_beat_softmax = phase_d.get("n_beat_softmax", 0)
    trust_report = _daily_trust_report(survivors, phase_c, phase_d)
    trust_counts = trust_report.get("counts", {})
    trust_rows = _format_trust_rows(trust_report)
    refresh_section = _format_refresh_section(phase_refresh)

    def _row(pid: str, data: dict[str, Any]) -> str:
        if data.get("status") != "ok":
            return f"| `{pid[:32]}` | {data.get('status', '?')} | — | — | — |"
        delta = data.get("delta_vs_softmax_blimp", 0.0)
        sign = "+" if delta >= 0 else ""
        return (
            f"| `{data['name'][:48]}` | "
            f"{data['blimp_overall_accuracy']:.4f} | "
            f"{sign}{delta:.4f} | "
            f"{data['wikitext_post_ppl']:.1f} | "
            f"{data['n_params']} |"
        )

    blimp_rows = "\n".join(
        _row(pid, data) for pid, data in phase_d.get("results", {}).items()
    )
    return f"""# Fab Daily Report — {date}

**Run wall-clock:** {phase_a["started_iso"]} → {_now_iso()}
**Phase B duration:** {phase_b.get("elapsed_s", 0) / 60.0:.1f} min
**Ledger growth:** {phase_a["pre_promoted_count"]} promoted before → measured below

## Learning-State Refresh (pre-run)

{refresh_section}

## Phase B: Autonomous Loop

- subprocess returncode: {phase_b.get("returncode", "?")}
- log: `{phase_b.get("log_path", "")}`

## Phase C: Tier-2 Binding (6 discrete-symbolic tasks, 200 train steps each)

- candidates evaluated: {phase_c.get("n_evaluated", 0)}
- tier-2 survivors (>= {phase_c.get("pass_threshold", 4)}/6 tasks beat best baseline): **{len(survivors)}**

## Phase D: BLiMP Cohort (TinyLM + wikitext-103 + BLiMP 67 subtasks)

- softmax_attention baseline BLiMP: **{softmax_baseline:.4f}**
- best candidate BLiMP: **{best_blimp:.4f}**
- candidates beating softmax: **{n_beat_softmax} / {phase_d.get("n_evaluated", 0)}**
- trust tiers: trusted={trust_counts.get("trusted", 0)}, promising={trust_counts.get("promising", 0)}, screened={trust_counts.get("screened", 0)}, rejected={trust_counts.get("rejected", 0)}

| Candidate | BLiMP | Δ vs softmax | wikitext PPL | n_params |
|---|---:|---:|---:|---:|
{blimp_rows}

## Trust Certification

| Candidate | Trust tier | Evidence status | Tier-2 Δ | BLiMP Δ | Reason |
|---|---|---|---:|---:|---|
{trust_rows}

## Stretch Acceptance

- BLiMP ≥ 0.545 (existing-plan Phase 6 target): {"YES" if best_blimp >= 0.545 else "no"}
- BLiMP ≥ 0.57 (cf3e6bc6-class north star): {"YES" if best_blimp >= 0.57 else "no"}
- ≥ +0.02 delta vs softmax_attention: {"YES" if best_blimp - softmax_baseline >= 0.02 else "no"}

## Tomorrow's Anchor Suggestions

The top Tier-2 + BLiMP candidates above should seed tomorrow's
`--use-promoted-as-anchors` pool. If none beat softmax, expand the
search by adding more axis variants (Phase 2 math knob families).

## Cohort Artifacts

- `component_fab/catalog/cohort_tier2_{date}.json`
- `component_fab/catalog/cohort_blimp_{date}.json`
- `research/reports/fab_daily_phaseB_{date}.log`
"""


def _phase_e_report(
    phase_a: dict[str, Any],
    phase_b: dict[str, Any],
    phase_c: dict[str, Any],
    phase_d: dict[str, Any],
    quiet: bool,
    phase_refresh: dict[str, Any] | None = None,
) -> Path:
    """Write the markdown daily report and append the Obsidian log line."""
    date = _today()
    report_path = _REPORTS / f"fab_daily_{date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    body = _render_daily_markdown(
        date, phase_a, phase_b, phase_c, phase_d, phase_refresh
    )
    report_path.write_text(body, encoding="utf-8")
    survivors = phase_c.get("survivors", [])
    log_line = (
        f"- {date}: phaseB={phase_b.get('elapsed_s', 0) / 60:.0f}m "
        f"tier2_survivors={len(survivors)} "
        f"best_blimp={phase_d.get('best_candidate_blimp', 0.0):.4f} "
        f"softmax_baseline={phase_d.get('softmax_baseline_blimp', 0.0):.4f} "
        f"beat_softmax={phase_d.get('n_beat_softmax', 0)}\n"
    )
    _DAILY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _DAILY_LOG.open("a", encoding="utf-8") as fh:
        fh.write(log_line)
    if not quiet:
        print(f"[E] report -> {report_path}")
    return report_path


def _daily_trust_report(
    survivors: list[str], phase_c: dict[str, Any], phase_d: dict[str, Any]
) -> dict[str, Any]:
    """Classify daily candidates with the shared fab trust policy."""
    if not survivors:
        return {"counts": {}, "decisions": []}
    from component_fab.validator.trust import TrustThresholds, build_trust_report
    from research.tools.run_tier2_binding_cohort import _load_proposals_by_id

    ledger = Ledger(_LEDGER_PATH, include_rotated=True)
    return build_trust_report(
        survivors,
        ledger=ledger,
        proposals_by_id=_load_proposals_by_id(),
        tier2_summary=phase_c,
        blimp_summary=phase_d,
        thresholds=TrustThresholds(min_seed_count=2),
    )


def _format_trust_rows(trust_report: dict[str, Any]) -> str:
    rows = trust_report.get("decisions") or []
    if not rows:
        return "| — | — | — | — | — | no candidates reached trust audit |"
    out: list[str] = []
    for row in rows:
        reason = "; ".join(row.get("reasons") or ())
        tier2 = row.get("tier2") or {}
        blimp = row.get("blimp") or {}
        out.append(
            f"| `{str(row.get('name') or row.get('proposal_id'))[:48]}` | "
            f"{row.get('trust_tier', '')} | "
            f"{row.get('evidence_status', '')} | "
            f"{float(tier2.get('mean_delta') or 0.0):.4f} | "
            f"{float(blimp.get('delta_vs_softmax') or 0.0):.4f} | "
            f"{reason[:96]} |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-hours", default=4.0, type=float)
    parser.add_argument("--tier2-top-k", default=8, type=int)
    parser.add_argument("--blimp-top-k", default=3, type=int)
    parser.add_argument("--blimp-train-steps", default=500, type=int)
    parser.add_argument("--tier2-seed-count", default=2, type=int)
    parser.add_argument("--blimp-seed-count", default=2, type=int)
    parser.add_argument(
        "--skip-phase-b",
        action="store_true",
        help="post-process only: assume Phase B was run separately",
    )
    parser.add_argument("--max-cross-pairs", default=60, type=int)
    parser.add_argument("--max-knob-specs", default=96, type=int)
    parser.add_argument("--halt-quiescent", default=6, type=int)
    parser.add_argument("--probe-steps", default=60, type=int)
    parser.add_argument("--use-promoted-as-anchors", action="store_true", default=True)
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="skip the pre-run learning-state refresh (axis_lift + failure_attribution)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    total_minutes = int(args.budget_hours * 60)
    # 90-min cushion for Phases C-E (Tier-2 + BLiMP + report).
    phase_b_minutes = max(30, total_minutes - 90)

    phase_a = _phase_a_bootstrap(quiet=args.quiet)

    phase_refresh = (
        None if args.skip_refresh else _phase_refresh_learning_state(quiet=args.quiet)
    )

    if args.skip_phase_b:
        phase_b: dict[str, Any] = {
            "returncode": "skipped",
            "elapsed_s": 0.0,
            "log_path": "",
        }
    else:
        phase_b = _phase_b_autonomous(
            phase_b_minutes,
            use_promoted_as_anchors=args.use_promoted_as_anchors,
            max_cross_pairs=args.max_cross_pairs,
            max_knob_specs=args.max_knob_specs,
            halt_quiescent=args.halt_quiescent,
            probe_steps=args.probe_steps,
            quiet=args.quiet,
        )

    # When phase B is skipped, the loop's promotions happened BEFORE the
    # orchestrator launched, so filtering by phase_a.started_iso would drop
    # them all. since_iso=None falls back to "all promoted regardless of
    # when," which is what we want for skip-mode.
    since_iso = None if args.skip_phase_b else phase_a["started_iso"]
    phase_c = _phase_c_tier2(
        top_k=args.tier2_top_k,
        since_iso=since_iso,
        seed_count=args.tier2_seed_count,
        quiet=args.quiet,
    )
    phase_d = _phase_d_blimp(
        phase_c.get("survivors", []),
        args.blimp_top_k,
        args.blimp_train_steps,
        args.blimp_seed_count,
        quiet=args.quiet,
    )
    report_path = _phase_e_report(
        phase_a,
        phase_b,
        phase_c,
        phase_d,
        quiet=args.quiet,
        phase_refresh=phase_refresh,
    )
    if not args.quiet:
        print(f"\ndaily loop complete: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
