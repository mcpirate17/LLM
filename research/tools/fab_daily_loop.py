"""Daily fab autonomous loop — five-phase orchestrator.

Each invocation runs one full day-cycle:

  A. Bootstrap — log start, snapshot pre-run ledger size.
  B. Autonomous run — subprocess `component_fab.tools.run_autonomous`
     with a wall-clock budget. Loop produces fab promotions.
  C. Tier-2 binding — take top-K promotions, run harder_binding suite.
     Drop those that don't beat the best baseline on >= pass_threshold of 6 tasks.
  D. BLiMP cohort — train each Tier-2 survivor as TinyLM on wikitext +
     BLiMP. Compute delta vs softmax_attention baseline at same budget.
  E. Daily report — write a markdown handoff to
     ``research/reports/fab_daily_<YYYY-MM-DD>.md`` and append the
     summary line to ``research/notes/fab_daily_log.md``.

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
_DAILY_LOG = _REPO / "research" / "notes" / "fab_daily_log.md"


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return _dt.date.today().isoformat()


def _phase_a_bootstrap(quiet: bool) -> dict[str, Any]:
    """Snapshot pre-run ledger size + timestamp."""
    ledger = Ledger(_LEDGER_PATH)
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
    ledger = Ledger(_LEDGER_PATH)
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


def _phase_c_tier2(top_k: int, since_iso: str | None, quiet: bool) -> dict[str, Any]:
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
        summary = run_tier2(pids, n_train_steps=200, quiet=quiet)
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
    survivors: list[str], top_k: int, n_train_steps: int, quiet: bool
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
        summary = run_blimp(pids, n_train_steps=n_train_steps, quiet=quiet)
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


def _phase_e_report(
    phase_a: dict[str, Any],
    phase_b: dict[str, Any],
    phase_c: dict[str, Any],
    phase_d: dict[str, Any],
    quiet: bool,
) -> Path:
    """Write the markdown daily report."""
    date = _today()
    report_path = _REPORTS / f"fab_daily_{date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    survivors = phase_c.get("survivors", [])
    best_blimp = phase_d.get("best_candidate_blimp", 0.0)
    softmax_baseline = phase_d.get("softmax_baseline_blimp", 0.0)
    n_beat_softmax = phase_d.get("n_beat_softmax", 0)

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
    body = f"""# Fab Daily Report — {date}

**Run wall-clock:** {phase_a["started_iso"]} → {_now_iso()}
**Phase B duration:** {phase_b.get("elapsed_s", 0) / 60.0:.1f} min
**Ledger growth:** {phase_a["pre_promoted_count"]} promoted before → measured below

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

| Candidate | BLiMP | Δ vs softmax | wikitext PPL | n_params |
|---|---:|---:|---:|---:|
{blimp_rows}

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
    report_path.write_text(body, encoding="utf-8")
    log_line = (
        f"- {date}: phaseB={phase_b.get('elapsed_s', 0) / 60:.0f}m "
        f"tier2_survivors={len(survivors)} "
        f"best_blimp={best_blimp:.4f} "
        f"softmax_baseline={softmax_baseline:.4f} "
        f"beat_softmax={n_beat_softmax}\n"
    )
    _DAILY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _DAILY_LOG.open("a", encoding="utf-8") as fh:
        fh.write(log_line)
    if not quiet:
        print(f"[E] report -> {report_path}")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-hours", default=4.0, type=float)
    parser.add_argument("--tier2-top-k", default=8, type=int)
    parser.add_argument("--blimp-top-k", default=3, type=int)
    parser.add_argument("--blimp-train-steps", default=500, type=int)
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
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    total_minutes = int(args.budget_hours * 60)
    # 90-min cushion for Phases C-E (Tier-2 + BLiMP + report).
    phase_b_minutes = max(30, total_minutes - 90)

    phase_a = _phase_a_bootstrap(quiet=args.quiet)

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
        quiet=args.quiet,
    )
    phase_d = _phase_d_blimp(
        phase_c.get("survivors", []),
        args.blimp_top_k,
        args.blimp_train_steps,
        quiet=args.quiet,
    )
    report_path = _phase_e_report(phase_a, phase_b, phase_c, phase_d, quiet=args.quiet)
    if not args.quiet:
        print(f"\ndaily loop complete: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
