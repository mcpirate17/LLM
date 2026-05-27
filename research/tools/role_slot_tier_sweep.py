#!/usr/bin/env python3
"""Breadth-first tier sweep for role-slot (neural_symbolic_retrieval_v2) fingerprints.

Goal: populate the leak-immune capability metrics
(induction_intermediate_auc, ar_curriculum_auc_pair_final,
binding_intermediate_auc) for the role-slot / neuro-symbolic template family so
that component can contribute to the arch-component statistical analysis. Those
fingerprints currently have zero coverage on those columns.

Reuses the proven queue/drain/poll primitives in
``research.tools.tier_orchestrator`` (no duplication). Unlike the orchestrator,
which runs s1->inv->cap->val DEEP per fingerprint, this sweep is breadth-first:

  Phase A: screening replay + investigation across ALL targets (best-first).
           Screening creates the candidate-confirmed S1 sibling the
           investigation rerun gate requires for backfill-provenance rows.
  Phase B: capability_ranking on the strongest investigated fingerprints.
  Phase C: validation on at most one clear winner (very picky).

Every phase is bounded by a hard wall-clock deadline so the session ends in
time for findings + shutdown. Per-stage durations are measured from the first
runs and used to decide whether the next item fits before the deadline.

Usage:
    python -m research.tools.role_slot_tier_sweep --deadline-epoch 1779596520 \
        --results-json research/reports/role_slot_session/sweep_results.json \
        --result-id da13664f-57c --result-id f2a96557-eac ...
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from research.tools import tier_orchestrator as O

DB_PATH = O.DB_PATH

# Conservative initial per-stage reserves (seconds); replaced by measured maxima.
_INIT_SCREEN_S = 240.0
_INIT_INV_S = 720.0
_INIT_CAP_S = 720.0
_INIT_VAL_S = 1800.0


def latest_capability(fp: str) -> dict:
    """Newest leak-immune capability metrics for a fingerprint."""
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row
        r = c.execute(
            "SELECT result_id, timestamp, stage1_passed, loss_ratio, "
            "induction_intermediate_auc AS ind, ar_curriculum_auc_pair_final AS arc, "
            "binding_intermediate_auc AS bind, ar_validation_rank_score AS arval "
            "FROM program_results_compat WHERE graph_fingerprint = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (fp,),
        ).fetchone()
    return dict(r) if r else {}


def cap_signal(m: dict) -> float:
    """Single scalar to rank investigated fingerprints for promotion.

    Induction is the cleanest retrieval signal; ar_curriculum is the
    highest-confidence dosage lever per the 05-24 extensions note. Weight both,
    add a small binding term.
    """
    ind = m.get("ind") or 0.0
    arc = m.get("arc") or 0.0
    bind = m.get("bind") or 0.0
    return 0.5 * ind + 0.4 * arc + 0.1 * bind


def _save(out_path: Path, state: dict) -> None:
    out_path.write_text(json.dumps(state, indent=2, default=str))


def phase_a(state: dict, out_path: Path, deadline: float) -> tuple[float, float]:
    """Screening replay + investigation across all targets, best-first."""
    screen_max, inv_max = _INIT_SCREEN_S, _INIT_INV_S
    targets = state["targets"]
    O.log(
        f"[sweep] Phase A start: {len(targets)} targets, "
        f"{deadline - time.time():.0f}s to deadline"
    )
    for i, rid in enumerate(targets, 1):
        left = deadline - time.time()
        need = screen_max + inv_max + 60.0
        if left < need:
            O.log(
                f"[sweep] deadline guard: {left:.0f}s < {need:.0f}s; stop Phase A "
                f"before target {i}/{len(targets)} ({rid})"
            )
            break
        fp = O.get_fp_for_result(rid)
        if not fp:
            state["phase_a"][rid] = {"ok": False, "reason": "not_in_db"}
            _save(out_path, state)
            continue
        O.log(
            f"[sweep] --- target {i}/{len(targets)} rid={rid} fp={fp[:16]} "
            f"({left:.0f}s left) ---"
        )

        s = O.run_tier(rid, "screening", fp)
        if s.get("ok") and s.get("elapsed_s"):
            screen_max = max(screen_max, float(s["elapsed_s"]))
        entry = {"fp": fp, "screening": s}

        if s.get("ok") and (deadline - time.time()) > (inv_max + 60.0):
            inv = O.run_tier(rid, "investigation", fp)
            if inv.get("ok") and inv.get("elapsed_s"):
                inv_max = max(inv_max, float(inv["elapsed_s"]))
            entry["investigation"] = inv
            entry["capability"] = latest_capability(fp)
            entry["cap_signal"] = cap_signal(entry["capability"])
        else:
            entry["investigation"] = {"ok": False, "reason": "skipped_deadline_or_s1"}
        state["phase_a"][rid] = entry
        _save(out_path, state)
    return screen_max, inv_max


def phase_b(state: dict, out_path: Path, deadline: float, cap_top_k: int) -> None:
    """capability_ranking on the strongest investigated fingerprints."""
    ranked = sorted(
        (
            (rid, e)
            for rid, e in state["phase_a"].items()
            if e.get("investigation", {}).get("ok")
        ),
        key=lambda kv: kv[1].get("cap_signal", 0.0),
        reverse=True,
    )
    O.log(
        f"[sweep] Phase B: {len(ranked)} investigated; cap top-k={cap_top_k}; "
        f"{deadline - time.time():.0f}s left"
    )
    cap_max = _INIT_CAP_S
    for rid, e in ranked[:cap_top_k]:
        if (deadline - time.time()) < (cap_max + 60.0):
            O.log(f"[sweep] deadline guard: stop Phase B before {rid}")
            break
        fp = e["fp"]
        c = O.run_tier(rid, "capability_ranking", fp)
        if c.get("ok") and c.get("elapsed_s"):
            cap_max = max(cap_max, float(c["elapsed_s"]))
        state["phase_b"][rid] = {
            "fp": fp,
            "capability_ranking": c,
            "capability": latest_capability(fp),
        }
        _save(out_path, state)


def phase_c(
    state: dict, out_path: Path, deadline: float, validate_top_k: int, min_signal: float
) -> None:
    """validation on at most ``validate_top_k`` clear-signal winners."""
    candidates = sorted(
        (
            (rid, e)
            for rid, e in state["phase_a"].items()
            if e.get("cap_signal", 0.0) >= min_signal
        ),
        key=lambda kv: kv[1].get("cap_signal", 0.0),
        reverse=True,
    )
    O.log(
        f"[sweep] Phase C: {len(candidates)} clear-signal candidates "
        f"(>= {min_signal}); {deadline - time.time():.0f}s left"
    )
    for rid, e in candidates[:validate_top_k]:
        if (deadline - time.time()) < (_INIT_VAL_S + 60.0):
            O.log(f"[sweep] deadline guard: no room for validation; skip {rid}")
            break
        fp = e["fp"]
        v = O.run_tier(rid, "validation", fp)
        state["phase_c"][rid] = {
            "fp": fp,
            "validation": v,
            "capability": latest_capability(fp),
        }
        _save(out_path, state)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-epoch", type=float, required=True)
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--result-id", action="append", dest="result_ids", default=[])
    ap.add_argument("--cap-top-k", type=int, default=3)
    ap.add_argument("--validate-top-k", type=int, default=1)
    ap.add_argument(
        "--validate-min-signal",
        type=float,
        default=0.30,
        help="only validate a fingerprint whose cap_signal clears this",
    )
    args = ap.parse_args()

    deadline = args.deadline_epoch
    out_path = Path(args.results_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state: dict = {
        "started_at": time.time(),
        "deadline_epoch": deadline,
        "targets": args.result_ids,
        "phase_a": {},
        "phase_b": {},
        "phase_c": {},
    }
    _save(out_path, state)

    phase_a(state, out_path, deadline)
    phase_b(state, out_path, deadline, args.cap_top_k)
    phase_c(state, out_path, deadline, args.validate_top_k, args.validate_min_signal)

    state["finished_at"] = time.time()
    _save(out_path, state)
    O.log(f"[sweep] complete; results at {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
