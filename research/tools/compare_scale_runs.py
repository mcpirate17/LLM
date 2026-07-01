"""Scale-run capability comparison table — novel-mechanism leaderboard.

Pulls the keeper scale-runs from ``research/runs.db`` (``scale_run_blimp`` +
``scale_run_probe_metrics``) and, optionally, a freshly-evaluated run from its
``*_post_eval.json`` + ``*_blimp.json``, then emits one markdown table ranked by
BLiMP with a **recursion-depth (r)** column.

The recursion column exists to answer a specific question: do the stronger novel
models win because they *recurse deeper* (Mixture-of-Recursions r4/r7) rather than
because the carrier is better? A single-pass mechanism (r1) that trails r7 on BLiMP
but matches it per-FLOP is a depth gap to CLOSE (give the mechanism recursion), not
a reason to abandon the mechanism.

Usage (keepers only):
    python -m research.tools.compare_scale_runs --out research/notes/scale_comparison.md

Usage (with a new run folded in):
    python -m research.tools.compare_scale_runs \
        --new-post-eval research/reports/frontier_probes/rdr_padic_46Mactive_seed0_post_eval.json \
        --new-blimp     research/reports/frontier_probes/rdr_padic_46Mactive_blimp.json \
        --new-label rdr_padic_46Mactive --new-active-m 42.7 --new-total-m 106.9 \
        --new-tokens-m 819.2 --new-recursion 1 \
        --out research/notes/scale_run_comparison_rdr_padic.md
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DB = _REPO / "research" / "runs.db"

# probe headline keys (the metrics of record for this project)
_AR_KEY = "ar_validation_held_pair_acc"  # THE AR-gate no-go metric
_BIND_KEY = "binding_intermediate_auc"
_IND_KEY = "induction_intermediate_auc"


def _recursion_depth(lane: str) -> int:
    """Effective recursion passes from a lane/mixer name. MoR lanes carry ``_rN_``.

    Single-pass mechanisms (no ``_rN`` token) are r1. The recursive_depth_router
    family is a width-mixture / single forward pass — r1, NOT iterative recursion.
    """
    m = re.search(r"_r(\d+)", lane or "")
    if m:
        return int(m.group(1))
    return 1


def _mechanism(lane: str) -> str:
    """Short human label for the carrier mechanism."""
    n = (lane or "").lower()
    table = [
        ("recursive_depth_router_padic", "p-adic gated mixer (novel)"),
        ("recursive_depth_router", "SSM+conv+softmax depth-router"),
        ("hyper_mor", "hyperbolic MoR + semiring"),
        ("mor_surprise", "MoR + semiring (surprise)"),
        ("mor_refine", "MoR + semiring"),
        ("native_semiring", "native semiring"),
        ("semiring_winner", "semiring (annealed)"),
        ("native_adaptive_reciprocal_slot_delta", "reciprocal slot-delta"),
        ("slot_table_mh_dplr", "slot table DPLR"),
        ("slot_table_mh", "slot table multi-head"),
        ("reciprocal_rank_attention", "reciprocal-rank attn [softmax-twin]"),
        ("pq_rope_winner", "product-quant RoPE"),
    ]
    for key, label in table:
        if key in n:
            return label
    return lane or "?"


def _keeper_rows(conn: sqlite3.Connection) -> list[dict]:
    """Best BLiMP row per run from scale_run_blimp, joined to probe headlines."""
    rows = []
    seen: dict[str, dict] = {}
    for run, lane, n_params_m, blimp, step in conn.execute(
        "select run_name, lane, n_params_m, blimp_overall, step from scale_run_blimp"
    ):
        # keep the best-BLiMP checkpoint per run_name
        if run in seen and seen[run]["blimp"] >= (blimp or 0):
            continue
        seen[run] = {
            "label": run,
            "lane": lane,
            "mechanism": _mechanism(lane),
            "recursion": _recursion_depth(lane),
            "total_m": round(n_params_m, 1) if n_params_m else None,
            "active_m": None,  # filled from leaderboard if available
            "blimp": round(blimp, 4) if blimp is not None else None,
            "step": step,
            "ar_held": None,
            "binding": None,
            "induction": None,
            "is_new": False,
        }
    # probe headlines (best value per run/metric)
    for run, mkey, val in conn.execute(
        "select run_name, metric_key, value_num from scale_run_probe_metrics "
        "where metric_key in (?,?,?)",
        (_AR_KEY, _BIND_KEY, _IND_KEY),
    ):
        if run not in seen or val is None:
            continue
        slot = {"ar_held": _AR_KEY, "binding": _BIND_KEY, "induction": _IND_KEY}
        for col, key in slot.items():
            if mkey == key:
                cur = seen[run][col]
                seen[run][col] = max(cur, val) if cur is not None else val
    # active_m from the normalized leaderboard (label prefix match, best effort)
    lb = list(conn.execute("select model, active_m from scale_run_leaderboard"))
    for r in seen.values():
        for model, active_m in lb:
            if model and (model in r["label"] or r["label"] in model):
                r["active_m"] = round(active_m, 1) if active_m else None
                break
    rows = list(seen.values())
    return rows


def _new_row(args) -> dict | None:
    if not args.new_post_eval:
        return None
    pe = json.loads(Path(args.new_post_eval).read_text())
    probes = pe.get("probes", {})

    def _g(probe, key):
        v = (probes.get(probe) or {}).get(key)
        return round(v, 4) if isinstance(v, (int, float)) else None

    blimp = None
    if args.new_blimp and Path(args.new_blimp).exists():
        bd = json.loads(Path(args.new_blimp).read_text())
        # blimp file is a list of per-ckpt dicts; take max blimp_overall
        cand = bd if isinstance(bd, list) else [bd]
        vals: list[float] = [
            float(c["blimp_overall"])
            for c in cand
            if isinstance(c, dict) and isinstance(c.get("blimp_overall"), (int, float))
        ]
        blimp = round(max(vals), 4) if vals else None
    lane = pe.get("mixer", args.new_label)
    return {
        "label": args.new_label,
        "lane": lane,
        "mechanism": _mechanism(lane),
        "recursion": args.new_recursion
        if args.new_recursion is not None
        else _recursion_depth(lane),
        "total_m": args.new_total_m or (round(pe.get("n_params", 0) / 1e6, 1) or None),
        "active_m": args.new_active_m,
        "blimp": blimp,
        "step": pe.get("checkpoint", "").split("step")[-1].split(".")[0] or None,
        "ar_held": _g("ar_validation", _AR_KEY),
        "binding": _g("binding_v2", _BIND_KEY),
        "induction": _g("induction_intermediate", _IND_KEY),
        "is_new": True,
    }


def _fmt(v, nd=3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _cap_avg(r: dict) -> float | None:
    vals = [r["blimp"], r["ar_held"], r["binding"], r["induction"]]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def build_table(args) -> str:
    conn = sqlite3.connect(str(_DB))
    rows = _keeper_rows(conn)
    new = _new_row(args)
    if new:
        # drop any keeper row already ingested under the same label (avoid dup)
        rows = [r for r in rows if r["label"] != new["label"]]
        rows.append(new)
    conn.close()
    for r in rows:
        r["cap_avg"] = _cap_avg(r)
    # rank by BLiMP (primary), then capability avg
    rows.sort(key=lambda r: (r["blimp"] or 0, r["cap_avg"] or 0), reverse=True)

    lines = []
    lines.append(
        "| # | model | mechanism | r | active_M | total_M | BLiMP | AR_held | binding | induction | cap_avg |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        star = " ⭐" if r["is_new"] else ""
        lines.append(
            f"| {i} | **{r['label']}**{star} | {r['mechanism']} | "
            f"{r['recursion']} | {_fmt(r['active_m'], 1)} | {_fmt(r['total_m'], 1)} | "
            f"{_fmt(r['blimp'])} | {_fmt(r['ar_held'])} | {_fmt(r['binding'])} | "
            f"{_fmt(r['induction'])} | {_fmt(r['cap_avg'])} |"
        )

    # recursion-vs-BLiMP trend (the hypothesis test)
    paired = [(r["recursion"], r["blimp"]) for r in rows if r["blimp"] is not None]
    trend = ""
    if len(paired) >= 3:
        rs = [p[0] for p in paired]
        bs = [p[1] for p in paired]
        mr = sum(rs) / len(rs)
        mb = sum(bs) / len(bs)
        cov = sum((a - mr) * (b - mb) for a, b in paired)
        vr = sum((a - mr) ** 2 for a in rs) ** 0.5
        vb = sum((b - mb) ** 2 for b in bs) ** 0.5
        rho = cov / (vr * vb) if vr and vb else 0.0
        trend = (
            f"\n**Recursion-depth vs BLiMP** (Pearson over {len(paired)} runs): "
            f"r = {rho:+.2f}. "
            + (
                "Positive — deeper recursion tracks higher BLiMP; a single-pass "
                "mechanism's gap is a *depth* gap to close (add recursion), not a "
                "carrier failure."
                if rho > 0.2
                else "Weak/none — depth alone does not explain the ranking."
            )
        )
    return "\n".join(lines) + "\n" + trend + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-post-eval")
    p.add_argument("--new-blimp")
    p.add_argument("--new-label", default="new_run")
    p.add_argument("--new-active-m", type=float, default=None)
    p.add_argument("--new-total-m", type=float, default=None)
    p.add_argument("--new-tokens-m", type=float, default=None)
    p.add_argument("--new-recursion", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    table = build_table(args)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
