"""Run the project's induction / binding / AR probes on the multi-token loss-monster archs.

User worry: several of these may actually form induction heads but can't show it at nano size
/ low steps. This runs the SAME capability probes existing runs.db rows use — induction
(induction_score → AUC), binding (nano_bind), AR (ar_gate held_pair + gMQAR AUDC/D50) — on the
loss-monster architectures, with a HIGHER step budget than the historical screening, to test
whether the capability is there but was hidden.

Note: these probes train the architecture ON the synthetic capability task (that's how the
project measures induction/binding/AR), so they measure architecture CAPACITY — directly the
"can it form induction heads" question. (The nano/bAbI multi-token checkpoints can't transfer
here: different vocab. So this is a fresh, higher-budget capacity probe, comparable to runs.db.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.scientist.native_runner import compile_model_native_first
from research.synthesis.serializer import graph_from_json
from research.eval.induction_probe import induction_score
from research.eval.ar_gate import ar_gate
from research.eval.nano_bind import nano_bind
from research.tools.loss_monster_screen import (
    _OUT_DIR,
    _RUNS_DB,
    select_family_champions,
)

_HIST = {  # historical runs.db screening (for comparison): induction_screening_auc
    "residual_block": 0.002,
    "parallel_split": 0.004,
    "recursive_depth_router": 0.008,
    "sparse_ffn": 0.004,
    "conditional_compute": 0.004,
}


def probe_one(family: str, graph_json: str, args) -> dict:
    rec: dict = {"family": family, "hist_induction_screen": _HIST.get(family)}
    # --- induction (live model, vocab >= 256 for the restricted induction vocab) ---
    try:
        model = compile_model_native_first(
            [graph_from_json(graph_json)] * args.n_layers,
            vocab_size=512,
            max_seq_len=256,
        ).to(args.device)
        ind = induction_score(
            model, n_train_steps=args.ind_steps, device=args.device, seed=0
        )
        rec["induction_auc"] = round(float(ind.auc), 4)
        rec["induction_status"] = ind.status
    except Exception as exc:  # loud per-probe
        rec["induction_auc"] = None
        rec["induction_err"] = f"{type(exc).__name__}: {exc}"
    # --- AR gate (held_pair) ---
    try:
        ar = ar_gate(graph_json=graph_json, device=args.device)
        rec["ar_held_pair"] = round(float(ar.held_pair_acc), 4)
        rec["ar_status"] = ar.status
    except Exception as exc:
        rec["ar_held_pair"] = None
        rec["ar_err"] = f"{type(exc).__name__}: {exc}"
    # --- binding (nano_bind) ---
    try:
        nb = nano_bind(graph_json, device=args.device)
        rec["binding_best"] = round(float(max(nb.scores) if nb.scores else 0.0), 4)
        rec["binding_is_no_go"] = bool(nb.is_no_go)
    except Exception as exc:
        rec["binding_best"] = None
        rec["binding_err"] = f"{type(exc).__name__}: {exc}"
    print(
        f"  {family:24s} induction_auc={rec.get('induction_auc')} "
        f"(hist {rec['hist_induction_screen']})  ar_held={rec.get('ar_held_pair')}  "
        f"binding_best={rec.get('binding_best')} no_go={rec.get('binding_is_no_go')}",
        flush=True,
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--families",
        nargs="*",
        default=[
            "residual_block",
            "parallel_split",
            "recursive_depth_router",
            "sparse_ffn",
        ],
    )
    ap.add_argument(
        "--ind-steps",
        type=int,
        default=3000,
        help="induction probe train steps (hist was 1000)",
    )
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "capability_probe_monsters.json"))
    args = ap.parse_args()

    champs = {c.family: c for c in select_family_champions(_RUNS_DB)}
    print(
        f"capability probes (induction {args.ind_steps} steps, ar_gate, nano_bind) on "
        f"{args.families}\n"
    )
    results = []
    for fam in args.families:
        if fam not in champs:
            print(f"  {fam}: no champion graph")
            continue
        results.append(probe_one(fam, champs[fam].graph_json, args))

    Path(args.out).write_text(
        json.dumps({"config": vars(args), "results": results}, indent=2)
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
