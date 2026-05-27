#!/usr/bin/env python
"""Full S1 (s1 + the 7 core post-S1 probes) on the TOP-N cheap-probe funnel winners.

Phase 5 of the shortlist cheap-probe funnel (see
``shortlist_cheap_probe_funnel.py``). Takes the funnel's ranked ``done`` graphs,
selects the top-N by the cheap signals (nano_induction_nearest primary; nb1.0 /
nb0.5 / ar_gate tiebreak), rebuilds each from its shortlist graph_json, and runs
the REAL stage-1 screening experiment -- reusing ``screen_template``'s
S0/S0.5/S1 + probe helpers. Records a COMPLETE stage1 write (the write-path
guard stays active, so a probe failure blocks the row rather than writing
partial data). The funnel already registered each fingerprint, so the S1 write
is an INTENTIONAL rerun (a second result row carrying the full screening).

Run from repo root, AFTER the funnel finishes:
  python -m research.tools.s1_top_shortlist \
      --results-json research/reports/shortlist_cheap_probe_results.json \
      --shortlist research/reports/cpu_cascade_million_shortlist_clean.jsonl \
      --top-n 10
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from research.scientist.notebook import LabNotebook
from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json
from research.tools._wikitext_batches import load_wikitext_batch_source
from research.tools.screen_template import _run_probes, _s0_s05_check, _train_loop
from research.training.loss_ops import next_token_cross_entropy

VOCAB_SIZE = 100277
RERUN_REASON = "full_s1_on_cheap_probe_winner"


def _rank_key(rec: dict[str, Any]) -> tuple:
    def f(x: Any) -> float:
        return float(x) if isinstance(x, (int, float)) else -1.0

    return (
        f(rec.get("nano_max")),
        f(rec.get("nb10")),
        f(rec.get("nb05")),
        f(rec.get("ar_gate_score")),
    )


def _top_fingerprints(results_json: Path, top_n: int) -> list[dict]:
    results = json.loads(results_json.read_text(encoding="utf-8"))
    gated = [r for r in results if r.get("verdict") == "done"]
    gated.sort(key=_rank_key, reverse=True)
    return gated[:top_n]


def _shortlist_graphs(shortlist: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with shortlist.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["fingerprint"])] = rec["graph"]
    return out


def _eval_val(model, batch_source, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    for vb in batch_source.iter_val_batches(device=device):
        with torch.no_grad():
            total += next_token_cross_entropy(model(vb), vb, VOCAB_SIZE).item()
            n += 1
    model.train()
    return total / max(n, 1)


def _record_full_s1(
    nb: LabNotebook,
    exp_id: str,
    fp: str,
    graph_json_str: str,
    metrics: dict[str, Any],
) -> str:
    """Record a complete stage1 row as an intentional rerun over the funnel fp."""
    rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint=fp,
        graph_json=graph_json_str,
        bypass_quality_gate=True,
        intentional_rerun_reason=RERUN_REASON,
        model_source="cpu_cascade_shortlist_s1",
        trust_label="shortlist_full_s1",
        **metrics,
    )
    nb.flush_writes()
    try:
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="cpu_cascade_shortlist_s1",
            architecture_desc=f"cascade-shortlist {fp[:12]}",
            tier="screening",
            tags=f"cascade_shortlist,cheap_probe_winner,{fp[:12]}",
            screening_passed=int(bool(metrics.get("stage1_passed"))),
            **{
                k: v
                for k, v in metrics.items()
                if k
                in (
                    "wikitext_perplexity",
                    "param_count",
                    "induction_screening_auc",
                    "binding_screening_auc",
                    "hellaswag_acc",
                    "blimp_overall_accuracy",
                )
                and v is not None
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  leaderboard upsert skipped: {type(exc).__name__}: {exc}", flush=True)
    return rid


def _run_one_s1(
    nb: LabNotebook,
    exp_id: str,
    fp: str,
    graph_dict: dict,
    args: argparse.Namespace,
) -> dict:
    gj = json.dumps(graph_dict)
    graphs = [graph_from_json(gj) for _ in range(args.layers)]
    model = compile_model(graphs, vocab_size=VOCAB_SIZE, max_seq_len=512).to(
        args.device
    )
    n_params = sum(p.numel() for p in model.parameters())
    out: dict[str, Any] = {"fingerprint": fp, "n_params": n_params}

    s0, s05 = _s0_s05_check(model, VOCAB_SIZE, args.device)
    if not (s0 and s05):
        out["verdict"] = "s0_s05_failed"
        out.update(stage0_passed=s0, stage05_passed=s05)
        del model
        torch.cuda.empty_cache()
        return out

    batch_source = load_wikitext_batch_source(
        batch_size=args.batch_size, seq_len=args.seq_len, vocab_size=VOCAB_SIZE
    )
    pre_val = _eval_val(model, batch_source, args.device)
    _, elapsed = _train_loop(
        model,
        batch_source,
        n_steps=args.screening_steps,
        lr=args.lr,
        lr_warmup=args.lr,
        vocab_size=VOCAB_SIZE,
        device=args.device,
        seed=args.seed,
        label="S1",
        eval_fn=lambda m=model: _eval_val(m, batch_source, args.device),
    )
    s_val = _eval_val(model, batch_source, args.device)
    loss_ratio = s_val / pre_val if pre_val > 0 else 1.0
    ppl = math.exp(min(s_val, 20))
    s1_passed = loss_ratio < 0.95
    model.eval()
    probes = _run_probes(model, VOCAB_SIZE, args.device)
    del model
    torch.cuda.empty_cache()

    metrics: dict[str, Any] = {
        "stage0_passed": True,
        "stage05_passed": True,
        "stage1_passed": s1_passed,
        "loss_ratio": loss_ratio,
        "final_loss": s_val,
        "initial_loss": pre_val,
        "param_count": n_params,
        "n_train_steps": args.screening_steps,
        "wikitext_perplexity": ppl,
        **{k: v for k, v in probes.items() if v is not None},
    }
    out.update(
        verdict="recorded" if s1_passed else "s1_failed",
        stage1_passed=s1_passed,
        wikitext_perplexity=round(ppl, 2),
        loss_ratio=round(loss_ratio, 4),
        elapsed_s=round(elapsed, 1),
        **{
            k: probes.get(k)
            for k in (
                "induction_screening_auc",
                "binding_screening_auc",
                "binding_screening_composite",
                "ar_legacy_auc",
                "hellaswag_acc",
                "blimp_overall_accuracy",
            )
        },
    )
    # Only claim stage1 (and write the completeness-guarded row) when the
    # probes are all present; otherwise record a stage1_passed=False row so the
    # guard never sees a partial S1-pass write.
    required = (
        "wikitext_perplexity",
        "hellaswag_acc",
        "blimp_overall_accuracy",
        "induction_screening_auc",
        "binding_screening_auc",
        "binding_screening_composite",
        "ar_legacy_auc",
    )
    if s1_passed and any(metrics.get(c) is None for c in required):
        metrics["stage1_passed"] = False
        out["verdict"] = "s1_pass_but_probe_incomplete"
    out["result_id"] = _record_full_s1(nb, exp_id, fp, gj, metrics)
    return out


def _emit_report(rows: list[dict], funnel_rows: list[dict], out_path: Path) -> None:
    fmap = {r["fingerprint"]: r for r in funnel_rows}
    lines = [
        "# Full-S1 results for cheap-probe funnel top-N",
        "",
        "Cheap probes (funnel) vs full S1. `lf_*` = cascade label-free prediction.",
        "",
        "| fp | nano | nb1.0 | ar_gate | lf_rec | S1 ppl | s1_pass | ind_auc | bind_auc | hella | blimp | ar_legacy |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        f = fmap.get(r["fingerprint"], {})
        lines.append(
            f"| {r['fingerprint']} | {f.get('nano_max')} | {f.get('nb10')} | "
            f"{f.get('ar_gate_score')} | {f.get('label_free_probe_recommendation')} | "
            f"{r.get('wikitext_perplexity')} | {r.get('stage1_passed')} | "
            f"{r.get('induction_screening_auc')} | {r.get('binding_screening_auc')} | "
            f"{r.get('hellaswag_acc')} | {r.get('blimp_overall_accuracy')} | "
            f"{r.get('ar_legacy_auc')} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-json",
        type=Path,
        default=Path("research/reports/shortlist_cheap_probe_results.json"),
    )
    ap.add_argument(
        "--shortlist",
        type=Path,
        default=Path("research/reports/cpu_cascade_million_shortlist_clean.jsonl"),
    )
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=256)  # informational; graph carries dim
    ap.add_argument("--screening-steps", type=int, default=750)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("research/reports/shortlist_full_s1_top.md"),
    )
    args = ap.parse_args()

    top = _top_fingerprints(args.results_json, args.top_n)
    graphs = _shortlist_graphs(args.shortlist)
    print(
        f"full S1 on top {len(top)} funnel winners (layers={args.layers})", flush=True
    )

    nb = LabNotebook()
    exp_id = nb.start_experiment(
        experiment_type="shortlist_full_s1",
        config={
            "results_json": str(args.results_json),
            "top_n": int(args.top_n),
            "layers": int(args.layers),
            "screening_steps": int(args.screening_steps),
            "source_script": "s1_top_shortlist",
        },
        hypothesis="Full S1 on the cheap-probe funnel's top winners.",
    )
    rows: list[dict] = []
    try:
        for i, frec in enumerate(top, 1):
            fp = frec["fingerprint"]
            if fp not in graphs:
                print(f"[{i}/{len(top)}] {fp} MISSING from shortlist; skip", flush=True)
                continue
            try:
                row = _run_one_s1(nb, exp_id, fp, graphs[fp], args)
            except Exception as exc:  # noqa: BLE001
                row = {
                    "fingerprint": fp,
                    "verdict": f"error:{type(exc).__name__}:{exc}",
                }
                torch.cuda.empty_cache()
            rows.append(row)
            _emit_report(rows, top, args.out)
            print(
                f"[{i}/{len(top)}] {fp} ppl={row.get('wikitext_perplexity')} "
                f"s1={row.get('stage1_passed')} ind={row.get('induction_screening_auc')} "
                f"{row.get('verdict')}",
                flush=True,
            )
        nb.complete_experiment(
            exp_id,
            results={
                # "total" is the program count complete_experiment gates on;
                # omitting it marks a fully-evaluated shortlist run as failed.
                "total": len(rows),
                "evaluated": len(rows),
                "s1_passed": sum(1 for r in rows if r.get("stage1_passed")),
            },
            aria_summary=f"full S1 on {len(rows)} shortlist winners -> {args.out.name}",
        )
    except Exception as exc:
        nb.fail_experiment(exp_id, error=str(exc))
        raise
    finally:
        nb.close()
    print(f"\nDONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
