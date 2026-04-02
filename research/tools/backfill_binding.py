#!/usr/bin/env python3
"""Backfill binding probes (induction head + AR + binding range) for top leaderboard entries.

Reconstructs models from graph_json, runs all three probes, stores results
in program_results + leaderboard, and rescores.

Usage:
    python -m research.tools.backfill_binding [--top N] [--tier validation,investigation] [--dry-run] [--device cuda]
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.defaults import VOCAB_SIZE
from research.eval.associative_recall import associative_recall_score
from research.eval.binding_range import binding_range_profile
from research.eval.induction_probe import induction_score
from research.scientist.leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite,
    prefetch_program_results,
)
from research.scientist.notebook import LabNotebook
from research.scientist.thresholds import (
    BINDING_AR_SOFT_GATE,
    BINDING_BINDING_AUC_SOFT_GATE,
    BINDING_INDUCTION_SOFT_GATE,
)
from research.tools._backfill_shared import DB_PATH, reconstruct_model


def _micro_train(
    model,
    steps: int,
    device: str,
    seq_len: int = 128,
    batch_size: int = 8,
    lr: float = 3e-4,
):
    """Simulate screening micro-training so binding probes run on a trained model."""
    model.train()
    vs = model.vocab_size if hasattr(model, "vocab_size") else VOCAB_SIZE
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        data = torch.randint(0, vs, (batch_size, seq_len), device=device)
        logits = model(data)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vs), data[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()


def _run_probes(model, device: str):
    """Run all three binding probes. Returns (ar_result, ind_result, br_result)."""
    ar = associative_recall_score(
        model, n_pairs=20, n_eval=200, n_train_steps=500, batch_size=16, device=device
    )
    ind = induction_score(
        model,
        gaps=(4, 8, 16, 32, 64),
        n_train_steps=1000,
        n_eval=200,
        batch_size=32,
        device=device,
    )
    br = binding_range_profile(
        model, distances=(2, 4, 8, 16, 32, 64), n_eval=200, device=device
    )
    return ar, ind, br


def _compute_binding_composite(ar_auc: float, ind_auc: float, br_auc: float):
    """Compute binding composite and local_only flag."""
    bc = 0.4 * ar_auc + 0.3 * ind_auc + 0.3 * br_auc
    is_local = int(
        ar_auc < BINDING_AR_SOFT_GATE
        and ind_auc < BINDING_INDUCTION_SOFT_GATE
        and br_auc < BINDING_BINDING_AUC_SOFT_GATE
    )
    return round(bc, 4), is_local


def _store_results(
    nb,
    result_id: str,
    ar_auc: float,
    ind_auc: float,
    br_auc: float,
    bc: float,
    is_local: int,
):
    """Write probe results to program_results and leaderboard."""
    nb.conn.execute(
        "UPDATE program_results SET ar_auc=?, induction_auc=?, binding_auc=?, "
        "binding_composite=?, local_only=? WHERE result_id=?",
        (ar_auc, ind_auc, br_auc, bc, is_local, result_id),
    )
    nb.conn.execute(
        "UPDATE leaderboard SET ar_auc=?, induction_auc=?, binding_auc=?, "
        "binding_composite=?, local_only=? WHERE result_id=?",
        (ar_auc, ind_auc, br_auc, bc, is_local, result_id),
    )


def _rescore_entry(
    nb,
    entry_id: str,
    result_id: str,
    ar_auc: float,
    ind_auc: float,
    is_ref: bool,
    pr_cache: dict,
):
    """Recompute composite score with new binding data. Returns (new_score, old_score)."""
    existing = nb.conn.execute(
        "SELECT * FROM leaderboard WHERE entry_id=?", (entry_id,)
    ).fetchone()
    if not existing:
        return None, None
    d = dict(existing)
    old_score = float(d.get("composite_score") or 0)
    pr_dict = dict(pr_cache.get(result_id, {}))
    pr_dict["ar_auc"] = ar_auc
    pr_dict["induction_auc"] = ind_auc
    score_kw = build_score_kwargs_from_prefetch(pr_dict, d, is_ref)
    new_score = compute_composite(**score_kw)
    nb.conn.execute(
        "UPDATE leaderboard SET composite_score=? WHERE entry_id=?",
        (new_score, entry_id),
    )
    return new_score, old_score


def _query_candidates(nb, tiers: list[str], top: int, force: bool):
    """Query and filter leaderboard entries needing backfill."""
    tier_ph = ",".join("?" for _ in tiers)
    rows = nb.conn.execute(
        f"SELECT l.entry_id, l.result_id, l.tier, l.composite_score, "
        f"l.is_reference, l.model_source, "
        f"pr.graph_json, pr.induction_auc, pr.graph_fingerprint "
        f"FROM leaderboard l "
        f"LEFT JOIN program_results pr ON l.result_id = pr.result_id "
        f"WHERE l.tier IN ({tier_ph}) "
        f"ORDER BY l.composite_score DESC",
        tuple(tiers),
    ).fetchall()

    if not force:
        rows = [r for r in rows if r["induction_auc"] is None]

    # Limit per tier
    by_tier: dict[str, list] = {}
    for r in rows:
        t = r["tier"]
        tier_list = by_tier.setdefault(t, [])
        if len(tier_list) < top:
            tier_list.append(r)

    result = []
    for t in tiers:
        result.extend(by_tier.get(t, []))
    return result, by_tier


def main():
    parser = argparse.ArgumentParser(
        description="Backfill binding probes for top leaderboard entries"
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--tier",
        default="validation,investigation,breakthrough,screening,investigation_failed",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Re-evaluate even if data exists"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--train-steps", type=int, default=500)
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tier.split(",")]
    nb = LabNotebook(DB_PATH)
    rows, by_tier = _query_candidates(nb, tiers, args.top, args.force)

    total = len(rows)
    print(f"Entries to backfill: {total}  (device={args.device})")
    for t in tiers:
        n = len(by_tier.get(t, []))
        if n:
            print(f"  {t}: {n}")
    print()

    if total == 0:
        print("Nothing to backfill.")
        return

    if args.dry_run:
        for r in rows:
            fp = (r["graph_fingerprint"] or "")[:12]
            print(
                f"  [{fp}] tier={r['tier']} score={r['composite_score']:.1f} ref={bool(r['is_reference'])}"
            )
        print(f"\nDry run: would evaluate {total} entries.")
        return

    pr_cache = prefetch_program_results(nb.conn, [r["result_id"] for r in rows])
    evaluated, failed, no_graph, local_only_count = 0, 0, 0, 0
    t0 = time.time()

    for i, row in enumerate(rows):
        entry_id, result_id = row["entry_id"], row["result_id"]
        graph_json = row["graph_json"]
        fp = (row["graph_fingerprint"] or "")[:12]
        is_ref = bool(row["is_reference"])

        if not graph_json or graph_json == "{}":
            no_graph += 1
            print(f"  [{fp}] skip: no graph_json")
            continue

        try:
            model = reconstruct_model(graph_json, args.device)
            _micro_train(model, steps=args.train_steps, device=args.device)
            ar, ind, br = _run_probes(model, args.device)
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

            bc, is_local = _compute_binding_composite(ar.auc, ind.auc, br.auc)
            if is_local:
                local_only_count += 1

            _store_results(nb, result_id, ar.auc, ind.auc, br.auc, bc, is_local)
            new_score, old_score = _rescore_entry(
                nb, entry_id, result_id, ar.auc, ind.auc, is_ref, pr_cache
            )

            if new_score is not None:
                delta = new_score - old_score
                marker = " LOCAL" if is_local else ""
                print(
                    f"  [{fp}] ind={ind.auc:.3f} ar={ar.auc:.3f} bc={bc:.3f} "
                    f"score={old_score:.1f}->{new_score:.1f} ({delta:+.1f}){marker}"
                )
            evaluated += 1

        except (RuntimeError, KeyError, ValueError) as e:
            failed += 1
            print(f"  [{fp}] error: {e}")
            if args.device == "cuda":
                torch.cuda.empty_cache()

        if (i + 1) % 5 == 0:
            nb.conn.commit()
            print(f"  ... {i + 1}/{total} ({time.time() - t0:.0f}s)")

    nb.conn.commit()
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("BACKFILL COMPLETE")
    print(f"  Evaluated: {evaluated}, Failed: {failed}, No graph: {no_graph}")
    print(f"  Local-only (no binding): {local_only_count}/{evaluated}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / max(evaluated, 1):.1f}s/entry)")


if __name__ == "__main__":
    main()
