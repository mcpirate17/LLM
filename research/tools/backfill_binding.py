#!/usr/bin/env python3
"""Backfill binding-family metrics for S1-surviving leaderboard entries.

Defaults to backfilling `binding` only. Optional metrics such as `induction`
and `ar` must be explicitly requested.

Usage:
    python -m research.tools.backfill_binding [--metrics binding] [--top N] [--tier validation,investigation] [--dry-run] [--device cuda]
"""

import argparse
import json
import os
import sys
import time
from typing import Any

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.eval.binding_pipeline import (
    compute_binding_composite,
    compute_local_only,
    run_screening_binding_probes,
)
from research.scientist.leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite,
    prefetch_program_results,
)
from research.tools._backfill_shared import DB_PATH, reconstruct_model
from research.tools._script_audit import (
    build_metric_backfill_context,
    complete_script_experiment,
    fail_script_experiment,
    start_script_experiment,
)
from research.tools.backfill import micro_train, store_probe_results


_VALID_METRICS = ("binding", "induction", "ar")


def _parse_metrics(raw: str) -> tuple[str, ...]:
    metrics = tuple(metric.strip() for metric in raw.split(",") if metric.strip())
    if not metrics:
        raise ValueError("At least one metric must be requested")
    invalid = [metric for metric in metrics if metric not in _VALID_METRICS]
    if invalid:
        raise ValueError(
            f"Unsupported metrics: {invalid}. Valid metrics: {list(_VALID_METRICS)}"
        )
    # Preserve order but dedupe.
    return tuple(dict.fromkeys(metrics))


def _requested_metric_is_missing(row, metrics: tuple[str, ...]) -> bool:
    for metric in metrics:
        if metric == "binding" and row["binding_auc"] is None:
            return True
        if metric == "induction" and row["induction_auc"] is None:
            return True
        if metric == "ar" and row["ar_auc"] is None:
            return True
    return False


def _run_requested_probes(
    model, *, device: str, metrics: tuple[str, ...]
) -> dict[str, Any]:
    if metrics == ("binding",):
        return run_screening_binding_probes(model, device=device)
    result: dict[str, Any] = {}
    if "ar" in metrics:
        from research.eval.associative_recall import associative_recall_score

        ar = associative_recall_score(
            model,
            n_pairs=20,
            n_eval=200,
            n_train_steps=500,
            batch_size=16,
            device=device,
        )
        result.update(
            {
                "ar_auc": ar.auc,
                "ar_final_acc": ar.final_acc,
                "ar_timed_out": int(ar.timed_out),
                "ar_above_chance": int(ar.above_chance),
            }
        )
    if "induction" in metrics:
        from research.eval.native_induction import (
            induction_result_metadata,
            induction_score_gold,
        )

        ind = induction_score_gold(model, device=device)
        result.update(induction_result_metadata(ind))
    if "binding" in metrics:
        from research.eval.binding_curriculum import (
            CURRICULUM_BINDING_DISTANCES,
            CURRICULUM_BINDING_EVAL_FULL,
            CURRICULUM_BINDING_STEPS_FULL,
            curriculum_binding_range_profile,
        )

        br = curriculum_binding_range_profile(
            model,
            distances=CURRICULUM_BINDING_DISTANCES,
            n_train_steps=CURRICULUM_BINDING_STEPS_FULL,
            n_eval=CURRICULUM_BINDING_EVAL_FULL,
            device=device,
        )
        result.update(
            {
                "binding_auc": br.auc,
                "binding_distance_accuracies": br.distance_accuracies,
                "binding_probe_distances": list(CURRICULUM_BINDING_DISTANCES),
                "binding_probe_eval_examples": CURRICULUM_BINDING_EVAL_FULL,
                "binding_probe_elapsed_ms": br.elapsed_ms,
            }
        )
    return result


def _merged_binding_fields(
    nb, result_id: str, overrides: dict[str, Any]
) -> tuple[float | None, int | None]:
    row = nb.conn.execute(
        "SELECT ar_auc, induction_auc, binding_auc FROM program_results WHERE result_id=?",
        (result_id,),
    ).fetchone()
    merged = dict(row or {})
    merged.update(
        {
            "ar_auc": overrides.get("ar_auc", merged.get("ar_auc")),
            "induction_auc": overrides.get(
                "induction_auc", merged.get("induction_auc")
            ),
            "binding_auc": overrides.get("binding_auc", merged.get("binding_auc")),
        }
    )
    binding_auc = merged.get("binding_auc")
    induction_auc = merged.get("induction_auc")
    ar_auc = merged.get("ar_auc")
    if binding_auc is None or induction_auc is None:
        return None, None
    bc = compute_binding_composite(ar_auc, induction_auc, binding_auc)
    local_only = (
        compute_local_only(ar_auc, induction_auc, binding_auc)
        if ar_auc is not None
        else None
    )
    return bc, local_only


def _store_results(
    nb,
    result_id: str,
    updates: dict[str, Any],
    provenance_context: dict,
):
    """Write only requested probe results to program_results and leaderboard."""
    bc, is_local = _merged_binding_fields(nb, result_id, updates)
    row_updates = dict(updates)
    if "induction_gap_accuracies" in row_updates:
        row_updates["induction_gap_accuracies_json"] = json.dumps(
            row_updates.pop("induction_gap_accuracies"),
            sort_keys=True,
            separators=(",", ":"),
        )
    if "induction_probe_gaps" in row_updates:
        row_updates["induction_probe_gaps_json"] = json.dumps(
            row_updates.pop("induction_probe_gaps"),
            sort_keys=True,
            separators=(",", ":"),
        )
    if bc is not None:
        row_updates["binding_composite"] = bc
    if is_local is not None:
        row_updates["local_only"] = is_local
    store_probe_results(
        nb,
        result_id,
        row_updates,
        write_leaderboard=False,
        provenance_context=provenance_context,
    )
    leaderboard_updates = {
        key: value
        for key, value in (
            ("ar_auc", row_updates.get("ar_auc")),
            ("induction_auc", row_updates.get("induction_auc")),
            ("binding_auc", row_updates.get("binding_auc")),
            ("binding_composite", row_updates.get("binding_composite")),
            ("local_only", row_updates.get("local_only")),
        )
        if value is not None
    }
    if leaderboard_updates:
        assignments = ", ".join(f"{key}=?" for key in leaderboard_updates)
        nb.conn.execute(
            f"UPDATE leaderboard SET {assignments} WHERE result_id=?",
            tuple(leaderboard_updates.values()) + (result_id,),
        )
    if "induction_auc" in updates:
        fp_row = nb.conn.execute(
            "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        nb.upsert_induction_metric_v2(
            graph_fingerprint=str(fp_row["graph_fingerprint"] if fp_row else ""),
            result_id=str(result_id),
            row={
                "induction_auc": updates["induction_auc"],
                "induction_gap_accuracies": updates.get("induction_gap_accuracies", {}),
                "induction_probe_train_steps": updates.get(
                    "induction_probe_train_steps"
                ),
                "induction_probe_eval_examples": updates.get(
                    "induction_probe_eval_examples"
                ),
                "induction_probe_batch_size": updates.get("induction_probe_batch_size"),
                "induction_probe_gaps": updates.get("induction_probe_gaps", []),
                "induction_probe_elapsed_ms": updates.get("induction_probe_elapsed_ms"),
                "induction_probe_metric_version": updates.get(
                    "induction_probe_metric_version"
                ),
                "induction_probe_speed_mode": updates.get("induction_probe_speed_mode"),
                "induction_probe_pool_size": updates.get("induction_probe_pool_size"),
            },
            source_cohort="runtime_backfill",
        )


def _rescore_entry(
    nb,
    entry_id: str,
    result_id: str,
    overrides: dict[str, Any],
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
    for key in (
        "ar_auc",
        "induction_auc",
        "binding_auc",
        "binding_composite",
        "local_only",
    ):
        if key in overrides:
            pr_dict[key] = overrides[key]
    score_kw = build_score_kwargs_from_prefetch(pr_dict, d, is_ref)
    new_score = compute_composite(**score_kw)
    nb.conn.execute(
        "UPDATE leaderboard SET composite_score=? WHERE entry_id=?",
        (new_score, entry_id),
    )
    return new_score, old_score


def _query_candidates(
    nb, tiers: list[str], top: int, force: bool, metrics: tuple[str, ...]
):
    """Query and filter S1-surviving leaderboard entries needing backfill."""
    tier_ph = ",".join("?" for _ in tiers)
    rows = nb.conn.execute(
        f"SELECT l.entry_id, l.result_id, l.tier, l.composite_score, "
        f"l.is_reference, l.model_source, "
        f"pr.graph_json, pr.binding_auc, pr.induction_auc, pr.ar_auc, "
        f"pr.graph_fingerprint, pr.stage1_passed "
        f"FROM leaderboard l "
        f"LEFT JOIN program_results pr ON l.result_id = pr.result_id "
        f"WHERE l.tier IN ({tier_ph}) "
        f"AND COALESCE(pr.stage1_passed, 0) = 1 "
        f"ORDER BY l.composite_score DESC",
        tuple(tiers),
    ).fetchall()

    if not force:
        rows = [r for r in rows if _requested_metric_is_missing(r, metrics)]

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
    parser.add_argument(
        "--metrics",
        default="binding",
        help="Comma-separated metrics to backfill: binding, induction, ar. Default: binding",
    )
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tier.split(",")]
    metrics = _parse_metrics(args.metrics)
    nb, exp_id = start_script_experiment(
        db_path=DB_PATH,
        experiment_type="binding_backfill",
        config={
            "tiers": tiers,
            "top": args.top,
            "device": args.device,
            "force": bool(args.force),
            "train_steps": args.train_steps,
            "metrics": list(metrics),
        },
        source_script="backfill_binding",
        hypothesis="Backfill binding probes on leaderboard entries",
    )
    rows, by_tier = _query_candidates(nb, tiers, args.top, args.force, metrics)
    provenance_context = build_metric_backfill_context(
        kind="binding_backfill",
        source_script="backfill_binding",
        experiment_id=exp_id,
        device=args.device,
        tiers=tiers,
        top=args.top,
        force=bool(args.force),
        train_steps=args.train_steps,
        metrics=list(metrics),
    )

    total = len(rows)
    print(
        f"Entries to backfill: {total}  (device={args.device}, metrics={','.join(metrics)})"
    )
    for t in tiers:
        n = len(by_tier.get(t, []))
        if n:
            print(f"  {t}: {n}")
    print()

    if total == 0:
        print("Nothing to backfill.")
        complete_script_experiment(
            nb,
            exp_id,
            results={"total": 0, "evaluated": 0, "failed": 0, "no_graph": 0},
            summary="Binding backfill found no candidates",
        )
        nb.close()
        return

    if args.dry_run:
        for r in rows:
            fp = (r["graph_fingerprint"] or "")[:12]
            print(
                f"  [{fp}] tier={r['tier']} score={r['composite_score']:.1f} ref={bool(r['is_reference'])}"
            )
        print(f"\nDry run: would evaluate {total} entries.")
        fail_script_experiment(
            nb,
            exp_id,
            error="Dry-run invocation does not write results",
            results={"total": total, "evaluated": 0, "dry_run": True},
        )
        nb.close()
        return

    pr_cache = prefetch_program_results(nb.conn, [r["result_id"] for r in rows])
    evaluated, failed, no_graph, local_only_count = 0, 0, 0, 0
    t0 = time.time()

    try:
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
                micro_train(model, steps=args.train_steps, device=args.device)
                updates = _run_requested_probes(
                    model,
                    device=args.device,
                    metrics=metrics,
                )
                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()

                _store_results(
                    nb,
                    result_id,
                    updates,
                    provenance_context,
                )
                merged_bc, merged_local = _merged_binding_fields(nb, result_id, updates)
                rescore_overrides = dict(updates)
                if merged_bc is not None:
                    rescore_overrides["binding_composite"] = merged_bc
                if merged_local is not None:
                    rescore_overrides["local_only"] = merged_local
                if merged_local:
                    local_only_count += 1
                new_score, old_score = _rescore_entry(
                    nb,
                    entry_id,
                    result_id,
                    rescore_overrides,
                    is_ref,
                    pr_cache,
                )

                if new_score is not None:
                    delta = new_score - old_score
                    marker = " LOCAL" if merged_local else ""
                    ind_display = updates.get("induction_auc")
                    ar_display = updates.get("ar_auc")
                    bc_display = merged_bc
                    print(
                        f"  [{fp}] ind={ind_display if ind_display is not None else 'keep'} "
                        f"ar={ar_display if ar_display is not None else 'keep'} "
                        f"bc={bc_display if bc_display is not None else 'keep'} "
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
    except KeyboardInterrupt:
        fail_script_experiment(
            nb,
            exp_id,
            error="KeyboardInterrupt",
            results={
                "total": total,
                "evaluated": evaluated,
                "failed": failed,
                "no_graph": no_graph,
            },
        )
        nb.close()
        raise
    except Exception as exc:
        fail_script_experiment(
            nb,
            exp_id,
            error=str(exc),
            results={
                "total": total,
                "evaluated": evaluated,
                "failed": failed,
                "no_graph": no_graph,
            },
        )
        nb.close()
        raise

    nb.conn.commit()
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("BACKFILL COMPLETE")
    print(f"  Evaluated: {evaluated}, Failed: {failed}, No graph: {no_graph}")
    print(f"  Local-only (no binding): {local_only_count}/{evaluated}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / max(evaluated, 1):.1f}s/entry)")
    complete_script_experiment(
        nb,
        exp_id,
        results={
            "total": total,
            "evaluated": evaluated,
            "failed": failed,
            "no_graph": no_graph,
            "local_only": local_only_count,
            "elapsed_s": round(elapsed, 3),
        },
        summary=(
            f"Binding backfill: evaluated={evaluated} failed={failed} "
            f"local_only={local_only_count}"
        ),
    )
    nb.close()


if __name__ == "__main__":
    main()
