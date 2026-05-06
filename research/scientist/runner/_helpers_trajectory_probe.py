"""Validation trajectory probe helpers."""

from __future__ import annotations

import logging
import math
from typing import Any

from ._helpers_gate import clear_gpu_memory

logger = logging.getLogger(__name__)


def _compile_trajectory_model(*, graph_json_str: str, config, dev):
    from ...synthesis.serializer import graph_from_json
    from ..native_runner import compile_model_native_first as _compile

    traj_graph = graph_from_json(graph_json_str)
    traj_layers = [traj_graph] * config.n_layers
    traj_model = _compile(traj_layers, vocab_size=config.vocab_size, max_seq_len=128)
    return traj_model.to(dev)


def _run_wikitext_trajectory(traj_model, *, config, dev_str: str) -> dict[str, Any]:
    from ...eval.wikitext_eval import evaluate_wikitext_trajectory

    return evaluate_wikitext_trajectory(
        traj_model,
        config.vocab_size,
        dev_str,
        checkpoints=(200, 500, 1000, 2000, 4000),
        seq_len=128,
    )


def _run_validation_hellaswag(traj_model, *, config, dev_str: str) -> dict[str, Any]:
    try:
        from ...eval.hellaswag_eval import evaluate_hellaswag

        result = evaluate_hellaswag(
            traj_model, config.vocab_size, dev_str, n_examples=200
        )
        acc = result.get("hellaswag_acc")
        if acc is not None:
            logger.info(
                "Validation HellaSwag acc=%.1f%% (%d/%d, %.0fms)",
                acc * 100,
                result.get("hellaswag_correct", 0),
                result.get("hellaswag_total", 0),
                result.get("elapsed_ms", 0),
            )
        return result
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.warning("Validation HellaSwag eval skipped: %s", exc)
        return {}


def _run_validation_binding_probes(traj_model, *, dev_str: str) -> dict[str, Any]:
    try:
        from ...eval.binding_pipeline import (
            compute_binding_composite,
            compute_local_only,
            run_full_binding_probes,
        )

        probe = run_full_binding_probes(traj_model, device=dev_str)
        ar_auc = probe.ar_auc
        induction_auc = probe.induction_auc
        binding_auc = probe.binding_auc
        binding_composite = compute_binding_composite(
            ar_auc, induction_auc, binding_auc
        )
        logger.info(
            "Validation binding probes: ar=%.3f ind=%.3f bind=%.3f bc=%.3f local=%s (%.0f+%.0f+%.0fms)",
            ar_auc,
            induction_auc,
            binding_auc,
            binding_composite,
            bool(compute_local_only(ar_auc, induction_auc, binding_auc)),
            probe.ar_elapsed_ms,
            probe.induction_elapsed_ms,
            probe.binding_elapsed_ms,
        )
        return {
            "ar_auc": ar_auc,
            "induction_auc": induction_auc,
            "binding_auc": binding_auc,
            "binding_composite": binding_composite,
            "local_only": compute_local_only(ar_auc, induction_auc, binding_auc),
            "induction_metadata": probe.induction_metadata,
            "probe": probe,
        }
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.warning("Validation binding probes skipped: %s", exc)
        return {}


def _trajectory_metric_update(
    *,
    traj_result: dict[str, Any],
    hellaswag_result: dict[str, Any],
    binding_result: dict[str, Any],
    config,
) -> dict[str, Any]:
    update: dict[str, Any] = {}
    peak_ppl = traj_result.get("peak_ppl")
    if peak_ppl is not None:
        update["peak_ppl"] = peak_ppl
        vocab = config.vocab_size or 32000
        update["wikitext_score"] = round(
            max(0.0, math.log(vocab / peak_ppl) / math.log(vocab)),
            4,
        )
    if traj_result.get("peak_step") is not None:
        update["peak_step"] = traj_result["peak_step"]
    if traj_result.get("steps_to_divergence") is not None:
        update["steps_to_divergence"] = traj_result["steps_to_divergence"]

    checkpoints = traj_result.get("checkpoints", {})
    if 500 in checkpoints and checkpoints[500].get("ppl") is not None:
        update["ppl_500"] = checkpoints[500].get("ppl")

    update.update(_hellaswag_update_fields(hellaswag_result))
    update.update(_binding_update_fields(binding_result))
    return update


def _hellaswag_update_fields(result: dict[str, Any]) -> dict[str, Any]:
    update: dict[str, Any] = {}
    if result.get("hellaswag_acc") is not None:
        update["hellaswag_acc"] = result.get("hellaswag_acc")
    for key in (
        "hellaswag_metric_version",
        "hellaswag_tokenizer_mode",
        "hellaswag_tiktoken_encoding",
    ):
        if result.get(key) is not None:
            update[key] = result.get(key)
    return update


def _binding_update_fields(result: dict[str, Any]) -> dict[str, Any]:
    update: dict[str, Any] = {}
    probe = result.get("probe")
    if result.get("ar_auc") is not None:
        update["ar_auc"] = result.get("ar_auc")
        update["ar_final_acc"] = probe.ar_final_acc
        update["ar_timed_out"] = int(probe.ar_timed_out)
        update["ar_above_chance"] = int(probe.ar_above_chance)
    if result.get("induction_auc") is not None:
        update.update(
            result.get("induction_metadata")
            or {"induction_auc": result.get("induction_auc")}
        )
    if result.get("binding_auc") is not None:
        update.update(
            {
                "binding_auc": result.get("binding_auc"),
                "binding_distance_accuracies": probe.binding_distance_accuracies,
                "binding_probe_distances": [4, 8, 16, 32],
                "binding_probe_eval_examples": 200,
                "binding_probe_elapsed_ms": probe.binding_elapsed_ms,
                "binding_auc_curriculum": probe.binding_auc_curriculum,
                "binding_distance_accuracies_curriculum": (
                    probe.binding_distance_accuracies_curriculum
                ),
                "binding_probe_curriculum_steps": (
                    probe.binding_curriculum_train_steps
                ),
                "binding_probe_curriculum_elapsed_ms": (
                    probe.binding_curriculum_elapsed_ms
                ),
                "binding_probe_curriculum_protocol_version": "copy_curriculum_v1",
            }
        )
    if result.get("local_only") is not None:
        update["local_only"] = result.get("local_only")
        update["binding_composite"] = round(
            0.4 * (result.get("ar_auc") or 0)
            + 0.3 * (result.get("induction_auc") or 0)
            + 0.3 * (result.get("binding_auc") or 0),
            4,
        )
    return update


def _promote_trajectory_update(
    *,
    nb,
    source_result_id: str,
    tier: str,
    update: dict[str, Any],
) -> float | None:
    entry = nb.get_leaderboard_entry(source_result_id)
    if not entry or not update:
        return None

    nb.promote_to_tier(entry_id=entry["entry_id"], tier=tier, **update)
    row = nb.conn.execute(
        "SELECT composite_score FROM leaderboard WHERE entry_id = ?",
        (entry["entry_id"],),
    ).fetchone()
    return row["composite_score"] if row else None


def run_trajectory_probe(
    *,
    graph_json_str: str | None,
    config,  # RunConfig
    dev,  # torch.device
    dev_str: str,
    nb,
    source_result_id: str,
    tier: str,
    passed_seeds: list,
) -> float | None:
    """Run wikitext trajectory probe and update leaderboard.

    Returns trajectory_composite or None.
    """
    if not graph_json_str or len(passed_seeds) == 0:
        return None

    try:
        traj_model = _compile_trajectory_model(
            graph_json_str=graph_json_str,
            config=config,
            dev=dev,
        )
        try:
            traj_result = _run_wikitext_trajectory(
                traj_model,
                config=config,
                dev_str=dev_str,
            )
            hellaswag_result = _run_validation_hellaswag(
                traj_model,
                config=config,
                dev_str=dev_str,
            )
            binding_result = _run_validation_binding_probes(
                traj_model,
                dev_str=dev_str,
            )
        finally:
            del traj_model
            clear_gpu_memory()

        update = _trajectory_metric_update(
            traj_result=traj_result,
            hellaswag_result=hellaswag_result,
            binding_result=binding_result,
            config=config,
        )
        trajectory_composite = _promote_trajectory_update(
            nb=nb,
            source_result_id=source_result_id,
            tier=tier,
            update=update,
        )

        checkpoints = traj_result.get("checkpoints", {})
        ppl_500 = checkpoints[500].get("ppl") if 500 in checkpoints else None
        logger.info(
            "Trajectory probe %s: peak_ppl=%.1f steps_to_div=%s ppl_500=%s composite=%.1f",
            source_result_id[:8],
            traj_result.get("peak_ppl") or 0,
            traj_result.get("steps_to_divergence"),
            ppl_500,
            trajectory_composite or 0,
        )
        return trajectory_composite
    except Exception as exc:  # top-level error boundary: probe must not crash caller
        logger.warning("Trajectory probe failed for %s: %s", source_result_id[:8], exc)
        return None
