from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from flask import jsonify

from research.defaults import (
    VALIDATION_BATCH_SIZE,
    VALIDATION_SEQ_LEN,
    VALIDATION_STEPS,
)

from ..capability_ranker_metrics import enable_capability_rankers
from ..runner._types import RunConfig
from ..runtime_events import publish_lifecycle_event
from ._helpers import (
    _BATCH_RERUN_STATE,
    get_runner,
    normalize_result_ids,
    record_run_trigger,
    reset_runner_launch_state,
)
from ._strategy_preflight import (
    apply_compact_synthesis_bias,
    apply_live_screening_bias,
    apply_sparse_morph_bias,
    build_start_mode_eligibility,
    extract_hypothesis_missing_fields,
    normalize_start_mode,
    resolve_scale_up_result_ids,
)
from .deps import get_notebook

logger = logging.getLogger(__name__)


@dataclass
class StartExperimentRequest:
    body: Dict[str, Any]
    mode: str
    config: RunConfig
    auto_harden: bool
    preflight_override: bool
    enforce_preflight: bool
    preflight_sample_n: int
    hypothesis: Any
    preregistration: Any
    exploratory: bool
    compact_changes: Dict[str, Any]
    live_screening_changes: Dict[str, Any]
    sparse_morph_changes: Dict[str, Any]
    intermediate_probe_changes: Dict[str, Any]


@dataclass(frozen=True)
class LaunchLifecycleContext:
    launch_id: str


def create_launch_lifecycle_context() -> LaunchLifecycleContext:
    return LaunchLifecycleContext(launch_id=f"launch-{uuid.uuid4().hex}")


def publish_launch_requested(
    start: StartExperimentRequest,
    *,
    notebook_path: str,
    context: LaunchLifecycleContext,
) -> None:
    publish_lifecycle_event(
        notebook_path=notebook_path,
        event_type="experiment_start_requested",
        producer="api.experiment_launch",
        run_id=context.launch_id,
        payload={
            "mode": start.mode,
            "config": start.config.to_dict(),
            "hypothesis": start.hypothesis,
            "exploratory": start.exploratory,
        },
        sequence=1,
    )


def publish_launch_failed(
    error: Exception,
    *,
    mode: str,
    notebook_path: str,
    context: Optional[LaunchLifecycleContext],
) -> None:
    if context is None:
        return
    try:
        publish_lifecycle_event(
            notebook_path=notebook_path,
            event_type="experiment_start_failed",
            producer="api.experiment_launch",
            run_id=context.launch_id,
            payload={
                "mode": mode,
                "error": str(error),
            },
            sequence=2,
        )
    except Exception:
        logger.warning(
            "Runtime lifecycle publish failed for launch failure %s",
            context.launch_id,
            exc_info=True,
        )


def parse_start_request(body: Dict[str, Any]) -> StartExperimentRequest:
    body = dict(body)
    auto_harden = bool(body.pop("auto_harden", True))
    preflight_override = bool(body.pop("preflight_override", False))
    enforce_preflight = bool(body.pop("enforce_preflight", True))
    preflight_sample_n = int(body.pop("preflight_sample_n", 4) or 4)
    hypothesis = body.pop("hypothesis", None)
    preregistration = body.pop("preregistration", None)
    exploratory = bool(body.pop("exploratory", False))
    enable_intermediate_probes = bool(body.pop("enable_intermediate_probes", False))
    refine_analysis_json = body.pop("refine_analysis_json", "")
    mode = normalize_start_mode(body.pop("mode", "single"))
    if mode == "confirmation":
        body["mode"] = "confirmation"
        body.setdefault("scale_up_steps", VALIDATION_STEPS * 4)
        body.setdefault("scale_up_batch_size", VALIDATION_BATCH_SIZE)
        body.setdefault("scale_up_seq_len", VALIDATION_SEQ_LEN)
        scale_steps = int(body.get("scale_up_steps") or (VALIDATION_STEPS * 4))
        body["early_stop_min_steps"] = max(
            int(body.get("early_stop_min_steps") or 0), scale_steps + 1
        )
        body["early_stop_patience"] = max(
            int(body.get("early_stop_patience") or 0), scale_steps + 1
        )
        body["phase_checkpoint_step_interval"] = 10_000
    config = RunConfig.from_dict(body) if body else RunConfig()
    if refine_analysis_json:
        config.refine_analysis_json = (
            refine_analysis_json
            if isinstance(refine_analysis_json, str)
            else json.dumps(refine_analysis_json)
        )
    compact_changes: Dict[str, Any] = {}
    live_screening_changes: Dict[str, Any] = {}
    sparse_morph_changes: Dict[str, Any] = {}
    intermediate_probe_changes: Dict[str, Any] = {}
    if mode == "live_screening":
        live_screening_changes = apply_live_screening_bias(config)
        mode = "single"
    if mode == "compact_synthesis":
        compact_changes = apply_compact_synthesis_bias(config)
        mode = "single"
    if mode == "sparse_morph":
        sparse_morph_changes = apply_sparse_morph_bias(config)
        mode = "single"
    if enable_intermediate_probes:
        if not getattr(config, "run_ar_intermediate", False):
            config.run_ar_intermediate = True
            intermediate_probe_changes["run_ar_intermediate"] = True
        if not getattr(config, "run_binding_multislot", False):
            config.run_binding_multislot = True
            intermediate_probe_changes["run_binding_multislot"] = True
    return StartExperimentRequest(
        body=body,
        mode=mode,
        config=config,
        auto_harden=auto_harden,
        preflight_override=preflight_override,
        enforce_preflight=enforce_preflight,
        preflight_sample_n=preflight_sample_n,
        hypothesis=hypothesis,
        preregistration=preregistration,
        exploratory=exploratory,
        compact_changes=compact_changes,
        live_screening_changes=live_screening_changes,
        sparse_morph_changes=sparse_morph_changes,
        intermediate_probe_changes=intermediate_probe_changes,
    )


def maybe_block_preflight(
    start: StartExperimentRequest, prescreen: Dict[str, Any], preflight: Dict[str, Any]
):
    if not (
        start.enforce_preflight
        and preflight.get("verdict") in {"warn", "fail"}
        and not start.preflight_override
    ):
        return None
    return (
        jsonify(
            {
                "error": (
                    "Preflight gate blocked launch."
                    if preflight.get("verdict") == "fail"
                    else "Preflight produced warnings; override required to start."
                ),
                "preflight_blocked": True,
                "preflight": preflight,
                "config": start.config.to_dict(),
                "prescreen": prescreen,
            }
        ),
        409,
    )


def _require_result_ids(start: StartExperimentRequest, mode: str):
    result_ids = normalize_result_ids(start.body.get("result_ids", []))
    if result_ids:
        return result_ids, None
    return None, (jsonify({"error": f"result_ids required for {mode} mode"}), 400)


def _maybe_block_ineligible(nb, mode: str, result_ids: list[str], force: bool):
    if force:
        return None, None
    eligibility = build_start_mode_eligibility(nb, mode, result_ids)
    if eligibility.get("all_eligible"):
        return eligibility, None
    return eligibility, (
        jsonify(
            {
                "error": f"Ineligible result_ids for {mode} mode",
                "eligibility": eligibility,
            }
        ),
        409,
    )


def _launch_simple_mode(start: StartExperimentRequest, *, runner):
    mode = start.mode
    config = start.config
    if mode == "continuous":
        config.continuous = True
        return runner.start_continuous(config)
    if mode == "evolve":
        return runner.start_evolution(
            config,
            hypothesis=start.hypothesis,
            preregistration=start.preregistration,
            exploratory=start.exploratory,
        )
    if mode == "novelty":
        return runner.start_novelty_search(
            config,
            hypothesis=start.hypothesis,
            preregistration=start.preregistration,
            exploratory=start.exploratory,
        )
    return None


def _enable_investigation_deep_probes(config: RunConfig) -> None:
    """Ensure investigation rows carry real post-S1 probe evidence."""
    config.investigation_run_capability_rankers = True
    config.run_ar_intermediate = True
    config.run_binding_multislot = True
    enable_capability_rankers(config)


def _launch_result_id_mode(start: StartExperimentRequest, *, nb, runner):
    mode = start.mode
    config = start.config
    if mode == "investigation":
        result_ids, error = _require_result_ids(start, mode)
        if error:
            return None, None, error
        force = bool(start.body.get("force") or start.body.get("force_reinvestigate"))
        eligibility, error = _maybe_block_ineligible(nb, mode, result_ids, force)
        if error:
            return None, eligibility, error
        _enable_investigation_deep_probes(config)
        exp_id = runner.start_investigation(
            result_ids,
            config,
            hypothesis=start.hypothesis,
            preregistration=start.preregistration,
            exploratory=start.exploratory,
            force=force,
        )
        return exp_id, eligibility, None
    if mode == "capability_ranking":
        result_ids, error = _require_result_ids(start, mode)
        if error:
            return None, None, error
        force = bool(
            start.body.get("force")
            or start.body.get("force_capability_ranking")
            or start.body.get("force_override")
            or start.body.get("allow_ineligible")
            or start.body.get("override_ineligible")
        )
        eligibility, error = _maybe_block_ineligible(nb, mode, result_ids, force)
        if error:
            return None, eligibility, error
        exp_id = runner.start_capability_ranking(
            result_ids,
            config,
            hypothesis=start.hypothesis,
            preregistration=start.preregistration,
            exploratory=start.exploratory,
            force=force,
        )
        return exp_id, eligibility, None
    if mode == "confirmation":
        result_ids, error = _require_result_ids(start, mode)
        if error:
            return None, None, error
        force = bool(
            start.body.get("force")
            or start.body.get("force_confirmation")
            or start.body.get("force_override")
            or start.body.get("allow_ineligible")
            or start.body.get("override_ineligible")
        )
        eligibility, error = _maybe_block_ineligible(nb, mode, result_ids, force)
        if error:
            return None, eligibility, error
        config.scale_up = True
        config.scale_up_result_ids = ",".join(result_ids)
        hypothesis = start.hypothesis or (
            f"Champion confirmation: post-validation scale training for "
            f"{len(result_ids)} candidate(s) at 4x validation steps."
        )
        exp_id = runner.start_scale_up(
            result_ids,
            config,
            hypothesis=hypothesis,
            preregistration=start.preregistration,
            exploratory=start.exploratory,
            workflow_mode="confirmation",
        )
        return exp_id, eligibility, None
    if mode != "validation":
        return None, None, None
    result_ids, error = _require_result_ids(start, mode)
    if error:
        return None, None, error
    force = bool(
        start.body.get("force")
        or start.body.get("force_validation")
        or start.body.get("force_override")
        or start.body.get("allow_ineligible")
        or start.body.get("override_ineligible")
    )
    eligibility, error = _maybe_block_ineligible(nb, mode, result_ids, force)
    if error:
        return None, eligibility, error
    if eligibility and eligibility.get("eligible_result_ids"):
        result_ids = list(eligibility["eligible_result_ids"])
    exp_id = runner.start_validation(
        result_ids,
        config,
        hypothesis=start.hypothesis,
        preregistration=start.preregistration,
        exploratory=start.exploratory,
        force=force,
    )
    return exp_id, eligibility, None


def _launch_fingerprint_mode(start: StartExperimentRequest, *, nb, runner):
    mode = start.mode
    config = start.config
    result_ids = normalize_result_ids(start.body.get("result_ids", []))
    graph_fingerprints = normalize_result_ids(
        start.body.get("graph_fingerprints", start.body.get("fingerprints", []))
    )
    resolution = resolve_scale_up_result_ids(
        nb,
        result_ids=result_ids,
        graph_fingerprints=graph_fingerprints,
    )
    result_ids = resolution.get("result_ids", [])
    if mode == "scale_up":
        if not result_ids:
            return (
                None,
                resolution,
                (
                    jsonify(
                        {
                            "error": "result_ids or graph_fingerprints required for scale_up mode",
                            "scale_up_resolution": resolution,
                        }
                    ),
                    400,
                ),
            )
        config.scale_up = True
        config.scale_up_result_ids = ",".join(result_ids)
        return (
            runner.start_scale_up(
                result_ids,
                config,
                hypothesis=start.hypothesis,
                preregistration=start.preregistration,
                exploratory=start.exploratory,
            ),
            resolution,
            None,
        )
    if mode != "refine_fingerprint":
        return None, None, None
    if not result_ids:
        return (
            None,
            resolution,
            (
                jsonify(
                    {
                        "error": "result_ids or graph_fingerprints required for refine_fingerprint mode",
                        "refine_resolution": resolution,
                    }
                ),
                400,
            ),
        )
    return (
        runner.start_fingerprint_refinement(
            result_ids,
            config,
            hypothesis=start.hypothesis,
        ),
        resolution,
        None,
    )


def launch_experiment_mode(start: StartExperimentRequest, *, nb, runner):
    config = start.config
    simple_exp_id = _launch_simple_mode(start, runner=runner)
    if simple_exp_id is not None:
        return simple_exp_id, None, None, None, None

    if start.mode in {
        "investigation",
        "capability_ranking",
        "validation",
        "confirmation",
    }:
        exp_id, eligibility, error = _launch_result_id_mode(start, nb=nb, runner=runner)
        return exp_id, eligibility, None, None, error

    if start.mode in {"scale_up", "refine_fingerprint"}:
        exp_id, resolution, error = _launch_fingerprint_mode(
            start, nb=nb, runner=runner
        )
        if start.mode == "scale_up":
            return exp_id, None, resolution, None, error
        return exp_id, None, None, resolution, error

    exp_id = runner.start_experiment(
        config,
        hypothesis=start.hypothesis,
        preregistration=start.preregistration,
        exploratory=start.exploratory,
    )
    return exp_id, None, None, None, None


def build_start_success_response(
    start: StartExperimentRequest,
    exp_id: str,
    *,
    runner,
    prescreen: Dict[str, Any],
    preflight: Dict[str, Any],
    eligibility: Optional[Dict[str, Any]],
    scale_up_resolution: Optional[Dict[str, Any]],
    refine_resolution: Optional[Dict[str, Any]],
):
    record_run_trigger(
        experiment_id=exp_id,
        source="ui_start",
        mode=start.mode,
        details={
            "endpoint": "/api/experiments/start",
            "auto_harden": start.auto_harden,
        },
    )
    critique = (
        runner.progress.hypothesis_critique
        if isinstance(runner.progress.hypothesis_critique, dict)
        else None
    )
    return jsonify(
        {
            "experiment_id": exp_id,
            "status": "started",
            "config": start.config.to_dict(),
            "prescreen": prescreen,
            "compact_synthesis_bias": start.compact_changes,
            "live_screening_bias": start.live_screening_changes,
            "sparse_morph_bias": start.sparse_morph_changes,
            "intermediate_probe_bias": start.intermediate_probe_changes,
            "scale_up_resolution": scale_up_resolution,
            "refine_resolution": refine_resolution,
            "aria_message": runner.progress.aria_message,
            "hypothesis_critique": critique,
            "hypothesis_review_gate": critique.get("gate") if critique else None,
            "hypothesis_missing_fields": extract_hypothesis_missing_fields(critique),
            "preflight": preflight,
            "preflight_override": start.preflight_override,
            "eligibility": eligibility,
        }
    )


def build_start_error_response(
    error: Exception,
    *,
    mode: str,
    notebook_path: str,
    runner,
    should_autospawn_self_repair,
    spawn_code_agent_task,
):
    logger.error("Error starting experiment: %s\n%s", error, traceback.format_exc())
    error_text = str(error)
    reset_runner_launch_state(runner, error=error_text)
    auto_repair_task: Optional[Dict[str, Any]] = None
    if should_autospawn_self_repair(error_text):
        try:
            auto_repair_task = spawn_code_agent_task(
                goal=(
                    "Experiment start failed with runtime/code error. "
                    f"mode={mode}, error={error_text}. "
                    "Identify root cause, apply safe code/config fixes, and report validation."
                ),
                notebook_path=notebook_path,
                allow_write=True,
                session_id="",
            )
        except Exception as spawn_err:
            logger.warning("Auto self-repair spawn failed: %s", spawn_err)
    return (
        jsonify(
            {
                "error": error_text,
                "auto_repair_started": bool(auto_repair_task),
                "auto_repair_task": auto_repair_task,
            }
        ),
        500,
    )


def load_rerun_source(nb, experiment_id: str):
    return nb.get_resumable_experiment(experiment_id) or nb.get_experiment(
        experiment_id
    )


def _load_rerun_config(source):
    try:
        config_dict = json.loads(source.get("config_json") or "{}")
    except (json.JSONDecodeError, TypeError, ValueError):
        config_dict = {}
    config = RunConfig.from_dict(config_dict)
    hypothesis = source.get("hypothesis")
    exp_type = str(source.get("experiment_type") or "synthesis").strip().lower()
    return config, hypothesis, exp_type


def start_rerun_from_source(*, source, experiment_id: str, notebook_path: str):
    runner = get_runner(notebook_path)
    config, hypothesis, exp_type = _load_rerun_config(source)
    if exp_type == "continuous":
        config.continuous = True
        new_id = runner.start_continuous(config)
        mode = "continuous"
    elif exp_type == "evolution":
        new_id = runner.start_evolution(config, hypothesis=hypothesis)
        mode = "evolve"
    elif exp_type == "novelty":
        new_id = runner.start_novelty_search(config, hypothesis=hypothesis)
        mode = "novelty"
    else:
        new_id = runner.start_experiment(config, hypothesis=hypothesis)
        mode = "single"
    record_run_trigger(
        experiment_id=new_id,
        source="ui_rerun",
        mode=mode,
        details={
            "endpoint": f"/api/experiments/{experiment_id}/rerun",
            "source_experiment_id": experiment_id,
        },
    )
    return new_id, mode, config


def _run_batch_rerun_single(eid: str, notebook_path: str):
    runner = get_runner(notebook_path)
    nb = get_notebook(notebook_path, read_only=False)
    source = load_rerun_source(nb, eid)
    if source is None:
        return None
    try:
        config, hypothesis, exp_type = _load_rerun_config(source)
        if str(source.get("status") or "").strip().lower() == "running":
            nb.cancel_experiment(eid)
        if exp_type == "continuous":
            config.continuous = True
            new_id = runner.start_continuous(config)
            trigger_mode = "continuous"
        elif exp_type == "evolution":
            new_id = runner.start_evolution(config, hypothesis=hypothesis)
            trigger_mode = "evolution"
        elif exp_type == "novelty":
            new_id = runner.start_novelty_search(config, hypothesis=hypothesis)
            trigger_mode = "novelty"
        else:
            new_id = runner.start_experiment(config, hypothesis=hypothesis)
            trigger_mode = exp_type
        record_run_trigger(
            experiment_id=new_id,
            source="ui_batch_rerun",
            mode=trigger_mode,
            details={"source_experiment_id": eid},
        )
        return new_id
    except Exception as exc:
        logger.error(
            "Batch rerun error for %s: %s\n%s", eid, exc, traceback.format_exc()
        )
        if runner.is_running:
            logger.debug("Runner remained active after batch rerun failure for %s", eid)
        return None


def start_batch_rerun(experiment_ids: list[str], *, notebook_path: str):
    queue = list(experiment_ids)
    first_id = queue.pop(0)
    _BATCH_RERUN_STATE.update(
        {
            "active": True,
            "total": len(experiment_ids),
            "completed": 0,
            "current": first_id,
            "remaining": queue,
            "results": [],
        }
    )

    def batch_worker():
        try:
            new_id = _run_batch_rerun_single(first_id, notebook_path)
            _BATCH_RERUN_STATE["results"].append(
                {"source_id": first_id, "new_id": new_id, "ok": new_id is not None}
            )
            for next_id in list(_BATCH_RERUN_STATE["remaining"]):
                runner = get_runner(notebook_path)
                while runner.is_running:
                    time.sleep(5)
                _BATCH_RERUN_STATE["completed"] += 1
                _BATCH_RERUN_STATE["current"] = next_id
                _BATCH_RERUN_STATE["remaining"] = [
                    queued_id
                    for queued_id in _BATCH_RERUN_STATE["remaining"]
                    if queued_id != next_id
                ]
                new_id = _run_batch_rerun_single(next_id, notebook_path)
                _BATCH_RERUN_STATE["results"].append(
                    {"source_id": next_id, "new_id": new_id, "ok": new_id is not None}
                )
            runner = get_runner(notebook_path)
            while runner.is_running:
                time.sleep(5)
            _BATCH_RERUN_STATE["completed"] += 1
        except Exception as exc:
            logger.error(
                "Batch rerun worker error: %s\n%s", exc, traceback.format_exc()
            )
        finally:
            _BATCH_RERUN_STATE["active"] = False
            _BATCH_RERUN_STATE["current"] = None
            _BATCH_RERUN_STATE["remaining"] = []

    threading.Thread(target=batch_worker, daemon=True).start()
    return {
        "status": "queued",
        "total": len(experiment_ids),
        "started": first_id,
        "queued": queue,
    }
