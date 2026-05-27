from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from research.defaults import RUNS_DB
from research.scientist.json_utils import json_safe
from research.scientist.runner import RunConfig


DEFAULT_REPORT_DIR = Path("research/reports/measured_rescue_ab")


@dataclass(frozen=True)
class RescueABSettings:
    db_path: str = str(RUNS_DB)
    report_dir: Path = DEFAULT_REPORT_DIR
    n_programs: int = 50
    model_dim: int = 256
    n_layers: int = 4
    stage1_steps: int = 120
    device: str = "cuda"
    tau: float = 0.01
    max_rescue: int = 4
    probe_budget: int = 24
    p_pass_floor: float = 0.0
    allow_unproven_screening_ensemble: bool = False
    require_prescreener_active: bool = True
    use_learned_grammar: bool = True
    dry_run: bool = False
    backup_max_age_hours: float = 24.0
    skip_backup_check: bool = False


def build_bounded_config(settings: RescueABSettings) -> RunConfig:
    """Build the shared config for both bounded A/B arms."""
    config = RunConfig(
        mode="single",
        n_programs=int(settings.n_programs),
        model_dim=int(settings.model_dim),
        n_layers=int(settings.n_layers),
        stage1_steps=int(settings.stage1_steps),
        device=str(settings.device),
        gbm_prescreener_enabled=True,
        screening_ensemble_p_pass_floor=float(settings.p_pass_floor),
    )
    config.allow_unproven_ml_influence = bool(
        settings.allow_unproven_screening_ensemble
    )

    # Keep the harness a single direct screening pass per arm. These switches
    # prevent the normal dashboard automation from queueing extra work around it.
    config.continuous = False
    config.max_experiments = 1
    config.rest_between_experiments = 0
    config.control_experiment_interval = 0
    config.auto_scale_up = False
    config.auto_report = False
    config.auto_investigate = False
    config.auto_validate = False
    config.auto_preregister = False
    config.auto_novelty_calibration = False
    config.enable_campaigns = False
    config.auto_go_no_go = False
    config.enable_causal_ablation = False
    config.causal_ablation_interval = 0
    config.enable_llm_decision_planner = False
    config.llm_decision_interval = 0
    config.require_preregistration = False
    return config


@contextmanager
def scoped_measured_rescue_env(
    *,
    enabled: bool,
    tau: float,
    max_rescue: int,
    probe_budget: int,
) -> Iterator[None]:
    updates = {
        "ARIA_MEASURED_RESCUE": "1" if enabled else "0",
        "ARIA_MEASURED_RESCUE_TAU": str(float(tau)),
        "ARIA_MEASURED_RESCUE_MAX": str(int(max_rescue)),
        "ARIA_MEASURED_RESCUE_PROBE_BUDGET": str(int(probe_budget)),
    }
    sentinel = object()
    previous: dict[str, object | str] = {
        key: os.environ.get(key, sentinel) for key in updates
    }
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, value in previous.items():
            if value is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)


def run_bounded_ab(
    settings: RescueABSettings,
    *,
    runner_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    config = build_bounded_config(settings)
    payload: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": bool(settings.dry_run),
        "settings": json_safe(settings.__dict__),
        "config": json_safe(config.to_dict()),
        "arms": [],
    }
    preflight = _prescreener_preflight(config)
    payload["prescreener_preflight"] = preflight

    if settings.dry_run:
        payload["arms"] = [
            _planned_arm_payload("off", config, settings),
            _planned_arm_payload("on", config, settings),
        ]
        _write_reports(payload, settings.report_dir)
        return payload

    if settings.require_prescreener_active and not preflight.get("active"):
        raise RuntimeError(
            "Measured-rescue A/B requires an active screening ensemble, but preflight "
            f"reported inactive: {preflight.get('reason')}. Pass "
            "--allow-unproven-screening-ensemble only for a measurement run that "
            "intentionally bypasses the ML trust-policy block."
        )

    if not settings.skip_backup_check:
        _check_backup_freshness(settings.backup_max_age_hours)

    from research.scientist.runner import ExperimentRunner

    factory = runner_factory or ExperimentRunner
    runner = factory(str(settings.db_path))
    if hasattr(runner, "_ensure_math_spaces"):
        runner._ensure_math_spaces()

    arms = [
        ("off", False),
        ("on", True),
    ]
    for arm_name, enabled in arms:
        arm_config = config.copy()
        arm_payload = _run_arm(
            runner=runner,
            config=arm_config,
            arm_name=arm_name,
            rescue_enabled=enabled,
            settings=settings,
        )
        payload["arms"].append(arm_payload)

    payload["comparison"] = _compare_arms(payload["arms"])
    _write_reports(payload, settings.report_dir)
    return payload


def _planned_arm_payload(
    arm_name: str,
    config: RunConfig,
    settings: RescueABSettings,
) -> dict[str, Any]:
    enabled = arm_name == "on"
    return {
        "arm": arm_name,
        "planned": True,
        "measured_rescue_enabled": enabled,
        "env": _arm_env(enabled, settings),
        "config": json_safe(config.to_dict()),
    }


def _run_arm(
    *,
    runner: Any,
    config: RunConfig,
    arm_name: str,
    rescue_enabled: bool,
    settings: RescueABSettings,
) -> dict[str, Any]:
    nb = runner._make_notebook()
    exp_id = nb.start_experiment(
        experiment_type="synthesis",
        config=config.to_dict(),
        hypothesis=(
            "Bounded measured-rescue A/B "
            f"arm={arm_name}: direct single experiment, no continuous follow-up."
        ),
        hypothesis_metadata={
            "source": "measured_rescue_ab",
            "arm": arm_name,
            "measured_rescue_enabled": rescue_enabled,
        },
    )
    started = time.time()
    try:
        with scoped_measured_rescue_env(
            enabled=rescue_enabled,
            tau=settings.tau,
            max_rescue=settings.max_rescue,
            probe_budget=settings.probe_budget,
        ):
            results = runner._execute_experiment(
                exp_id,
                config,
                nb,
                use_learned_grammar=bool(settings.use_learned_grammar),
            )
        _complete_direct_experiment(nb, exp_id, results, arm_name=arm_name)
        return {
            "arm": arm_name,
            "experiment_id": exp_id,
            "status": "completed",
            "measured_rescue_enabled": rescue_enabled,
            "elapsed_seconds": time.time() - started,
            "summary": _summarize_results(results),
        }
    except BaseException as exc:
        nb.fail_experiment(exp_id, error=str(exc))
        raise
    finally:
        close = getattr(nb, "close", None)
        if callable(close):
            close()


def _complete_direct_experiment(
    nb: Any,
    exp_id: str,
    results: Mapping[str, Any],
    *,
    arm_name: str,
) -> None:
    nb.complete_experiment(
        experiment_id=exp_id,
        results=dict(results),
        aria_summary=(
            f"Measured-rescue A/B arm={arm_name}: "
            f"{results.get('stage1_passed', 0)}/{results.get('total', 0)} S1."
        ),
    )
    s0_op_counts = dict(results).pop("_s0_op_counts", None)
    if s0_op_counts:
        nb.merge_op_failure_counts(s0_op_counts)
    else:
        nb.update_op_success_rates(exp_id)
    nb.strip_graph_json_for_failures(exp_id)
    nb.update_failure_signatures(exp_id)
    flush = getattr(nb, "flush_writes", None)
    if callable(flush):
        flush()


def _summarize_results(results: Mapping[str, Any]) -> dict[str, Any]:
    funnel = results.get("funnel_counts") or {}
    records = results.get("measured_rescue_records") or []
    summary = {
        "total": results.get("total", 0),
        "stage0_passed": results.get("stage0_passed", 0),
        "stage05_passed": results.get("stage05_passed", 0),
        "stage1_passed": results.get("stage1_passed", 0),
        "best_loss_ratio": results.get("best_loss_ratio"),
        "gbm_prescreener_skipped": funnel.get("gbm_prescreener_skipped", 0),
        "post_gbm_prescreener": funnel.get("post_gbm_prescreener", 0),
        "measured_rescued": funnel.get("measured_rescued", len(records)),
        "measured_rescue_records": records,
        "elapsed_seconds": results.get("elapsed_seconds"),
    }
    return json_safe(summary)


def _compare_arms(arms: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_name = {str(arm.get("arm")): arm for arm in arms}
    off = (by_name.get("off") or {}).get("summary") or {}
    on = (by_name.get("on") or {}).get("summary") or {}
    return {
        "delta_stage1_passed": (on.get("stage1_passed") or 0)
        - (off.get("stage1_passed") or 0),
        "delta_measured_rescued": (on.get("measured_rescued") or 0)
        - (off.get("measured_rescued") or 0),
        "delta_best_loss_ratio": _delta_optional(
            on.get("best_loss_ratio"), off.get("best_loss_ratio")
        ),
    }


def _prescreener_preflight(config: RunConfig) -> dict[str, Any]:
    from research.scientist.ml_influence_policy import build_ml_influence_policy
    from research.scientist.ml_influence_policy import component_is_allowed

    policy = build_ml_influence_policy(config)
    component = (policy.get("components") or {}).get("screening_ensemble", {})
    if not component_is_allowed("screening_ensemble", config):
        return {
            "active": False,
            "reason": "blocked_by_ml_trust_policy",
            "screening_ensemble": json_safe(component),
        }

    try:
        from research.scientist.intelligence.predictor import load_runtime_ensemble

        ensemble = load_runtime_ensemble(
            profiling_db="research/profiling/component_profiles.db"
        )
        if ensemble is None or not ensemble.is_fitted():
            return {
                "active": False,
                "reason": "runtime_ensemble_not_fitted",
                "screening_ensemble": json_safe(component),
            }
    except Exception as exc:
        return {
            "active": False,
            "reason": f"runtime_ensemble_load_failed: {exc}",
            "screening_ensemble": json_safe(component),
        }

    return {
        "active": True,
        "reason": "screening_ensemble_allowed_and_fitted",
        "screening_ensemble": json_safe(component),
    }


def _delta_optional(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _arm_env(enabled: bool, settings: RescueABSettings) -> dict[str, str]:
    return {
        "ARIA_MEASURED_RESCUE": "1" if enabled else "0",
        "ARIA_MEASURED_RESCUE_TAU": str(float(settings.tau)),
        "ARIA_MEASURED_RESCUE_MAX": str(int(settings.max_rescue)),
        "ARIA_MEASURED_RESCUE_PROBE_BUDGET": str(int(settings.probe_budget)),
    }


def _write_reports(payload: Mapping[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    json_path = report_dir / f"measured_rescue_ab_{stamp}.json"
    md_path = report_dir / f"measured_rescue_ab_{stamp}.md"
    json_path.write_text(json.dumps(json_safe(payload), indent=2) + "\n")
    md_path.write_text(_render_markdown(payload) + "\n")


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Measured Rescue Bounded A/B",
        "",
        f"- created_at: `{payload.get('created_at')}`",
        f"- dry_run: `{payload.get('dry_run')}`",
        "",
        "## Arms",
    ]
    for arm in payload.get("arms", []):
        summary = arm.get("summary") or {}
        lines.extend(
            [
                "",
                f"### {arm.get('arm')}",
                f"- experiment_id: `{arm.get('experiment_id', 'planned')}`",
                f"- status: `{arm.get('status', 'planned')}`",
                f"- measured_rescue_enabled: `{arm.get('measured_rescue_enabled')}`",
                f"- total: `{summary.get('total', 0)}`",
                f"- stage1_passed: `{summary.get('stage1_passed', 0)}`",
                f"- measured_rescued: `{summary.get('measured_rescued', 0)}`",
            ]
        )
    comparison = payload.get("comparison")
    if comparison:
        lines.extend(["", "## Comparison"])
        for key, value in comparison.items():
            lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)


def _check_backup_freshness(max_age_hours: float) -> None:
    from research.tools import check_backup_freshness

    rc = check_backup_freshness.main(["--max-age-hours", str(float(max_age_hours))])
    if rc != 0:
        raise RuntimeError(
            "Backup freshness check failed; rerun after creating a fresh DB backup "
            "or pass --skip-backup-check intentionally."
        )


def parse_args(argv: list[str] | None = None) -> RescueABSettings:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded measured-rescue OFF/ON A/B via direct single-experiment "
            "execution, without dashboard continuous follow-up."
        )
    )
    parser.add_argument("--db-path", default=str(RUNS_DB))
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--n-programs", type=int, default=50)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--stage1-steps", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--max-rescue", type=int, default=4)
    parser.add_argument("--probe-budget", type=int, default=24)
    parser.add_argument("--p-pass-floor", type=float, default=0.0)
    parser.add_argument(
        "--allow-unproven-screening-ensemble",
        action="store_true",
        help=(
            "Allow the screening ensemble despite the ML trust-policy gate. Use only "
            "for measurement/audit runs."
        ),
    )
    parser.add_argument(
        "--no-require-prescreener-active",
        action="store_true",
        help="Do not fail fast when preflight says the prescreener is inactive.",
    )
    parser.add_argument("--no-learned-grammar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup-max-age-hours", type=float, default=24.0)
    parser.add_argument("--skip-backup-check", action="store_true")
    args = parser.parse_args(argv)
    return RescueABSettings(
        db_path=args.db_path,
        report_dir=args.report_dir,
        n_programs=args.n_programs,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        stage1_steps=args.stage1_steps,
        device=args.device,
        tau=args.tau,
        max_rescue=args.max_rescue,
        probe_budget=args.probe_budget,
        p_pass_floor=args.p_pass_floor,
        allow_unproven_screening_ensemble=args.allow_unproven_screening_ensemble,
        require_prescreener_active=not args.no_require_prescreener_active,
        use_learned_grammar=not args.no_learned_grammar,
        dry_run=args.dry_run,
        backup_max_age_hours=args.backup_max_age_hours,
        skip_backup_check=args.skip_backup_check,
    )


def main(argv: list[str] | None = None) -> int:
    settings = parse_args(argv)
    payload = run_bounded_ab(settings)
    print(json.dumps(json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
