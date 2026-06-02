"""Controlled screening batches for dynamic component lowerings.

This tool creates a temporary candidate artifact filtered to one lowering
family, then runs the normal screening pipeline with dynamic candidates forced.
It is meant to collect comparable DB outcome rows for branch lowerings without
adding another topology family first.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from research.scientist.runner import RunConfig
from research.tools.backfill_templates import _start_backfill_experiment_with_retry


DEFAULT_DB = Path("research/runs.db")
DEFAULT_CANDIDATES = Path(
    "research/data/synthesis_candidates/dynamic_component_candidates.json"
)
DEFAULT_ARTIFACT_DIR = Path(tempfile.gettempdir()) / "dynamic_component_controlled"
KNOWN_LOWERINGS = (
    "rmsnorm_chain_with_binary_skip",
    "trunk_sidecar_merge_v1",
    "mixer_sidecar_restore_v1",
    "router_lane_blend_v1",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def filtered_candidate_artifact(
    *,
    source_path: str | Path = DEFAULT_CANDIDATES,
    output_path: str | Path,
    lowering: str,
    max_candidates: int | None = None,
    require_backward_passed: bool = True,
) -> dict[str, Any]:
    """Write a dynamic candidate artifact containing only ``lowering`` rows."""
    source = Path(source_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    ready_rows = _candidate_rows(payload, key="ready_for_registration")
    candidate_rows = _candidate_rows(payload, key="candidates")

    ready = _filter_rows(
        ready_rows,
        lowering=lowering,
        max_candidates=max_candidates,
        require_backward_passed=require_backward_passed,
    )
    candidates = _filter_rows(
        candidate_rows,
        lowering=lowering,
        max_candidates=max_candidates,
        require_backward_passed=False,
    )
    if not ready:
        raise ValueError(
            f"No ready dynamic candidates found for lowering {lowering!r} in {source}"
        )

    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "controlled_lowering": lowering,
            "controlled_source_path": str(source),
            "controlled_ready_count": len(ready),
            "controlled_candidate_count": len(candidates),
            "controlled_require_backward_passed": bool(require_backward_passed),
            "controlled_created_at": time.time(),
        }
    )
    artifact = {
        "schema_version": payload.get(
            "schema_version", "dynamic_component_candidates_v1"
        ),
        "metadata": metadata,
        "candidates": candidates,
        "ready_for_registration": ready,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "lowering": lowering,
        "source_path": str(source),
        "output_path": str(out),
        "ready_count": len(ready),
        "candidate_count": len(candidates),
        "example_components": [
            _component_id(row) for row in ready[: min(5, len(ready))]
        ],
    }


def build_controlled_run_config(
    *,
    n_programs: int,
    device: str,
    candidate_path: str | Path,
    dynamic_prob: float = 1.0,
    dynamic_strength: float = 0.0,
    max_candidates: int = 32,
    min_lowered_ops: int = 8,
    composition_depth: int = 3,
    model_dim: int | None = None,
    n_layers: int | None = None,
    stage1_steps: int | None = None,
    max_ops: int | None = None,
    max_depth: int | None = None,
) -> RunConfig:
    """Build the neutral screening config used by controlled lowering runs."""
    kwargs: dict[str, Any] = {
        "n_programs": int(n_programs),
        "device": device,
        "mode": "single",
        "composition_depth": int(composition_depth),
        "template_weights": {},
        "op_weights": {},
        "use_dynamic_template_candidates": True,
        "dynamic_template_candidate_path": str(candidate_path),
        "dynamic_template_candidate_prob": float(dynamic_prob),
        "dynamic_template_candidate_strength": float(dynamic_strength),
        "dynamic_template_max_candidates": int(max_candidates),
        "dynamic_template_min_lowered_ops": int(min_lowered_ops),
        "use_learned_candidate_weights": False,
        "use_screening_signal_weights": False,
        "routing_mandatory": False,
        "persist_screening_failures": True,
        "disable_runtime_dedup": True,
        "enable_stage09_cheap_train_gate": False,
        "gbm_prescreener_enabled": False,
    }
    if model_dim is not None:
        kwargs["model_dim"] = int(model_dim)
    if n_layers is not None:
        kwargs["n_layers"] = int(n_layers)
    if stage1_steps is not None:
        kwargs["stage1_steps"] = int(stage1_steps)
    if max_ops is not None:
        kwargs["max_ops"] = int(max_ops)
    if max_depth is not None:
        kwargs["max_depth"] = int(max_depth)
    return RunConfig(**kwargs)


def run_lowering_batch(
    *,
    lowering: str,
    n_programs: int,
    device: str,
    db_path: str | Path = DEFAULT_DB,
    source_candidates: str | Path = DEFAULT_CANDIDATES,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    dynamic_prob: float = 1.0,
    dynamic_strength: float = 0.0,
    max_candidates: int = 32,
    min_lowered_ops: int = 8,
    composition_depth: int = 3,
    model_dim: int | None = None,
    n_layers: int | None = None,
    stage1_steps: int | None = None,
    max_ops: int | None = None,
    max_depth: int | None = None,
    runner_factory: Callable[[str], Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one controlled screening batch for a lowering family."""
    artifact_path = _artifact_path(artifact_dir, lowering)
    artifact_summary = filtered_candidate_artifact(
        source_path=source_candidates,
        output_path=artifact_path,
        lowering=lowering,
        max_candidates=max_candidates,
    )
    config = build_controlled_run_config(
        n_programs=n_programs,
        device=device,
        candidate_path=artifact_path,
        dynamic_prob=dynamic_prob,
        dynamic_strength=dynamic_strength,
        max_candidates=max_candidates,
        min_lowered_ops=min_lowered_ops,
        composition_depth=composition_depth,
        model_dim=model_dim,
        n_layers=n_layers,
        stage1_steps=stage1_steps,
        max_ops=max_ops,
        max_depth=max_depth,
    )
    config_payload = config.to_dict()
    config_payload.update(
        {
            "controlled_dynamic_lowering": lowering,
            "controlled_dynamic_artifact": str(artifact_path),
            "controlled_dynamic_source_candidates": str(source_candidates),
        }
    )
    result: dict[str, Any] = {
        "lowering": lowering,
        "artifact": artifact_summary,
        "config": config_payload,
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return result

    from research.scientist.runner import ExperimentRunner

    factory = runner_factory or ExperimentRunner
    runner = factory(str(db_path))
    runner._grammar_weight_overrides = {}
    runner._op_weights_overrides = {}
    runner._ensure_math_spaces()

    hypothesis = (
        "Controlled dynamic lowering screening: "
        f"{lowering} ({artifact_summary['ready_count']} ready candidates)"
    )
    exp_id, nb = _start_backfill_experiment_with_retry(
        runner,
        experiment_type="backfill",
        config=config_payload,
        hypothesis=hypothesis,
        hypothesis_metadata={
            "source": "dynamic_component_controlled_screening",
            "lowering": lowering,
            "artifact": str(artifact_path),
        },
        created_by="dynamic_component_controlled_screening",
    )
    nb.close()

    nb = runner._make_notebook()
    try:
        results = runner._execute_experiment(
            exp_id,
            config,
            nb,
            use_learned_grammar=False,
        )
        nb.complete_experiment(
            experiment_id=exp_id,
            results=results,
            aria_summary=(
                f"Controlled dynamic lowering {lowering}: "
                f"{results.get('stage1_passed', 0)}/{results.get('total', 0)} S1"
            ),
        )
        s0_op_counts = results.pop("_s0_op_counts", None)
        if s0_op_counts:
            nb.merge_op_failure_counts(s0_op_counts)
        else:
            nb.update_op_success_rates(exp_id)
        nb.strip_graph_json_for_failures(exp_id)
        nb.update_failure_signatures(exp_id)
        result.update({"experiment_id": exp_id, "results": results})
        return result
    except KeyboardInterrupt:
        nb.fail_experiment(exp_id, error="KeyboardInterrupt")
        raise
    except Exception as exc:
        nb.fail_experiment(exp_id, error=str(exc))
        result.update({"experiment_id": exp_id, "error": str(exc)})
        return result
    finally:
        nb.close()


def _candidate_rows(payload: Mapping[str, Any], *, key: str) -> list[Mapping[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _filter_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    lowering: str,
    max_candidates: int | None,
    require_backward_passed: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if _row_lowering(row) != lowering:
            continue
        validation = row.get("validation")
        if (
            require_backward_passed
            and isinstance(validation, Mapping)
            and not validation.get("backward_passed")
        ):
            continue
        out.append(dict(row))
        if max_candidates is not None and len(out) >= int(max_candidates):
            break
    return out


def _row_lowering(row: Mapping[str, Any]) -> str:
    descriptor = row.get("component_descriptor")
    if not isinstance(descriptor, Mapping):
        descriptor = {}
    return str(row.get("lowering") or descriptor.get("lowering") or "unknown")


def _component_id(row: Mapping[str, Any]) -> str:
    descriptor = row.get("component_descriptor")
    if not isinstance(descriptor, Mapping):
        descriptor = {}
    return str(
        row.get("component_id")
        or descriptor.get("component_id")
        or row.get("proposed_template_name")
        or "unknown"
    )


def _artifact_path(artifact_dir: str | Path, lowering: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in lowering)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return Path(artifact_dir) / f"{safe}_{stamp}.json"


def _parse_lowerings(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all:
        return KNOWN_LOWERINGS
    if args.lowering:
        return tuple(args.lowering)
    raise SystemExit("pass --lowering LOWERING or --all")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lowering",
        action="append",
        choices=KNOWN_LOWERINGS,
        help="Lowering family to run. Repeat to run multiple families.",
    )
    parser.add_argument("--all", action="store_true", help="Run all known lowerings")
    parser.add_argument("--n", type=int, default=24, help="Programs per lowering")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--dynamic-prob", type=float, default=1.0)
    parser.add_argument("--dynamic-strength", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--min-lowered-ops", type=int, default=8)
    parser.add_argument("--composition-depth", type=int, default=3)
    parser.add_argument("--model-dim", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--stage1-steps", type=int)
    parser.add_argument("--max-ops", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write filtered artifacts and print config without running screening.",
    )
    args = parser.parse_args(argv)

    summaries = []
    for lowering in _parse_lowerings(args):
        logger.info("Preparing controlled dynamic lowering batch: %s", lowering)
        summary = run_lowering_batch(
            lowering=lowering,
            n_programs=args.n,
            device=args.device,
            db_path=args.db,
            source_candidates=args.candidates,
            artifact_dir=args.artifact_dir,
            dynamic_prob=args.dynamic_prob,
            dynamic_strength=args.dynamic_strength,
            max_candidates=args.max_candidates,
            min_lowered_ops=args.min_lowered_ops,
            composition_depth=args.composition_depth,
            model_dim=args.model_dim,
            n_layers=args.n_layers,
            stage1_steps=args.stage1_steps,
            max_ops=args.max_ops,
            max_depth=args.max_depth,
            dry_run=args.dry_run,
        )
        summaries.append(summary)
        if args.dry_run:
            print(
                "dynamic_component_controlled_screening "
                f"lowering={lowering} dry_run=1 "
                f"ready={summary['artifact']['ready_count']} "
                f"artifact={summary['artifact']['output_path']}"
            )
        else:
            results = summary.get("results") or {}
            print(
                "dynamic_component_controlled_screening "
                f"lowering={lowering} exp={summary.get('experiment_id')} "
                f"total={results.get('total', 0)} "
                f"s1={results.get('stage1_passed', 0)} "
                f"artifact={summary['artifact']['output_path']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
