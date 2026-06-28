#!/usr/bin/env python3
"""Run bounded class-aware replacement ablations for top graphs.

This is intentionally narrower than champion_exhaustive_ablation: it targets
component classes already present in the parent graphs and ranks alternatives
by class fit, historical S1 rate, and parameter efficiency before compiling
and evaluating at most ~100 unique child graphs.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.scientist.causal_attribution import (  # noqa: E402
    CausalAblationCandidate,
    run_ablation_suite,
)
from research.scientist.construction_priors import (  # noqa: E402
    assess_local_edit_prior,
    get_active_construction_prior,
)
from research.scientist.notebook import LabNotebook  # noqa: E402
from research.scientist.runner import ExperimentRunner  # noqa: E402
from research.scientist.runner._helpers_metrics import (  # noqa: E402
    _rebuild_graph_with_overrides,
)
from research.scientist.runner._types import RunConfig  # noqa: E402
from research.scientist.native_runner import compile_model_native_first as compile_model  # noqa: E402
from research.synthesis.graph import ComputationGraph  # noqa: E402
from research.synthesis.primitives import (  # noqa: E402
    PrimitiveOp,
    estimate_op_params,
    get_primitive,
    list_primitives,
)
from research.synthesis.validator import validate_graph  # noqa: E402
from research.tools.champion_exhaustive_ablation import (  # noqa: E402
    ensure_ablation_metric_completeness,
)
from research.tools.focused_op_deletion_ablation import (  # noqa: E402
    ParentCandidate,
    json_dump,
    prepare_config,
    select_top_parents,
)
from research.tools.db_backup import backup_database  # noqa: E402


DB_PATH = PROJECT_ROOT / "research/runs.db"
RUNTIME_DIR = PROJECT_ROOT / "research/runtime"
GOOGLE_BACKUP_ROOT = Path("/home/tim/GoogleDrive/Backups/LLM_Research")
LOGGER = logging.getLogger("smart_component_replacement_ablation")
NODE_ID_PATTERN = re.compile(r"\(id=\d+\)")

CLASS_CANDIDATES: dict[str, tuple[str, ...]] = {
    "normalization": (
        "rmsnorm",
        "layernorm",
        "learnable_scale",
        "learnable_bias",
        "hyperbolic_norm",
    ),
    "activation": (
        "gelu",
        "relu",
        "silu",
        "sigmoid",
        "tanh",
        "swiglu_mlp",
    ),
    "projection": (
        "linear_proj",
        "fused_linear_gelu",
        "ternary_projection",
        "semi_structured_2_4_linear",
        "nm_sparse_linear",
        "block_sparse_linear",
        "low_rank_proj",
        "grouped_linear",
        "kronecker_linear",
        "shared_basis_proj",
        "tied_proj",
        "bottleneck_proj",
        "gated_linear",
    ),
    "sequence_mixer": (
        "spectral_filter",
        "chebyshev_spectral_mix",
        "state_space",
        "linear_attention",
        "gated_delta",
        "gated_linear_attention",
        "softmax_attention",
        "diff_attention",
        "graph_attention",
        "conv1d_seq",
        "long_conv_hyena",
        "local_window_attn",
        "latent_attention_compressor",
        "rwkv_time_mixing",
        "adjacent_token_merge",
        "rope_rotate",
    ),
    "binary_merge": (
        "add",
        "mul",
        "sub",
        "maximum",
        "minimum",
        "calibrated_branch_merge",
        "dual_compression_blend",
        "score_depth_blend",
        "difficulty_blend_3way",
        "geometric_product",
        "tropical_add",
        "tropical_matmul",
    ),
    "routing_signal": (
        "token_class_proj",
        "topk_gate",
        "learned_token_gate",
        "confidence_token_gate",
        "hybrid_token_gate",
        "depth_token_mask",
        "feature_sparsity",
        "adaptive_rank_gate",
    ),
    "scalar_signal": (
        "token_entropy",
        "mean_last",
        "norm_last",
        "max_last",
        "sum_last",
        "softmax_last",
    ),
}


@dataclasses.dataclass(slots=True)
class ReplacementChild:
    parent: ParentCandidate
    graph: ComputationGraph
    node_id: int
    original_op: str
    replacement_op: str
    component_class: str
    fingerprint: str
    estimated_param_delta: int
    rank_score: float


@dataclasses.dataclass(slots=True)
class PlannedSuite:
    parent: ParentCandidate
    component_class: str
    node_id: int
    original_op: str
    children: list[ReplacementChild]


def configure_logging(log_file: str) -> Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    path = (
        Path(log_file)
        if log_file
        else RUNTIME_DIR / "smart_component_replacement_ablation.log"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(path, encoding="utf-8"),
        ],
    )
    return path


def log(message: str) -> None:
    LOGGER.info(message)


def _table_columns(nb: LabNotebook, table: str) -> set[str]:
    return {str(row["name"]) for row in nb.conn.execute(f"PRAGMA table_info({table})")}


def _op_stats(nb: LabNotebook) -> dict[str, dict[str, float]]:
    if "op_success_rates" not in {
        str(row["name"])
        for row in nb.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }:
        return {}
    cols = _table_columns(nb, "op_success_rates")
    loss_expr = "avg_loss_ratio" if "avg_loss_ratio" in cols else "NULL"
    rows = nb.conn.execute(
        f"""SELECT op_name, n_used, n_stage1_passed, {loss_expr} AS avg_loss_ratio
            FROM op_success_rates"""
    ).fetchall()
    stats: dict[str, dict[str, float]] = {}
    for row in rows:
        used = float(row["n_used"] or 0.0)
        passed = float(row["n_stage1_passed"] or 0.0)
        stats[str(row["op_name"])] = {
            "n_used": used,
            "s1_rate": passed / used if used > 0 else 0.0,
            "avg_loss_ratio": float(row["avg_loss_ratio"])
            if row["avg_loss_ratio"] is not None
            else 1.0,
        }
    return stats


def _validation_error_key(error: Any) -> str:
    return NODE_ID_PATTERN.sub("(id=*)", str(error))


def _existing_fingerprints(nb: LabNotebook) -> set[str]:
    rows = nb.conn.execute(
        """SELECT graph_fingerprint FROM program_results_compat
           WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
           UNION
           SELECT child_fingerprint FROM causal_ablation_child_observations
           WHERE TRIM(COALESCE(child_fingerprint, '')) <> ''"""
    ).fetchall()
    return {str(row[0]) for row in rows if row[0]}


def _component_class(op_name: str, primitive: PrimitiveOp) -> str:
    name = str(op_name)
    if name in {"rmsnorm", "layernorm"} or "norm" in name:
        return "normalization"
    if name in {"gelu", "relu", "silu", "sigmoid", "tanh", "swiglu_mlp"}:
        return "activation"
    if (
        "linear" in name
        or "proj" in name
        or "sparse_linear" in name
        or name in {"ternary_projection", "semi_structured_2_4_linear"}
    ):
        return "projection"
    if name in {
        "add",
        "mul",
        "matmul",
        "sub",
        "maximum",
        "minimum",
        "dual_compression_blend",
        "score_depth_blend",
        "difficulty_blend_3way",
    }:
        return "binary_merge" if primitive.n_inputs == 2 else "projection"
    if primitive.shape_rule == "reduce_last" or name in {
        "entropy_score",
        "token_entropy",
    }:
        return "scalar_signal"
    if (
        primitive.binding_range_class != "none"
        or primitive.category.value in {"mixing", "frequency", "sequence"}
        or name
        in {"rope_rotate", "adjacent_token_merge", "latent_attention_compressor"}
    ):
        return "sequence_mixer"
    if "gate" in name or "router" in name or "score" in name or "token" in name:
        return "routing_signal"
    return primitive.category.value


def _default_config(
    op: PrimitiveOp,
    *,
    original_config: dict[str, Any],
    output_dim: int,
    model_dim: int,
) -> dict[str, Any]:
    cfg = {
        key: value
        for key, value in dict(original_config or {}).items()
        if key in set(op.config_keys)
    }
    if "out_dim" in op.config_keys:
        cfg.setdefault("out_dim", int(output_dim or model_dim))
    defaults: dict[str, Any] = {
        "n": 2,
        "m": 4,
        "block_size": 16,
        "block_density": 0.25,
        "n_heads": 4,
        "window_size": 32,
        "num_experts": 4,
        "top_k": 1,
        "k": max(1, min(64, model_dim // 4)),
        "n_lanes": 3,
        "max_depth": 3,
        "n_keep": 0.75,
        "capacity_factor": 0.5,
        "threshold": 0.5,
        "span_width": 2,
        "fallback_behavior": "default_path",
        "lane_count": 3,
        "confidence_threshold": 0.5,
        "lane_id": 0,
        "n_classes": int(original_config.get("n_classes") or 4),
        "n_ways": 4,
        "chebyshev_order": 4,
        "kernel_scale": 1.0,
        "n_iters": 3,
        "damping": 0.5,
    }
    for key in op.config_keys:
        if key in defaults:
            cfg.setdefault(key, defaults[key])
    return cfg


def _candidate_ops(
    *,
    original_op: str,
    original_primitive: PrimitiveOp,
    component_class: str,
) -> list[PrimitiveOp]:
    primitives = {op.name: op for op in list_primitives()}
    candidates: dict[str, PrimitiveOp] = {}
    for name in CLASS_CANDIDATES.get(component_class, ()):
        op = primitives.get(name)
        if op is not None and op.n_inputs == original_primitive.n_inputs:
            candidates[op.name] = op
    candidates.pop(original_op, None)
    return list(candidates.values())


def _select_balanced_candidates(
    candidates: list[ReplacementChild],
    *,
    max_children: int,
) -> list[ReplacementChild]:
    """Keep the run bounded while ensuring both top parents are represented."""
    limit = max(1, int(max_children))
    by_parent: dict[str, list[ReplacementChild]] = {}
    for child in sorted(candidates, key=lambda child: child.rank_score, reverse=True):
        by_parent.setdefault(child.parent.result_id, []).append(child)
    if len(by_parent) <= 1:
        return candidates[:limit]

    selected: list[ReplacementChild] = []
    parent_ids = sorted(by_parent)
    per_parent_floor = max(1, limit // len(parent_ids))
    selected_keys: set[tuple[str, str]] = set()
    for parent_id in parent_ids:
        for child in by_parent[parent_id][:per_parent_floor]:
            selected.append(child)
            selected_keys.add((child.parent.result_id, child.fingerprint))
            if len(selected) >= limit:
                return sorted(
                    selected, key=lambda child: child.rank_score, reverse=True
                )

    remainder = [
        child
        for child in sorted(
            candidates, key=lambda child: child.rank_score, reverse=True
        )
        if (child.parent.result_id, child.fingerprint) not in selected_keys
    ]
    selected.extend(remainder[: max(0, limit - len(selected))])
    return sorted(selected, key=lambda child: child.rank_score, reverse=True)


def _rank_op(
    op: PrimitiveOp,
    *,
    component_class: str,
    stats: dict[str, dict[str, float]],
    model_dim: int,
    original_params: int,
) -> float:
    row = stats.get(op.name, {})
    s1_rate = float(row.get("s1_rate") or 0.0)
    avg_loss = float(row.get("avg_loss_ratio") or 1.0)
    params = max(0, estimate_op_params(op, model_dim))
    param_bonus = 0.0
    if original_params > 0:
        param_bonus = max(
            -0.25, min(0.35, (original_params - params) / original_params)
        )
    curated_bonus = (
        0.30 if op.name in CLASS_CANDIDATES.get(component_class, ()) else 0.0
    )
    safe_bonus = 0.08 if not op.numerically_risky else -0.20
    standalone_bonus = 0.05 if op.standalone else -0.10
    loss_bonus = max(-0.15, min(0.15, 0.7 - avg_loss))
    return (
        curated_bonus
        + s1_rate
        + param_bonus
        + safe_bonus
        + standalone_bonus
        + loss_bonus
    )


def _try_child(
    parent: ParentCandidate,
    *,
    node_id: int,
    replacement: PrimitiveOp,
    component_class: str,
    config: RunConfig,
    stats: dict[str, dict[str, float]],
    baseline_validation_errors: set[str],
) -> tuple[ReplacementChild | None, dict[str, Any]]:
    node = parent.graph.nodes[node_id]
    original = get_primitive(node.op_name)
    replacement_config = _default_config(
        replacement,
        original_config=dict(node.config or {}),
        output_dim=int(node.output_shape.dim or config.model_dim),
        model_dim=int(config.model_dim),
    )
    rebuilt = _rebuild_graph_with_overrides(
        parent.graph,
        {node_id: {"op_name": replacement.name, "config": replacement_config}},
    )
    meta = {
        "parent_result_id": parent.result_id,
        "node_id": int(node_id),
        "component_class": component_class,
        "original_op": node.op_name,
        "replacement_op": replacement.name,
        "replacement_config": replacement_config,
    }
    if rebuilt is None:
        return None, {**meta, "reason": "rebuild_failed"}
    try:
        validation = validate_graph(
            rebuilt,
            max_ops=max(1, int(config.max_ops)),
            max_depth=max(1, int(config.max_depth)),
            min_splits=config.min_splits,
        )
        if not validation.valid:
            errors = [str(error) for error in validation.errors]
            new_errors = [
                error
                for error in errors
                if _validation_error_key(error) not in baseline_validation_errors
            ]
            if not new_errors:
                validation = None
            else:
                return None, {
                    **meta,
                    "reason": "validation_failed",
                    "errors": new_errors,
                    "baseline_errors": sorted(baseline_validation_errors),
                }
        if validation is not None and not validation.valid:
            return None, {
                **meta,
                "reason": "validation_failed",
                "errors": list(validation.errors),
            }
        compile_model(
            [rebuilt],
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
        )
        fp = rebuilt.fingerprint()
    except (RuntimeError, ValueError, TypeError) as exc:
        return None, {**meta, "reason": "compile_failed", "error": str(exc)}
    original_params = estimate_op_params(original, int(config.model_dim))
    replacement_params = estimate_op_params(replacement, int(config.model_dim))
    score = _rank_op(
        replacement,
        component_class=component_class,
        stats=stats,
        model_dim=int(config.model_dim),
        original_params=original_params,
    )
    return (
        ReplacementChild(
            parent=parent,
            graph=rebuilt,
            node_id=node_id,
            original_op=node.op_name,
            replacement_op=replacement.name,
            component_class=component_class,
            fingerprint=fp,
            estimated_param_delta=int(replacement_params - original_params),
            rank_score=float(score),
        ),
        meta,
    )


def build_plan(
    nb: LabNotebook,
    *,
    parents: list[ParentCandidate],
    max_children: int,
    per_node_limit: int,
    allow_existing_fingerprints: bool = False,
) -> tuple[list[PlannedSuite], list[dict[str, Any]]]:
    stats = _op_stats(nb)
    existing = set() if allow_existing_fingerprints else _existing_fingerprints(nb)
    seen = set(existing)
    rejected: list[dict[str, Any]] = []
    by_suite: dict[tuple[str, int, str], list[ReplacementChild]] = {}
    candidates: list[ReplacementChild] = []
    for parent in parents:
        config = parent.config
        baseline_validation = validate_graph(
            parent.graph,
            max_ops=max(1, int(config.max_ops)),
            max_depth=max(1, int(config.max_depth)),
            min_splits=config.min_splits,
        )
        baseline_validation_errors = {
            _validation_error_key(error) for error in baseline_validation.errors
        }
        for node_id in parent.graph.topological_order():
            node = parent.graph.nodes[node_id]
            if node.is_input:
                continue
            try:
                primitive = get_primitive(node.op_name)
            except (KeyError, ValueError) as exc:
                rejected.append(
                    {
                        "parent_result_id": parent.result_id,
                        "node_id": int(node_id),
                        "op_name": node.op_name,
                        "reason": "primitive_lookup_failed",
                        "error": str(exc),
                    }
                )
                continue
            component_class = _component_class(node.op_name, primitive)
            replacements = _candidate_ops(
                original_op=node.op_name,
                original_primitive=primitive,
                component_class=component_class,
            )
            replacements = sorted(
                replacements,
                key=lambda op: _rank_op(
                    op,
                    component_class=component_class,
                    stats=stats,
                    model_dim=int(config.model_dim),
                    original_params=estimate_op_params(
                        primitive, int(config.model_dim)
                    ),
                ),
                reverse=True,
            )
            accepted_here = 0
            for replacement in replacements:
                if accepted_here >= max(1, int(per_node_limit)):
                    break
                child, meta = _try_child(
                    parent,
                    node_id=node_id,
                    replacement=replacement,
                    component_class=component_class,
                    config=config,
                    stats=stats,
                    baseline_validation_errors=baseline_validation_errors,
                )
                if child is None:
                    rejected.append(meta)
                    continue
                if child.fingerprint == parent.fingerprint:
                    rejected.append({**meta, "reason": "duplicate_parent_fingerprint"})
                    continue
                if child.fingerprint in seen:
                    rejected.append(
                        {
                            **meta,
                            "reason": "duplicate_existing_or_planned_fingerprint",
                            "fingerprint": child.fingerprint,
                        }
                    )
                    continue
                seen.add(child.fingerprint)
                accepted_here += 1
                candidates.append(child)

    candidates = sorted(candidates, key=lambda child: child.rank_score, reverse=True)
    selected = _select_balanced_candidates(candidates, max_children=max_children)
    for child in selected:
        by_suite.setdefault(
            (child.parent.result_id, child.node_id, child.component_class), []
        ).append(child)
    suites = [
        PlannedSuite(
            parent=children[0].parent,
            component_class=component_class,
            node_id=node_id,
            original_op=children[0].original_op,
            children=sorted(children, key=lambda child: child.rank_score, reverse=True),
        )
        for (parent_id, node_id, component_class), children in sorted(
            by_suite.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
        )
        if parent_id
    ]
    return suites, rejected


def _suite_rule_key(suite: PlannedSuite) -> str:
    return f"{suite.component_class}:{suite.node_id}:{suite.original_op}"


def make_candidate(
    suite: PlannedSuite,
    *,
    active_prior: dict[str, Any] | None = None,
) -> CausalAblationCandidate:
    rule_key = _suite_rule_key(suite)
    replacements = [child.replacement_op for child in suite.children]
    prior_assessment = assess_local_edit_prior(
        active_prior,
        rule_type="component_replace",
        rule_key=rule_key,
    )
    return CausalAblationCandidate(
        parent_experiment_id=suite.parent.experiment_id,
        parent_result_id=suite.parent.result_id,
        parent_fingerprint=suite.parent.fingerprint,
        parent_loss_ratio=suite.parent.loss_ratio,
        graph=suite.parent.graph,
        rule_type="component_replace",
        rule_key=rule_key,
        hypothesis=f"component_replace:{rule_key}",
        context={
            "campaign": "smart_component_replacement_ablation",
            "node_id": suite.node_id,
            "original_op": suite.original_op,
            "component_class": suite.component_class,
            "replacement_ops": replacements,
            "selection_policy": "class_fit + historical_s1_rate + parameter_efficiency",
            "prior_assessment": prior_assessment,
        },
    )


def child_meta(
    child: ReplacementChild,
    *,
    active_prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rule_key = f"{child.component_class}:{child.node_id}:{child.original_op}"
    return {
        "campaign": "smart_component_replacement_ablation",
        "parent_result_id": child.parent.result_id,
        "parent_fingerprint": child.parent.fingerprint,
        "node_id": child.node_id,
        "original_op": child.original_op,
        "replacement_op": child.replacement_op,
        "component_class": child.component_class,
        "estimated_param_delta": child.estimated_param_delta,
        "rank_score": child.rank_score,
        "prior_assessment": assess_local_edit_prior(
            active_prior,
            rule_type="component_replace",
            rule_key=rule_key,
        ),
    }


def make_backups(db_path: Path, *, dry_run: bool) -> dict[str, str]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_name = f"pre_smart_component_replacement_ablation_{ts}"
    return backup_database(
        db_path,
        backup_name,
        project_root=PROJECT_ROOT,
        google_backup_root=GOOGLE_BACKUP_ROOT,
        dry_run=dry_run,
    )


def inventory_payload(
    *,
    suites: list[PlannedSuite],
    rejected: list[dict[str, Any]],
    active_prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "created_at": time.time(),
        "suite_count": len(suites),
        "child_count": sum(len(suite.children) for suite in suites),
        "rejected_count": len(rejected),
        "suites": [
            {
                "parent_result_id": suite.parent.result_id,
                "node_id": suite.node_id,
                "original_op": suite.original_op,
                "component_class": suite.component_class,
                "prior_assessment": assess_local_edit_prior(
                    active_prior,
                    rule_type="component_replace",
                    rule_key=_suite_rule_key(suite),
                ),
                "children": [
                    {
                        "replacement_op": child.replacement_op,
                        "fingerprint": child.fingerprint,
                        "estimated_param_delta": child.estimated_param_delta,
                        "rank_score": child.rank_score,
                    }
                    for child in suite.children
                ],
            }
            for suite in suites
        ],
        "rejected_sample": rejected[:200],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--rank-offset", type=int, default=0)
    parser.add_argument(
        "--parent-result-id",
        action="append",
        default=[],
        help="Restrict planning/runs to selected parent result_id values.",
    )
    parser.add_argument("--max-children", type=int, default=100)
    parser.add_argument("--per-node-limit", type=int, default=5)
    parser.add_argument("--include-references", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help=(
            "Rebuild the deterministic plan without rejecting existing child "
            "fingerprints; run_ablation_suite will use historical-child dedupe "
            "and execute only missing children."
        ),
    )
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = configure_logging(args.log_file)
    db_path = Path(args.db)
    status_path = RUNTIME_DIR / "smart_component_replacement_ablation_status.json"
    inventory_path = RUNTIME_DIR / "smart_component_replacement_ablation_inventory.json"
    nb = LabNotebook(str(db_path), use_native=False)
    try:
        active_prior = get_active_construction_prior(nb)
        parents = select_top_parents(
            nb,
            top_k=max(1, int(args.top_k)),
            include_references=bool(args.include_references),
            rank_offset=max(0, int(args.rank_offset)),
            parent_result_ids=set(args.parent_result_id or []) or None,
        )
        if not parents:
            raise SystemExit("no eligible parent graphs found")
        for parent in parents:
            ensure_ablation_metric_completeness(nb, parent_result_id=parent.result_id)
            parent.config = prepare_config(parent.config, device=args.device)
        suites, rejected = build_plan(
            nb,
            parents=parents,
            max_children=max(1, int(args.max_children)),
            per_node_limit=max(1, int(args.per_node_limit)),
            allow_existing_fingerprints=bool(args.resume_partial),
        )
        inventory = inventory_payload(
            suites=suites,
            rejected=rejected,
            active_prior=active_prior,
        )
        json_dump(inventory_path, inventory)
        status = {
            "created_at": time.time(),
            "inventory_path": str(inventory_path),
            "log_path": str(log_path),
            "top_k": int(args.top_k),
            "rank_offset": max(0, int(args.rank_offset)),
            "parent_result_ids_filter": list(args.parent_result_id or []),
            "max_children": int(args.max_children),
            "planned_suites": inventory["suite_count"],
            "planned_children": inventory["child_count"],
            "rejected_count": inventory["rejected_count"],
            "parent_result_ids": [parent.result_id for parent in parents],
            "dry_run": bool(args.dry_run),
            "audit_only": bool(args.audit_only),
            "resume_partial": bool(args.resume_partial),
        }
        json_dump(status_path, status)
        log(
            "planned smart component replacement ablation "
            f"parents={len(parents)} suites={inventory['suite_count']} "
            f"children={inventory['child_count']} rejected={inventory['rejected_count']} "
            f"inventory={inventory_path}"
        )
        if args.dry_run or args.audit_only:
            log("audit/dry run complete; no DB writes or child training launched")
            return 0
        if not suites:
            log("no replacement suites generated")
            return 1
        backup_paths = {} if args.no_backup else make_backups(db_path, dry_run=False)
        if backup_paths:
            log(f"database backups created: {backup_paths}")
        runner = ExperimentRunner(str(db_path))
        completed: list[dict[str, Any]] = []
        started_at = time.time()
        for index, suite in enumerate(suites, start=1):
            candidate = make_candidate(suite, active_prior=active_prior)
            graphs = [child.graph for child in suite.children]
            meta = {
                child.fingerprint: child_meta(child, active_prior=active_prior)
                for child in suite.children
            }
            status.update(
                {
                    "started_at": started_at,
                    "updated_at": time.time(),
                    "completed_suites": len(completed),
                    "current_suite": {
                        "index": index,
                        "parent_result_id": suite.parent.result_id,
                        "rule_type": candidate.rule_type,
                        "rule_key": candidate.rule_key,
                        "children": len(suite.children),
                        "prior_assessment": assess_local_edit_prior(
                            active_prior,
                            rule_type=candidate.rule_type,
                            rule_key=candidate.rule_key,
                        ),
                        "replacement_ops": [
                            child.replacement_op for child in suite.children
                        ],
                    },
                    "latest_results": completed[-20:],
                }
            )
            json_dump(status_path, status)
            log(
                f"running suite {index}/{len(suites)} "
                f"{candidate.rule_type}:{candidate.rule_key} "
                f"parent={suite.parent.result_id} children={len(graphs)}"
            )
            baseline_validation = validate_graph(
                suite.parent.graph,
                max_ops=max(1, int(suite.parent.config.max_ops)),
                max_depth=max(1, int(suite.parent.config.max_depth)),
                min_splits=suite.parent.config.min_splits,
            )
            baseline_error_keys = {
                _validation_error_key(error) for error in baseline_validation.errors
            }
            result = run_ablation_suite(
                nb=nb,
                runner=runner,
                config=suite.parent.config,
                candidate=candidate,
                graphs=graphs,
                child_meta_by_fingerprint=meta,
                campaign="smart_component_replacement_ablation",
                extra_evidence_fields={
                    "component_class": suite.component_class,
                    "replacement_ops": [
                        child.replacement_op for child in suite.children
                    ],
                    "selection_policy": "class_fit + historical_s1_rate + parameter_efficiency",
                    "prior_assessment": assess_local_edit_prior(
                        active_prior,
                        rule_type=candidate.rule_type,
                        rule_key=candidate.rule_key,
                    ),
                },
                allowed_validation_error_keys=baseline_error_keys,
                exclude_failed_observations=True,
            )
            if result is not None:
                completed.append(result)
                log(
                    f"recorded evidence={result['evidence_id']} "
                    f"outcome={result['outcome']} effect={result['effect_size']} "
                    f"stage1={result['ablation_stage1_pass_count']}/{result['ablation_total']}"
                )
            status.update(
                {
                    "updated_at": time.time(),
                    "completed_suites": len(completed),
                    "latest_results": completed[-20:],
                }
            )
            json_dump(status_path, status)
        final_audit = {
            parent.result_id: ensure_ablation_metric_completeness(
                nb, parent_result_id=parent.result_id
            )
            for parent in parents
        }
        status.update(
            {
                "completed_at": time.time(),
                "completed_suites": len(completed),
                "final_metric_audit": final_audit,
                "latest_results": completed[-50:],
            }
        )
        json_dump(status_path, status)
        log(
            "smart component replacement ablation complete "
            f"completed_suites={len(completed)} final_metric_audit={final_audit}"
        )
        return 0
    finally:
        nb.close()


if __name__ == "__main__":
    raise SystemExit(main())
