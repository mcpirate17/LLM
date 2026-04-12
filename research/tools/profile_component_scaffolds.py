#!/usr/bin/env python
"""Controlled component scaffold profiler.

Profiles candidate operations inside small, fixed scaffolds so we can measure
compatibility and quality without confounding the whole architecture search.

Scaffold families:
    - gpt2_attn: GPT-2-style block with variable attention slot
    - gpt2_ffn: GPT-2-style block with variable FFN slot
    - gpt2_replace: GPT-2-style block with arbitrary post-attention replacement op
    - mamba_mixer: Mamba-style block with variable mixer/SSM slot
    - pair_residual: two residualized candidate ops in sequence

Usage:
    python -m research.tools.profile_component_scaffolds --top 20
    python -m research.tools.profile_component_scaffolds --ops local_window_attn linear_attention diff_attention
    python -m research.tools.profile_component_scaffolds --family gpt2_attn gpt2_ffn --json-out research/reports/scaffold_profile.json
    python -m research.tools.profile_component_scaffolds --family gpt2_ffn --ops block_sparse_linear kronecker_linear --allow-arbitrary-ops
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from itertools import islice, product
from pathlib import Path
from typing import Any, Iterable

from research.scientist.native_runner import compile_model_native_first
from research.scientist.runner import ExperimentRunner, RunConfig
from research.scientist.shared_utils import resolve_device
from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph
from research.synthesis.primitives import OP_NAME_ALIASES, PRIMITIVE_REGISTRY
from research.synthesis.serializer import graph_to_json
from research.tools._script_audit import (
    complete_script_experiment,
    fail_script_experiment,
    start_script_experiment,
)


_DEFAULT_MODEL_DIM = 128
_DEFAULT_SEQ_LEN = 128
_DEFAULT_VOCAB_SIZE = 8192
_DEFAULT_BATCH_SIZE = 4
_DEFAULT_STAGE1_STEPS = 40
_DEFAULT_LOG_PATH = Path("research/runtime/scaffold_profile/scaffold_profile.log")

_ATTN_OPS = (
    "softmax_attention",
    "local_window_attn",
    "linear_attention",
    "diff_attention",
    "graph_attention",
    "latent_attention_compressor",
)
_FFN_OPS = (
    "swiglu_mlp",
    "chebyshev_spectral_mix",
    "conv1d_seq",
    "hetero_moe",
    "kronecker_linear",
    "rwkv_channel",
    "nm_sparse_linear",
    "low_rank_proj",
    "bottleneck_proj",
    "sparse_bottleneck_moe",
    "spectral_filter",
)
_REPLACEMENT_OPS = (
    "arch_router",
    "compute_budget_router",
    "linear_attention",
    "diff_attention",
    "local_window_attn",
    "graph_attention",
    "latent_attention_compressor",
    "conv1d_seq",
    "rwkv_channel",
    "nm_sparse_linear",
)
_MIXER_OPS = (
    "selective_scan",
    "linear_attention",
    "diff_attention",
    "rwkv_channel",
    "latent_attention_compressor",
)
_PAIR_OPS = (
    "swiglu_mlp",
    "conv1d_seq",
    "rwkv_channel",
    "low_rank_proj",
    "bottleneck_proj",
)

_CATALOG_SCAFFOLD_FAMILY_BY_OP: dict[str, str] = {
    "adaptive_rank_gate": "gpt2_replace",
    "adjacent_token_merge": "gpt2_replace",
    "arch_router": "gpt2_replace",
    "cheap_verify_blend": "gpt2_replace",
    "chebyshev_spectral_mix": "gpt2_ffn",
    "compute_budget_router": "gpt2_replace",
    "confidence_token_gate": "gpt2_replace",
    "depth_gated_transform": "mamba_mixer",
    "depth_token_mask": "gpt2_replace",
    "depth_weighted_proj": "mamba_mixer",
    "difficulty_blend_3way": "gpt2_replace",
    "dual_compression_blend": "gpt2_ffn",
    "feature_sparsity": "gpt2_replace",
    "gated_lane_blend": "mamba_mixer",
    "hetero_moe": "gpt2_ffn",
    "kronecker_linear": "gpt2_ffn",
    "learned_token_gate": "gpt2_replace",
    "n_way_sparse_router": "gpt2_ffn",
    "relu_gated_moe": "gpt2_ffn",
    "score_depth_blend": "gpt2_replace",
    "signal_conditioned_compression": "gpt2_replace",
    "sparse_bottleneck_moe": "gpt2_ffn",
    "spectral_filter": "gpt2_ffn",
    "token_class_proj": "gpt2_replace",
    "token_entropy": "gpt2_replace",
}

_OP_CONFIGS: dict[str, dict[str, Any]] = {
    "linear_proj": {"out_dim": _DEFAULT_MODEL_DIM},
    "swiglu_mlp": {"mlp_ratio": 4.0},
    "local_window_attn": {"window_size": 16},
    "nm_sparse_linear": {"out_dim": _DEFAULT_MODEL_DIM},
    "low_rank_proj": {"out_dim": _DEFAULT_MODEL_DIM, "rank": 32},
    "bottleneck_proj": {"out_dim": _DEFAULT_MODEL_DIM // 2},
    "latent_attention_compressor": {"compression_ratio": 4},
}


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if mins:
        return f"{mins:d}m{secs:02d}s"
    return f"{secs:d}s"


def _append_log(log_path: Path | None, line: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        if not line.endswith("\n"):
            fh.write("\n")


def _emit(line: str, *, log_path: Path | None = None) -> None:
    print(line, flush=True)
    _append_log(log_path, line)


@dataclass(frozen=True)
class ScaffoldCase:
    family: str
    name: str
    op_a: str | None = None
    op_b: str | None = None


def recommended_scaffold_family(op_name: str) -> str | None:
    canonical = OP_NAME_ALIASES.get(op_name, op_name)
    if canonical in _ATTN_OPS:
        return "gpt2_attn"
    if canonical in _FFN_OPS:
        return "gpt2_ffn"
    if canonical in _REPLACEMENT_OPS:
        return "gpt2_replace"
    if canonical in _MIXER_OPS:
        return "mamba_mixer"
    if canonical in _PAIR_OPS:
        return "pair_residual"
    return _CATALOG_SCAFFOLD_FAMILY_BY_OP.get(canonical)


def catalog_scaffold_ops(families: Iterable[str]) -> list[str]:
    selected = set(families)
    return sorted(
        op
        for op, family in _CATALOG_SCAFFOLD_FAMILY_BY_OP.items()
        if family in selected
    )


def canonical_missing_profile_ops(profiled_ops: Iterable[str]) -> list[str]:
    profiled_canonical = {OP_NAME_ALIASES.get(op, op) for op in profiled_ops}
    primitive_canonical = {OP_NAME_ALIASES.get(op, op) for op in PRIMITIVE_REGISTRY}
    return sorted(primitive_canonical - profiled_canonical)


def _config_for(op_name: str, *, model_dim: int) -> dict[str, Any]:
    config = dict(_OP_CONFIGS.get(op_name, {}))
    if "out_dim" in config:
        raw_out_dim = int(config["out_dim"])
        config["out_dim"] = (
            model_dim
            if raw_out_dim == _DEFAULT_MODEL_DIM
            else max(4, raw_out_dim * model_dim // _DEFAULT_MODEL_DIM)
        )
    if "rank" in config:
        raw_rank = int(config["rank"])
        config["rank"] = max(4, raw_rank * model_dim // _DEFAULT_MODEL_DIM)
    return config


def _add(
    graph: ComputationGraph, op_name: str, input_id: int, *, model_dim: int
) -> int:
    return graph.add_op(
        op_name, [input_id], config=_config_for(op_name, model_dim=model_dim)
    )


def _fix_dim(graph: ComputationGraph, node_id: int) -> int:
    node = graph.nodes[node_id]
    if node.output_shape.dim == graph.model_dim and node.output_shape.seq == "S":
        return node_id
    return graph.add_op(
        "linear_proj",
        [node_id],
        config={"out_dim": graph.model_dim},
    )


def build_gpt2_attn_scaffold(
    attn_op: str,
    *,
    model_dim: int = _DEFAULT_MODEL_DIM,
) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    norm1 = _add(graph, "rmsnorm", inp, model_dim=model_dim)
    attended = _add(graph, attn_op, norm1, model_dim=model_dim)
    post_attn = _add(graph, "rmsnorm", _fix_dim(graph, attended), model_dim=model_dim)
    proj = _add(graph, "linear_proj", post_attn, model_dim=model_dim)
    mid = graph.add_op("add", [inp, _fix_dim(graph, proj)])
    norm2 = _add(graph, "rmsnorm", mid, model_dim=model_dim)
    ffn = _add(graph, "swiglu_mlp", norm2, model_dim=model_dim)
    out = graph.add_op("add", [mid, _fix_dim(graph, ffn)])
    graph.set_output(out)
    graph.metadata.update(
        {
            "scaffold_family": "gpt2_attn",
            "candidate_ops": [attn_op],
        }
    )
    return graph


def build_gpt2_ffn_scaffold(
    ffn_op: str,
    *,
    model_dim: int = _DEFAULT_MODEL_DIM,
) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    norm1 = _add(graph, "rmsnorm", inp, model_dim=model_dim)
    attended = _add(graph, "softmax_attention", norm1, model_dim=model_dim)
    post_attn = _add(graph, "rmsnorm", _fix_dim(graph, attended), model_dim=model_dim)
    proj = _add(graph, "linear_proj", post_attn, model_dim=model_dim)
    mid = graph.add_op("add", [inp, _fix_dim(graph, proj)])
    norm2 = _add(graph, "rmsnorm", mid, model_dim=model_dim)
    ffn = _add(graph, ffn_op, norm2, model_dim=model_dim)
    out = graph.add_op("add", [mid, _fix_dim(graph, ffn)])
    graph.set_output(out)
    graph.metadata.update(
        {
            "scaffold_family": "gpt2_ffn",
            "candidate_ops": [ffn_op],
        }
    )
    return graph


def build_gpt2_replace_scaffold(
    replacement_op: str,
    *,
    model_dim: int = _DEFAULT_MODEL_DIM,
) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    norm1 = _add(graph, "rmsnorm", inp, model_dim=model_dim)
    attended = _add(graph, "softmax_attention", norm1, model_dim=model_dim)
    post_attn = _add(graph, "rmsnorm", _fix_dim(graph, attended), model_dim=model_dim)
    proj = _add(graph, "linear_proj", post_attn, model_dim=model_dim)
    mid = graph.add_op("add", [inp, _fix_dim(graph, proj)])
    norm2 = _add(graph, "rmsnorm", mid, model_dim=model_dim)
    replaced = _add(graph, replacement_op, norm2, model_dim=model_dim)
    out = graph.add_op("add", [mid, _fix_dim(graph, replaced)])
    graph.set_output(out)
    graph.metadata.update(
        {
            "scaffold_family": "gpt2_replace",
            "candidate_ops": [replacement_op],
        }
    )
    return graph


def build_mamba_mixer_scaffold(
    mixer_op: str,
    *,
    model_dim: int = _DEFAULT_MODEL_DIM,
) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    norm1 = _add(graph, "rmsnorm", inp, model_dim=model_dim)
    conv = _add(graph, "conv1d_seq", norm1, model_dim=model_dim)
    conv_norm = _add(graph, "rmsnorm", _fix_dim(graph, conv), model_dim=model_dim)
    mixed = _add(graph, mixer_op, conv_norm, model_dim=model_dim)
    proj = _add(graph, "linear_proj", _fix_dim(graph, mixed), model_dim=model_dim)
    mid = graph.add_op("add", [inp, _fix_dim(graph, proj)])
    norm2 = _add(graph, "rmsnorm", mid, model_dim=model_dim)
    ffn = _add(graph, "swiglu_mlp", norm2, model_dim=model_dim)
    out = graph.add_op("add", [mid, _fix_dim(graph, ffn)])
    graph.set_output(out)
    graph.metadata.update(
        {
            "scaffold_family": "mamba_mixer",
            "candidate_ops": [mixer_op],
        }
    )
    return graph


def build_pair_residual_scaffold(
    op_a: str,
    op_b: str,
    *,
    model_dim: int = _DEFAULT_MODEL_DIM,
) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    norm1 = _add(graph, "rmsnorm", inp, model_dim=model_dim)
    a_out = _add(graph, op_a, norm1, model_dim=model_dim)
    mid = graph.add_op("add", [inp, _fix_dim(graph, a_out)])
    norm2 = _add(graph, "rmsnorm", mid, model_dim=model_dim)
    b_out = _add(graph, op_b, norm2, model_dim=model_dim)
    out = graph.add_op("add", [mid, _fix_dim(graph, b_out)])
    graph.set_output(out)
    graph.metadata.update(
        {
            "scaffold_family": "pair_residual",
            "candidate_ops": [op_a, op_b],
        }
    )
    return graph


def build_scaffold(case: ScaffoldCase, *, model_dim: int) -> ComputationGraph:
    if case.family == "gpt2_attn":
        return build_gpt2_attn_scaffold(
            case.op_a or "softmax_attention", model_dim=model_dim
        )
    if case.family == "gpt2_ffn":
        return build_gpt2_ffn_scaffold(case.op_a or "swiglu_mlp", model_dim=model_dim)
    if case.family == "gpt2_replace":
        return build_gpt2_replace_scaffold(
            case.op_a or "linear_attention", model_dim=model_dim
        )
    if case.family == "mamba_mixer":
        return build_mamba_mixer_scaffold(
            case.op_a or "selective_scan", model_dim=model_dim
        )
    if case.family == "pair_residual":
        return build_pair_residual_scaffold(
            case.op_a or "swiglu_mlp",
            case.op_b or "swiglu_mlp",
            model_dim=model_dim,
        )
    raise ValueError(f"Unknown scaffold family: {case.family}")


def generate_cases(
    families: Iterable[str],
    ops: list[str] | None,
    *,
    max_pairs: int,
    allow_arbitrary_ops: bool = False,
) -> list[ScaffoldCase]:
    def _family_candidates(
        supplied_ops: tuple[str, ...],
        defaults: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not supplied_ops:
            return defaults
        if allow_arbitrary_ops:
            return supplied_ops
        filtered = tuple(op for op in supplied_ops if op in defaults)
        return filtered or defaults

    selected_families = tuple(families)
    op_list = tuple(ops or ())
    cases: list[ScaffoldCase] = []
    for family in selected_families:
        if family == "gpt2_attn":
            cases.append(
                ScaffoldCase(
                    family=family, name="gpt2_attn:control", op_a="softmax_attention"
                )
            )
            candidates = _family_candidates(op_list, _ATTN_OPS)
            cases.extend(
                ScaffoldCase(family=family, name=f"gpt2_attn:{op}", op_a=op)
                for op in candidates
            )
        elif family == "gpt2_ffn":
            cases.append(
                ScaffoldCase(family=family, name="gpt2_ffn:control", op_a="swiglu_mlp")
            )
            candidates = _family_candidates(op_list, _FFN_OPS)
            cases.extend(
                ScaffoldCase(family=family, name=f"gpt2_ffn:{op}", op_a=op)
                for op in candidates
            )
        elif family == "gpt2_replace":
            cases.append(
                ScaffoldCase(
                    family=family,
                    name="gpt2_replace:control",
                    op_a="swiglu_mlp",
                )
            )
            candidates = _family_candidates(op_list, _REPLACEMENT_OPS)
            cases.extend(
                ScaffoldCase(family=family, name=f"gpt2_replace:{op}", op_a=op)
                for op in candidates
            )
        elif family == "mamba_mixer":
            cases.append(
                ScaffoldCase(
                    family=family, name="mamba_mixer:control", op_a="selective_scan"
                )
            )
            candidates = _family_candidates(op_list, _MIXER_OPS)
            cases.extend(
                ScaffoldCase(family=family, name=f"mamba_mixer:{op}", op_a=op)
                for op in candidates
            )
        elif family == "pair_residual":
            cases.append(
                ScaffoldCase(
                    family=family,
                    name="pair_residual:control",
                    op_a="swiglu_mlp",
                    op_b="swiglu_mlp",
                )
            )
            pair_ops = tuple(op_list or _PAIR_OPS)
            for op_a, op_b in islice(product(pair_ops, pair_ops), max_pairs):
                cases.append(
                    ScaffoldCase(
                        family=family,
                        name=f"pair_residual:{op_a}+{op_b}",
                        op_a=op_a,
                        op_b=op_b,
                    )
                )
        else:
            raise ValueError(f"Unknown family: {family}")
    return cases


def _make_config(args: argparse.Namespace) -> RunConfig:
    config = RunConfig(
        n_programs=1,
        device=args.device,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        stage1_steps=args.stage1_steps,
        stage1_batch_size=args.batch_size,
        stage1_val_batches=0,
        stage1_discovery_batches=0,
        stage1_compute_val_loss=False,
        stage1_compute_discovery_loss=False,
        profile_disable_post_eval=True,
        enable_stage09_cheap_train_gate=False,
        progressive_screening=False,
        gbm_prescreener_enabled=False,
        persist_screening_failures=False,
        data_mode=args.data_mode,
        optimizer_type=args.optimizer,
    )
    return config


def _baseline_key(family: str) -> str:
    return {
        "gpt2_attn": "gpt2_attn:control",
        "gpt2_ffn": "gpt2_ffn:control",
        "gpt2_replace": "gpt2_replace:control",
        "mamba_mixer": "mamba_mixer:control",
        "pair_residual": "pair_residual:control",
    }[family]


def evaluate_case(
    case: ScaffoldCase,
    *,
    runner: ExperimentRunner,
    config: RunConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    graph = build_scaffold(case, model_dim=config.model_dim)
    graph_json = graph_to_json(graph)
    graph_fingerprint = graph.fingerprint()
    try:
        dev = resolve_device(config.device)
        compile_started = time.perf_counter()
        layer_graphs = [graph] * config.n_layers
        if str(dev) == "cuda":
            model = compile_model_native_first(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            )
        else:
            model = compile_model(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            )
        compile_ms = (time.perf_counter() - compile_started) * 1000.0
        sandbox = runner._safe_eval_for_stage(
            model,
            stage_tag=f"scaffold_{case.family}",
            batch_size=min(config.stage1_batch_size, _DEFAULT_BATCH_SIZE),
            seq_len=min(config.max_seq_len, _DEFAULT_SEQ_LEN),
            vocab_size=config.vocab_size,
            device=str(dev),
            timeout_seconds=30,
        )
        result: dict[str, Any] = {
            "name": case.name,
            "family": case.family,
            "op_a": case.op_a,
            "op_b": case.op_b,
            "graph_json": graph_json,
            "graph_fingerprint": graph_fingerprint,
            "compile_time_ms": compile_ms,
            "sandbox_passed": bool(getattr(sandbox, "passed", False)),
            "stability_score": getattr(sandbox, "stability_score", None),
            "causality_passed": getattr(sandbox, "causality_passed", None),
            "param_count": getattr(sandbox, "param_count", None),
        }
        if not result["sandbox_passed"]:
            result["status"] = "screen_fail"
            result["elapsed_s"] = time.perf_counter() - started
            return result
        train = runner._micro_train(model, config, dev, graph_json=graph_json)
        result.update(
            {
                "status": "ok",
                "loss_ratio": train.get("loss_ratio"),
                "validation_loss_ratio": train.get("validation_loss_ratio"),
                "discovery_loss_ratio": train.get("discovery_loss_ratio"),
                "final_loss": train.get("final_loss"),
                "avg_step_time_ms": train.get("avg_step_time_ms"),
                "throughput_tok_s": train.get("throughput")
                or train.get("throughput_tok_s"),
                "passed": bool(train.get("passed", False)),
                "error": train.get("error"),
            }
        )
        result["elapsed_s"] = time.perf_counter() - started
        return result
    except Exception as exc:
        return {
            "name": case.name,
            "family": case.family,
            "op_a": case.op_a,
            "op_b": case.op_b,
            "graph_json": graph_json,
            "graph_fingerprint": graph_fingerprint,
            "status": "error",
            "error": str(exc),
            "elapsed_s": time.perf_counter() - started,
        }


def annotate_with_baselines(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_family = {
        item["family"]: item
        for item in results
        if item["name"] == _baseline_key(item["family"])
    }
    annotated: list[dict[str, Any]] = []
    for item in results:
        baseline = baseline_by_family.get(item["family"])
        merged = dict(item)
        if baseline and item["name"] != baseline["name"]:
            base_loss = baseline.get("validation_loss_ratio") or baseline.get(
                "loss_ratio"
            )
            cur_loss = item.get("validation_loss_ratio") or item.get("loss_ratio")
            if base_loss is not None and cur_loss is not None:
                merged["delta_loss_ratio_vs_control"] = float(cur_loss) - float(
                    base_loss
                )
            base_tp = baseline.get("throughput_tok_s")
            cur_tp = item.get("throughput_tok_s")
            if base_tp is not None and cur_tp is not None:
                merged["delta_throughput_tok_s_vs_control"] = float(cur_tp) - float(
                    base_tp
                )
        annotated.append(merged)
    return annotated


def render_report(results: list[dict[str, Any]], *, top: int) -> str:
    ordered = sorted(
        results,
        key=lambda item: (
            0 if item.get("name", "").endswith(":control") else 1,
            item.get("delta_loss_ratio_vs_control", 999.0)
            if item.get("delta_loss_ratio_vs_control") is not None
            else 999.0,
            item.get("validation_loss_ratio")
            if item.get("validation_loss_ratio") is not None
            else item.get("loss_ratio")
            if item.get("loss_ratio") is not None
            else 999.0,
            item["name"],
        ),
    )
    lines = []
    lines.append(
        f"{'case':<38} {'status':<10} {'loss':>7} {'val':>7} {'dloss':>7} {'tok/s':>8} {'dtok':>8}"
    )
    lines.append("-" * 96)
    for item in ordered[:top]:
        loss = item.get("loss_ratio")
        val = item.get("validation_loss_ratio")
        dloss = item.get("delta_loss_ratio_vs_control")
        tp = item.get("throughput_tok_s")
        dtp = item.get("delta_throughput_tok_s_vs_control")
        lines.append(
            f"{item['name']:<38} {item.get('status', '?'):<10} "
            f"{(f'{loss:.3f}' if isinstance(loss, (int, float)) else 'n/a'):>7} "
            f"{(f'{val:.3f}' if isinstance(val, (int, float)) else 'n/a'):>7} "
            f"{(f'{dloss:+.3f}' if isinstance(dloss, (int, float)) else 'n/a'):>7} "
            f"{(f'{tp:.1f}' if isinstance(tp, (int, float)) else 'n/a'):>8} "
            f"{(f'{dtp:+.1f}' if isinstance(dtp, (int, float)) else 'n/a'):>8}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile controlled component scaffolds"
    )
    parser.add_argument(
        "--family",
        nargs="*",
        default=["gpt2_attn", "gpt2_ffn", "mamba_mixer", "pair_residual"],
        choices=[
            "gpt2_attn",
            "gpt2_ffn",
            "gpt2_replace",
            "mamba_mixer",
            "pair_residual",
        ],
    )
    parser.add_argument("--ops", nargs="*", default=None)
    parser.add_argument(
        "--include-unprofiled-catalog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append catalog-mapped unprofiled ops for the selected families.",
    )
    parser.add_argument(
        "--allow-arbitrary-ops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use --ops exactly as provided instead of filtering to each family's default candidate set",
    )
    parser.add_argument("--db", type=Path, default=Path("research/lab_notebook.db"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--data-mode", choices=["random", "corpus"], default="corpus")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--model-dim", type=int, default=_DEFAULT_MODEL_DIM)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=_DEFAULT_SEQ_LEN)
    parser.add_argument("--vocab-size", type=int, default=_DEFAULT_VOCAB_SIZE)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--stage1-steps", type=int, default=_DEFAULT_STAGE1_STEPS)
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=_DEFAULT_LOG_PATH,
        help="Append progress and final report to this log file",
    )
    parser.add_argument(
        "--persist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist scaffold profiling runs/results into the main notebook DB",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream per-case progress with elapsed time and ETA",
    )
    args = parser.parse_args()

    runner = ExperimentRunner(str(args.db))
    config = _make_config(args)
    requested_ops = list(args.ops or [])
    if args.include_unprofiled_catalog:
        requested_ops.extend(catalog_scaffold_ops(args.family))
        requested_ops = sorted(dict.fromkeys(requested_ops))

    cases = generate_cases(
        args.family,
        requested_ops or None,
        max_pairs=args.max_pairs,
        allow_arbitrary_ops=bool(args.allow_arbitrary_ops),
    )
    run_id = f"scaffold_{uuid.uuid4().hex[:10]}"
    log_path = args.log_file
    notebook = None
    audit_nb = None
    exp_id = None
    try:
        if args.persist:
            audit_nb, exp_id = start_script_experiment(
                db_path=args.db,
                experiment_type="scaffold_profiling",
                config={
                    "families": list(args.family),
                    "ops": requested_ops,
                    "allow_arbitrary_ops": bool(args.allow_arbitrary_ops),
                    "include_unprofiled_catalog": bool(args.include_unprofiled_catalog),
                    "device": args.device,
                    "data_mode": args.data_mode,
                    "optimizer": args.optimizer,
                    "model_dim": args.model_dim,
                    "n_layers": args.n_layers,
                    "seq_len": args.seq_len,
                    "vocab_size": args.vocab_size,
                    "batch_size": args.batch_size,
                    "stage1_steps": args.stage1_steps,
                    "max_pairs": args.max_pairs,
                    "top": args.top,
                    "persist": bool(args.persist),
                },
                source_script="profile_component_scaffolds",
                hypothesis="Profile controlled component scaffolds",
            )
            notebook = audit_nb
            notebook.save_scaffold_profile_run(
                run_id=run_id,
                config=config.to_dict(),
                device=args.device,
                metadata={
                    "experiment_id": exp_id,
                    "families": list(args.family),
                    "ops": requested_ops,
                    "allow_arbitrary_ops": bool(args.allow_arbitrary_ops),
                    "include_unprofiled_catalog": bool(args.include_unprofiled_catalog),
                    "max_pairs": int(args.max_pairs),
                    "top": int(args.top),
                },
            )
        if args.progress:
            _emit(
                f"Scaffold profiler starting: {len(cases)} cases "
                f"(families={','.join(args.family)} device={args.device} "
                f"steps={args.stage1_steps} dim={args.model_dim} layers={args.n_layers}"
                f" arbitrary_ops={'on' if args.allow_arbitrary_ops else 'off'}"
                f" include_unprofiled={'on' if args.include_unprofiled_catalog else 'off'}"
                f"{f' run_id={run_id}' if args.persist else ''})",
                log_path=log_path,
            )
        results = []
        started_all = time.perf_counter()
        for idx, case in enumerate(cases, start=1):
            case_started = time.perf_counter()
            if args.progress:
                _emit(
                    f"[{idx}/{len(cases)}] start {case.name}",
                    log_path=log_path,
                )
            result = evaluate_case(case, runner=runner, config=config)
            results.append(result)
            if notebook is not None:
                notebook.save_scaffold_profile_result(
                    run_id=run_id,
                    family=case.family,
                    case_name=case.name,
                    status=str(result.get("status") or "unknown"),
                    metrics=result,
                    graph_json=result.get("graph_json"),
                    graph_fingerprint=result.get("graph_fingerprint"),
                    op_a=case.op_a,
                    op_b=case.op_b,
                )
            if args.progress:
                elapsed_all = time.perf_counter() - started_all
                avg_case = elapsed_all / max(idx, 1)
                remaining = avg_case * max(len(cases) - idx, 0)
                loss = result.get("validation_loss_ratio")
                if loss is None:
                    loss = result.get("loss_ratio")
                loss_str = f"{loss:.3f}" if isinstance(loss, (int, float)) else "n/a"
                tp = result.get("throughput_tok_s")
                tp_str = f"{tp:.1f}" if isinstance(tp, (int, float)) else "n/a"
                _emit(
                    f"[{idx}/{len(cases)}] done  {case.name} "
                    f"status={result.get('status', '?')} "
                    f"loss={loss_str} tok/s={tp_str} "
                    f"case={_format_duration(time.perf_counter() - case_started)} "
                    f"elapsed={_format_duration(elapsed_all)} "
                    f"eta={_format_duration(remaining)}",
                    log_path=log_path,
                )
        results = annotate_with_baselines(results)
        report = render_report(results, top=args.top)
        _emit(report, log_path=log_path)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
        if audit_nb is not None and exp_id is not None:
            ok = sum(1 for item in results if item.get("status") == "ok")
            errors = sum(1 for item in results if item.get("status") == "error")
            screen_fail = sum(
                1 for item in results if item.get("status") == "screen_fail"
            )
            complete_script_experiment(
                audit_nb,
                exp_id,
                results={
                    "cases": len(results),
                    "ok": ok,
                    "errors": errors,
                    "screen_fail": screen_fail,
                    "families": list(args.family),
                },
                summary=(
                    f"Scaffold profiling complete: ok={ok} "
                    f"errors={errors} screen_fail={screen_fail}"
                ),
            )
    except KeyboardInterrupt:
        if audit_nb is not None and exp_id is not None:
            fail_script_experiment(
                audit_nb,
                exp_id,
                error="KeyboardInterrupt",
            )
        raise
    except Exception as exc:
        if audit_nb is not None and exp_id is not None:
            fail_script_experiment(
                audit_nb,
                exp_id,
                error=str(exc),
            )
        raise
    finally:
        if audit_nb is not None:
            audit_nb.close()
        elif notebook is not None:
            notebook.close()


if __name__ == "__main__":
    main()
