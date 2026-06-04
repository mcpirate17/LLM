#!/usr/bin/env python
"""Published-architecture sanity set for the NAS gate stack.

Builds graph-DSL approximations of published sequence-model families, scores them
with the same label-free NAS oracle used by the CPU cascade, checks whether the
fingerprint already exists in the local DB, and emits a JSONL consumable by
``shortlist_cheap_probe_funnel`` and ``s1_top_shortlist``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.defaults import RUNS_DB
from research.synthesis.graph import ComputationGraph
from research.tools.cpu_screening_cascade import (
    CpuMechanismScorer,
    _capability_gate_passes,
)
from research.tools.learned_rules import score_template_quality


@dataclass(frozen=True)
class PublishedSpec:
    key: str
    paper: str
    url: str
    builder: Callable[[int], ComputationGraph]


def _residual(
    g: ComputationGraph, inp: int, op_name: str, config: dict | None = None
) -> int:
    norm = g.add_op("rmsnorm", [inp])
    mixed = g.add_op(op_name, [norm], config or {})
    return g.add_op("add", [inp, mixed])


def _ffn(g: ComputationGraph, inp: int) -> int:
    norm = g.add_op("rmsnorm", [inp])
    ff = g.add_op("swiglu_mlp", [norm])
    return g.add_op("add", [inp, ff])


def build_transformer(dim: int) -> ComputationGraph:
    g = ComputationGraph(dim)
    inp = g.add_input()
    out = _residual(g, inp, "softmax_attention")
    out = _ffn(g, out)
    g.set_output(out)
    g.metadata["published_family"] = "transformer"
    return g


def build_mamba(dim: int) -> ComputationGraph:
    g = ComputationGraph(dim)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    conv = g.add_op("conv1d_seq", [norm])
    act = g.add_op("silu", [conv])
    scan = g.add_op("selective_scan", [act])
    gate = g.add_op("gated_linear", [scan], {"out_dim": dim})
    out = g.add_op("add", [inp, gate])
    g.set_output(out)
    g.metadata["published_family"] = "mamba"
    return g


def build_rwkv(dim: int) -> ComputationGraph:
    g = ComputationGraph(dim)
    inp = g.add_input()
    out = _residual(g, inp, "rwkv_time_mixing")
    out = _residual(g, out, "rwkv_channel")
    g.set_output(out)
    g.metadata["published_family"] = "rwkv"
    return g


def build_hyena(dim: int) -> ComputationGraph:
    g = ComputationGraph(dim)
    inp = g.add_input()
    out = _residual(g, inp, "long_conv_hyena")
    out = _ffn(g, out)
    g.set_output(out)
    g.metadata["published_family"] = "hyena"
    return g


def build_retnet(dim: int) -> ComputationGraph:
    g = ComputationGraph(dim)
    inp = g.add_input()
    out = _residual(g, inp, "retention_mix")
    out = _ffn(g, out)
    g.set_output(out)
    g.metadata["published_family"] = "retnet"
    return g


def build_gla_surrogate(dim: int) -> ComputationGraph:
    g = ComputationGraph(dim)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    attn = g.add_op("linear_attention", [norm])
    gate = g.add_op("gated_linear", [norm], {"out_dim": dim})
    mixed = g.add_op("mul", [attn, gate])
    out = g.add_op("add", [inp, mixed])
    out = _ffn(g, out)
    g.set_output(out)
    g.metadata["published_family"] = "gated_linear_attention_surrogate"
    return g


def build_jamba_style_hybrid(dim: int) -> ComputationGraph:
    g = ComputationGraph(dim)
    inp = g.add_input()
    out = _residual(g, inp, "softmax_attention")
    norm = g.add_op("rmsnorm", [out])
    conv = g.add_op("conv1d_seq", [norm])
    scan = g.add_op("selective_scan", [conv])
    out = g.add_op("add", [out, scan])
    out = _ffn(g, out)
    g.set_output(out)
    g.metadata["published_family"] = "transformer_mamba_hybrid"
    return g


SPECS: tuple[PublishedSpec, ...] = (
    PublishedSpec(
        "transformer",
        "Attention Is All You Need",
        "https://arxiv.org/abs/1706.03762",
        build_transformer,
    ),
    PublishedSpec(
        "mamba",
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "https://arxiv.org/abs/2312.00752",
        build_mamba,
    ),
    PublishedSpec(
        "rwkv",
        "RWKV: Reinventing RNNs for the Transformer Era",
        "https://arxiv.org/abs/2305.13048",
        build_rwkv,
    ),
    PublishedSpec(
        "hyena",
        "Hyena Hierarchy: Towards Larger Convolutional Language Models",
        "https://arxiv.org/abs/2302.10866",
        build_hyena,
    ),
    PublishedSpec(
        "retnet",
        "Retentive Network: A Successor to Transformer for Large Language Models",
        "https://arxiv.org/abs/2307.08621",
        build_retnet,
    ),
    PublishedSpec(
        "gla_surrogate",
        "Gated Linear Attention Transformers with Hardware-Efficient Training",
        "https://arxiv.org/abs/2312.06635",
        build_gla_surrogate,
    ),
    PublishedSpec(
        "jamba_style_hybrid",
        "Hybrid Transformer/Mamba-style block",
        "https://arxiv.org/abs/2312.00752",
        build_jamba_style_hybrid,
    ),
)


def _graph_exists(conn: sqlite3.Connection, fingerprint: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM graphs WHERE graph_fingerprint = ? LIMIT 1",
        (fingerprint,),
    ).fetchone()
    return row is not None


def _ops(graph_dict: dict[str, Any]) -> list[str]:
    nodes = graph_dict.get("nodes", {})
    values = nodes.values() if isinstance(nodes, dict) else nodes
    return sorted({str(n["op_name"]) for n in values if not n.get("is_input")})


def build_records(dim: int, db_path: str, meta_db: str) -> list[dict[str, Any]]:
    scorer = CpuMechanismScorer(db_path, meta_db, use_probe_oracle=True)
    records: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        for spec in SPECS:
            graph = spec.builder(dim)
            graph_dict = graph.to_dict()
            fp = graph.fingerprint()
            profile = scorer.profile(graph_dict["nodes"])
            quality = score_template_quality(graph_dict["nodes"])
            probe = scorer.probe_oracle_score(graph_dict) or {}
            records.append(
                {
                    "fingerprint": fp,
                    "published_key": spec.key,
                    "published_paper": spec.paper,
                    "published_url": spec.url,
                    "db_present": _graph_exists(conn, fp),
                    "ops": _ops(graph_dict),
                    "mech_score": round(profile.mech_score, 3),
                    "novelty": profile.novelty,
                    "mixer_depth": profile.mixer_depth,
                    "n_mixers_on_path": profile.n_mix,
                    "n_novel_mixers": profile.n_novel_mix,
                    "lit_family": profile.lit_family,
                    "lit_model": profile.lit_model,
                    "lit_match_type": profile.lit_match_type,
                    "template_quality": quality["score"],
                    "failure_risk": quality["failure_risk"],
                    "nas_capability_gate_pass": _capability_gate_passes(probe),
                    "nas_pass": _capability_gate_passes(probe),
                    **probe,
                    "graph": graph_dict,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--db", default=str(RUNS_DB))
    parser.add_argument("--meta", default="research/meta_analysis.db")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("research/reports/published_nas_sanity.jsonl"),
    )
    args = parser.parse_args()

    records = build_records(args.dim, args.db, args.meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    summary = {
        "out": args.out.as_posix(),
        "n": len(records),
        "nas_pass": sum(1 for r in records if r["nas_pass"]),
        "db_present": sum(1 for r in records if r["db_present"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
