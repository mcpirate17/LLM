"""Multi-capability screening of parallel-sum ensembles of top-AR scaffolds.

Builds N-way parallel-sum ensembles from the DB's top-AR-curriculum graphs and
runs them through the standard mixer_fingerprint probe suite at screening
regime (vocab=cl100k_base, dim=256, n_blocks=1, 1000 wikitext steps).

Each ensemble lane is `mean_k( CompiledLayer(g_k)(x) )` over K top-AR graphs;
output back into the TinyLM body, then standard probe suite.

Probes reported per variant:
- AR-curriculum at production budget (1000 steps/stage, 32 eval batches)
- AR-curriculum at screening budget (mixer_fingerprint default, 200/stage)
- induction_intermediate, induction_screening, NB05, NI05
- binding_v2, binding_range, binding_curriculum, binding_multislot
- wikitext PPL, BLiMP, HellaSwag
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from research.defaults import VOCAB_SIZE
from research.eval.ar_curriculum_probe import (
    ARCurriculumConfig,
    ar_curriculum_probe,
    required_vocab_size,
)
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.synthesis.compiler import _compile_layer_module
from research.synthesis.serializer import graph_from_json
from research.tools.mixer_fingerprint import (
    _WarmupCosineSchedule,
    _cheap_evals,
    _expensive_core_evals,
    _expensive_enrichment_evals,
    _make_optimizer,
)
from research.tools.scaling_blimp_study import (
    _RandomWindowBatcher,
    _build_lane_factory,
    _build_tinylm,
    _load_wikitext_tokens,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "research" / "runs.db"

TOP_AR_FPS: tuple[tuple[str, str, float], ...] = (
    (
        "7fb0412ec57a1213",  # pragma: allowlist secret
        "dual: tropical + local_window + conv + swiglu",
        0.9046,
    ),  # pragma: allowlist secret
    (
        "13021b4ebe7adabe",  # pragma: allowlist secret
        "linear_attention + block_sparse_linear + softmax_last",
        0.8411,
    ),
    ("bb0b8d5856da1f29", "ablation variant", 0.7975),  # pragma: allowlist secret
    (
        "3b42e14e72f0fd95",  # pragma: allowlist secret
        "fingerprint_refine variant",
        0.7949,
    ),  # pragma: allowlist secret
)


def _load_graphs_by_fingerprint(
    graph_specs: tuple[tuple[str, str, float], ...],
) -> list[tuple[str, str, float, Any]]:
    out = []
    with sqlite3.connect(str(DB_PATH)) as conn:
        cur = conn.cursor()
        for fp, desc, db_auc in graph_specs:
            cur.execute(
                "SELECT graph_json FROM program_results "
                "WHERE graph_fingerprint=? AND graph_json IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT 1",
                (fp,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"no graph_json for fp {fp}")
            graph_json = resolve_graph_json_value(conn, DB_PATH, row[0])
            out.append((fp, desc, db_auc, graph_from_json(graph_json)))
    return out


# 2026-05-21: graphs from runs.db.program_results that simultaneously clear
# ar_curriculum_auc_pair_final > 0.4 AND binding_intermediate_auc > 0.7 AND
# induction_intermediate_auc > 0.5. Both share the `local_attn_ssm_hybrid`
# template (research/synthesis/_templates_attention_hybrid.py:468) — ops:
# local_window_attn + selective_scan + diff_attention + conv1d_seq + silu +
# linear_proj + rmsnorm. Powers the `cross_bias_Nway` lane.
CROSS_BIAS_FPS: tuple[tuple[str, str, float], ...] = (
    (
        "bb0b8d5856da1f29",  # pragma: allowlist secret
        "local_attn_ssm_hybrid (ar=0.797, bi=0.984, ii=1.0)",
        0.7975,
    ),
    (
        "5c5013c79d1f0a51",  # pragma: allowlist secret
        "local_attn_ssm_hybrid (ar=0.407, bi=0.995, ii=1.0)",
        0.4073,
    ),
)

# 2026-05-21: cluster 2 — bind > 0.7 AND ind > 0.5 AND ar > 0.3. Share
# `latent_compress_block + difficulty_routed_block + routed_bottleneck` templates
# — ops: softmax_attention + token_type_classifier + entropy_score + rope_rotate
# + spectral_filter. Powers the `routed_compress_Nway` lane (codex track).
ROUTED_COMPRESS_FPS: tuple[tuple[str, str, float], ...] = (
    (
        "176ec33b18a580bd",  # pragma: allowlist secret
        "latent_compress + difficulty_routed (ar=0.346, bi=0.995, ii=0.923)",
        0.3464,
    ),
    (
        "0c07a4d2c6ea74fa",  # pragma: allowlist secret
        "latent_compress + difficulty_routed (ar=0.300, bi=1.000, ii=0.985)",
        0.3001,
    ),
)


def _load_top_graphs(n: int):
    return _load_graphs_by_fingerprint(TOP_AR_FPS[:n])


def _load_cross_bias_graphs(n: int):
    return _load_graphs_by_fingerprint(CROSS_BIAS_FPS[:n])


def _load_routed_compress_graphs(n: int):
    return _load_graphs_by_fingerprint(ROUTED_COMPRESS_FPS[:n])


# 2026-05-21: ops whose input gets a `rope_rotate` injected when a graph passes
# through `_inject_rope_before_ops` — anything `attention`-shaped where Q/K
# scoring would benefit from explicit positional info. `selective_scan` and
# `conv1d_seq` are intentionally excluded: they're order-aware by construction
# (recurrent state / causal kernel), so RoPE adds nothing.
_ROPE_INJECTION_TARGETS = frozenset(
    {"local_window_attn", "diff_attention", "softmax_attention"}
)


def _inject_rope_before_ops(
    graph, target_op_names: frozenset[str] = _ROPE_INJECTION_TARGETS
):
    """Return a copy of `graph` with `rope_rotate` nodes inserted before targets.

    For every node whose `op_name` is in `target_op_names`, each input edge is
    redirected through a fresh `rope_rotate` node. Used by the `local_ssm_diff_rope`
    lane to give the cluster-1 attention ops the same position-aware Q/K
    scoring the three_lane blocks get via `_attach_rope_to_attention`.

    Why a graph-level mutation rather than patching `_op_local_window_attn`:
    the synthesis-graph attention ops use Q=K=V=x (no projections), so RoPE
    has to be applied at the input — at which point it's structurally identical
    to inserting a `rope_rotate` node on the input edge.
    """
    g = graph.copy()
    targets = [n for n in list(g.nodes.values()) if n.op_name in target_op_names]
    for tgt in targets:
        new_input_ids = []
        for src_id in tgt.input_ids:
            rope_id = g.add_op("rope_rotate", [src_id], {})
            new_input_ids.append(rope_id)
        tgt.input_ids[:] = new_input_ids
    g._ir_version += 1
    if g._cache:
        g._cache.clear()
    return g


def _rescaled_graph_copy(graph, new_dim: int):
    """Deep-copy a ComputationGraph and rewrite all dim-dependent fields to new_dim.

    Why: synthesis graph_jsons are baked at model_dim=256 (the native dim of the
    AR-curriculum experiments). To screen at the standard 8M-24M block range we
    need dim=512/640. Only two field families are dim-dependent across the 4 top
    AR graphs (audited 2026-05-19): node.output_shape['dim'] and
    node.config['out_dim'] (used by linear_proj / block_sparse_linear /
    shared_basis_proj). Everything else (rmsnorm, attention, conv1d, swiglu_mlp
    with mlp_ratio, sliding_window_mask) is dim-parametric.
    """
    g = copy.deepcopy(graph)
    g.model_dim = int(new_dim)
    for node in g.nodes.values():
        if isinstance(node.output_shape, dict) and "dim" in node.output_shape:
            node.output_shape["dim"] = int(new_dim)
        elif hasattr(node.output_shape, "dim"):
            node.output_shape.dim = int(new_dim)
        if isinstance(getattr(node, "config", None), dict) and "out_dim" in node.config:
            node.config["out_dim"] = int(new_dim)
    return g


def _make_ensemble_lane_factory(graph_specs) -> Callable[[int], nn.Module]:
    """Equal-weight parallel-sum of CompiledLayers from the supplied graphs.

    Why: empirical 2026-05-19 finding — uniform mean beat learned softmax gate
    at the AR-curriculum probe budget (gate doesn't converge in 6000 steps).
    Graphs are rescaled per-call via _rescaled_graph_copy so the same factory
    can be invoked at any TinyLM block dim (e.g., 640 for screening).
    """
    graphs = [spec[3] for spec in graph_specs]

    def factory(dim: int) -> nn.Module:
        rescaled = [
            (g if g.model_dim == dim else _rescaled_graph_copy(g, dim)) for g in graphs
        ]
        branch_executor = os.environ.get("SYNTHESIS_ENSEMBLE_EXECUTOR", "compiled")
        branches = nn.ModuleList(
            [
                _compile_layer_module(g, executor_variant=branch_executor)
                for g in rescaled
            ]
        )
        w = 1.0 / len(branches)

        class _ParallelSum(nn.Module):
            """Equal-weight parallel-sum over branches.

            Sequential by design. Two stream-parallel attempts (2026-05-20,
            see [[feedback_parallel_sum_threads_dont_help]]):
              1. Streams + inline join: 28.5 → 27.3 ms/block (~no overlap).
              2. Streams + deferred join + ``@_dynamo.disable``: +14 ms
                 regression because the disable boundary cost the outer
                 model's compile fusion more than parallelism saved.
            Both failed because each branch's CompiledLayer.forward is a
            Python orchestration loop; the CPU still enters each branch
            sequentially even when streams overlap on the GPU. Only L2
            code-gen (collapse the per-branch Python loop to a flat
            function) lets parallelism actually pay off.
            """

            def __init__(self) -> None:
                super().__init__()
                self.branches = branches
                self._w = w

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                acc = self.branches[0](x) * self._w
                for b in self.branches[1:]:
                    acc = acc + b(x) * self._w
                return acc

        return _ParallelSum()

    return factory


class _SwiGLU(nn.Module):
    """Standalone SwiGLU FFN matching component_fab.harness.top_ar_block.SwiGLU."""

    def __init__(self, dim: int, mlp_ratio: float = 3.0) -> None:
        super().__init__()
        h = int(round(dim * float(mlp_ratio)))
        self.w1 = nn.Linear(dim, h, bias=False)
        self.w2 = nn.Linear(dim, h, bias=False)
        self.w3 = nn.Linear(h, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class _ThreeLaneAsBlock(nn.Module):
    """`three_lane` wrapped as a self-contained block (RMSNorm + mixer + RMSNorm + FFN + residuals).

    Makes a generic mixer structurally comparable to the ensemble_4way branches
    (which contain their own RMSNorm/FFN/residual via the graph). Required for
    the cross-family hybrid so both sides contribute block-equivalent
    transformations. Uses RMSNorm to match the top AR-curriculum scaffolds
    (per [[feedback-rope-or-pe-required]]).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        from component_fab.generator.block_templates import ThreeLaneAdaptive
        from component_fab.generator.primitive_templates import (
            MultiscaleWaveletLane,
            SparsemaxAttention,
            TropicalAttention,
        )
        from component_fab.harness.top_ar_block import RMSNorm as _RMSNorm

        self.norm1 = _RMSNorm(dim)
        self.three_lane = ThreeLaneAdaptive(
            lambda d: TropicalAttention(d),
            lambda d: SparsemaxAttention(d),
            lambda d: MultiscaleWaveletLane(d),
            dim,
        )
        self.norm2 = _RMSNorm(dim)
        self.ffn = _SwiGLU(dim, mlp_ratio=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.three_lane(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def _make_ensemble_plus_three_lane_factory(graph_specs) -> Callable[[int], nn.Module]:
    """Cross-family hybrid: parallel-sum of (ensemble_4way of top-AR graphs) and (three_lane block).

    Tests whether AR specialty (from the graph ensemble) survives mixing with
    binding specialty (from three_lane). Both branches output block-equivalent
    transformations of the same input; summed with equal weight.
    """
    ensemble_factory = _make_ensemble_lane_factory(graph_specs)

    def factory(dim: int) -> nn.Module:
        ensemble_branch = ensemble_factory(dim)
        three_lane_branch = _ThreeLaneAsBlock(dim)

        class _CrossFamily(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ensemble_branch = ensemble_branch
                self.three_lane_branch = three_lane_branch

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 0.5 * self.ensemble_branch(x) + 0.5 * self.three_lane_branch(x)

        return _CrossFamily()

    return factory


def _train_wikitext(
    *,
    model: nn.Module,
    train_batcher,
    n_steps: int,
    learning_rate: float,
    warmup_steps: int,
    min_lr: float,
    device: torch.device,
    log_every: int = 100,
) -> None:
    opt, _ = _make_optimizer(model, learning_rate=learning_rate, device=device)
    sched = _WarmupCosineSchedule(
        opt,
        learning_rate=learning_rate,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        total_steps=n_steps,
    )
    model.train()
    for step in range(1, n_steps + 1):
        sched.apply(step - 1)
        batch = train_batcher.next()
        logits = model(batch[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1 or step == n_steps:
            ppl = float(torch.exp(loss).item())
            print(
                f"    step={step:5d}/{n_steps} loss={loss.item():.4f} "
                f"ppl={ppl:.1f} lr={opt.param_groups[0]['lr']:.2e}"
            )
    model.eval()


def _resolve_variant(name: str):
    """Map a variant name to (factory, graph_specs_or_None, n_branches_for_label, use_ffn).

    Ensemble names (`ensemble_Nway`, N in 1..len(TOP_AR_FPS)) build a
    parallel-sum lane over the top-N AR-curriculum scaffolds (graphs from
    runs.db). Their branches already contain norm/residual/FFN internally,
    so use_ffn=False on the TinyLM outer LaneBlock.

    Any other name is passed to `scaling_blimp_study._build_lane_factory`,
    which returns a generic mixer (TropicalAttention, three_lane,
    causal_conv, softmax_attention, etc.). These are mixers without
    internal FFN — use_ffn=True so the standard screening regime adds
    the outer FFN.
    """
    if name.startswith("ensemble_") and name.endswith("way"):
        n = int(name[len("ensemble_") : -len("way")])
        specs = _load_top_graphs(n)
        return _make_ensemble_lane_factory(specs), specs, n, False
    if name == "ensemble_4way_plus_three_lane":
        # Cross-family hybrid: top-4 AR graphs parallel-sum + three_lane block.
        # Both branches are block-equivalent transformations (have internal
        # norm/FFN/residual), so the outer TinyLM LaneBlock skips FFN.
        specs = _load_top_graphs(4)
        return _make_ensemble_plus_three_lane_factory(specs), specs, 5, False
    return _build_lane_factory(name), None, 1, True


def _swap_layernorms_to_rmsnorm(model: nn.Module) -> int:
    """Walk model and replace every nn.LayerNorm with component_fab RMSNorm.

    Why: validated 2026-05-19 ablation — wrapper LayerNorms cost ~0.02 AUC vs
    SynthesizedModel gold; RMSNorm swap closes the gap and matches the
    normalization scheme of the top AR-curriculum graphs (fp 7fb0412 + fp
    13021b4 are both RMSNorm-dominant). Per
    [[feedback-rope-or-pe-required]].
    """
    from component_fab.harness.top_ar_block import RMSNorm as _RMSNorm

    n_swapped = 0
    for _, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.LayerNorm):
                shape = child.normalized_shape
                d = shape[0] if isinstance(shape, tuple) else int(shape)
                replacement = _RMSNorm(int(d), eps=float(child.eps)).to(
                    child.weight.device
                )
                setattr(mod, cname, replacement)
                n_swapped += 1
    return n_swapped


def _build_screening_model_and_batchers(
    *,
    lane_factory: Callable[[int], nn.Module],
    use_ffn: bool,
    dim: int,
    n_blocks: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    seed: int,
):
    model = _build_tinylm(
        lane_factory,
        dim=dim,
        n_blocks=n_blocks,
        vocab_size=VOCAB_SIZE,
        use_ffn=use_ffn,
        use_rope=True,
        use_position_embedding=False,
    ).to(device)
    _swap_layernorms_to_rmsnorm(model)
    train_batcher = _RandomWindowBatcher(
        train_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        device=str(device),
        seed=42 + seed,
    )
    val_batcher = _RandomWindowBatcher(
        val_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        device=str(device),
        seed=123 + seed,
    )
    val_batches = val_batcher.fixed_batches(8)
    return model, train_batcher, val_batches, lane_factory


def _run_probe_battery(
    *,
    model: nn.Module,
    lane_factory: Callable[[int], nn.Module],
    val_batches: list[torch.Tensor],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    t0 = time.time()
    cheap_out = _cheap_evals(
        model=model,
        factory=lane_factory,
        val_batches=val_batches,
        device=device,
        seed=seed,
        amp=False,
        amp_dtype=torch.bfloat16,
    )
    out["wikitext_ppl"] = cheap_out.get("wikitext_ppl")
    out["cheap"] = cheap_out
    out["cheap_wall_s"] = round(time.time() - t0, 1)

    core_out: dict[str, Any] = {}
    t0 = time.time()
    _expensive_core_evals(model=model, device=device, out=core_out)
    out["core"] = core_out
    out["core_wall_s"] = round(time.time() - t0, 1)

    enrich_out: dict[str, Any] = {}
    t0 = time.time()
    _expensive_enrichment_evals(model=model, device=device, out=enrich_out)
    out["enrichment"] = enrich_out
    out["enrichment_wall_s"] = round(time.time() - t0, 1)
    return out


def _run_fresh_ar_prod(
    *,
    lane_factory: Callable[[int], nn.Module],
    use_ffn: bool,
    dim: int,
    n_blocks: int,
    device: torch.device,
    seed: int,
    steps_per_stage: int,
) -> dict[str, Any]:
    """Fresh model, no wikitext pretrain — matches the protocol that reproduces fp 7fb0412 at AUC 0.83."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    fresh_vocab = max(required_vocab_size(), 2048)
    model = _build_tinylm(
        lane_factory,
        dim=dim,
        n_blocks=n_blocks,
        vocab_size=fresh_vocab,
        use_ffn=use_ffn,
        use_rope=True,
        use_position_embedding=False,
    ).to(device)
    _swap_layernorms_to_rmsnorm(model)
    model = model.eval()
    t0 = time.time()
    r = ar_curriculum_probe(
        model,
        cfg=ARCurriculumConfig(
            seed=seed,
            steps_per_stage=steps_per_stage,
            batch_size=16,
            eval_batches=32,
            mode="cumulative",
        ),
        device=str(device),
    )
    n_params = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    return {
        "wall_s": round(time.time() - t0, 1),
        "vocab": fresh_vocab,
        "n_params": n_params,
        "result": r.to_dict(),
        "auc_pair_final": r.auc_pair_final,
        "s0": r.s0_held_pair_acc,
        "max_pass": r.max_passing_stage,
    }


def _make_headline(
    *, screening: dict[str, Any], ar_prod: dict[str, Any]
) -> dict[str, Any]:
    cheap = screening.get("cheap", {})
    core = screening.get("core", {})
    enrich = screening.get("enrichment", {})
    return {
        "wikitext_ppl": screening.get("wikitext_ppl"),
        "ar_prod_auc": ar_prod["auc_pair_final"],
        "ar_prod_s0": ar_prod["s0"],
        "ar_prod_max_pass": ar_prod["max_pass"],
        "ar_screening_auc": core.get("ar_curriculum", {}).get(
            "ar_curriculum_auc_pair_final"
        ),
        "induction_screening_auc": cheap.get("induction_screening_auc"),
        "induction_intermediate_auc": core.get("induction_intermediate", {}).get(
            "induction_intermediate_auc"
        ),
        "binding_v2_auc": core.get("binding_v2", {}).get("binding_intermediate_auc"),
        "binding_range_auc": enrich.get("binding_range", {}).get(
            "binding_screening_auc"
        ),
        "binding_curriculum_auc": enrich.get("binding_curriculum", {}).get(
            "binding_screening_auc"
        ),
        "binding_multislot_two_plus_slots_acc": enrich.get("binding_multislot", {}).get(
            "binding_multislot_two_plus_slots_acc"
        ),
        "induction_validation_auc": enrich.get("induction_validation", {}).get(
            "induction_validation_auc"
        ),
        "blimp_overall": cheap.get("blimp_overall"),
        "hellaswag_acc": cheap.get("hellaswag_acc"),
    }


def _print_variant_header(
    label: str, variant_name: str, seed: int, graph_specs
) -> None:
    print(f"\n=== variant: {label} ({variant_name}, seed={seed}) ===")
    if graph_specs is not None:
        print("  branches:")
        for fp, desc, db_auc, _ in graph_specs:
            print(f"    {fp[:12]} (db_auc={db_auc:.4f}) {desc}")
    else:
        print(f"  generic lane: {variant_name}")


def _run_variant(
    *,
    variant_name: str,
    label: str,
    dim: int,
    n_blocks: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    learning_rate: float,
    warmup_steps: int,
    min_lr: float,
    device: torch.device,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    seed: int,
    ar_prod_steps_per_stage: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    lane_factory, graph_specs, n_branches, use_ffn = _resolve_variant(variant_name)
    _print_variant_header(label, variant_name, seed, graph_specs)

    model, train_batcher, val_batches, _ = _build_screening_model_and_batchers(
        lane_factory=lane_factory,
        use_ffn=use_ffn,
        dim=dim,
        n_blocks=n_blocks,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        seed=seed,
    )
    n_total = sum(p.numel() for p in model.parameters())
    n_block = sum(p.numel() for b in model.blocks for p in b.parameters())
    print(f"  params total={n_total:,} block={n_block:,} use_ffn={use_ffn}")

    t0 = time.time()
    _train_wikitext(
        model=model,
        train_batcher=train_batcher,
        n_steps=steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        device=device,
    )
    train_wall = time.time() - t0
    screening = _run_probe_battery(
        model=model,
        lane_factory=lane_factory,
        val_batches=val_batches,
        device=device,
        seed=seed,
    )
    del model
    torch.cuda.empty_cache()

    print(f"  AR-curriculum prod budget ({ar_prod_steps_per_stage}/stage) on FRESH...")
    ar_prod = _run_fresh_ar_prod(
        lane_factory=lane_factory,
        use_ffn=use_ffn,
        dim=dim,
        n_blocks=n_blocks,
        device=device,
        seed=seed,
        steps_per_stage=ar_prod_steps_per_stage,
    )
    out: dict[str, Any] = {
        "label": label,
        "variant": variant_name,
        "n_branches": n_branches,
        "seed": seed,
        "n_params_total": n_total,
        "n_params_block": n_block,
        "use_ffn": use_ffn,
        "train_wall_s": round(train_wall, 1),
        "screening": screening,
        "ar_prod": ar_prod,
    }
    out["headline"] = _make_headline(screening=screening, ar_prod=ar_prod)
    print(f"  HEADLINE: {json.dumps(out['headline'], indent=2)}")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument(
        "--variants",
        type=str,
        default="ensemble_2way,ensemble_4way,tropical_sparsemax_wavelet_three_lane,causal_conv",
        help="Comma-separated variant names. Ensemble names: ensemble_{1,2,3,4}way. "
        "Generic lane names: any name supported by scaling_blimp_study._build_lane_factory.",
    )
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--ar-prod-steps-per-stage", type=int, default=1000)
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "research"
        / "reports"
        / "ensemble_screening_2026-05-19.jsonl",
    )
    return p.parse_args()


def _load_corpus():
    print(f"loading wikitext tokens at vocab={VOCAB_SIZE}...")
    train_tokens, val_tokens, _, _ = _load_wikitext_tokens(
        variant="wikitext-103-raw-v1",
        vocab_size=VOCAB_SIZE,
        max_chars_train=200_000_000,
        max_chars_val=2_000_000,
    )
    print(f"  train_tokens={train_tokens.numel():,} val_tokens={val_tokens.numel():,}")
    return train_tokens, val_tokens


def _print_summary(rows: list[dict[str, Any]], output: Path) -> None:
    print("\n==== FINAL SUMMARY ====")
    cols = (
        "label",
        "n_params_block",
        "wikitext_ppl",
        "ar_prod_auc",
        "ar_prod_max_pass",
        "ar_screening_auc",
        "induction_intermediate_auc",
        "binding_curriculum_auc",
        "binding_multislot_two_plus_slots_acc",
        "blimp_overall",
        "hellaswag_acc",
    )
    print(f"{'variant':<28} " + " ".join(f"{c:>14}" for c in cols[1:]))
    for r in rows:
        h = r["headline"]
        vals = [r["n_params_block"]] + [h.get(c) for c in cols[2:]]
        cells = []
        for v in vals:
            if isinstance(v, int):
                cells.append(f"{v:>14,}")
            elif isinstance(v, float):
                cells.append(f"{v:>14.4f}")
            else:
                cells.append(f"{str(v):>14}")
        print(f"{r['label']:<28} " + " ".join(cells))
    print(f"\nwrote {output}")


def main() -> int:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tokens, val_tokens = _load_corpus()

    variants = [s.strip() for s in args.variants.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for variant_name in variants:
            for seed in seeds:
                label = f"{variant_name}_seed{seed}"
                row = _run_variant(
                    variant_name=variant_name,
                    label=label,
                    dim=args.dim,
                    n_blocks=args.n_blocks,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    learning_rate=args.learning_rate,
                    warmup_steps=args.warmup_steps,
                    min_lr=args.min_lr,
                    device=device,
                    train_tokens=train_tokens,
                    val_tokens=val_tokens,
                    seed=seed,
                    ar_prod_steps_per_stage=args.ar_prod_steps_per_stage,
                )
                rows.append(row)
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()

    _print_summary(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
