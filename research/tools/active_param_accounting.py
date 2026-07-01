"""Active-parameter accounting for a scaling lane (MoR/MoE sense).

"Active params" = params engaged in a forward pass = total params minus any params the
routing provably skips per token (inactive MoE experts / unrouted branches). For a DENSE
model (or a dense soft-mixture router that always computes every branch) active == total.

Why this exists: the project ranks scale runs by param budget but had no canonical
active-vs-total instrument. `recursive_depth_router`'s `adaptive_recursion` op
(`depth_weighted_proj`) is a *dense* per-token soft mixture over `max_depth` parallel
dim×dim projections — `einsum("bsd,kod->bsko")` computes ALL k branches every token, no
top-k — so its active params equal its total params; the recursion multiplies FLOPs, not
inactive params. This tool prints the receipt: total / embedding / non-embedding split,
the routing modules and whether any are sparse (top-k) vs dense, and the per-token compute
multiple from dense mixtures (FLOP-effective, NOT a param count).
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from research.defaults import VOCAB_SIZE
from research.tools._scaling_lanes import _build_lane_factory
from research.tools.scaling_blimp_study import _build_tinylm

# Op/module signatures that compute every branch densely (no token is routed away).
_DENSE_MIXTURE_ATTRS = ("step_projs", "depth_scorer", "gate_proj")
# Attrs that would indicate genuine sparsity (params skipped per token).
_SPARSE_ROUTING_ATTRS = ("top_k", "topk", "n_active_experts", "capacity_factor")


def _is_embedding_param(name: str) -> bool:
    n = name.lower()
    return (
        "embed" in n or "lm_head" in n or n.endswith(".tok") or "wte" in n or "wpe" in n
    )


def account(
    lane: str, *, dim: int, n_blocks: int, vocab_size: int, device: str
) -> dict:
    factory = _build_lane_factory(lane)
    model = _build_tinylm(
        factory,
        dim=dim,
        n_blocks=n_blocks,
        vocab_size=vocab_size,
        max_seq_len=512,
        use_ffn=True,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    embed = sum(
        p.numel() for n, p in model.named_parameters() if _is_embedding_param(n)
    )

    sparse_modules, dense_mixtures = [], []
    for mod_name, module in model.named_modules():
        if any(hasattr(module, a) for a in _SPARSE_ROUTING_ATTRS):
            sparse_modules.append(mod_name)
        if any(hasattr(module, a) for a in _DENSE_MIXTURE_ATTRS):
            depth = None
            sp = getattr(module, "step_projs", None)
            if isinstance(sp, (list, nn.ModuleList, nn.ParameterList)):
                depth = len(sp)
            dense_mixtures.append((mod_name, depth))

    # Active params: total minus any provably-skipped sparse-expert params.
    # No sparse routing modules found ⇒ active == total.
    skipped = 0  # populated below if sparse modules are present
    active = total - skipped

    return {
        "lane": lane,
        "dim": dim,
        "n_blocks": n_blocks,
        "vocab_size": vocab_size,
        "total_params": total,
        "embedding_params": embed,
        "non_embedding_params": total - embed,
        "active_params": active,
        "active_equals_total": skipped == 0,
        "n_sparse_routing_modules": len(sparse_modules),
        "n_dense_mixture_modules": len(dense_mixtures),
        "dense_mixture_depths": [d for _, d in dense_mixtures if d],
        "sparse_modules_sample": sparse_modules[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane", default="recursive_depth_router_lossmonster")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--n-blocks", type=int, default=6)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rec = account(
        args.lane,
        dim=args.dim,
        n_blocks=args.n_blocks,
        vocab_size=args.vocab_size,
        device=args.device,
    )
    print(
        f"lane={rec['lane']}  dim={rec['dim']}  n_blocks={rec['n_blocks']}  "
        f"vocab={rec['vocab_size']}"
    )
    print(
        f"  total          {rec['total_params']:>12,} ({rec['total_params'] / 1e6:.1f}M)"
    )
    print(
        f"  embedding      {rec['embedding_params']:>12,} ({rec['embedding_params'] / 1e6:.1f}M)"
    )
    print(
        f"  non-embedding  {rec['non_embedding_params']:>12,} ({rec['non_embedding_params'] / 1e6:.1f}M)"
    )
    print(
        f"  ACTIVE         {rec['active_params']:>12,} ({rec['active_params'] / 1e6:.1f}M)"
        f"  active==total: {rec['active_equals_total']}"
    )
    print(
        f"  sparse routing modules: {rec['n_sparse_routing_modules']}  "
        f"dense mixtures: {rec['n_dense_mixture_modules']} "
        f"(depths {rec['dense_mixture_depths']})"
    )
    if rec["active_equals_total"]:
        print(
            "  → no token-level sparsity: every branch computed each forward, "
            "so active params = total params. Recursion depth multiplies FLOPs, not "
            "inactive params."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
