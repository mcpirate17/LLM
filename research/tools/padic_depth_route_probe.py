"""Probe the p-adic depth-routing variant of recursive_depth_router vs the softmax original.

Two questions (the second is the whole point — today's scale run died because the softmax
depth-router collapsed to one expert in 6/8 blocks):
  1. CAPABILITY: does swapping the softmax depth-scorer for intrinsic p-adic ultrametric
     routing (`padic_depth_route`) preserve induction / AR-gate / nano-bind?
  2. ROUTER-ALIVE: after a short induction-task training, does the router stay SPREAD across
     the depth projections (high entropy) instead of collapsing to one?

The novelty claim: routing by the token's fixed p-adic valuation (not a free learned linear
gate) should resist the single-expert collapse the softmax router suffers.
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from research.tools.loss_monster_screen import select_family_champions, _RUNS_DB
from research.synthesis.serializer import graph_from_json
from research.scientist.native_runner import compile_model_native_first
from research.eval.induction_probe import induction_score, _generate_induction_batch
from research.eval.ar_gate import ar_gate, ar_gate_score, ar_gate_is_no_go
from research.eval.nano_bind import nano_bind


def _variant_graph(base: dict, swap_to: str | None) -> str:
    """Clone the champion graph; optionally swap the adaptive_recursion op for `swap_to`."""
    g = json.loads(json.dumps(base))  # deep copy
    if swap_to is not None:
        for n in g["nodes"].values():
            if n["op_name"] == "adaptive_recursion":
                n["op_name"] = swap_to
    return json.dumps(g)


def _router_weights(module, x: torch.Tensor) -> torch.Tensor | None:
    """Recompute the per-token depth-weight distribution for either router type."""
    if hasattr(module, "depth_anchors"):  # padic reciprocal router
        k = len(module.step_projs)
        val = (
            __import__("research.mathspaces.padic", fromlist=["padic_valuation"])
            .padic_valuation(x.float())
            .mean(dim=-1)
        )
        val = (val - val.mean()) / (val.std() + 1e-5)
        anchors = module.depth_anchors[:k].to(val.dtype)
        sharp = F.softplus(module.route_log_sharpness.to(val.dtype)) + 0.5
        dist = (val.unsqueeze(-1) - anchors).abs()
        inv = 1.0 / (1.0 + (dist * sharp).pow(2))
        return (inv / inv.sum(dim=-1, keepdim=True)).reshape(-1, k)
    if hasattr(module, "depth_scorer"):  # softmax router
        logits = x.to(module.depth_scorer.dtype) @ module.depth_scorer.t()
        return F.softmax(logits, dim=-1).reshape(-1, logits.shape[-1])
    return None


def _mean_router_entropy(model, device: str) -> float:
    """Mean depth-router entropy as % of uniform, over all router blocks, on real-ish input."""
    caps = []
    routers = [
        (n, m)
        for n, m in model.named_modules()
        if hasattr(m, "depth_anchors") or hasattr(m, "depth_scorer")
    ]
    hooks = []
    for _, m in routers:

        def mk(mod):
            def h(module, inp, out):
                x = inp[0] if isinstance(inp, tuple) and inp else None
                if x is None or x.dim() != 3:
                    return
                w = _router_weights(module, x)
                if w is not None:
                    ent = -(w * (w + 1e-9).log()).sum(-1).mean().item()
                    caps.append(ent / math.log(w.shape[-1]))

            return h

        hooks.append(m.register_forward_hook(mk(m)))
    ids = torch.randint(0, 256, (8, 64), device=device)
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    return round(sum(caps) / max(1, len(caps)), 4)


def _train_induction(model, steps: int, device: str, lr: float = 1e-3) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        ids, tgt = _generate_induction_batch(32, 8, device, generator=None)
        opt.zero_grad(set_to_none=True)
        logits = model(ids)
        loss = F.cross_entropy(logits[:, -1, :256], tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def probe(graph_json: str, label: str, args) -> dict:
    rec = {"variant": label}
    model = compile_model_native_first(
        [graph_from_json(graph_json)] * args.n_layers, vocab_size=512, max_seq_len=256
    ).to(args.device)
    rec["router_entropy_init"] = _mean_router_entropy(model, args.device)
    _train_induction(model, args.entropy_train_steps, args.device)
    rec["router_entropy_trained"] = _mean_router_entropy(model, args.device)

    ind = induction_score(
        model, n_train_steps=args.ind_steps, device=args.device, seed=0
    )
    rec["induction_auc"] = round(float(ind.auc), 4)
    ar = ar_gate(graph_json=graph_json, device=args.device)
    rec["ar_score"] = round(float(ar_gate_score(ar)), 4)
    rec["ar_is_no_go"] = bool(ar_gate_is_no_go(ar))
    nb = nano_bind(graph_json, device=args.device)
    rec["binding_no_go"] = bool(nb.is_no_go)
    print(
        f"  {label:28s} induction={rec['induction_auc']}  ar={rec['ar_score']}  "
        f"router_entropy init={rec['router_entropy_init']} -> trained="
        f"{rec['router_entropy_trained']} (% of uniform)",
        flush=True,
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--ind-steps", type=int, default=3000)
    ap.add_argument("--entropy-train-steps", type=int, default=1500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out",
        default="research/reports/loss_monsters/padic_depth_route_probe.json",
    )
    args = ap.parse_args()

    champ = next(
        c
        for c in select_family_champions(_RUNS_DB)
        if c.family == "recursive_depth_router"
    )
    base = json.loads(champ.graph_json)
    print("Probe: softmax depth-router vs p-adic ultrametric depth-router\n")
    results = [
        probe(_variant_graph(base, None), "softmax (original)", args),
        probe(
            _variant_graph(base, "padic_depth_route"), "padic_depth_route (fixed)", args
        ),
        probe(
            _variant_graph(base, "padic_gated_mixer"),
            "padic_gated_mixer (learned)",
            args,
        ),
    ]
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"config": vars(args), "results": results}, indent=2)
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
