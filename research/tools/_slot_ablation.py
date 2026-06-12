"""Ablation of SlotTableMemoryLane improvements, at a scale with signal.

Nano (dim64) hardened tasks are floored (~0.5 for everyone), so improvements
can't be measured there. This runs a CUMULATIVE ablation at larger dim (where
signal may emerge — also probing the open 'does scale unfloor it' question):
  base -> +multihead(#1) -> +decay(#2) -> +hard_route(#4)
on interference + compositional, with softmax_4h as the attention reference.
(#3 delta-rule within a slot is sequential; added later only if scale shows signal.)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn

from component_fab.harness.binding_validity import (
    DEFAULT_BINDING_VALIDITY_TASKS,
    run_binding_validity_task,
)
from component_fab.harness.tiny_lm import MultiHeadCausalAttention


class SlotAblation(nn.Module):
    """Configurable slot-table lane: multi-head slots, per-head decay, hard routing."""

    def __init__(
        self,
        dim: int,
        n_slots: int = 16,
        memory_dim: int | None = None,
        n_heads: int = 1,
        use_decay: bool = False,
        hard_route: bool = False,
    ) -> None:
        super().__init__()
        memory_dim = memory_dim or dim
        if memory_dim % n_heads != 0:
            n_heads = 1
        self.h = n_heads
        self.hd = memory_dim // n_heads
        self.s = n_slots
        self.m = memory_dim
        self.use_decay = use_decay
        self.hard_route = hard_route
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)
        self.route = nn.Linear(memory_dim, n_heads * n_slots)
        self.out = nn.Linear(memory_dim, dim, bias=False)
        self.route_log_temp = nn.Parameter(torch.zeros(1))
        if use_decay:
            self.decay_logit = nn.Parameter(
                torch.full((n_heads,), 3.0)
            )  # sigmoid(3)~0.95
        self.read_scale = float(self.hd) ** -0.5

    def _heads(self, t):  # [B,L,m] -> [B,L,H,hd]
        b, l, _ = t.shape
        return t.view(b, l, self.h, self.hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        k = self._heads(torch.tanh(self.k(x)))  # [B,L,H,hd]
        v = self._heads(self.v(x))
        q = self._heads(torch.tanh(self.q(x)))
        temp = self.route_log_temp.exp().clamp(min=1e-2)
        rl = self.route(torch.tanh(self.k(x))).view(b, l, self.h, self.s) / temp
        route = torch.softmax(rl, dim=-1)  # [B,L,H,S]
        if self.hard_route:  # straight-through onehot
            idx = route.argmax(dim=-1, keepdim=True)
            hard = torch.zeros_like(route).scatter_(-1, idx, 1.0)
            route = hard + route - route.detach()

        wk = route.unsqueeze(-1) * k.unsqueeze(3)  # [B,L,H,S,hd]
        wv = route.unsqueeze(-1) * v.unsqueeze(3)
        rw = route  # [B,L,H,S]
        if self.use_decay:
            gamma = torch.sigmoid(self.decay_logit)  # [H]
            pos = torch.arange(l, device=x.device)
            rel = (pos.view(l, 1) - pos.view(1, l)).clamp(min=0)  # [L,L] t-t'
            # decay_mat[h,t,t'] = gamma_h^(t-t') for t'<=t else 0
            mask = (pos.view(l, 1) >= pos.view(1, l)).float()  # lower-tri
            dm = (gamma.view(self.h, 1, 1) ** rel.view(1, l, l)) * mask  # [H,L,L]
            slot_key = torch.einsum("htu,buhsd->bthsd", dm, wk)
            slot_val = torch.einsum("htu,buhsd->bthsd", dm, wv)
            slot_w = torch.einsum("htu,buhs->bths", dm, rw).clamp(min=1e-6)
        else:
            slot_key = wk.cumsum(dim=1)
            slot_val = wv.cumsum(dim=1)
            slot_w = rw.cumsum(dim=1).clamp(min=1e-6)
        slot_key = slot_key / slot_w.unsqueeze(-1)
        slot_val = slot_val / slot_w.unsqueeze(-1)

        scores = torch.einsum("blhd,blhsd->blhs", q, slot_key) * self.read_scale
        attn = torch.softmax(scores, dim=-1)
        read = torch.einsum("blhs,blhsd->blhd", attn, slot_val)  # [B,L,H,hd]
        return self.out(read.reshape(b, l, self.m))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seeds", type=str, default="0,1")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    D = args.dim
    configs = {
        "base": lambda d: SlotAblation(d, memory_dim=d),
        "+multihead": lambda d: SlotAblation(d, memory_dim=d, n_heads=4),
        "+decay": lambda d: SlotAblation(d, memory_dim=d, n_heads=4, use_decay=True),
        "+hard_route": lambda d: SlotAblation(
            d, memory_dim=d, n_heads=4, use_decay=True, hard_route=True
        ),
        "softmax_4h(ref)": lambda d: MultiHeadCausalAttention(d, n_heads=4),
    }
    tmap = {t.name: t for t in DEFAULT_BINDING_VALIDITY_TASKS}
    hard = ["episodic_distinct_key_interference", "episodic_compositional"]
    t0 = time.monotonic()
    rows = {}
    print(
        f"dim={D} steps={args.steps} seeds={seeds}\n{'config':18s}{'params':>7s}  interf  compos  mean",
        flush=True,
    )
    for name, fac in configs.items():
        per = {}
        for tn in hard:
            a = [
                run_binding_validity_task(
                    fac,
                    tmap[tn],
                    mixer_label=name,
                    dim=D,
                    n_train_steps=args.steps,
                    seed=s,
                    device="cuda",
                ).eval_accuracy
                for s in seeds
            ]
            per[tn] = sum(a) / len(a)
        p = sum(pp.numel() for pp in fac(D).parameters())
        i, c = per[hard[0]], per[hard[1]]
        rows[name] = {"params": p, "interf": i, "compos": c, "mean": (i + c) / 2}
        print(
            f"{name:18s}{p:>7d}  {i:.3f}   {c:.3f}   {(i + c) / 2:.3f}  ({time.monotonic() - t0:.0f}s)",
            flush=True,
        )
    Path("research/reports/_slot_ablation.json").write_text(
        json.dumps({"dim": D, "rows": rows}, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
