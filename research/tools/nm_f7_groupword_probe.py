"""NM-F7 falsification probe: does the nonabelian group structure carry weight?

The structured-vs-scrambled test NM-F next-steps §6 calls for. Task: a sequence
of group-element tokens ``g_1 ... g_L``; at every position ``t`` predict the
index of the ordered product ``g_t . g_{t-1} . ... . g_1`` (dense supervision).
A model whose mixer transports state by the *true* left-regular representation of
the group computes this running product exactly (that is literally what
``NonabelianGroupConv`` does at one-hot selection). The **scrambled** control has
the same shapes / param count / doubly-stochastic transport but its permutation
set is not a group (closure broken), so it cannot represent a consistent product.

Decisive reading:
  * ``groupconv`` (true) >> ``groupconv_scram`` (scrambled)  -> the nonabelian
    structure is load-bearing; F7's mechanism claim holds.
  * ``groupconv`` ~ ``groupconv_scram``                      -> F7 falsified cheaply.
``attn`` (causal self-attention) and ``mlp`` (order-blind) are capability
references. >=3 seeds, median reported; the task label needs genuine ordered
composition so a recency/positional shortcut sits near chance (1/|G|).

CPU-feasible (nano scale). Writes JSON to research/reports/ (auto-pruned).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch import nn

from research.synthesis.nonabelian_group_conv import (
    NonabelianGroupConv,
    _dihedral_mult_table,
)

DIM = 64
GROUP_ORDER = 8


def _make_batch(
    batch: int, seq_len: int, order: int, table: torch.Tensor, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """tokens g_t in [0, order); dense label = ordered product g_t...g_1 index.

    ``table`` is the (order, order) Cayley LongTensor; the running product is
    left-multiplied by each new element and vectorized over the batch via
    advanced indexing ``table[g_t, acc]``.
    """
    tokens = torch.randint(0, order, (batch, seq_len), generator=gen)
    labels = torch.empty_like(tokens)
    acc = tokens.new_zeros(batch)  # running product index, starts at identity=0
    for t in range(seq_len):
        acc = table[tokens[:, t], acc]  # product_t = g_t . product_{t-1}
        labels[:, t] = acc
    return tokens, labels


class _Block(nn.Module):
    def __init__(self, mixer_kind: str) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(DIM)
        self.mixer_kind = mixer_kind
        if mixer_kind == "groupconv":
            self.mix = NonabelianGroupConv(DIM, group_order=GROUP_ORDER)
        elif mixer_kind == "groupconv_scram":
            self.mix = NonabelianGroupConv(
                DIM, group_order=GROUP_ORDER, scramble_seed=1234
            )
        elif mixer_kind == "attn":
            self.attn = nn.MultiheadAttention(DIM, 4, batch_first=True)
        elif mixer_kind == "mlp":
            self.mix = nn.Sequential(
                nn.Linear(DIM, 2 * DIM), nn.GELU(), nn.Linear(2 * DIM, DIM)
            )
        else:
            raise ValueError(f"unknown mixer {mixer_kind!r}")
        self.ffn = nn.Sequential(
            nn.Linear(DIM, 2 * DIM), nn.GELU(), nn.Linear(2 * DIM, DIM)
        )
        self.ffn_norm = nn.RMSNorm(DIM)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.norm(h)
        if self.mixer_kind == "attn":
            mask = torch.triu(
                torch.ones(h.shape[1], h.shape[1], device=h.device, dtype=torch.bool),
                diagonal=1,
            )
            delta, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
        elif self.mixer_kind == "mlp":
            delta = self.mix(y)
        else:  # groupconv / groupconv_scram already return y + readout
            delta = self.mix(y) - y
        h = h + delta
        return h + self.ffn(self.ffn_norm(h))


class _Model(nn.Module):
    def __init__(self, mixer_kind: str, order: int, n_blocks: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(order, DIM)
        self.blocks = nn.ModuleList(_Block(mixer_kind) for _ in range(n_blocks))
        self.head = nn.Linear(DIM, order)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


def _train_eval(
    mixer_kind: str, *, seed: int, steps: int, order: int, device: str
) -> float:
    torch.manual_seed(seed)
    table = torch.tensor(_dihedral_mult_table(order), dtype=torch.long)
    model = _Model(mixer_kind, order, n_blocks=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(seed + 1)
    seq_len = 24
    for _ in range(steps):
        tokens, labels = _make_batch(64, seq_len, order, table, gen)
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(tokens)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, order), labels.reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        eg = torch.Generator().manual_seed(seed + 999)
        tokens, labels = _make_batch(256, seq_len, order, table, eg)
        tokens, labels = tokens.to(device), labels.to(device)
        pred = model(tokens).argmax(-1)
        # accuracy over the second half (needs a real running product, not warmup)
        acc = (pred[:, seq_len // 2 :] == labels[:, seq_len // 2 :]).float().mean()
    return float(acc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mixers", default="groupconv,groupconv_scram,attn,mlp")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reports"
        / "nm_f7_groupword_probe.json",
    )
    args = ap.parse_args()

    results: dict = {"task": "groupword", "chance": round(1.0 / GROUP_ORDER, 4)}
    for mixer in args.mixers.split(","):
        accs = [
            _train_eval(
                mixer, seed=s, steps=args.steps, order=GROUP_ORDER, device=args.device
            )
            for s in range(args.seeds)
        ]
        results[mixer] = {
            "per_seed": [round(a, 4) for a in accs],
            "median": round(statistics.median(accs), 4),
        }
        print(
            f"{mixer:18s} per_seed={[round(a, 3) for a in accs]} median={statistics.median(accs):.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    true_med = results.get("groupconv", {}).get("median", 0.0)
    scram_med = results.get("groupconv_scram", {}).get("median", 0.0)
    verdict = (
        "STRUCTURE LOAD-BEARING (F7 holds)"
        if true_med - scram_med > 0.15
        else "NOT SEPARATED (F7 falsified on this task)"
    )
    print(f"\ntrue={true_med:.3f} scrambled={scram_med:.3f} -> {verdict}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
