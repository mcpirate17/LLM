"""NM-F nano-contract capability probes: F9 binding curve + F4 retention flatness.

Trains tiny LMs (nano contract: dim 256, vocab 256, 2 blocks) around the shipped
NM-F operators on synthetic tasks with randomized-query controls, and checks the
two falsifiable predictions made in ``tasks/nm_f_operator_families_2026-07-01.md``:

  * **Probe A (NM-F9)** — gMQAR-style multi-slot binding: accuracy vs number of
    bound pairs {2,4,8,16} for ``CDMASlotBinding`` at chips {32,64,128}. The
    Welch/Gold interference prediction: degradation with pair count, milder at
    longer codes; reported next to the analytic interference ratio per config.
  * **Probe B (NM-F4)** — bind → distractors → query retention: train at gaps
    ≤128, eval at gaps {16,64,256,1024} (length extrapolation). Prediction:
    ``IntegralControlMixer`` is FLAT vs gap; the matched learned-decay EMA (the
    decay-by-parameterization baseline, identical except for the state law)
    falls off. A tiny 2-layer RoPE transformer runs as the positive-control
    probe in both (a control, NOT a new baseline training run).

Pair/query positions are randomized every sequence — a recency/positional
shortcut cannot pass either task. 3 seeds, median-of-seeds reported. Writes JSON
to ``research/reports/nm_f_probes/`` (auto-pruned); durable conclusions go to
``research/notes/``. NO runs.db / notebook writes — this is a standalone probe.

Usage:
    python research/tools/nm_f_capability_probes.py --probe all --seeds 3
    python research/tools/nm_f_capability_probes.py --probe binding --steps 30 --seeds 1
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.synthesis.cdma_slot_binding import (  # noqa: E402
    CDMASlotBinding,
    gold_cross_correlation_bound,
)
from research.synthesis.integral_control_gate import IntegralControlMixer  # noqa: E402

VOCAB = 256
DIM = 256
N_BLOCKS = 2
QUERY_TOK = 1
KEYS = (8, 72)  # 64 keys
VALUES = (128, 192)  # 64 values
FILLER = (200, 250)
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "nm_f_probes"


# ── task generators (positions randomized per sequence — no recency shortcut) ──


def make_binding_batch(
    batch: int,
    n_pairs: int,
    seq_len: int,
    device: torch.device,
    gen: torch.Generator,
    n_keys: int = 64,
    n_values: int = 64,
    layout: str = "block",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-query AR (gMQAR-style, dense supervision): ``n_pairs`` of (k, v) in
    a ``seq_len`` filler body, then ``[QRY, k_π(1) … k_π(n)]`` with the queried
    keys in RANDOM order; labels ``-100`` except at each queried key's position
    (predict its value). Layouts: ``"block"`` = contiguous pair block at a RANDOM
    offset, pair order shuffled (learnable at nano budget; diagnosed 2026-07-02:
    fully random scatter defeats even the 2-layer attention control at 30K
    steps); ``"scatter"`` = pairs at random non-overlapping slots — the harder
    layout-generalization EVAL: a positional shortcut learned on "block" cannot
    transfer to it, content binding can."""
    if 2 * n_pairs + 2 > seq_len:
        raise ValueError(f"seq_len={seq_len} too short for n_pairs={n_pairs}")
    if not (
        n_pairs <= n_keys <= KEYS[1] - KEYS[0] and n_values <= VALUES[1] - VALUES[0]
    ):
        raise ValueError(f"bad difficulty n_keys={n_keys}, n_values={n_values}")
    total_len = seq_len + 1 + n_pairs
    x = torch.randint(FILLER[0], FILLER[1], (batch, total_len), generator=gen).to(
        device
    )
    keys = (
        torch.stack(
            [torch.randperm(n_keys, generator=gen)[:n_pairs] for _ in range(batch)]
        ).to(device)
        + KEYS[0]
    )
    values = (
        torch.randint(0, n_values, (batch, n_pairs), generator=gen).to(device)
        + VALUES[0]
    )
    rows = torch.arange(batch, device=device).unsqueeze(1)
    if layout == "block":
        offset = torch.randint(
            0, seq_len - 2 * n_pairs + 1, (batch, 1), generator=gen
        ).to(device)
        pos = offset + 2 * torch.arange(n_pairs, device=device).unsqueeze(0)
    elif layout == "scatter":
        slot_idx = torch.stack(
            [
                torch.randperm(seq_len // 2, generator=gen)[:n_pairs]
                for _ in range(batch)
            ]
        ).to(device)
        pos = slot_idx * 2
    else:
        raise ValueError(f"unknown layout {layout!r}")
    x[rows, pos] = keys
    x[rows, pos + 1] = values
    x[:, seq_len] = QUERY_TOK
    order = torch.stack(
        [torch.randperm(n_pairs, generator=gen) for _ in range(batch)]
    ).to(device)
    x[:, seq_len + 1 :] = torch.gather(keys, 1, order)
    labels = torch.full((batch, total_len), -100, dtype=torch.long, device=device)
    labels[:, seq_len + 1 :] = torch.gather(values, 1, order)
    return x, labels


def make_retention_batch(
    batch: int, gap: int, device: torch.device, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """[filler prefix, k, v, filler x gap, QRY, k] -> v. Prefix length random;
    labels are ``-100`` except at the final (query key) position."""
    prefix = int(torch.randint(1, 9, (1,), generator=gen))
    seq_len = prefix + 2 + gap + 2
    x = torch.randint(FILLER[0], FILLER[1], (batch, seq_len), generator=gen).to(device)
    keys = (
        torch.randint(0, KEYS[1] - KEYS[0], (batch,), generator=gen).to(device)
        + KEYS[0]
    )
    values = (
        torch.randint(0, VALUES[1] - VALUES[0], (batch,), generator=gen).to(device)
        + VALUES[0]
    )
    x[:, prefix] = keys
    x[:, prefix + 1] = values
    x[:, -2] = QUERY_TOK
    x[:, -1] = keys
    labels = torch.full((batch, seq_len), -100, dtype=torch.long, device=device)
    labels[:, -1] = values
    return x, labels


# ── mixers under test (each returns a DELTA; the block adds it to the stream) ──


class DeltaWrap(nn.Module):
    """Adapts an internal-residual op (forward(x) = x + f(x)) to delta form."""

    def __init__(self, op: nn.Module) -> None:
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x) - x


class EMAMixer(nn.Module):
    """Decay-by-parameterization baseline: identical lifts/gate to
    IntegralControlMixer, but the state law is a learned-decay EMA — the
    geometric-forgetting pathway the integral controller structurally lacks."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.d = dim
        self.in_lift = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.in_lift.weight.copy_(torch.eye(dim))
        self.out_lift = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out_lift.weight)
        self.raw_decay = nn.Parameter(torch.full((dim,), 3.0))  # sigmoid(3) ≈ 0.95
        self.gate_weight = nn.Parameter(torch.zeros(dim))
        self.gate_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.in_lift(x)
        gate = torch.sigmoid(x @ self.gate_weight + self.gate_bias)
        decay = torch.sigmoid(self.raw_decay)
        s = x.new_zeros(x.shape[0], self.d)
        states = []
        for t in range(x.shape[1]):
            s = decay * s + (1.0 - decay) * (gate[:, t : t + 1] * u[:, t])
            states.append(s)
        return self.out_lift(torch.stack(states, dim=1))


class RoPEAttention(nn.Module):
    """Tiny causal MHA with RoPE — the positive-control probe (a control that a
    known-good mechanism clears the task at this budget, NOT a baseline run)."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        inv = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2) / self.head_dim))
        self.register_buffer("inv_freq", inv)

    def _rope(self, t: torch.Tensor, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len, device=t.device, dtype=torch.float32)
        ang = pos.unsqueeze(-1) * self.inv_freq  # (L, hd/2)
        cos, sin = ang.cos(), ang.sin()
        t1, t2 = t[..., 0::2], t[..., 1::2]
        out = torch.empty_like(t)
        out[..., 0::2] = t1 * cos - t2 * sin
        out[..., 1::2] = t1 * sin + t2 * cos
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, seq_len, self.n_heads, self.head_dim)
        q = self._rope(q.view(shape).transpose(1, 2), seq_len)
        k = self._rope(k.view(shape).transpose(1, 2), seq_len)
        v = v.view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(b, seq_len, d))


def build_mixer(name: str) -> nn.Module:
    if name.startswith("cdma"):
        return DeltaWrap(
            CDMASlotBinding(DIM, n_slots=32, chips=int(name[4:]), code_family="gold")
        )
    if name == "integral":
        return DeltaWrap(IntegralControlMixer(DIM))
    if name == "ema":
        return EMAMixer(DIM)
    if name == "attn":
        return RoPEAttention(DIM)
    raise ValueError(f"unknown mixer {name!r}")


# ── nano-contract model ──


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class LocalConv(nn.Module):
    """Kernel-4 causal depthwise conv — the standard local-fuse companion of
    recurrent/state mixers (H3/Mamba convention). Without it a state mixer has
    no pathway to fuse an adjacent (key, value) token pair before binding;
    attention gets this for free via previous-token heads, so attn blocks skip
    it (diagnosed 2026-07-02). ~4·D params."""

    def __init__(self, dim: int, kernel: int = 4) -> None:
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x.transpose(1, 2), (self.kernel - 1, 0))
        return self.conv(padded).transpose(1, 2)


class ProbeBlock(nn.Module):
    def __init__(self, mixer: nn.Module, local: bool) -> None:
        super().__init__()
        self.norm0 = RMSNorm(DIM) if local else None
        self.local = LocalConv(DIM) if local else None
        self.norm1 = RMSNorm(DIM)
        self.mixer = mixer
        self.norm2 = RMSNorm(DIM)
        self.mlp = nn.Sequential(
            nn.Linear(DIM, 4 * DIM, bias=False),
            nn.GELU(),
            nn.Linear(4 * DIM, DIM, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.local is not None:
            x = x + self.local(self.norm0(x))
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ProbeLM(nn.Module):
    def __init__(self, mixer_name: str) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList(
            [
                ProbeBlock(build_mixer(mixer_name), local=mixer_name != "attn")
                for _ in range(N_BLOCKS)
            ]
        )
        self.norm = RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def non_embedding_params(self) -> int:
        return sum(p.numel() for b in self.blocks for p in b.parameters()) + sum(
            p.numel() for p in self.norm.parameters()
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))  # (B, L, V); loss masks via labels == -100


# ── train / eval ──


def train_model(
    model: ProbeLM,
    sample_batch,
    steps: int,
    batch: int,
    device: torch.device,
    gen: torch.Generator,
    lr: float = 3e-3,
) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr / 10
    )
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        x, y = sample_batch(batch, gen)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def eval_accuracy(model: ProbeLM, sample_batch, n_seq: int, batch: int, gen) -> float:
    model.eval()
    correct = total = 0
    for _ in range(math.ceil(n_seq / batch)):
        x, y = sample_batch(batch, gen)
        mask = y != -100
        correct += int(((model(x).argmax(dim=-1) == y) & mask).sum())
        total += int(mask.sum())
    return correct / total


def run_binding_probe(args, device: torch.device) -> dict:
    """Probe A: accuracy vs pair count per mixer; Welch interference alongside."""
    train_pairs, eval_pairs, seq_len = (2, 4, 8), (2, 4, 8, 16), args.body_len
    mixers = (
        args.mixers.split(",")
        if args.mixers
        else ["cdma32", "cdma64", "cdma128", "attn"]
    )
    nk, nv = args.n_keys, args.n_values
    results: dict = {
        "task": "binding",
        "train_pairs": train_pairs,
        "seq_len": seq_len,
        "n_keys": nk,
        "n_values": nv,
    }
    for name in mixers:
        per_seed: dict[int, dict] = {}
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = ProbeLM(name).to(device)

            def sample(b: int, g) -> tuple[torch.Tensor, torch.Tensor]:
                n = train_pairs[
                    int(torch.randint(0, len(train_pairs), (1,), generator=g))
                ]
                return make_binding_batch(b, n, seq_len, device, g, nk, nv)

            losses = train_model(
                model, sample, args.steps, args.batch, device, gen, args.lr
            )
            accs = {
                layout: {
                    n: eval_accuracy(
                        model,
                        lambda b, g, n=n, lay=layout: make_binding_batch(
                            b, n, seq_len, device, g, nk, nv, lay
                        ),
                        args.eval_seqs,
                        args.batch,
                        torch.Generator().manual_seed(10_000 + seed),
                    )
                    for n in eval_pairs
                }
                for layout in ("block", "scatter")
            }
            per_seed[seed] = {
                "acc_by_layout_pairs": accs,
                "final_loss": losses[-1],
                "first_loss": losses[0],
            }
            print(f"[binding] {name} seed={seed} acc={accs}", flush=True)
        entry: dict = {
            "per_seed": per_seed,
            "median_acc_by_layout_pairs": {
                layout: {
                    n: statistics.median(
                        per_seed[s]["acc_by_layout_pairs"][layout][n] for s in per_seed
                    )
                    for n in eval_pairs
                }
                for layout in ("block", "scatter")
            },
            "non_embedding_params": ProbeLM(name).non_embedding_params(),
        }
        if name.startswith("cdma"):
            op = CDMASlotBinding(DIM, n_slots=32, chips=int(name[4:]))
            entry["welch_interference_ratio_by_pairs"] = {
                n: (n - 1) * gold_cross_correlation_bound(op.degree) / op.chips
                for n in eval_pairs
            }
        results[name] = entry
    return results


def run_retention_probe(args, device: torch.device) -> dict:
    """Probe B: accuracy vs gap (incl. 8x length extrapolation) per mixer."""
    train_gaps, eval_gaps = (16, 64, 128), (16, 64, 256, 1024)
    mixers = args.mixers.split(",") if args.mixers else ["integral", "ema", "attn"]
    results: dict = {"task": "retention", "train_gaps": train_gaps}
    for name in mixers:
        per_seed: dict[int, dict] = {}
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = ProbeLM(name).to(device)

            def sample(b: int, g) -> tuple[torch.Tensor, torch.Tensor]:
                gap = train_gaps[
                    int(torch.randint(0, len(train_gaps), (1,), generator=g))
                ]
                return make_retention_batch(b, gap, device, g)

            losses = train_model(model, sample, args.steps, args.batch, device, gen)
            accs = {
                gap: eval_accuracy(
                    model,
                    lambda b, g, gap=gap: make_retention_batch(b, gap, device, g),
                    args.eval_seqs,
                    args.batch,
                    torch.Generator().manual_seed(10_000 + seed),
                )
                for gap in eval_gaps
            }
            per_seed[seed] = {
                "acc_by_gap": accs,
                "final_loss": losses[-1],
                "first_loss": losses[0],
            }
            print(f"[retention] {name} seed={seed} acc={accs}", flush=True)
        results[name] = {
            "per_seed": per_seed,
            "median_acc_by_gap": {
                gap: statistics.median(per_seed[s]["acc_by_gap"][gap] for s in per_seed)
                for gap in eval_gaps
            },
            "non_embedding_params": ProbeLM(name).non_embedding_params(),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", choices=["binding", "retention", "all"], default="all"
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eval-seqs", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n-keys", type=int, default=64, help="binding difficulty")
    parser.add_argument("--n-values", type=int, default=64, help="binding difficulty")
    parser.add_argument("--body-len", type=int, default=96, help="binding filler body")
    parser.add_argument(
        "--mixers", type=str, default="", help="comma list; empty = probe default set"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = {
        "config": vars(args) | {"dim": DIM, "vocab": VOCAB, "n_blocks": N_BLOCKS},
        "device": str(device),
    }
    t0 = time.time()
    if args.probe in ("binding", "all"):
        out["binding"] = run_binding_probe(args, device)
    if args.probe in ("retention", "all"):
        out["retention"] = run_retention_probe(args, device)
    out["wall_seconds"] = round(time.time() - t0, 1)
    path = REPORT_DIR / f"{stamp}_nm_f_probes.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path} ({out['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
