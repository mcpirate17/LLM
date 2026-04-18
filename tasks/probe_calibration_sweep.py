"""In-depth binding / induction probe calibration sweep (2026-04-17).

Goal: produce an empirical map of:
  - what layer depth is required to score on each probe
  - how training-step budget interacts with AUC ceiling
  - how architectural family (attn/conv/ssm/rwkv/hybrid) separates under the
    probes, and where each family's ceiling lives
  - the minimum steps at which a probe discriminates architectures well

Runs on GPU, saves incremental CSVs to tasks/probe_calibration_results/.
Findings are synthesized in a markdown report after the sweep completes.

Usage:
    python tasks/probe_calibration_sweep.py

This is a research script; not wired into any production path.
"""

from __future__ import annotations

import copy
import csv
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── Setup ─────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = 512  # probes slice to :256 or :vocab internally
D_MODEL = 96
N_HEADS = 4
MAX_SEQ_LEN = 512

RESULTS_DIR = Path("tasks/probe_calibration_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INDUCTION_CSV = RESULTS_DIR / "induction_sweep.csv"
BINDING_CURR_CSV = RESULTS_DIR / "binding_curriculum_sweep.csv"
AR_CSV = RESULTS_DIR / "associative_recall_sweep.csv"
INDUCTION_EXTENDED_CSV = RESULTS_DIR / "induction_extended_sweep.csv"

print(f"Device: {DEVICE}")
print(f"Results dir: {RESULTS_DIR.absolute()}")


# ── Model zoo ─────────────────────────────────────────────────────────


def _causal_mask(S: int, device) -> torch.Tensor:
    return torch.triu(
        torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1
    )


class CausalAttnLM(nn.Module):
    """Standard N-layer causal-attention language model with absolute pos embed."""

    def __init__(self, n_layers: int, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 vocab: int = VOCAB, max_seq_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.vocab_size = vocab
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ln2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                ),
            })
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        pos_ids = torch.arange(S, device=x.device)
        h = self.embed(x) + self.pos(pos_ids)
        mask = _causal_mask(S, x.device)
        for L in self.layers:
            a, _ = L['attn'](
                L['ln1'](h), L['ln1'](h), L['ln1'](h),
                attn_mask=mask, need_weights=False,
            )
            h = h + a
            h = h + L['ffn'](L['ln2'](h))
        return self.head(self.ln_f(h))


class CausalConvLM(nn.Module):
    """Stacked causal-1D-conv LM — deliberately no attention for baseline."""

    def __init__(self, n_layers: int, k: int, d_model: int = D_MODEL,
                 vocab: int = VOCAB):
        super().__init__()
        self.vocab_size = vocab
        self.k = k
        self.embed = nn.Embedding(vocab, d_model)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k - 1)
            for _ in range(n_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
            for _ in range(n_layers)
        ])
        self.lns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        h = self.embed(x).transpose(1, 2)  # (B, D, S)
        for conv, ffn, ln in zip(self.convs, self.ffns, self.lns):
            c = conv(h)[:, :, :S]  # causal truncate
            h = h + F.gelu(c)
            h_bsd = h.transpose(1, 2)
            h_bsd = h_bsd + ffn(ln(h_bsd))
            h = h_bsd.transpose(1, 2)
        h = h.transpose(1, 2)
        return self.head(self.ln_f(h))


class MiniSSM(nn.Module):
    """Tiny selective-state-space layer (Mamba-flavored, not exact)."""

    def __init__(self, d_model: int, state_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
        )
        self.B_proj = nn.Linear(d_model, state_dim, bias=False)
        self.C_proj = nn.Linear(d_model, state_dim, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.dt_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        A = -torch.exp(self.A_log)  # (N,)
        dt = F.softplus(self.dt_proj(x))  # (B, S, D)
        Bcoef = self.B_proj(x)  # (B, S, N)
        Ccoef = self.C_proj(x)  # (B, S, N)
        # Discretize A per (B, S, D, N)
        h = torch.zeros(B, D, self.state_dim, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(S):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A.view(1, 1, -1))
            dB = dt[:, t].unsqueeze(-1) * Bcoef[:, t].unsqueeze(1)
            h = dA * h + dB * x[:, t].unsqueeze(-1)
            y = (h * Ccoef[:, t].unsqueeze(1)).sum(dim=-1)
            outs.append(y + self.D * x[:, t])
        return torch.stack(outs, dim=1)


class CausalSSMLM(nn.Module):
    def __init__(self, n_layers: int, d_model: int = D_MODEL, vocab: int = VOCAB,
                 state_dim: int = 16):
        super().__init__()
        self.vocab_size = vocab
        self.embed = nn.Embedding(vocab, d_model)
        self.ssms = nn.ModuleList([MiniSSM(d_model, state_dim) for _ in range(n_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for ssm, ln in zip(self.ssms, self.lns):
            h = h + ssm(ln(h))
        return self.head(self.ln_f(h))


class MiniRWKVTimeMix(nn.Module):
    """Tiny RWKV-flavored time mixer (linear recurrent)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.time_decay = nn.Parameter(-torch.ones(d_model))
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        k = self.W_k(x)
        v = self.W_v(x)
        r = torch.sigmoid(self.W_r(x))
        decay = torch.exp(self.time_decay)  # (D,)
        outs = []
        num = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        den = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        for t in range(S):
            e_k = torch.exp(k[:, t].clamp(max=30))
            num = decay * num + e_k * v[:, t]
            den = decay * den + e_k
            wkv = num / (den + 1e-8)
            outs.append(r[:, t] * wkv)
        y = torch.stack(outs, dim=1)
        return self.W_o(y)


class CausalRWKVLM(nn.Module):
    def __init__(self, n_layers: int, d_model: int = D_MODEL, vocab: int = VOCAB):
        super().__init__()
        self.vocab_size = vocab
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'mix': MiniRWKVTimeMix(d_model),
                'ln2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                ),
            })
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for B in self.blocks:
            h = h + B['mix'](B['ln1'](h))
            h = h + B['ffn'](B['ln2'](h))
        return self.head(self.ln_f(h))


class _ConvBlock(nn.Module):
    def __init__(self, d_model: int, k: int = 5):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, k, padding=k - 1)

    def forward(self, h: torch.Tensor, *, mask=None) -> torch.Tensor:
        S = h.shape[1]
        c = self.conv(self.ln(h).transpose(1, 2))[:, :, :S]
        return h + F.gelu(c.transpose(1, 2))


class _AttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, h: torch.Tensor, *, mask=None) -> torch.Tensor:
        q = self.ln(h)
        a, _ = self.attn(q, q, q, attn_mask=mask, need_weights=False)
        return h + a


class _FFNBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, h: torch.Tensor, *, mask=None) -> torch.Tensor:
        return h + self.ffn(self.ln(h))


class HybridConvAttnLM(nn.Module):
    """Conv layer + attention layer — simple hybrid baseline.

    Each "macro layer" = [conv_or_attn] + FFN. For n_layers=2, that's
    (conv, FFN, attn, FFN). For n_layers=4, (conv, FFN, attn, FFN) × 2.
    """

    def __init__(self, n_layers: int = 2, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 vocab: int = VOCAB, max_seq_len: int = MAX_SEQ_LEN):
        super().__init__()
        assert n_layers >= 2 and n_layers % 2 == 0
        self.vocab_size = vocab
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        blocks: list[nn.Module] = []
        self._is_attn: list[bool] = []
        for i in range(n_layers):
            if i % 2 == 0:
                blocks.append(_ConvBlock(d_model))
                self._is_attn.append(False)
            else:
                blocks.append(_AttnBlock(d_model, n_heads))
                self._is_attn.append(True)
            blocks.append(_FFNBlock(d_model))
            self._is_attn.append(False)
        self.layers = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        h = self.embed(x) + self.pos(torch.arange(S, device=x.device))
        mask = _causal_mask(S, x.device)
        for L, is_attn in zip(self.layers, self._is_attn):
            h = L(h, mask=mask if is_attn else None)
        return self.head(self.ln_f(h))


ARCHITECTURES: Dict[str, Callable[[], nn.Module]] = {
    "attn_1l": lambda: CausalAttnLM(1),
    "attn_2l": lambda: CausalAttnLM(2),
    "attn_4l": lambda: CausalAttnLM(4),
    "conv3_2l": lambda: CausalConvLM(2, k=3),
    "conv7_2l": lambda: CausalConvLM(2, k=7),
    "conv7_4l": lambda: CausalConvLM(4, k=7),
    "ssm_2l": lambda: CausalSSMLM(2),
    "ssm_4l": lambda: CausalSSMLM(4),
    "rwkv_2l": lambda: CausalRWKVLM(2),
    "hybrid_2l": lambda: HybridConvAttnLM(2),
    "hybrid_4l": lambda: HybridConvAttnLM(4),
}


# ── Induction probe (parameterized by training gap mode) ──────────────


def _gen_induction_batch(batch_size: int, gap: int, device, vocab: int = 256):
    """Generate [A][B][noise×gap][A] → predict B."""
    seq_len = gap + 3
    batch = torch.randint(1, vocab, (batch_size, seq_len), device=device)
    A = torch.randint(1, vocab, (batch_size,), device=device)
    Bt = torch.randint(1, vocab, (batch_size,), device=device)
    batch[:, 0] = A
    batch[:, 1] = Bt
    # Avoid A-in-noise collisions
    noise = batch[:, 2 : gap + 2]
    collisions = noise == A.unsqueeze(1)
    if collisions.any():
        offsets = torch.randint(1, vocab - 1, collisions.shape, device=device)
        noise[collisions] = (A.unsqueeze(1).expand_as(noise)[collisions] + offsets[collisions]) % (vocab - 1) + 1
        batch[:, 2 : gap + 2] = noise
    batch[:, gap + 2] = A
    return batch, Bt


def run_induction(
    model: nn.Module,
    *,
    n_train_steps: int,
    train_mode: str,  # "fixed8" or "mixed"
    eval_gaps: Tuple[int, ...] = (4, 8, 16, 32, 64),
    batch_size: int = 32,
    n_eval: int = 200,
    lr: float = 1e-3,
    device: str = DEVICE,
    vocab: int = 256,
) -> Dict[str, Any]:
    """Train the model on the induction task, return per-gap accuracy + AUC."""
    t0 = time.perf_counter()
    try:
        m = copy.deepcopy(model).to(device)
    except Exception as e:
        return {"status": f"copy_fail: {e}", "auc": 0.0, "gap_acc": {}, "elapsed_s": 0.0}
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    train_gaps = (8,) if train_mode == "fixed8" else eval_gaps
    first_loss = None
    last_loss = None
    status = "ok"
    for step in range(n_train_steps):
        g = train_gaps[step % len(train_gaps)]
        inp, tgt = _gen_induction_batch(batch_size, g, device, vocab=vocab)
        opt.zero_grad(set_to_none=True)
        logits = m(inp)
        pred_logits = logits[:, inp.shape[1] - 1, :vocab]
        loss = F.cross_entropy(pred_logits.float(), tgt)
        if not torch.isfinite(loss):
            status = "diverged"
            break
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if first_loss is None:
            first_loss = float(loss.item())
        last_loss = float(loss.item())
    m.eval()
    accs: Dict[int, float] = {}
    with torch.inference_mode():
        for g in eval_gaps:
            correct = total = 0
            remaining = n_eval
            while remaining > 0:
                bs = min(batch_size, remaining)
                inp, tgt = _gen_induction_batch(bs, g, device, vocab=vocab)
                preds = m(inp)[:, inp.shape[1] - 1, :vocab].argmax(-1)
                correct += (preds == tgt).sum().item()
                total += tgt.numel()
                remaining -= bs
            accs[g] = round(correct / max(total, 1), 4)
    auc = round(sum(accs.values()) / len(accs), 4)
    max_gap_acc = max(accs.values()) if accs else 0.0
    min_gap_acc = min(accs.values()) if accs else 0.0
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "status": status,
        "auc": auc,
        "gap_acc": accs,
        "max_gap_acc": round(max_gap_acc, 4),
        "min_gap_acc": round(min_gap_acc, 4),
        "first_loss": round(first_loss or 0.0, 4),
        "last_loss": round(last_loss or 0.0, 4),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


# ── Binding curriculum (mirrors production, but parameterized) ────────


def _gen_copy_batch(batch_size: int, seq_len: int, distance: int, vocab: int, device):
    seed = torch.randint(1, vocab, (batch_size, distance), device=device)
    n_rep = (seq_len + distance - 1) // distance
    return seed.repeat(1, n_rep)[:, :seq_len]


def run_binding_curriculum(
    model: nn.Module,
    *,
    n_train_steps: int,
    distances: Tuple[int, ...] = (4, 8, 16, 32),
    seq_len: int = 128,
    batch_size: int = 16,
    n_eval: int = 100,
    lr: float = 3e-4,
    device: str = DEVICE,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        m = copy.deepcopy(model).to(device)
    except Exception as e:
        return {"status": f"copy_fail: {e}", "auc": 0.0, "dist_acc": {}, "elapsed_s": 0.0}
    vocab = int(getattr(m, "vocab_size", 256) or 256)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    first_loss = None
    last_loss = None
    status = "ok"
    for step in range(n_train_steps):
        d = distances[step % len(distances)]
        batch = _gen_copy_batch(batch_size, seq_len, d, vocab, device)
        opt.zero_grad(set_to_none=True)
        logits = m(batch)
        pred = logits[:, d - 1 : seq_len - 1, :vocab]
        tgt = batch[:, d:seq_len]
        loss = F.cross_entropy(pred.reshape(-1, vocab), tgt.reshape(-1))
        if not torch.isfinite(loss):
            status = "diverged"
            break
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if first_loss is None:
            first_loss = float(loss.item())
        last_loss = float(loss.item())
    m.eval()
    accs: Dict[int, float] = {}
    with torch.inference_mode():
        for d in distances:
            if d + 1 >= seq_len:
                accs[d] = 0.0
                continue
            correct = total = 0
            remaining = n_eval
            while remaining > 0:
                bs = min(batch_size, remaining)
                b = _gen_copy_batch(bs, seq_len, d, vocab, device)
                preds = m(b)[:, d - 1 : seq_len - 1, :vocab].argmax(-1)
                tgt = b[:, d:seq_len]
                correct += (preds == tgt).sum().item()
                total += tgt.numel()
                remaining -= bs
            accs[d] = round(correct / max(total, 1), 4)
    auc = round(sum(accs.values()) / len(accs), 4)
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "status": status,
        "auc": auc,
        "dist_acc": accs,
        "first_loss": round(first_loss or 0.0, 4),
        "last_loss": round(last_loss or 0.0, 4),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


# ── Associative recall (production-compatible) ────────────────────────


def run_associative_recall(
    model: nn.Module,
    *,
    n_train_steps: int,
    n_pairs: int = 20,
    batch_size: int = 16,
    n_eval: int = 200,
    lr: float = 1e-3,
    device: str = DEVICE,
) -> Dict[str, Any]:
    from research.eval.associative_recall import (
        _generate_ar_batch,
        _generate_eval_set,
        _get_special_tokens,
    )

    t0 = time.perf_counter()
    try:
        m = copy.deepcopy(model).to(device)
    except Exception as e:
        return {"status": f"copy_fail: {e}", "auc": 0.0, "final_acc": 0.0, "elapsed_s": 0.0}
    m.train()
    sep, ans = _get_special_tokens(m)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    try:
        eval_ids, eval_tgts = _generate_eval_set(n_eval, n_pairs, sep, ans, device)
    except Exception as e:
        return {"status": f"eval_gen_fail: {e}", "auc": 0.0, "final_acc": 0.0, "elapsed_s": 0.0}
    ans_pos = 3 * n_pairs + 3
    first_loss = None
    last_loss = None
    curve: List[Tuple[int, float]] = []
    status = "ok"

    def _eval():
        m.eval()
        correct = 0
        total = eval_ids.shape[0]
        with torch.no_grad():
            for s in range(0, total, batch_size):
                e = min(s + batch_size, total)
                inp = eval_ids[s:e]
                tgt = eval_tgts[s:e]
                pred = m(inp)[:, ans_pos, 100:356].argmax(-1) + 100
                correct += (pred == tgt).sum().item()
        m.train()
        return correct / total

    curve.append((0, round(_eval(), 4)))
    for step in range(1, n_train_steps + 1):
        inp, tgt = _generate_ar_batch(batch_size, n_pairs, sep, ans, device)
        opt.zero_grad(set_to_none=True)
        logits = m(inp)
        pred = logits[:, ans_pos, 100:356]
        loss = F.cross_entropy(pred, tgt - 100)
        if not torch.isfinite(loss):
            status = "diverged"
            break
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if first_loss is None:
            first_loss = float(loss.item())
        last_loss = float(loss.item())
        if step % max(1, n_train_steps // 10) == 0 or step == n_train_steps:
            curve.append((step, round(_eval(), 4)))
    final_acc = curve[-1][1] if curve else 0.0
    if len(curve) >= 2:
        area = 0.0
        for i in range(1, len(curve)):
            dt = curve[i][0] - curve[i - 1][0]
            area += 0.5 * dt * (curve[i - 1][1] + curve[i][1])
        auc = round(area / max(n_train_steps, 1), 4)
    else:
        auc = final_acc
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "status": status,
        "auc": auc,
        "final_acc": round(final_acc, 4),
        "curve": curve,
        "first_loss": round(first_loss or 0.0, 4),
        "last_loss": round(last_loss or 0.0, 4),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


# ── Sweep drivers ─────────────────────────────────────────────────────


def _init_csv(path: Path, fieldnames: List[str]):
    first = not path.exists()
    f = path.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if first:
        w.writeheader()
        f.flush()
    return f, w


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sweep_induction():
    fieldnames = [
        "arch", "n_params", "n_train_steps", "train_mode",
        "auc", "max_gap_acc", "min_gap_acc",
        "acc_4", "acc_8", "acc_16", "acc_32", "acc_64",
        "first_loss", "last_loss", "status", "elapsed_s",
    ]
    f, w = _init_csv(INDUCTION_CSV, fieldnames)
    step_grid = (100, 250, 500, 1000, 2000)
    mode_grid = ("fixed8", "mixed")
    total = len(ARCHITECTURES) * len(step_grid) * len(mode_grid)
    done = 0
    print(f"\n=== Induction sweep: {total} runs ===")
    for arch_name, arch_fn in ARCHITECTURES.items():
        base = arch_fn()
        n_params = _param_count(base)
        for steps in step_grid:
            for mode in mode_grid:
                done += 1
                try:
                    res = run_induction(base, n_train_steps=steps, train_mode=mode)
                except Exception as e:
                    res = {
                        "status": f"exception: {type(e).__name__}: {e}",
                        "auc": 0.0, "gap_acc": {},
                        "max_gap_acc": 0.0, "min_gap_acc": 0.0,
                        "first_loss": 0.0, "last_loss": 0.0, "elapsed_s": 0.0,
                    }
                gap_acc = res.get("gap_acc", {})
                row = {
                    "arch": arch_name,
                    "n_params": n_params,
                    "n_train_steps": steps,
                    "train_mode": mode,
                    "auc": res["auc"],
                    "max_gap_acc": res.get("max_gap_acc", 0.0),
                    "min_gap_acc": res.get("min_gap_acc", 0.0),
                    "acc_4": gap_acc.get(4, 0.0),
                    "acc_8": gap_acc.get(8, 0.0),
                    "acc_16": gap_acc.get(16, 0.0),
                    "acc_32": gap_acc.get(32, 0.0),
                    "acc_64": gap_acc.get(64, 0.0),
                    "first_loss": res["first_loss"],
                    "last_loss": res["last_loss"],
                    "status": res["status"],
                    "elapsed_s": res["elapsed_s"],
                }
                w.writerow(row)
                f.flush()
                print(f"[{done}/{total}] {arch_name:<12} steps={steps:4} "
                      f"{mode:<6} auc={res['auc']:.3f} "
                      f"peak={res.get('max_gap_acc', 0):.3f} "
                      f"time={res['elapsed_s']}s")
        del base
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    f.close()


def sweep_induction_extended():
    """Extended: fine-grained step sweep for the most promising archs,
    so we can see the learning curve shape."""
    fieldnames = [
        "arch", "n_params", "n_train_steps", "train_mode",
        "auc", "max_gap_acc", "min_gap_acc",
        "acc_4", "acc_8", "acc_16", "acc_32", "acc_64",
        "last_loss", "status", "elapsed_s",
    ]
    f, w = _init_csv(INDUCTION_EXTENDED_CSV, fieldnames)
    # Dense step grid on the architectures most relevant for discrimination.
    target_archs = ["attn_1l", "attn_2l", "attn_4l", "hybrid_2l", "ssm_4l", "rwkv_2l", "conv7_4l"]
    step_grid = (50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400)
    mode_grid = ("mixed",)  # mixed is the better training regime per earlier finding
    total = len(target_archs) * len(step_grid) * len(mode_grid)
    done = 0
    print(f"\n=== Induction extended (learning-curve) sweep: {total} runs ===")
    for arch_name in target_archs:
        base = ARCHITECTURES[arch_name]()
        n_params = _param_count(base)
        for steps in step_grid:
            for mode in mode_grid:
                done += 1
                try:
                    res = run_induction(base, n_train_steps=steps, train_mode=mode)
                except Exception as e:
                    res = {
                        "status": f"exception: {type(e).__name__}: {e}",
                        "auc": 0.0, "gap_acc": {},
                        "max_gap_acc": 0.0, "min_gap_acc": 0.0,
                        "last_loss": 0.0, "elapsed_s": 0.0,
                    }
                gap_acc = res.get("gap_acc", {})
                row = {
                    "arch": arch_name,
                    "n_params": n_params,
                    "n_train_steps": steps,
                    "train_mode": mode,
                    "auc": res["auc"],
                    "max_gap_acc": res.get("max_gap_acc", 0.0),
                    "min_gap_acc": res.get("min_gap_acc", 0.0),
                    "acc_4": gap_acc.get(4, 0.0),
                    "acc_8": gap_acc.get(8, 0.0),
                    "acc_16": gap_acc.get(16, 0.0),
                    "acc_32": gap_acc.get(32, 0.0),
                    "acc_64": gap_acc.get(64, 0.0),
                    "last_loss": res.get("last_loss", 0.0),
                    "status": res["status"],
                    "elapsed_s": res["elapsed_s"],
                }
                w.writerow(row)
                f.flush()
                print(f"[{done}/{total}] {arch_name:<12} steps={steps:4} "
                      f"auc={res['auc']:.3f} peak={res.get('max_gap_acc', 0):.3f} "
                      f"time={res['elapsed_s']}s")
        del base
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    f.close()


def sweep_binding_curriculum():
    fieldnames = [
        "arch", "n_params", "n_train_steps",
        "auc", "acc_4", "acc_8", "acc_16", "acc_32",
        "first_loss", "last_loss", "status", "elapsed_s",
    ]
    f, w = _init_csv(BINDING_CURR_CSV, fieldnames)
    step_grid = (200, 400, 800, 1600)
    total = len(ARCHITECTURES) * len(step_grid)
    done = 0
    print(f"\n=== Binding curriculum sweep: {total} runs ===")
    for arch_name, arch_fn in ARCHITECTURES.items():
        base = arch_fn()
        n_params = _param_count(base)
        for steps in step_grid:
            done += 1
            try:
                res = run_binding_curriculum(base, n_train_steps=steps)
            except Exception as e:
                res = {
                    "status": f"exception: {type(e).__name__}: {e}",
                    "auc": 0.0, "dist_acc": {},
                    "first_loss": 0.0, "last_loss": 0.0, "elapsed_s": 0.0,
                }
            dist_acc = res.get("dist_acc", {})
            row = {
                "arch": arch_name,
                "n_params": n_params,
                "n_train_steps": steps,
                "auc": res["auc"],
                "acc_4": dist_acc.get(4, 0.0),
                "acc_8": dist_acc.get(8, 0.0),
                "acc_16": dist_acc.get(16, 0.0),
                "acc_32": dist_acc.get(32, 0.0),
                "first_loss": res["first_loss"],
                "last_loss": res["last_loss"],
                "status": res["status"],
                "elapsed_s": res["elapsed_s"],
            }
            w.writerow(row)
            f.flush()
            print(f"[{done}/{total}] {arch_name:<12} steps={steps:4} "
                  f"auc={res['auc']:.3f} time={res['elapsed_s']}s")
        del base
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    f.close()


def sweep_associative_recall():
    fieldnames = [
        "arch", "n_params", "n_train_steps",
        "auc", "final_acc",
        "first_loss", "last_loss", "status", "elapsed_s",
    ]
    f, w = _init_csv(AR_CSV, fieldnames)
    step_grid = (500, 1000, 2000)
    total = len(ARCHITECTURES) * len(step_grid)
    done = 0
    print(f"\n=== Associative-recall sweep: {total} runs ===")
    for arch_name, arch_fn in ARCHITECTURES.items():
        base = arch_fn()
        n_params = _param_count(base)
        for steps in step_grid:
            done += 1
            try:
                res = run_associative_recall(base, n_train_steps=steps)
            except Exception as e:
                res = {
                    "status": f"exception: {type(e).__name__}: {e}",
                    "auc": 0.0, "final_acc": 0.0,
                    "first_loss": 0.0, "last_loss": 0.0, "elapsed_s": 0.0,
                }
            row = {
                "arch": arch_name,
                "n_params": n_params,
                "n_train_steps": steps,
                "auc": res["auc"],
                "final_acc": res["final_acc"],
                "first_loss": res["first_loss"],
                "last_loss": res["last_loss"],
                "status": res["status"],
                "elapsed_s": res["elapsed_s"],
            }
            w.writerow(row)
            f.flush()
            print(f"[{done}/{total}] {arch_name:<12} steps={steps:4} "
                  f"auc={res['auc']:.3f} final={res['final_acc']:.3f} "
                  f"time={res['elapsed_s']}s")
        del base
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    f.close()


# ── Main ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    t_start = time.perf_counter()
    print(f"Total architectures: {len(ARCHITECTURES)}")
    for name, fn in ARCHITECTURES.items():
        try:
            m = fn()
            n = _param_count(m)
            print(f"  {name:<12} {n:>10,} params")
            del m
        except Exception as e:
            print(f"  {name:<12} BUILD FAILED: {e}")
    try:
        sweep_induction()
    except Exception as e:
        print(f"INDUCTION SWEEP CRASH: {e}")
        traceback.print_exc()
    try:
        sweep_induction_extended()
    except Exception as e:
        print(f"INDUCTION EXTENDED CRASH: {e}")
        traceback.print_exc()
    try:
        sweep_binding_curriculum()
    except Exception as e:
        print(f"BINDING CURRICULUM CRASH: {e}")
        traceback.print_exc()
    try:
        sweep_associative_recall()
    except Exception as e:
        print(f"AR SWEEP CRASH: {e}")
        traceback.print_exc()
    total_s = time.perf_counter() - t_start
    print(f"\n=== All sweeps complete in {total_s / 60:.1f} min ===")
    print(f"Results in {RESULTS_DIR.absolute()}")
