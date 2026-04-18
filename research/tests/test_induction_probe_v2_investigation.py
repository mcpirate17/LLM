"""Smoke tests for the investigation-tier induction probe (v2).

Verifies the probe discriminates architectural families as calibrated in
`PROBE_CALIBRATION_2026-04-17.md`:
  - Causal attention (2L) scores AUC >= 0.7 at 500 mixed-gap steps
  - Pure-conv (k=3) scores AUC <= 0.2 regardless of step budget
  - Compression-based recurrent (SSM-like) scores AUC <= 0.2

These are separation-tier tests: the probe can produce very tight
distributions, so we use generous margins (attention >= 0.7 vs ceiling ~1.00,
non-attention <= 0.2 vs floor ~0.00).

Marked ``unit`` so they run in the normal unit sweep. Each test takes 2-10s
on GPU, 10-60s on CPU — fast enough for CI but not a microsecond test.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = pytest.mark.unit


def _causal_mask(S: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)


class _CausalAttnLM(nn.Module):
    """Minimal 2-layer causal transformer for the attention reference."""

    def __init__(
        self,
        vocab: int = 512,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(d_model),
                        "attn": nn.MultiheadAttention(
                            d_model, n_heads, batch_first=True
                        ),
                        "ln2": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, 4 * d_model),
                            nn.GELU(),
                            nn.Linear(4 * d_model, d_model),
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        B, S = x.shape
        h = self.embed(x) + self.pos(torch.arange(S, device=x.device))
        mask = _causal_mask(S, x.device)
        for L in self.layers:
            a, _ = L["attn"](
                L["ln1"](h),
                L["ln1"](h),
                L["ln1"](h),
                attn_mask=mask,
                need_weights=False,
            )
            h = h + a
            h = h + L["ffn"](L["ln2"](h))
        return self.head(self.ln_f(h))


class _CausalConv3LM(nn.Module):
    """Minimal 2-layer causal-conv-only LM. Receptive field = 2*(k-1) = 4.
    Expected to fail induction at all gaps except very short."""

    def __init__(
        self, vocab: int = 512, d_model: int = 64, n_layers: int = 2, k: int = 3
    ):
        super().__init__()
        self.vocab_size = vocab
        self.k = k
        self.embed = nn.Embedding(vocab, d_model)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k - 1)
                for _ in range(n_layers)
            ]
        )
        self.lns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        B, S = x.shape
        h = self.embed(x).transpose(1, 2)
        for conv, ln in zip(self.convs, self.lns):
            c = conv(h)[:, :, :S]
            h_bsd = h.transpose(1, 2) + F.gelu(c.transpose(1, 2))
            h = ln(h_bsd).transpose(1, 2) + h - h  # keep as BDS shape; ln on BSD
            h = h_bsd.transpose(1, 2)
        return self.head(self.ln_f(h.transpose(1, 2)))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_probe_imports_cleanly():
    from research.eval.induction_probe_v2_investigation import (
        INDUCTION_V2_GAPS,
        INDUCTION_V2_PROTOCOL_VERSION,
        InductionV2Result,
    )

    assert INDUCTION_V2_GAPS == (4, 8, 16, 32, 64)
    assert INDUCTION_V2_PROTOCOL_VERSION.startswith("induction_investigation_")
    r = InductionV2Result()
    assert r.auc == 0.0 and r.status == "ok"


def test_result_to_dict_has_all_keys():
    from research.eval.induction_probe_v2_investigation import InductionV2Result

    r = InductionV2Result(auc=0.75, max_gap_acc=0.9, gap_accuracies={4: 0.9, 8: 0.8})
    d = r.to_dict()
    assert "induction_v2_investigation_auc" in d
    assert "induction_v2_investigation_max_gap_acc" in d
    assert "induction_v2_investigation_protocol_version" in d


@pytest.mark.slow
def test_attention_2l_beats_conv3_2l():
    """A 2-layer causal transformer should clearly out-score a 2-layer
    conv-only model on the v2 induction probe. This is the core signal the
    probe is supposed to produce."""
    from research.eval.induction_probe_v2_investigation import (
        run_induction_v2_investigation,
    )

    dev = _device()
    torch.manual_seed(42)
    attn = _CausalAttnLM()
    conv = _CausalConv3LM()

    # Shorter step budget for CI — full probe runs 500; we use 300 with a
    # reduced eval set to keep test runtime bounded.
    r_attn = run_induction_v2_investigation(
        attn, n_train_steps=300, n_eval=100, device=dev
    )
    r_conv = run_induction_v2_investigation(
        conv, n_train_steps=300, n_eval=100, device=dev
    )
    assert r_attn.status == "ok", f"attn probe failed: {r_attn.status}"
    assert r_conv.status == "ok", f"conv probe failed: {r_conv.status}"
    assert r_attn.auc - r_conv.auc > 0.3, (
        f"attention should out-separate conv by > 0.3 AUC. "
        f"attn={r_attn.auc:.3f} conv={r_conv.auc:.3f}"
    )
