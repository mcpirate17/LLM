"""Shared transformer reference model and trainer."""

from __future__ import annotations

import gc
import os
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._reference_model_native import load_reference_model_native
from .training_core import run_training_loop
from .utils import language_model_loss


class SimpleTransformerLayer(nn.Module):
    """Minimal transformer layer for reference comparisons."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self._causal_mask_cache: dict[tuple[int, str, int | None], torch.Tensor] = {}

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (int(seq_len), device.type, device.index)
        mask = self._causal_mask_cache.get(key)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len,
                device=device,
            )
            self._causal_mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        h = self.ln1(x)
        h, _ = self.attn(
            h,
            h,
            h,
            attn_mask=self._causal_mask(seq_len, x.device),
            need_weights=False,
            is_causal=True,
        )
        x = x + h
        return x + self.ff(self.ln2(x))


class BaselineTransformer(nn.Module):
    """Minimal transformer used for baseline/reference training."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int = 2):
        super().__init__()
        self.n_heads = 4
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                SimpleTransformerLayer(d_model, n_heads=self.n_heads)
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not os.environ.get("ARIA_DISABLE_REFERENCE_MODEL_NATIVE"):
            try:
                native = load_reference_model_native()
                return native.baseline_transformer_forward(
                    input_ids,
                    self.embed.weight,
                    [layer.attn.in_proj_weight for layer in self.layers],
                    [layer.attn.in_proj_bias for layer in self.layers],
                    [layer.attn.out_proj.weight for layer in self.layers],
                    [layer.attn.out_proj.bias for layer in self.layers],
                    [layer.ff[0].weight for layer in self.layers],
                    [layer.ff[0].bias for layer in self.layers],
                    [layer.ff[2].weight for layer in self.layers],
                    [layer.ff[2].bias for layer in self.layers],
                    [layer.ln1.weight for layer in self.layers],
                    [layer.ln1.bias for layer in self.layers],
                    [layer.ln2.weight for layer in self.layers],
                    [layer.ln2.bias for layer in self.layers],
                    self.ln_f.weight,
                    self.ln_f.bias,
                    self.head.weight,
                    int(self.n_heads),
                )
            except Exception:
                pass
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln_f(x))


def train_reference_transformer(
    *,
    d_model: int,
    seq_len: int,
    n_steps: int,
    vocab_size: int,
    batch_size: int,
    lr: float,
    device: str | torch.device,
    n_layers: int = 2,
    optimizer_name: str = "adamw",
    weight_decay: float = 0.01,
    momentum: float = 0.0,
    betas: Optional[tuple[float, float]] = None,
    seed: int = 0,
    data_fn: Optional[Callable[[int, int, torch.device], torch.Tensor]] = None,
) -> tuple[float, int]:
    """Train one reference transformer seed and return (final_loss, param_count)."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = BaselineTransformer(vocab_size, d_model, n_layers=n_layers).to(dev)
    param_count = sum(p.numel() for p in model.parameters())
    model.train()
    data_gen = torch.Generator(device=dev).manual_seed(seed * 100000)

    def compute_loss(_step: int) -> torch.Tensor:
        if data_fn is not None:
            input_ids = data_fn(batch_size, seq_len, dev)
        else:
            input_ids = torch.randint(
                0,
                vocab_size,
                (batch_size, seq_len),
                device=dev,
                generator=data_gen,
            )
        with torch.amp.autocast(
            device_type=dev.type,
            dtype=torch.bfloat16,
            enabled=(dev.type == "cuda"),
        ):
            logits = model(input_ids)
            return language_model_loss(logits, input_ids, vocab_size)

    try:
        result = run_training_loop(
            model.parameters(),
            compute_loss,
            n_steps=n_steps,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            betas=betas,
            clip_grad=1.0,
        )
        return result.final_loss, param_count
    finally:
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
