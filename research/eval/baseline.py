"""
Transformer Baseline

Micro-trains a vanilla transformer as a reference point.
Caches results by model/training recipe so baseline runs are reused safely.
Persists cache in SQLite for reuse across runs.
"""

from __future__ import annotations

import gc
import math
import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.defaults import VOCAB_SIZE


class _SimpleTransformerLayer(nn.Module):
    """Minimal transformer layer for baseline comparison."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class _BaselineTransformer(nn.Module):
    """Minimal 2-layer transformer for baseline loss measurement."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [_SimpleTransformerLayer(d_model) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)


class TransformerBaseline:
    """Manages a cached transformer baseline for comparison.

    Caches the final loss by (d_model, seq_len, n_steps, vocab_size) so
    the baseline only needs to be trained once per configuration.
    """

    def __init__(self, cache_path: str = "research/baseline_cache.db"):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.cache_path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS baseline_results (
                config_key TEXT PRIMARY KEY,
                final_loss REAL NOT NULL,
                initial_loss REAL NOT NULL,
                trained_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def close(self):
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _config_key(
        self,
        d_model: int,
        seq_len: int,
        n_steps: int,
        vocab_size: int,
        n_layers: int = 2,
        device: str = "cuda",
        optimizer_name: str = "adamw",
        weight_decay: float = 0.01,
        momentum: float = 0.0,
        betas: Optional[Tuple[float, float]] = None,
        data_mode: str = "random",
        data_tag: str = "none",
    ) -> str:
        dev_tag = "gpu" if device == "cuda" else "cpu"
        opt = (optimizer_name or "adamw").lower()
        wd_tag = f"{weight_decay:.8f}"
        mom_tag = f"{momentum:.8f}"
        beta_tag = "none"
        if betas is not None and len(betas) == 2:
            beta_tag = f"{float(betas[0]):.6f}_{float(betas[1]):.6f}"
        return (
            f"{d_model}_{seq_len}_{n_steps}_{vocab_size}_{n_layers}_{dev_tag}_"
            f"{opt}_{wd_tag}_{mom_tag}_{beta_tag}_{data_mode}_{data_tag}"
        )

    def _get_cached(self, config_key: str) -> Optional[float]:
        row = self._conn.execute(
            "SELECT final_loss FROM baseline_results WHERE config_key = ?",
            (config_key,),
        ).fetchone()
        return row[0] if row else None

    def _save_cache(self, config_key: str, final_loss: float, initial_loss: float):
        self._conn.execute(
            """INSERT OR REPLACE INTO baseline_results
               (config_key, final_loss, initial_loss, trained_at)
               VALUES (?, ?, ?, ?)""",
            (config_key, final_loss, initial_loss, time.time()),
        )
        self._conn.commit()

    def get_baseline_loss(
        self,
        d_model: int = 256,
        seq_len: int = 128,
        n_steps: int = 500,
        vocab_size: int = VOCAB_SIZE,
        batch_size: int = 4,
        lr: float = 3e-4,
        device: str = "cuda",
        n_layers: int = 2,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.01,
        momentum: float = 0.0,
        betas: Optional[Tuple[float, float]] = None,
        data_fn=None,
        data_mode: str = "random",
        data_tag: str = "none",
        cache_data_fn: bool = True,
    ) -> float:
        """Get the baseline transformer final loss, training if needed.

        Args:
            data_fn: Optional callable(batch_size, seq_len, device) -> input_ids tensor.
                     Used for training on real data.
            data_mode: "random" or "corpus".
            data_tag: Cache key suffix for data source (e.g. "shakespeare").
        """
        config_key = self._config_key(
            d_model,
            seq_len,
            n_steps,
            vocab_size,
            n_layers,
            device,
            optimizer_name,
            weight_decay,
            momentum,
            betas,
            data_mode=data_mode,
            data_tag=data_tag,
        )

        if cache_data_fn:
            cached = self._get_cached(config_key)
            if cached is not None:
                return cached

        # Train multiple seeds and average for stability (#47)
        n_seeds = 3
        losses = []
        for seed in range(n_seeds):
            loss = self._train_baseline(
                d_model,
                seq_len,
                n_steps,
                vocab_size,
                batch_size,
                lr,
                device,
                n_layers,
                optimizer_name=optimizer_name,
                weight_decay=weight_decay,
                momentum=momentum,
                betas=betas,
                seed=seed,
                data_fn=data_fn,
            )
            if math.isfinite(loss):
                losses.append(loss)

        final_loss = sum(losses) / len(losses) if losses else float("inf")
        if math.isfinite(final_loss) and cache_data_fn:
            # NB: "initial_loss" column stores seed-0 final loss (historical misnomer)
            self._save_cache(config_key, final_loss, losses[0])
        return final_loss

    def _train_baseline(
        self,
        d_model: int,
        seq_len: int,
        n_steps: int,
        vocab_size: int,
        batch_size: int,
        lr: float,
        device: str,
        n_layers: int = 2,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.01,
        momentum: float = 0.0,
        betas: Optional[Tuple[float, float]] = None,
        seed: int = 0,
        data_fn=None,
    ) -> float:
        """Train a baseline transformer and return final loss."""
        dev = torch.device(device if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        model = _BaselineTransformer(vocab_size, d_model, n_layers=n_layers).to(dev)
        opt = (optimizer_name or "adamw").lower()
        if opt == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            adamw_betas = betas if betas is not None else (0.9, 0.999)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=adamw_betas,
            )
        model.train()
        _data_gen = torch.Generator(device=dev).manual_seed(seed * 100000)

        final_loss = float("inf")

        try:
            for step in range(n_steps):
                if data_fn is not None:
                    input_ids = data_fn(batch_size, seq_len, dev)
                else:
                    input_ids = torch.randint(
                        0,
                        vocab_size,
                        (batch_size, seq_len),
                        device=dev,
                        generator=_data_gen,
                    )
                with torch.amp.autocast(
                    device_type=dev.type,
                    dtype=torch.bfloat16,
                    enabled=(dev.type == "cuda"),
                ):
                    logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, vocab_size),
                        input_ids[:, 1:].reshape(-1),
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    break

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                final_loss = loss.item()

        finally:
            del model, optimizer
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        return final_loss

    def compare(
        self,
        program_loss: float,
        d_model: int = 256,
        seq_len: int = 128,
        n_steps: int = 500,
        vocab_size: int = VOCAB_SIZE,
        batch_size: int = 4,
        lr: float = 3e-4,
        device: str = "cuda",
        n_layers: int = 2,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.01,
        momentum: float = 0.0,
        betas: Optional[Tuple[float, float]] = None,
        data_fn=None,
        data_mode: str = "random",
        data_tag: str = "none",
        cache_data_fn: bool = True,
    ) -> float:
        """Compare program loss to baseline. Returns ratio (< 1.0 = better)."""
        baseline_loss = self.get_baseline_loss(
            d_model,
            seq_len,
            n_steps,
            vocab_size,
            batch_size,
            lr,
            device,
            n_layers=n_layers,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            betas=betas,
            data_fn=data_fn,
            data_mode=data_mode,
            data_tag=data_tag,
            cache_data_fn=cache_data_fn,
        )
        if baseline_loss <= 0 or math.isnan(baseline_loss):
            return 1.0

        random_chance = math.log(max(vocab_size, 2))
        if baseline_loss >= random_chance * 0.95:
            return 1.0

        return program_loss / baseline_loss

    def compare_normalized(
        self,
        program_loss: float,
        program_params: int,
        d_model: int = 256,
        seq_len: int = 128,
        n_steps: int = 500,
        vocab_size: int = VOCAB_SIZE,
        batch_size: int = 4,
        lr: float = 3e-4,
        device: str = "cuda",
        n_layers: int = 2,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.01,
        momentum: float = 0.0,
        betas: Optional[Tuple[float, float]] = None,
        data_fn=None,
        data_mode: str = "random",
        data_tag: str = "none",
        cache_data_fn: bool = True,
    ) -> Dict[str, float]:
        """Compare program loss to a parameter-matched baseline."""
        raw_ratio = self.compare(
            program_loss,
            d_model,
            seq_len,
            n_steps,
            vocab_size,
            batch_size,
            lr,
            device,
            n_layers,
            optimizer_name,
            weight_decay,
            momentum,
            betas,
            data_fn,
            data_mode=data_mode,
            data_tag=data_tag,
            cache_data_fn=cache_data_fn,
        )

        # Estimate how many layers the baseline needs to match program_params
        # Each transformer layer has ~12 * d_model^2 params (attn + FF + LN)
        params_per_layer = 12 * d_model * d_model
        embed_params = vocab_size * d_model
        non_embed_params = max(0, program_params - embed_params)
        matched_layers = max(
            2, min(12, int(math.ceil(non_embed_params / max(params_per_layer, 1))))
        )

        if matched_layers <= 2:
            # Program is smaller or equal to standard baseline — normalized = raw
            return {
                "raw_ratio": raw_ratio,
                "normalized_ratio": raw_ratio,
                "param_efficiency": (1.0 - raw_ratio) / max(1, program_params / 1e6),
                "matched_baseline_layers": 2,
            }

        # Train a matched-param baseline
        matched_loss = self.get_baseline_loss(
            d_model,
            seq_len,
            n_steps,
            vocab_size,
            batch_size,
            lr,
            device,
            n_layers=matched_layers,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            betas=betas,
            data_fn=data_fn,
            data_mode=data_mode,
            data_tag=data_tag,
        )

        random_chance = math.log(max(vocab_size, 2))
        if (
            matched_loss <= 0
            or math.isnan(matched_loss)
            or matched_loss >= random_chance * 0.95
        ):
            normalized_ratio = raw_ratio
        else:
            normalized_ratio = program_loss / matched_loss

        param_efficiency = (1.0 - raw_ratio) / max(1, program_params / 1e6)

        return {
            "raw_ratio": raw_ratio,
            "normalized_ratio": normalized_ratio,
            "param_efficiency": param_efficiency,
            "matched_baseline_layers": matched_layers,
        }
