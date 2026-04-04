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

from research.defaults import VOCAB_SIZE
from .reference_training import (
    train_reference_transformer,
)


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
        final_loss, _ = train_reference_transformer(
            d_model=d_model,
            seq_len=seq_len,
            n_steps=n_steps,
            vocab_size=vocab_size,
            batch_size=batch_size,
            lr=lr,
            device=device,
            n_layers=n_layers,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            betas=betas,
            seed=seed,
            data_fn=data_fn,
        )
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
