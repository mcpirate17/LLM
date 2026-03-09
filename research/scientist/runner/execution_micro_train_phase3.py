"""Micro-train helpers extracted from execution._micro_train."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class _ExecutionMicroTrainPhase3Mixin:
    """Split helpers for micro-train phase orchestration."""

    def _micro_train_make_random_batch(
        self,
        seed_int: int,
        step: int,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        dev: torch.device,
    ) -> torch.Tensor:
        """Generate deterministic random batch for a given step."""
        torch.manual_seed(seed_int * 100_000 + step)
        return torch.randint(0, int(vocab_size), (batch_size, seq_len), device=dev)

    def _micro_train_discovery_eval(
        self,
        model,
        config,
        dev: torch.device,
        seed_int: int,
        seq_len: int,
    ) -> Optional[float]:
        """Run fast discovery-loss evaluation on random batches."""
        discovery_steps = min(5, int(config.stage1_steps) // 10)
        if discovery_steps <= 0:
            return None

        losses = []
        model.eval()
        with torch.no_grad():
            for ds in range(discovery_steps):
                d_batch = self._micro_train_make_random_batch(
                    seed_int=seed_int,
                    step=ds + 9999,
                    batch_size=int(config.stage1_batch_size),
                    seq_len=seq_len,
                    vocab_size=int(config.vocab_size),
                    dev=dev,
                )
                with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=True):
                    d_logits = model(d_batch)
                    d_loss = F.cross_entropy(
                        d_logits[:, :-1].reshape(-1, int(config.vocab_size)),
                        d_batch[:, 1:].reshape(-1),
                    )
                losses.append(float(d_loss.item()))
        model.train()
        if not losses:
            return None
        return sum(losses) / len(losses)

    def _micro_train_optional_validation_loss(
        self,
        model,
        config,
        dev: torch.device,
        seq_len: int,
        seed: int,
    ) -> Optional[float]:
        """Compute optional heldout validation loss on corpus val split."""
        val_batches = max(1, int(getattr(config, "stage1_val_batches", 0) or 0))
        compute_val = bool(getattr(config, "stage1_compute_val_loss", True))
        val_batch_size = int(getattr(config, "stage1_val_batch_size", 0) or config.stage1_batch_size)
        val_frac = float(getattr(config, "corpus_val_fraction", 0.0) or 0.0)
        if not (compute_val and val_batches > 0 and val_frac > 0.0):
            return None
        if str(config.data_mode or "random").strip().lower() != "corpus":
            return None

        losses = []
        model.eval()
        try:
            with torch.no_grad():
                for i in range(val_batches):
                    input_ids = self._sample_training_input_ids(
                        config=config,
                        dev=dev,
                        batch_size=val_batch_size,
                        seq_len=seq_len,
                        seed=seed + 10_000 + i,
                        split="val",
                    )
                    if input_ids is None:
                        continue
                    with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                        logits = model(input_ids)
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            input_ids[:, 1:].reshape(-1),
                        )
                    if loss is not None and torch.isfinite(loss):
                        losses.append(float(loss.item()))
        finally:
            model.train()
        if not losses:
            return None
        return sum(losses) / len(losses)

    def _micro_train_optional_discovery_loss(
        self,
        model,
        config,
        dev: torch.device,
        seq_len: int,
        seed: int,
    ) -> Optional[float]:
        """Compute optional discovery loss on random tokens."""
        discovery_batches = max(1, int(getattr(config, "stage1_discovery_batches", 0) or 0))
        compute_discovery = bool(getattr(config, "stage1_compute_discovery_loss", True))
        discovery_batch_size = int(getattr(config, "stage1_discovery_batch_size", 0) or config.stage1_batch_size)
        if not (compute_discovery and discovery_batches > 0):
            return None

        losses = []
        model.eval()
        try:
            with torch.no_grad():
                for i in range(discovery_batches):
                    torch.manual_seed(int(seed) * 10_000 + 3_000 + i)
                    input_ids = torch.randint(
                        0,
                        int(config.vocab_size),
                        (discovery_batch_size, seq_len),
                        device=dev,
                    )
                    with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                        logits = model(input_ids)
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            input_ids[:, 1:].reshape(-1),
                        )
                    if loss is not None and torch.isfinite(loss):
                        losses.append(float(loss.item()))
        finally:
            model.train()
        if not losses:
            return None
        return sum(losses) / len(losses)
