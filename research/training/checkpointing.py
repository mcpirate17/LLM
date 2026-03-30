"""
Checkpoint Manager for Long-Running Experiments

Provides periodic checkpoint writing and resume support so interrupted
runs (continuous mode, investigation, validation) can continue from
where they left off instead of losing all progress.

Storage layout:
    {checkpoint_dir}/{experiment_id}/continuous.pt   -- continuous loop state
    {checkpoint_dir}/{experiment_id}/{phase}_{candidate_idx}_{seed_idx}.pt  -- phase state
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Save and load checkpoint state for experiment resumption.

    Uses atomic writes (save to .tmp, then os.replace) to prevent
    partial checkpoint corruption on crash.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)

    def _exp_dir(self, experiment_id: str) -> Path:
        return self.checkpoint_dir / experiment_id

    def _atomic_save(self, state: Dict[str, Any], path: Path) -> None:
        """Save state dict atomically via tmp file + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        torch.save(state, str(tmp_path))
        os.replace(str(tmp_path), str(path))
        logger.debug("Checkpoint saved: %s", path)

    # ── Continuous loop checkpoints ──

    def save_continuous(
        self,
        experiment_id: str,
        config_dict: Dict[str, Any],
        n_experiments: int,
        elapsed_seconds: float,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save continuous loop state (experiment count, timing, config)."""
        state = {
            "n_experiments": n_experiments,
            "elapsed_seconds": elapsed_seconds,
            "config_dict": config_dict,
        }
        if extra_state:
            state["extra"] = extra_state
        path = self._exp_dir(experiment_id) / "continuous.pt"
        self._atomic_save(state, path)

    def load_continuous(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load continuous loop checkpoint, or None if not found."""
        path = self._exp_dir(experiment_id) / "continuous.pt"
        if not path.exists():
            return None
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=False)
            logger.info(
                "Loaded continuous checkpoint: %s (n_experiments=%d)",
                path,
                state.get("n_experiments", 0),
            )
            return state
        except Exception as e:
            logger.error("Failed to load continuous checkpoint %s: %s", path, e)
            raise e

    # ── Phase checkpoints (investigation / validation) ──

    def save_phase(
        self,
        experiment_id: str,
        phase: str,
        candidate_idx: int,
        seed_idx: int,
        model_state_dict: Dict[str, Any],
        optimizer_state_dict: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save mid-phase training state for a specific candidate/seed."""
        state = {
            "phase": phase,
            "candidate_idx": candidate_idx,
            "seed_idx": seed_idx,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "step": step,
        }
        if metrics:
            state["metrics"] = metrics
        filename = f"{phase}_{candidate_idx}_{seed_idx}.pt"
        path = self._exp_dir(experiment_id) / filename
        self._atomic_save(state, path)

    def load_phase(
        self,
        experiment_id: str,
        phase: str,
        candidate_idx: int,
        seed_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Load phase checkpoint for a specific candidate/seed, or None."""
        filename = f"{phase}_{candidate_idx}_{seed_idx}.pt"
        path = self._exp_dir(experiment_id) / filename
        if not path.exists():
            return None
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=False)
            logger.info(
                "Loaded phase checkpoint: %s (step=%d)", path, state.get("step", 0)
            )
            return state
        except Exception as e:
            logger.error("Failed to load phase checkpoint %s: %s", path, e)
            raise e

    # ── Cleanup ──

    def cleanup(self, experiment_id: str) -> None:
        """Remove all checkpoints for a completed experiment."""
        exp_dir = self._exp_dir(experiment_id)
        if exp_dir.exists():
            shutil.rmtree(str(exp_dir), ignore_errors=True)
            logger.info("Cleaned up checkpoints for experiment %s", experiment_id)

    def has_checkpoint(self, experiment_id: str) -> bool:
        """Check if any checkpoint exists for the given experiment."""
        exp_dir = self._exp_dir(experiment_id)
        if not exp_dir.exists():
            return False
        return any(exp_dir.iterdir())
