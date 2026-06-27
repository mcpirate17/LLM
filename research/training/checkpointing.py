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
import torch.nn as nn
from torch.utils._pytree import tree_map_only

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

    def _artifact_dir(self, experiment_id: str) -> Path:
        return self.checkpoint_dir / "_investigation_artifacts" / experiment_id

    def _atomic_save(self, state: Dict[str, Any], path: Path) -> None:
        """Save state dict atomically via tmp file + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(path))
        logger.debug("Checkpoint saved: %s", path)

    @staticmethod
    def _cpu_tree(value: Any) -> Any:
        return tree_map_only(torch.Tensor, lambda t: t.detach().cpu(), value)

    @staticmethod
    def _move_tree_to_device(value: Any, device: torch.device) -> Any:
        return tree_map_only(torch.Tensor, lambda t: t.to(device=device), value)

    @staticmethod
    def _validate_phase_state(state: Dict[str, Any], path: Path) -> Dict[str, Any]:
        required = {
            "phase",
            "candidate_idx",
            "seed_idx",
            "model_state_dict",
            "optimizer_state_dict",
            "step",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(
                f"Checkpoint {path} is missing required fields: {', '.join(missing)}"
            )
        return state

    @classmethod
    def restore_phase_state(
        cls,
        checkpoint_state: Dict[str, Any],
        *,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str | torch.device] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Restore model/optimizer objects from a loaded phase checkpoint."""
        target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.empty(0)).device
        )
        model_state = cls._move_tree_to_device(
            checkpoint_state["model_state_dict"], target_device
        )
        model.load_state_dict(model_state, strict=strict)

        if optimizer is not None:
            optimizer_state = cls._move_tree_to_device(
                checkpoint_state["optimizer_state_dict"], target_device
            )
            optimizer.load_state_dict(optimizer_state)

        return {
            "step": int(checkpoint_state.get("step", 0)),
            "phase": checkpoint_state.get("phase"),
            "candidate_idx": int(checkpoint_state.get("candidate_idx", 0)),
            "seed_idx": int(checkpoint_state.get("seed_idx", 0)),
            "metrics": checkpoint_state.get("metrics") or {},
        }

    @staticmethod
    def phase_resume_candidate_idx(checkpoint_state: Dict[str, Any] | None) -> int:
        """Return the resume candidate index from a phase checkpoint payload."""
        if not checkpoint_state:
            return 0
        metrics = checkpoint_state.get("metrics") or {}
        candidate_idx = metrics.get(
            "candidate_idx", checkpoint_state.get("candidate_idx", 0)
        )
        return int(candidate_idx or 0)

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
        except Exception:
            logger.exception("Failed to load continuous checkpoint %s", path)
            raise
        logger.info(
            "Loaded continuous checkpoint: %s (n_experiments=%d)",
            path,
            state.get("n_experiments", 0),
        )
        return state

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
            "schema_version": 2,
            "phase": phase,
            "candidate_idx": candidate_idx,
            "seed_idx": seed_idx,
            "model_state_dict": self._cpu_tree(model_state_dict),
            "optimizer_state_dict": self._cpu_tree(optimizer_state_dict),
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
            state = self._validate_phase_state(state, path)
        except Exception:
            logger.exception("Failed to load phase checkpoint %s", path)
            raise
        logger.info("Loaded phase checkpoint: %s (step=%d)", path, state.get("step", 0))
        return state

    def load_phase_into(
        self,
        experiment_id: str,
        phase: str,
        candidate_idx: int,
        seed_idx: int,
        *,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str | torch.device] = None,
        strict: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Load a phase checkpoint and restore it into live objects."""
        state = self.load_phase(experiment_id, phase, candidate_idx, seed_idx)
        if state is None:
            return None
        return self.restore_phase_state(
            state,
            model=model,
            optimizer=optimizer,
            device=device,
            strict=strict,
        )

    def save_investigation_artifact(
        self,
        experiment_id: str,
        source_result_id: str,
        training_program_idx: int,
        payload: Dict[str, Any],
        model_state_dict: Optional[Dict[str, Any]] = None,
        artifact_kind: str = "program",
    ) -> Path:
        """Persist investigation artifacts outside normal checkpoint cleanup.

        Used to preserve expensive investigation work even if a later
        post-processing step crashes.
        """
        state: Dict[str, Any] = {
            "artifact_kind": artifact_kind,
            "experiment_id": experiment_id,
            "source_result_id": source_result_id,
            "training_program_idx": training_program_idx,
            "payload": payload,
        }
        if model_state_dict:
            state["model_state_dict"] = self._cpu_tree(model_state_dict)
        filename = f"{source_result_id}_tp{training_program_idx}_{artifact_kind}.pt"
        path = self._artifact_dir(experiment_id) / filename
        self._atomic_save(state, path)
        return path

    # ── Cleanup ──

    def cleanup(self, experiment_id: str) -> None:
        """Remove all checkpoints for a completed experiment."""
        exp_dir = self._exp_dir(experiment_id)
        if exp_dir.exists():
            shutil.rmtree(str(exp_dir), ignore_errors=True)
            logger.info("Cleaned up checkpoints for experiment %s", experiment_id)
