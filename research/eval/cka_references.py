"""
CKA Reference Artifact Store

Loads, validates, and caches reference activation artifacts for
artifact-backed CKA similarity computation in behavioral fingerprinting.

Artifact layout:
    artifacts/cka_references/<version>/
        manifest.json
        transformer.pt
        ssm.pt
        conv.pt

Each .pt file contains a dict with:
    {"activations": Tensor, "config": dict, "training_info": dict}
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Required reference families
REFERENCE_FAMILIES = ("transformer", "ssm", "conv")

# Manifest schema version this code understands
SUPPORTED_SCHEMA_VERSIONS = {"1"}


@dataclass
class ArtifactManifest:
    """Parsed and validated artifact manifest."""

    artifact_version: str
    schema_version: str
    created_at: str
    code_version: str
    reference_families: list
    probe_protocol_hash: str
    activation_shape: list  # [seq_len, dim] expected shape per family
    quality_flags: dict = field(default_factory=dict)

    # Populated after loading
    artifact_dir: Optional[Path] = None


def load_manifest(artifact_dir: Path) -> ArtifactManifest:
    """Load and validate manifest.json from an artifact directory.

    Raises ValueError on missing/malformed/incompatible manifests.
    """
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"No manifest.json in {artifact_dir}")

    try:
        with open(manifest_path, "r") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Cannot parse manifest.json: {e}")

    # Required fields
    required = [
        "artifact_version",
        "schema_version",
        "created_at",
        "code_version",
        "reference_families",
        "probe_protocol_hash",
        "activation_shape",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Manifest missing required fields: {missing}")

    # Schema version check
    schema_ver = str(raw["schema_version"])
    if schema_ver not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema version '{schema_ver}', "
            f"supported: {SUPPORTED_SCHEMA_VERSIONS}"
        )

    # Validate reference families
    families = raw["reference_families"]
    if not isinstance(families, list):
        raise ValueError("reference_families must be a list")
    missing_families = set(REFERENCE_FAMILIES) - set(families)
    if missing_families:
        raise ValueError(f"Manifest missing reference families: {missing_families}")

    # Validate activation shape
    shape = raw["activation_shape"]
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"activation_shape must be [seq_len, dim], got {shape}")
    if not all(isinstance(x, int) and x > 0 for x in shape):
        raise ValueError(f"activation_shape must be positive integers, got {shape}")

    manifest = ArtifactManifest(
        artifact_version=str(raw["artifact_version"]),
        schema_version=schema_ver,
        created_at=str(raw["created_at"]),
        code_version=str(raw["code_version"]),
        reference_families=families,
        probe_protocol_hash=str(raw["probe_protocol_hash"]),
        activation_shape=shape,
        quality_flags=raw.get("quality_flags", {}),
        artifact_dir=artifact_dir,
    )
    return manifest


def load_reference_activations(
    artifact_dir: Path, manifest: ArtifactManifest
) -> Dict[str, torch.Tensor]:
    """Load reference activation tensors for all families.

    Returns dict mapping family name -> activation tensor.
    Raises ValueError if files are missing or tensors don't match
    the declared activation_shape.
    """
    activations = {}
    expected_shape = tuple(manifest.activation_shape)

    for family in REFERENCE_FAMILIES:
        pt_path = artifact_dir / f"{family}.pt"
        if not pt_path.exists():
            raise ValueError(f"Missing artifact file: {pt_path}")

        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise ValueError(f"Cannot load {pt_path}: {e}")

        if not isinstance(data, dict) or "activations" not in data:
            raise ValueError(f"{pt_path} must contain a dict with 'activations' key")

        tensor = data["activations"]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"{family}.pt activations must be a Tensor, got {type(tensor)}"
            )

        actual_shape = tuple(tensor.shape[-2:])
        if actual_shape != expected_shape:
            raise ValueError(
                f"{family}.pt shape mismatch: expected {expected_shape}, "
                f"got {actual_shape}"
            )

        activations[family] = tensor.float()

    return activations


class ReferenceCkaStore:
    """Singleton-style loader and cache for CKA reference artifacts.

    Thread-safe. Lazily loads artifacts on first use.

    Usage:
        # relative to project root
        store = ReferenceCkaStore(artifact_dir="artifacts/cka_references/v1")
        refs = store.get_references()  # Dict[str, Tensor] or None
        meta = store.get_metadata()    # source/version info
    """

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        artifact_version: Optional[str] = None,
        allow_heuristic_fallback: bool = True,
    ):
        # Resolve from env or explicit args
        self._artifact_dir = artifact_dir or os.environ.get("CKA_REFERENCE_DIR")
        self._artifact_version = artifact_version or os.environ.get(
            "CKA_REFERENCE_VERSION", "v1"
        )
        self._allow_fallback = allow_heuristic_fallback or os.environ.get(
            "CKA_ALLOW_HEURISTIC_FALLBACK", "true"
        ).lower() in ("true", "1", "yes")

        self._lock = threading.Lock()
        self._loaded = False
        self._manifest: Optional[ArtifactManifest] = None
        self._references: Optional[Dict[str, torch.Tensor]] = None
        self._load_error: Optional[str] = None
        self._fallback_warned = False

    def _resolve_dir(self) -> Optional[Path]:
        """Resolve the artifact directory path."""
        if self._artifact_dir:
            p = Path(self._artifact_dir)
            if p.exists():
                return p
            return None

        # Default: look relative to project root
        # eval/cka_references.py -> project root is two levels up
        project_root = Path(__file__).parent.parent
        default = project_root / "artifacts" / "cka_references" / self._artifact_version
        if default.exists():
            return default
        return None

    def _load(self) -> None:
        """Attempt to load artifacts. Called once under lock."""
        if self._loaded:
            return
        self._loaded = True

        artifact_dir = self._resolve_dir()
        if artifact_dir is None:
            self._load_error = "Artifact directory not found"
            logger.info(
                "CKA reference artifacts not found, will use heuristic fallback"
            )
            return

        try:
            self._manifest = load_manifest(artifact_dir)
            self._references = load_reference_activations(artifact_dir, self._manifest)
            logger.info(
                "Loaded CKA reference artifacts v%s (%d families)",
                self._manifest.artifact_version,
                len(self._references),
            )
        except ValueError as e:
            self._load_error = str(e)
            self._manifest = None
            self._references = None
            logger.warning("Failed to load CKA reference artifacts: %s", e)

    def get_references(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get loaded reference activations, or None if unavailable."""
        with self._lock:
            self._load()
        return self._references

    def get_metadata(self) -> Dict[str, str]:
        """Get metadata about the CKA source for provenance tracking."""
        with self._lock:
            self._load()

        if self._references is not None and self._manifest is not None:
            return {
                "cka_source": "artifact",
                "cka_artifact_version": self._manifest.artifact_version,
                "cka_probe_protocol_hash": self._manifest.probe_protocol_hash,
                "cka_reference_quality": self._manifest.quality_flags.get(
                    "overall", "unknown"
                ),
                "cka_similarity_path": "_compute_reference_cka",
            }
        else:
            if not self._fallback_warned:
                self._fallback_warned = True
                logger.warning(
                    "Using heuristic CKA fallback: %s",
                    self._load_error or "unknown reason",
                )
            return {
                "cka_source": "heuristic_fallback",
                "cka_artifact_version": None,
                "cka_probe_protocol_hash": None,
                "cka_reference_quality": "heuristic",
                "cka_similarity_path": "_compute_reference_cka",
            }

    @property
    def is_artifact_backed(self) -> bool:
        """Whether real artifacts are loaded (vs heuristic fallback)."""
        with self._lock:
            self._load()
        return self._references is not None

    @property
    def allow_heuristic_fallback(self) -> bool:
        return self._allow_fallback

    def reset(self) -> None:
        """Reset cached state (for testing)."""
        with self._lock:
            self._loaded = False
            self._manifest = None
            self._references = None
            self._load_error = None
            self._fallback_warned = False


# Module-level default instance (lazy)
_default_store: Optional[ReferenceCkaStore] = None
_default_lock = threading.Lock()


def get_default_store() -> ReferenceCkaStore:
    """Get or create the module-level default ReferenceCkaStore."""
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = ReferenceCkaStore()
        return _default_store


def reset_default_store() -> None:
    """Reset the module-level default store (for testing)."""
    global _default_store
    with _default_lock:
        _default_store = None
