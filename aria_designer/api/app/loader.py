"""Component Loader — scans components/ directories and registers into DB.

On API startup, scans all components/<category>/<id>/manifest.yaml files,
validates them, and upserts into the component registry.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from . import database as db

logger = logging.getLogger(__name__)

COMPONENTS_ROOT = Path(__file__).resolve().parent.parent.parent / "components"

# Required top-level keys in manifest
REQUIRED_KEYS = {
    "id",
    "version",
    "name",
    "category",
    "inputs",
    "outputs",
    "implementation",
}

VALID_CATEGORIES = {
    "math",
    "linear_algebra",
    "structural",
    "routing",
    "mixing",
    "channel_mixing",
    "normalization",
    "positional",
    "blocks",
    "io",
    "representation",
    "topology",
    "sequence",
    "frequency",
    "functional",
    "math_space",
    "data_io",
    "data_transform",
    "control_flow",
}

VALID_STATUSES = {"draft", "approved", "deprecated", "quarantined"}


def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Validate a component manifest. Returns list of error strings."""
    errors = []
    missing = REQUIRED_KEYS - set(manifest.keys())
    if missing:
        errors.append(f"Missing required keys: {missing}")

    if "category" in manifest and manifest["category"] not in VALID_CATEGORIES:
        errors.append(f"Invalid category: {manifest['category']}")

    if "status" in manifest and manifest["status"] not in VALID_STATUSES:
        errors.append(f"Invalid status: {manifest['status']}")

    if "outputs" in manifest and len(manifest["outputs"]) < 1:
        errors.append("Component must have at least one output port")

    slots = manifest.get("slots")
    if slots is not None:
        if not isinstance(slots, list):
            errors.append("slots must be a list when provided")
        else:
            for idx, slot in enumerate(slots):
                if not isinstance(slot, dict):
                    errors.append(f"slots[{idx}] must be an object")
                    continue
                if not slot.get("name"):
                    errors.append(f"slots[{idx}] missing required name")

    templates = manifest.get("templates")
    if templates is not None:
        if not isinstance(templates, list):
            errors.append("templates must be a list when provided")
        else:
            for idx, template in enumerate(templates):
                if not isinstance(template, dict):
                    errors.append(f"templates[{idx}] must be an object")
                    continue
                if not template.get("id"):
                    errors.append(f"templates[{idx}] missing required id")

    return errors


def load_manifest(manifest_path: Path) -> Dict[str, Any] | None:
    """Load and validate a single manifest file. Returns None on error."""
    try:
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to parse %s: %s", manifest_path, e)
        return None

    if manifest is None:
        logger.error("Empty manifest: %s", manifest_path)
        return None

    errors = validate_manifest(manifest)
    if errors:
        logger.error("Invalid manifest %s: %s", manifest_path, "; ".join(errors))
        return None

    return manifest


def scan_and_load(root: Path | None = None) -> int:
    """Scan components directory tree, load all manifests into DB.

    Returns count of successfully loaded components.
    """
    root = root or COMPONENTS_ROOT
    if not root.exists():
        logger.warning("Components root does not exist: %s", root)
        return 0

    now = datetime.now(timezone.utc).isoformat()
    loaded = 0

    for manifest_path in sorted(root.rglob("manifest.yaml")):
        manifest = load_manifest(manifest_path)
        if manifest is None:
            continue

        try:
            db.upsert_component(manifest, created_at=now, updated_at=now)
            loaded += 1
        except Exception as e:
            logger.error("Failed to register %s: %s", manifest_path, e)

    logger.info("Loaded %d components from %s", loaded, root)
    return loaded
