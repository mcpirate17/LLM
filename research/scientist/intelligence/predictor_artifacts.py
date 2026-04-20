from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


STATE_DIR = Path("research/runtime/learning")
METRICS_REPORT_PATH = STATE_DIR / "predictor_metrics_report.json"

GBM_GATE_MODEL_PATH = STATE_DIR / "gbm_gate_model.txt"
GBM_RANK_MODEL_PATH = STATE_DIR / "gbm_rank_model.txt"
GBM_META_PATH = STATE_DIR / "gbm_predictor.json"

GRAPH_PREDICTOR_PATH = STATE_DIR / "graph_predictor.npz"
INTERACTION_MODEL_PATH = STATE_DIR / "interaction_model.npz"
OP_EMBEDDINGS_PATH = STATE_DIR / "op_embeddings.npz"
BAYESIAN_STATE_PATH = STATE_DIR / "bayesian_state.json"

ENSEMBLE_STATE_PATH = STATE_DIR / "ensemble_state.npz"
ENSEMBLE_META_PATH = STATE_DIR / "ensemble_state.json"


def ensure_state_dir(state_dir: str | Path = STATE_DIR) -> Path:
    path = Path(state_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: str | Path) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    return artifact_path


def metadata_sidecar_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".json")


def unlink_if_exists(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def unlink_paths(*paths: str | Path) -> None:
    for path in paths:
        unlink_if_exists(path)


def write_json(path: str | Path, payload: Any) -> Path:
    json_path = ensure_parent_dir(path)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return json_path


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_npz_archive(path: str | Path, **arrays: np.ndarray) -> Path:
    npz_path = ensure_parent_dir(path)
    np.savez_compressed(str(npz_path), **arrays)
    return npz_path


def load_npz_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def save_npz_with_metadata(
    path: str | Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> Path:
    npz_path = save_npz_archive(path, **arrays)
    write_json(metadata_sidecar_path(npz_path), dict(metadata))
    return npz_path


def load_npz_with_metadata(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = Path(path)
    return load_npz_archive(npz_path), dict(read_json(metadata_sidecar_path(npz_path)))
