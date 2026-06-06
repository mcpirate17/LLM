"""Consumer for the trained Tier-2 value predictor.

Loads the GradientBoostingRegressor written by ``tools.train_tier2_predictor``
and predicts a candidate's Tier-2 ``mean_delta`` (net win margin vs baseline) at
scoring time.

The loop SELF-ACTIVATES the moment a model is deployed: until then
``predict_mean_delta`` returns ``None`` and quality scoring falls back to its
heuristic estimate. Train and serve features come from the same builder
(``proposer.tier2_features``), so there is no train/serve skew. The model file is
written only when it clears the OOD deploy gate (>=60 labels, >=12 archs,
positive leave-architecture-out R²/Spearman), so its presence already means it
beats predict-the-mean out of distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from component_fab.proposer.spec_generator import ProposalSpec

_REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = _REPO / "research" / "runtime" / "tier2_value_predictor"
MODEL_PATH = MODEL_DIR / "model.joblib"

# {"mtime": float, "model": Any, "extractor": Any} — reloads when the file mtime
# changes so a freshly trained model is picked up without a restart.
_CACHE: dict[str, Any] = {}


def predictor_available() -> bool:
    return MODEL_PATH.exists()


def _load_model() -> Any | None:
    if not MODEL_PATH.exists():
        _CACHE.clear()
        return None
    mtime = MODEL_PATH.stat().st_mtime
    if _CACHE.get("mtime") != mtime:
        import joblib

        _CACHE["model"] = joblib.load(MODEL_PATH)
        _CACHE["mtime"] = mtime
        _CACHE.pop("extractor", None)  # rebuild extractor alongside a new model
    return _CACHE.get("model")


def _extractor() -> Any:
    ex = _CACHE.get("extractor")
    if ex is None:
        from research.tools.measured_descriptors import MeasuredDescriptorExtractor

        ex = MeasuredDescriptorExtractor(n_seeds=2)
        _CACHE["extractor"] = ex
    return ex


def predict_mean_delta(spec: ProposalSpec) -> float | None:
    """Predicted Tier-2 ``mean_delta`` for ``spec``.

    Returns ``None`` when no model is deployed or the candidate cannot be
    measured — callers must fall back, never treat ``None`` as 0.0.
    """
    model = _load_model()
    if model is None:
        return None
    from component_fab.proposer.tier2_features import features_for_spec

    feat = features_for_spec(spec, extractor=_extractor())
    if feat is None:
        return None
    import numpy as np

    return float(model.predict(np.asarray([feat], dtype=float))[0])
