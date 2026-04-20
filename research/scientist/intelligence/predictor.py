"""Performance predictors — aggregator.

Ridge regression on 18D fingerprint features, LightGBM on graph-structure
features, and an ensemble wrapper. Previously a 1,939-line god file; the
implementations now live in three focused sub-modules. This module
re-exports their public API so existing call sites are unchanged.
"""

from __future__ import annotations

from .predictor_ridge import (  # noqa: F401
    PerformancePredictor,
    _extract_features,
    _query_training_data,
    evaluate,
    predict,
    train,
)
from .predictor_gbm import (  # noqa: F401
    GBMPredictor,
    _graph_signature,
    _load_screening_predictor_corpus_rows,
    _query_graph_training_data,
    analyze_graph_label_quality,
    evaluate_gbm,
    evaluate_gbm_induction,
    train_gbm,
)
from .predictor_ensemble import (  # noqa: F401
    EnsemblePredictor,
    _calibrate_ensemble,
    load_runtime_ensemble,
    train_ensemble,
)
