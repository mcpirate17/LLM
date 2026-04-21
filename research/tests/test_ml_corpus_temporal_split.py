from __future__ import annotations

import numpy as np
import pytest

from research.scientist.intelligence.ml_corpus import grouped_temporal_split


pytestmark = pytest.mark.unit


def test_grouped_temporal_split_respects_time_and_preserves_class_presence():
    signatures = ["a", "b", "c", "d", "e", "f"]
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
    timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)

    train_idx, val_idx, stats = grouped_temporal_split(signatures, labels, timestamps)

    assert set(train_idx.tolist()).isdisjoint(set(val_idx.tolist()))
    assert stats["split_strategy"] == "temporal_grouped"
    assert stats["temporal_cutoff_timestamp"] <= stats["temporal_val_start_timestamp"]

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    assert 0 in train_labels and 1 in train_labels
    assert 0 in val_labels and 1 in val_labels
    assert float(np.max(timestamps[train_idx])) <= float(np.min(timestamps[val_idx]))
