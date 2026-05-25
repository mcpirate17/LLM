"""Capability screener: ungated corpus + OOD-honest pipeline (handoff regression).

Guards the two foundation fixes from tasks/induction_corpus_handoff.md:
  1. the training corpus is EVERY induction-labeled fingerprint (no provenance/
     completeness gate) — it must be far larger than the old 3143-row cap;
  2. the persisted scale-scoring contract (featurize_op_sets layout, load_screener
     returning a .predict-able model + op_vocab) is preserved for the generator.

These read research/runs.db (the live corpus); skip cleanly when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.tools import capability_screener as cs

_DB = "research/runs.db"
_OLD_CAP = 3143  # the deduped-predictor ∩ induction cap the fix removes


pytestmark = pytest.mark.skipif(
    not Path(_DB).exists(), reason="research/runs.db not present"
)


def test_label_corpus_is_ungated_and_large() -> None:
    """Corpus must hold all induction labels, not the old full-experiment ∩ cap."""
    cp, vocab = cs._label_corpus(_DB, "induction_screening_auc")
    assert len(cp.fps) > 5 * _OLD_CAP, (
        f"corpus {len(cp.fps)} not materially larger than old cap {_OLD_CAP}; "
        "provenance/completeness gate likely re-introduced"
    )
    # feature matrix is aligned and free (op-presence + op_count + pair_count)
    assert cp.X.shape[0] == len(cp.fps) == len(cp.y) == len(cp.clusters)
    assert cp.X.shape[1] == len(vocab) + 2
    assert (cp.y > 0.35).sum() > 0  # has positives to learn from


def test_featurize_layout_matches_static_matrix() -> None:
    """featurize_op_sets (scale path) must reproduce _static_matrix's column layout."""
    cp, vocab = cs._label_corpus(_DB, "induction_screening_auc")
    fps = cp.fps[:8]
    X_db, _, _ = cs._static_matrix(_DB, fps, op_vocab=vocab)
    # reconstruct op-sets/counts from the DB matrix and re-featurize in memory
    op_sets = [
        {vocab[j] for j in np.nonzero(X_db[i, : len(vocab)])[0]}
        for i in range(len(fps))
    ]
    counts = [int(X_db[i, -2]) for i in range(len(fps))]
    pairs = [int(X_db[i, -1]) for i in range(len(fps))]
    X_mem = cs.featurize_op_sets(op_sets, counts, pairs, vocab)
    assert X_mem.shape == X_db.shape
    np.testing.assert_array_equal(X_mem[:, : len(vocab)], X_db[:, : len(vocab)])


def test_novel_winner_check_ranks_stdp_above_median() -> None:
    """OOD go/no-go: a fast GBM on the full corpus must rank the headline STDP winner
    (real induction 0.894) above the corpus median — the in-dist screener did not."""
    from research.tools.induction_predictor_foundation import _fit_gbm

    cp, vocab = cs._label_corpus(_DB, "induction_screening_auc")
    model = _fit_gbm(cp.X, cp.y)
    corpus_pred = np.asarray(model.predict(cp.X), dtype=np.float64)
    check = cs._novel_winner_check(model, vocab, corpus_pred)
    headline = next(
        w
        for w in check["winners"]
        if w["fingerprint"] == "e656938e359ada50"  # pragma: allowlist secret
    )
    assert headline["above_median"], (
        f"STDP winner predicted {headline['predicted']} <= corpus median "
        f"{check['corpus_median_pred']} — regress-to-familiar pathology persists"
    )
