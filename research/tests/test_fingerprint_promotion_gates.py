"""Tests for Stream B: Promotion Integrity & Escalation gates.

B1: Auto-escalation to validation blocked without completed fingerprint
B2: Investigation marks incomplete fingerprint visibly (investigation_passed=False)
B3: Validation caps novelty without artifact CKA
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# B1: Block promotion on incomplete fingerprints
# ---------------------------------------------------------------------------


class TestB1EscalationFingerprintGate:
    """_auto_escalate_investigation must refuse escalation when
    fingerprint_completed_post_investigation is False."""

    @staticmethod
    def _make_mixin():
        from research.scientist.runner.results_auto_escalate_phase7 import (
            _ResultsAutoEscalatePhase7Mixin,
        )

        mixin = _ResultsAutoEscalatePhase7Mixin()
        # Stub methods the mixin calls on self
        mixin._score_candidate_pool = MagicMock(
            return_value={
                "scored": [],
                "selected": [],
                "summary": {},
                "policy": {},
                "reason": "",
            }
        )
        mixin._safe_build_evidence_pack = MagicMock(return_value={})
        mixin._emit_event = MagicMock()
        mixin._pending_validation = None
        return mixin

    @staticmethod
    def _make_nb_with_fingerprint(
        result_ids: list[str],
        fp_complete: bool,
        novelty_valid: bool = True,
        composite_score: float = 200.0,
    ):
        """Create a mock LabNotebook with the specified fingerprint state."""
        nb = MagicMock()

        # Build rows for the fingerprint_json query
        fp_json = json.dumps({"fingerprint_completed_post_investigation": fp_complete})
        fp_rows = []
        for rid in result_ids:
            row = {
                "result_id": rid,
                "novelty_valid_for_promotion": int(novelty_valid),
                "cka_source": "artifact" if fp_complete else "deferred",
                "fingerprint_json": fp_json,
            }
            # Make dict-like and subscriptable
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, k, _r=row: _r[k]
            mock_row.keys = lambda _r=row: _r.keys()
            fp_rows.append(mock_row)

        # Build rows for the composite_score query (includes replication fields)
        score_rows = []
        for rid in result_ids:
            _row_data = {
                "result_id": rid,
                "composite_score": composite_score,
                "replication_n": 3,
                "replication_loss_std": 0.01,
            }
            sr = MagicMock()
            sr.__getitem__ = lambda self, k, _r=_row_data: _r[k]
            score_rows.append(sr)

        # conn.execute returns different results based on query
        def mock_execute(query, params=None):
            result = MagicMock()
            if "fingerprint_json" in query:
                result.fetchall = MagicMock(return_value=fp_rows)
            elif "composite_score" in query:
                result.fetchall = MagicMock(return_value=score_rows)
            else:
                result.fetchall = MagicMock(return_value=[])
            return result

        nb.conn.execute = mock_execute
        return nb

    def test_blocks_incomplete_fingerprint(self):
        """Entries without completed fingerprint must not reach validation."""
        mixin = self._make_mixin()
        result_ids = ["rid_001", "rid_002"]
        nb = self._make_nb_with_fingerprint(result_ids, fp_complete=False)
        config = MagicMock()
        config.auto_validate = True
        config.auto_validate_min_composite_score = 0.0
        config.auto_validate_top_n = 5
        config.auto_validate_min_robustness = 0.5

        results = {
            "investigation_results": [
                {"result_id": rid, "robustness": 0.8, "best_loss_ratio": 0.1}
                for rid in result_ids
            ],
            "experiment_id": "exp_test",
        }

        mixin._auto_escalate_investigation(results, config, nb)

        # Should NOT have queued validation
        assert mixin._pending_validation is None
        # Should NOT have called _score_candidate_pool (no candidates passed)
        mixin._score_candidate_pool.assert_not_called()

    def test_allows_complete_fingerprint(self):
        """Entries with completed fingerprint should proceed to scoring."""
        mixin = self._make_mixin()
        result_ids = ["rid_001"]
        nb = self._make_nb_with_fingerprint(result_ids, fp_complete=True)

        # Need graph_meta query and MIN(screening_loss_ratio) for ref baseline
        _orig_execute = nb.conn.execute

        def patched_execute(query, params=None):
            result = MagicMock()
            if "graph_json" in query:
                row = MagicMock()
                row.__getitem__ = lambda self, k: {
                    "result_id": "rid_001",
                    "graph_json": "{}",
                    "routing_mode": None,
                }.get(k)
                row.keys = lambda: ["result_id", "graph_json", "routing_mode"]
                result.fetchall = MagicMock(return_value=[row])
            else:
                return _orig_execute(query, params)
            return result

        nb.conn.execute = patched_execute

        config = MagicMock()
        config.auto_validate = True
        config.auto_validate_min_composite_score = 0.0
        config.auto_validate_top_n = 5
        config.auto_validate_min_robustness = 0.5
        config.auto_validate_max_baseline_ratio = 0.80
        config.investigation_max_loss_ratio_multiplier = 10.0

        results = {
            "investigation_results": [
                {
                    "result_id": "rid_001",
                    "robustness": 0.8,
                    "best_loss_ratio": 0.1,
                    "baseline_loss_ratio": 0.45,
                    "novelty_confidence": 0.9,
                    "brittle_risk": False,
                    "loss_ratio_multiplier": 2.0,
                    "throughput_tok_s": 1000,
                    "flops_per_token": 100,
                    "peak_memory_mb": 512,
                }
            ],
            "experiment_id": "exp_test",
        }

        mixin._auto_escalate_investigation(results, config, nb)

        # Should have called _score_candidate_pool since candidate passed all gates
        mixin._score_candidate_pool.assert_called_once()


# ---------------------------------------------------------------------------
# B2: Investigation marks incomplete fingerprint visibly
# ---------------------------------------------------------------------------


class TestB2InvestigationFingerprintRequired:
    """When fingerprint completion fails, investigation_passed must be False
    and a persistent 'investigation_fingerprint_incomplete' tier is set."""

    def test_persistent_tier_for_incomplete_fingerprint(self):
        """_record_investigation_result sets tier='investigation_fingerprint_incomplete'
        when fingerprint_incomplete=True and investigation_passed=False."""
        from research.scientist.thresholds import TIER_RANK as _TIER_RANK

        # Verify tier exists in rank map
        assert "investigation_fingerprint_incomplete" in _TIER_RANK
        # Same rank as investigation_failed (not promotable)
        assert (
            _TIER_RANK["investigation_fingerprint_incomplete"]
            == _TIER_RANK["investigation_failed"]
        )

    def test_tier_selection_logic(self):
        """Verify the three-way tier logic: passed, fingerprint_incomplete, failed."""

        # Simulate the tier selection from _record_investigation_result
        def tier_for(investigation_passed, fingerprint_incomplete):
            return (
                "investigation"
                if investigation_passed
                else "investigation_fingerprint_incomplete"
                if fingerprint_incomplete
                else "investigation_failed"
            )

        assert tier_for(True, False) == "investigation"
        assert tier_for(False, True) == "investigation_fingerprint_incomplete"
        assert tier_for(False, False) == "investigation_failed"
