"""Behavior tests for selected execution_validation.py helpers."""

import json
import os
import sys
import unittest
from unittest.mock import patch

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_ROOT = os.path.dirname(_THIS_DIR)
_WORKSPACE_ROOT = os.path.dirname(_RESEARCH_ROOT)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from research.scientist.runner.execution_validation import (
    _ExecutionValidationMixin,
)
from research.scientist.runner._types import RunConfig

pytestmark = pytest.mark.unit


class TestChampionConfirmationPolicy(unittest.TestCase):
    def test_confirmation_uses_source_architecture_config(self):
        class _Stub(_ExecutionValidationMixin):
            pass

        class _Conn:
            def __init__(self):
                self._rows = []

            def execute(self, sql, params=None):
                if "program_results" in sql:
                    self._rows = [
                        {
                            "result_id": "replay-rid",
                            "experiment_id": "replay-exp",
                            "data_provenance_json": json.dumps(
                                {
                                    "n_layers": 6,
                                    "vocab_size": 100277,
                                    "model_dim": 256,
                                    "tokenizer_mode": "tiktoken",
                                    "tiktoken_encoding": "cl100k_base",
                                }
                            ),
                            "config_json": json.dumps(
                                {
                                    "n_layers": 6,
                                    "vocab_size": 100277,
                                    "model_dim": 256,
                                    "tokenizer_mode": "tiktoken",
                                    "tiktoken_encoding": "cl100k_base",
                                }
                            ),
                        }
                    ]
                else:
                    self._rows = [
                        {
                            "config_json": json.dumps(
                                {
                                    "n_layers": 3,
                                    "vocab_size": 100277,
                                    "model_dim": 256,
                                    "tokenizer_mode": "tiktoken",
                                    "tiktoken_encoding": "cl100k_base",
                                    "scale_up_steps": 5000,
                                }
                            )
                        }
                    ]
                return self

            def fetchone(self):
                return self._rows[0] if self._rows else None

            def fetchall(self):
                return self._rows

        class _Notebook:
            conn = _Conn()

        config = RunConfig(mode="confirmation", n_layers=4, scale_up_steps=40000)
        scale_config = config.copy()
        scale_config.stage1_steps = config.scale_up_steps
        source = {
            "result_id": "source-rid",
            "experiment_id": "source-exp",
            "graph_fingerprint": "same-fp",
            "data_provenance": {"n_layers": 6},
        }

        candidate, candidate_scale = _Stub()._scale_up_candidate_configs(
            _Notebook(), source, config, scale_config
        )

        self.assertEqual(candidate.n_layers, 6)
        self.assertEqual(candidate.scale_up_steps, 40000)
        self.assertEqual(candidate_scale.n_layers, 6)
        self.assertEqual(candidate_scale.stage1_steps, 40000)

    def test_confirmation_survivor_is_not_novelty_gated(self):
        class _Stub(_ExecutionValidationMixin):
            def __init__(self):
                self.events = []

            def _resolve_novelty_promotion_validity(self, *_args):
                return False, "duplicate_champion", False

            def _emit_event(self, event_type, payload):
                self.events.append((event_type, payload))

        class _Graph:
            def fingerprint(self):
                return "fp_parent"

        class _Novelty:
            novelty_valid_for_promotion = False
            novelty_validity_reason = "duplicate_champion"
            structural_novelty = 0.1
            behavioral_novelty = 0.2
            novelty_confidence = 0.3
            most_similar_to = "parent"

        class _Notebook:
            conn = None

            def get_program_detail(self, result_id):
                return {"graph_fingerprint": "fp_parent"}

            def record_program_result(self, **kwargs):
                self.recorded = kwargs
                return "child-confirm"

            def store_training_curve(self, *_args):
                raise AssertionError("no curve should be stored in this test")

        config = RunConfig(mode="confirmation")
        results = {
            "novel_count": 0,
            "confirmed_count": 0,
            "survivors": [],
            "best_loss_ratio": None,
            "best_novelty_score": None,
        }
        nb = _Notebook()

        with patch(
            "research.scientist.runner.execution_validation_scale.graph_to_json",
            return_value="{}",
        ):
            _Stub()._scale_up_record_result(
                exp_id="exp-confirm",
                source_result_id="parent-rid",
                prog_idx=0,
                total=1,
                config=config,
                nb=nb,
                results=results,
                graph=_Graph(),
                model=None,
                s1_passed=True,
                loss_ratio=0.53,
                final_loss=6.3,
                throughput=None,
                training_curve=None,
                n_score=0.1,
                nov=_Novelty(),
                program_metrics={},
            )

        self.assertEqual(results["confirmed_count"], 1)
        self.assertEqual(results["novel_count"], 1)
        self.assertTrue(results["survivors"][0]["confirmation"])
        self.assertEqual(
            nb.recorded["intentional_rerun_reason"], "champion_confirmation"
        )
        self.assertEqual(nb.recorded["graph_fingerprint"], "fp_parent")


if __name__ == "__main__":
    unittest.main()
