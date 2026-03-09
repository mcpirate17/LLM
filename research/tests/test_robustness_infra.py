
import pytest
import os
import sys
import unittest
import tempfile
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from research.scientist.notebook import LabNotebook

pytestmark = pytest.mark.unit

class TestRobustnessInfrastructure(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_lab_notebook.db"
        self.nb = LabNotebook(str(self.db_path))

    def tearDown(self):
        self.nb.close()
        self.tmp_dir.cleanup()

    def test_upsert_leaderboard_preserves_metrics(self):
        """Test that upsert_leaderboard doesn't overwrite existing data with NULL."""
        result_id = "test_res_123"
        
        # 1. Initial screening upsert
        self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            screening_loss_ratio=0.5,
            screening_novelty=0.8,
            tier="screening"
        )
        
        entry = self.nb.get_leaderboard_entry(result_id)
        self.assertEqual(entry["screening_loss_ratio"], 0.5)
        self.assertEqual(entry["screening_novelty"], 0.8)
        self.assertEqual(entry["tier"], "screening")
        
        # 2. Update with investigation data
        self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            investigation_loss_ratio=0.3,
            investigation_robustness=0.7,
            tier="investigation"
        )
        
        entry = self.nb.get_leaderboard_entry(result_id)
        self.assertEqual(entry["screening_loss_ratio"], 0.5) # Should be preserved
        self.assertEqual(entry["investigation_loss_ratio"], 0.3)
        self.assertEqual(entry["tier"], "investigation")
        
        # 3. Call upsert with only tier change (e.g. from some other automated path)
        # In the old code, this would set screening_loss_ratio and investigation_loss_ratio to NULL
        self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            tier="validation"
        )
        
        entry = self.nb.get_leaderboard_entry(result_id)
        self.assertEqual(entry["tier"], "validation")
        self.assertEqual(entry["screening_loss_ratio"], 0.5)
        self.assertEqual(entry["investigation_loss_ratio"], 0.3)
        self.assertEqual(entry["investigation_robustness"], 0.7)

    def test_promote_to_tier_preserves_metrics(self):
        """Test that promote_to_tier doesn't overwrite existing data with NULL."""
        result_id = "test_res_456"
        self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            screening_loss_ratio=0.5,
            tier="screening"
        )
        
        entry = self.nb.get_leaderboard_entry(result_id)
        entry_id = entry["entry_id"]
        
        # Set robustness metrics
        self.nb.promote_to_tier(
            entry_id=entry_id,
            tier="validation",
            quant_int8_retention=0.85,
            robustness_long_ctx_score=0.9
        )
        
        entry = self.nb.get_leaderboard_entry(result_id)
        self.assertEqual(entry["quant_int8_retention"], 0.85)
        
        # Re-promote without those metrics (e.g. updating notes or something)
        self.nb.promote_to_tier(
            entry_id=entry_id,
            tier="validation",
            notes="Updated notes"
        )
        
        entry = self.nb.get_leaderboard_entry(result_id)
        self.assertEqual(entry["quant_int8_retention"], 0.85)
        self.assertEqual(entry["notes"], "Updated notes")

    def test_robustness_columns_exist(self):
        """Test that all new robustness and scaling columns exist in the schema."""
        result_id = "test_res_789"
        self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            quant_int8_retention=0.88,
            robustness_noise_score=0.12,
            robustness_long_ctx_score=0.95,
            init_sensitivity_std=0.03,
            fp_jacobian_spectral_norm=1.2,
            scaling_param_efficiency=3.5,
            scaling_gate_passed=1
        )
        
        entry = self.nb.get_leaderboard_entry(result_id)
        self.assertEqual(entry["quant_int8_retention"], 0.88)
        self.assertEqual(entry["robustness_noise_score"], 0.12)
        self.assertEqual(entry["robustness_long_ctx_score"], 0.95)
        self.assertEqual(entry["init_sensitivity_std"], 0.03)
        self.assertEqual(entry["fp_jacobian_spectral_norm"], 1.2)
        self.assertEqual(entry["scaling_param_efficiency"], 3.5)
        self.assertEqual(entry["scaling_gate_passed"], 1)

if __name__ == "__main__":
    unittest.main()
