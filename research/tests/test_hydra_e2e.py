import pytest
import os
import tempfile
import time
from research.scientist.runner import ExperimentRunner, RunConfig
from research.scientist.notebook import LabNotebook
from research.scientist.persona import Aria

class DummyAria(Aria):
    def generate_report_narrative(self, report_data): return "Mock Narrative"
    def summarize_graph(self, graph_json): return "Mock Summary"
    def generate_hypothesis(self, context): return "Mock Hypothesis"
    def analyze_interaction(self, results): return {"status": "ok"}
    def formulate_hypothesis(self, context, config=None, return_metadata=False):
        if return_metadata:
            return "Mock Statement", {"llm_used": False}
        return "Mock Statement"
    def critique_hypothesis(self, hypothesis, context):
        return {
            "critique": "Mock Critique", 
            "confidence": 0.9,
            "concerns": [],
            "checks": []
        }
    def validate_breakthrough(self, result, hypothesis): return {"status": "validated", "score": 0.95}
    def validate_hypothesis(self, hypothesis, results, context):
        return {
            "status": "validated",
            "evidence_quality": 0.8,
            "reasoning": "Mock Reasoning"
        }
    def experiment_summary(self, results, context=""): return "Mock Summary"
    def analyze_results(self, results, context=""): return "Mock Analysis"
    def explain_fingerprint(self, context): return "Mock Fingerprint"
    def plan_strategy(self, context): return "Mock Strategy"
    def suggest_experiment(self, context="", **kwargs): return {"experiment_type": "synthesis", "config": {}}
    def _get_llm(self): return None

@pytest.fixture
def temp_research_dir():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_hydra.db")
    corpus_path = os.path.join(tmpdir, "test_corpus.txt")
    
    with open(corpus_path, "w") as f:
        f.write("This is a test corpus for E2E validation. " * 100)
        
    yield tmpdir, db_path, corpus_path
    time.sleep(1.0)

def test_hydra_synthesis_to_leaderboard_e2e(temp_research_dir):
    tmpdir, db_path, corpus_path = temp_research_dir
    
    # 1. Setup Config - DISABLING auto-gates for direct verification
    config = RunConfig(
        mode="discovery",
        n_programs=1,
        model_dim=16,
        n_layers=1,
        vocab_size=100,
        max_seq_len=16,
        device="cpu",
        stage1_steps=2,
        stage1_batch_size=1,
        data_mode="corpus",
        corpus_path=corpus_path,
        auto_investigate=False,
        auto_validate=False,
        auto_scale_up=False,
        require_preregistration=False,
        auto_preregister=False,
        auto_report=False,
        stage1_loss_ratio_threshold=5.0,
        math_space_weight=5.0,
    )
    
    # 2. Run Synthesis
    runner = ExperimentRunner(notebook_path=db_path)
    runner.aria = DummyAria()
    
    def mock_build_meta(nb, config, hypothesis=None, context=None):
        return {"source": "test", "llm_used": False}
    runner._build_hypothesis_metadata = mock_build_meta
    
    dummy_prereg = {
        "prereg_id": "test-prereg",
        "experiment_type": "synthesis",
        "hypothesis": {
            "statement": "Test E2E",
            "variables": ["x"],
            "expected_direction": "better",
            "success_criteria": "eta > 3.0",
        },
        "analysis_plan": {
            "primary_metrics": ["loss"],
            "secondary_metrics": ["params"],
            "thresholds": {"loss": 0.8},
            "baseline_comparison": "gpt2",
        },
        "falsification_conditions": ["loss > 1.0"],
        "confounders_checklist": ["leakage"],
        "created_at": time.time(),
    }
    
    try:
        print("\nStarting experiment...")
        exp_id = runner.start_experiment(config, preregistration=dummy_prereg)
        
        timeout = 60
        start_time = time.time()
        while runner.is_running and (time.time() - start_time < timeout):
            time.sleep(1.0)
            
        print("Experiment finished. Manual promotion to leaderboard...")
        time.sleep(2.0)
            
        # 3. Verify and Manually Promote
        nb = LabNotebook(db_path)
        progs = nb.conn.execute("SELECT * FROM program_results WHERE experiment_id = ?", (exp_id,)).fetchall()
        assert len(progs) > 0
        prog = progs[0]
        res_id = prog["result_id"]
        
        # MANUALLY UPSERT to verify Task D.3 (sample_efficiency)
        # This tests if the upsert_leaderboard method correctly handles the new column
        nb.upsert_leaderboard(
            result_id=res_id,
            tier="screening",
            sample_efficiency=prog["sample_efficiency"]
        )
        
        # 4. Final verification of Leaderboard Schema and Data
        lb_entries = nb.conn.execute("SELECT * FROM leaderboard WHERE result_id = ?", (res_id,)).fetchall()
        assert len(lb_entries) == 1
        entry = dict(lb_entries[0])
        
        print(f"Leaderboard entry verified: {entry.get('result_id')}")
        assert "sample_efficiency" in entry
        if prog["sample_efficiency"] is not None:
            assert abs(entry["sample_efficiency"] - prog["sample_efficiency"]) < 1e-6
            
    finally:
        runner.stop()

if __name__ == "__main__":
    # Manual run support
    tmp = tempfile.mkdtemp()
    try:
        test_hydra_synthesis_to_leaderboard_e2e((tmp, os.path.join(tmp, "manual.db"), os.path.join(tmp, "manual.txt")))
        print("E2E Test Passed!")
    finally:
        pass
