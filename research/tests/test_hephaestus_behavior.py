
import unittest
import torch
import torch.nn as nn
import numpy as np
import time
from research.synthesis.grammar import GrammarConfig, AdaptiveGenerator, EfficiencyPrior, batch_generate
from research.synthesis.primitives import OPCODE_MAP
from research.eval.fingerprint import compute_lightning_fingerprint

class TestHephaestusBehavior(unittest.TestCase):
    def setUp(self):
        self.config = GrammarConfig(
            model_dim=64,
            max_depth=10,
            max_ops=20,
            max_params_ratio=2.0 
        )

    def test_budget_pruning(self):
        print("\nTesting Budget Pruning...")
        tight_config = GrammarConfig(model_dim=64, max_params_ratio=0.01, max_ops=3)
        gen = AdaptiveGenerator(tight_config)
        
        graphs = [gen.generate(seed=i) for i in range(10)]
        
        for i, g in enumerate(graphs):
            params = g.n_params_estimate()
            # 0.01 ratio means max params = 0.01 * 64 * 64 = 40
            # Plus one final linear projection (64*64=4096)
            max_allowed = (0.01 * 64 * 64) + (64 * 64) 
            print(f"Graph {i} params: {params} (Limit: ~{max_allowed})")
            self.assertLessEqual(params, max_allowed * 2.0) 

    def test_efficiency_prior_bias(self):
        print("\nTesting Efficiency Prior Bias...")
        gen_no_prior = AdaptiveGenerator(self.config)
        graphs_no_prior = [gen_no_prior.generate(seed=i) for i in range(50)]
        
        def count_ops(graphs, motif):
            count = 0
            for g in graphs:
                for node in g.nodes.values():
                    if motif in node.op_name:
                        count += 1
            return count

        scan_count_base = count_ops(graphs_no_prior, "selective_scan")
        
        mock_frontier = [
            {"graph_json": '{"nodes": {"1": {"op_name": "selective_scan"}}}'} 
            for _ in range(10)
        ]
        prior = EfficiencyPrior(mock_frontier)
        prior.op_biases["selective_scan"] = 10.0 
        
        gen_prior = AdaptiveGenerator(self.config, prior=prior)
        graphs_prior = [gen_prior.generate(seed=i) for i in range(50)]
        
        scan_count_biased = count_ops(graphs_prior, "selective_scan")
        
        print(f"Selective Scan Count (Base):   {scan_count_base}")
        print(f"Selective Scan Count (Biased): {scan_count_biased}")
        
        self.assertGreaterEqual(scan_count_biased, scan_count_base)

    def test_lightning_fingerprint_speed(self):
        print("\nTesting Lightning Fingerprint Performance...")
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_mod = nn.Embedding(32000, 64)
                self.proj = nn.Linear(64, 64)
            def forward(self, x):
                return self.proj(self.embed_mod(x))
            def embed(self, x): # Compatibility for fingerprint.py
                return self.embed_mod(x)
        
        model = MockModel()
        
        start = time.time()
        # Ensure we have some reference activations for CKA
        # (This will fallback to heuristic if no artifacts found)
        fp = compute_lightning_fingerprint(model, model_dim=64, device="cpu", n_probes=4)
        duration = time.time() - start
        
        print(f"Lightning Fingerprint took: {duration*1000:.2f}ms")
        print(f"Novelty Score: {fp.novelty_score:.3f}")
        self.assertLess(duration, 1.0) 

if __name__ == "__main__":
    unittest.main()
