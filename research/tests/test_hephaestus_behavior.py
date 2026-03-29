import pytest
import unittest
import torch.nn as nn
import time
from research.synthesis.grammar import GrammarConfig, AdaptiveGenerator, EfficiencyPrior
from research.eval.fingerprint import compute_lightning_fingerprint

pytestmark = pytest.mark.unit


class TestHephaestusBehavior(unittest.TestCase):
    def setUp(self):
        self.config = GrammarConfig(model_dim=64, max_depth=10, max_ops=20)

    def test_budget_pruning(self):
        print("\nTesting Budget Pruning...")
        # Use a slightly relaxed budget — motif-based templates produce more
        # parameterized ops than the old random-walk grammar.
        tight_config = GrammarConfig(model_dim=64, max_ops=6, routing_mandatory=False)
        gen = AdaptiveGenerator(tight_config)

        graphs = [gen.generate(seed=i) for i in range(10)]

        for i, g in enumerate(graphs):
            params = g.n_params_estimate()
            # 5.0 ratio means max params = 5.0 * 64 * 64 = 20480
            # Validation applies 3x multiplier, so up to 61440 allowed
            max_allowed = (5.0 * 64 * 64) + (64 * 64)
            print(f"Graph {i} params: {params} (Limit: ~{max_allowed})")
            self.assertLessEqual(params, max_allowed * 3.0)

    def test_efficiency_prior_bias(self):
        print("\nTesting Efficiency Prior Bias...")
        gen_no_prior = AdaptiveGenerator(self.config)
        graphs_no_prior = []
        for i in range(100):
            try:
                graphs_no_prior.append(gen_no_prior.generate(seed=i))
            except ValueError:
                continue
            if len(graphs_no_prior) >= 50:
                break

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
        graphs_prior = []
        for i in range(100):
            try:
                graphs_prior.append(gen_prior.generate(seed=i))
            except ValueError:
                continue
            if len(graphs_prior) >= 50:
                break

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

            def embed(self, x):  # Compatibility for fingerprint.py
                return self.embed_mod(x)

        model = MockModel()

        start = time.time()
        # Ensure we have some reference activations for CKA
        # (This will fallback to heuristic if no artifacts found)
        fp = compute_lightning_fingerprint(
            model, model_dim=64, device="cpu", n_probes=4
        )
        duration = time.time() - start

        print(f"Lightning Fingerprint took: {duration * 1000:.2f}ms")
        print(f"Novelty Score: {fp.novelty_score:.3f}")
        self.assertLess(duration, 1.0)


if __name__ == "__main__":
    unittest.main()
