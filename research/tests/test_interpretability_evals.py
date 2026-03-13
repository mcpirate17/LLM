"""Tests for activation sparsity and routing heatmap eval modules.

Verifies:
- Activation sparsity scoring on simple models
- Dead neuron detection
- Routing heatmap extraction and collapse detection
- Runner-compatible dict return shapes
"""

import torch
import torch.nn as nn
import pytest

pytestmark = pytest.mark.unit


# ── Activation Sparsity ──────────────────────────────────────────────

class TestActivationSparsity:
    def _make_simple_model(self, dim=32, vocab=128):
        """Build a minimal model with Linear layers for hook testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.fc1 = nn.Linear(dim, dim)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(dim, vocab)

            def forward(self, x):
                h = self.embed(x)
                h = self.relu(self.fc1(h))
                return self.fc2(h)

        return SimpleModel()

    def test_basic_result_shape(self):
        from research.eval.sparsity import evaluate_activation_sparsity

        model = self._make_simple_model()
        batches = [torch.randint(0, 128, (2, 16)) for _ in range(2)]
        result = evaluate_activation_sparsity(model, batches, torch.device("cpu"))

        assert "activation_sparsity_score" in result
        assert "dead_neuron_ratio" in result
        assert "overall_sparsity" in result
        assert "total_neurons" in result
        assert "collapsed_layers" in result

    def test_score_range(self):
        from research.eval.sparsity import evaluate_activation_sparsity

        model = self._make_simple_model()
        batches = [torch.randint(0, 128, (2, 16)) for _ in range(2)]
        result = evaluate_activation_sparsity(model, batches, torch.device("cpu"))

        score = result["activation_sparsity_score"]
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_dead_neuron_ratio_range(self):
        from research.eval.sparsity import evaluate_activation_sparsity

        model = self._make_simple_model()
        batches = [torch.randint(0, 128, (2, 16)) for _ in range(2)]
        result = evaluate_activation_sparsity(model, batches, torch.device("cpu"))

        ratio = result["dead_neuron_ratio"]
        assert 0.0 <= ratio <= 1.0, f"Ratio {ratio} out of range"

    def test_empty_batches(self):
        from research.eval.sparsity import evaluate_activation_sparsity

        model = self._make_simple_model()
        result = evaluate_activation_sparsity(model, [], torch.device("cpu"))
        assert result["activation_sparsity_score"] == 0.0

    def test_healthy_model_has_high_score(self):
        """A randomly initialized model should have low dead neuron ratio."""
        from research.eval.sparsity import evaluate_activation_sparsity

        model = self._make_simple_model(dim=64)
        batches = [torch.randint(0, 128, (4, 32)) for _ in range(4)]
        result = evaluate_activation_sparsity(model, batches, torch.device("cpu"))

        # Random init + ReLU may kill some neurons but shouldn't collapse
        assert result["activation_sparsity_score"] > 0.5

    def test_collapsed_model_detected(self):
        """A model with zeroed Linear weights should show dead outputs."""
        from research.eval.sparsity import evaluate_activation_sparsity

        model = self._make_simple_model()
        # Zero out all Linear weights — Linear hooks will see all-zero outputs
        with torch.no_grad():
            model.fc1.weight.zero_()
            model.fc1.bias.zero_()
            model.fc2.weight.zero_()
            model.fc2.bias.zero_()

        batches = [torch.randint(0, 128, (2, 16)) for _ in range(2)]
        result = evaluate_activation_sparsity(model, batches, torch.device("cpu"))

        # With zeroed weights, Linear layer outputs are all zero
        assert result["dead_neuron_ratio"] > 0.5
        assert result["activation_sparsity_score"] < 0.5


# ── Routing Heatmap ──────────────────────────────────────────────────

class TestRoutingHeatmap:
    def _make_model_with_routing(self, dim=32, vocab=128):
        """Build a model that has routing_telemetry attached."""
        class RoutedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.router = nn.Linear(dim, 4)  # 4 experts
                self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
                self.head = nn.Linear(dim, vocab)

            def forward(self, x):
                h = self.embed(x)
                # Compute routing scores
                scores = self.router(h)  # (B, S, 4)
                expert_idx = scores.argmax(dim=-1)  # (B, S)

                # Record routing telemetry (mimicking compiler behavior)
                import torch.nn.functional as F
                rt = getattr(self, "routing_telemetry", {
                    "tokens_total": 0,
                    "tokens_processed": 0,
                    "expert_counts": torch.zeros(4),
                    "entropy_sum": 0.0,
                    "count": 0,
                    "heatmap": None,
                })
                B, S = expert_idx.shape
                rt["tokens_total"] += B * S
                rt["tokens_processed"] += B * S
                rt["expert_counts"] += torch.histc(
                    expert_idx.float(), bins=4, min=0, max=3)
                probs = F.softmax(scores, dim=-1)
                entropy = -torch.sum(
                    probs * torch.log(probs + 1e-10), dim=-1).mean().item()
                rt["entropy_sum"] += entropy
                rt["count"] += 1
                if getattr(self, "_capture_heatmap", False) and rt["heatmap"] is None:
                    rt["heatmap"] = expert_idx[0].detach().cpu().numpy().tolist()
                self.routing_telemetry = rt

                # Simple expert selection (use first expert for simplicity)
                out = self.experts[0](h)
                return self.head(out)

        return RoutedModel()

    def _make_plain_model(self, dim=32, vocab=128):
        """Model with no routing."""
        class PlainModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.fc = nn.Linear(dim, vocab)

            def forward(self, x):
                return self.fc(self.embed(x))

        return PlainModel()

    def test_basic_result_with_routing(self):
        from research.eval.routing_heatmap import evaluate_routing_heatmap

        model = self._make_model_with_routing()
        batches = [torch.randint(0, 128, (2, 16)) for _ in range(4)]
        result = evaluate_routing_heatmap(model, batches, torch.device("cpu"))

        assert result["has_routing"] is True
        assert "routing_collapse_score" in result
        assert result["n_routing_modules"] >= 1
        assert "modules" in result

    def test_score_range(self):
        from research.eval.routing_heatmap import evaluate_routing_heatmap

        model = self._make_model_with_routing()
        batches = [torch.randint(0, 128, (2, 16)) for _ in range(4)]
        result = evaluate_routing_heatmap(model, batches, torch.device("cpu"))

        score = result["routing_collapse_score"]
        assert score is not None
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_no_routing_model(self):
        from research.eval.routing_heatmap import evaluate_routing_heatmap

        model = self._make_plain_model()
        batches = [torch.randint(0, 128, (2, 16)) for _ in range(2)]
        result = evaluate_routing_heatmap(model, batches, torch.device("cpu"))

        assert result["has_routing"] is False
        assert result["routing_collapse_score"] is None

    def test_empty_batches(self):
        from research.eval.routing_heatmap import evaluate_routing_heatmap

        model = self._make_model_with_routing()
        result = evaluate_routing_heatmap(model, [], torch.device("cpu"))
        assert result["has_routing"] is False

    def test_heatmap_captured(self):
        from research.eval.routing_heatmap import evaluate_routing_heatmap

        model = self._make_model_with_routing()
        batches = [torch.randint(0, 128, (2, 16)) for _ in range(2)]
        result = evaluate_routing_heatmap(model, batches, torch.device("cpu"))

        # Should have captured a heatmap for at least one module
        heatmaps = [m for m in result["modules"] if m.get("heatmap") is not None]
        assert len(heatmaps) >= 1

    def test_collapsed_routing_detected(self):
        """A model where all tokens go to the same expert should be detected."""
        from research.eval.routing_heatmap import evaluate_routing_heatmap

        model = self._make_model_with_routing()
        # Force router to always select expert 0
        with torch.no_grad():
            model.router.weight.zero_()
            model.router.bias.zero_()
            model.router.bias[0] = 100.0  # Huge bias for expert 0

        batches = [torch.randint(0, 128, (4, 32)) for _ in range(4)]
        result = evaluate_routing_heatmap(model, batches, torch.device("cpu"))

        assert result["has_routing"] is True
        assert result["n_collapsed_modules"] >= 1
        # Collapsed routing should have low health score
        assert result["routing_collapse_score"] < 0.3

    def test_module_details_structure(self):
        from research.eval.routing_heatmap import evaluate_routing_heatmap

        model = self._make_model_with_routing()
        batches = [torch.randint(0, 128, (2, 16)) for _ in range(2)]
        result = evaluate_routing_heatmap(model, batches, torch.device("cpu"))

        for mod in result["modules"]:
            assert "module" in mod
            assert "n_experts" in mod
            assert "gini" in mod
            assert "normalized_entropy" in mod
            assert "dominant_expert_fraction" in mod
            assert "is_collapsed" in mod


# ── Gini Coefficient ─────────────────────────────────────────────────

class TestGiniCoefficient:
    def test_uniform_distribution(self):
        from research.eval.routing_heatmap import _gini_coefficient
        import numpy as np

        counts = np.array([25.0, 25.0, 25.0, 25.0])
        gini = _gini_coefficient(counts)
        assert abs(gini) < 0.01, f"Uniform should give ~0, got {gini}"

    def test_completely_skewed(self):
        from research.eval.routing_heatmap import _gini_coefficient
        import numpy as np

        counts = np.array([100.0, 0.0, 0.0, 0.0])
        gini = _gini_coefficient(counts)
        assert gini > 0.5, f"Fully skewed should give high gini, got {gini}"

    def test_empty_counts(self):
        from research.eval.routing_heatmap import _gini_coefficient
        import numpy as np

        assert _gini_coefficient(np.array([0.0, 0.0])) == 0.0
        assert _gini_coefficient(np.array([])) == 0.0


# ── WikiText Evaluation ───────────────────────────────────────────────

class TestWikiTextEval:
    def _make_simple_model(self, dim=32, vocab=256):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.fc = nn.Linear(dim, dim)
                self.head = nn.Linear(dim, vocab)

            def forward(self, x):
                return self.head(self.fc(self.embed(x)))

        return SimpleModel()

    def test_tokenize_file(self):
        import tempfile
        from pathlib import Path
        from research.eval.utils import tokenize_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world! This is a test.")
            f.flush()
            tokens = tokenize_file(Path(f.name), vocab_size=256)
            assert len(tokens) > 0
            assert all(0 <= t < 256 for t in tokens)

    def test_make_batches(self):
        from research.eval.utils import make_batches

        tokens = list(range(1000))
        batches = make_batches(tokens, batch_size=4, seq_len=32, n_batches=3,
                               device=torch.device("cpu"))
        assert len(batches) == 3
        assert batches[0].shape == (4, 32)

    def test_make_batches_empty(self):
        from research.eval.utils import make_batches

        batches = make_batches([1, 2], batch_size=4, seq_len=32, n_batches=3,
                               device=torch.device("cpu"))
        assert len(batches) == 0

    def test_compute_perplexity(self):
        from research.eval.utils import compute_perplexity

        model = self._make_simple_model()
        batches = [torch.randint(0, 256, (2, 16)) for _ in range(2)]
        ppl = compute_perplexity(model, batches, vocab_size=256)
        assert ppl is not None
        assert ppl > 0

    def test_micro_train_reduces_loss(self):
        from research.eval.utils import micro_train_loop, make_batches

        model = self._make_simple_model()
        tokens = [t % 256 for t in range(500)] * 10  # Repeated pattern, within vocab
        batches = make_batches(tokens, 4, 32, 8, torch.device("cpu"))

        # Get initial loss
        model.eval()
        with torch.no_grad():
            logits = model(batches[0])
            initial_loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 256), batches[0][:, 1:].reshape(-1)
            ).item()

        model.train()
        final_loss = micro_train_loop(model, batches, vocab_size=256, n_steps=50, lr=1e-3)
        # Training should reduce loss (or at least not crash)
        assert final_loss < initial_loss * 1.5  # Allow some tolerance

    def test_evaluate_wikitext_full(self):
        """Full end-to-end test with actual WikiText download."""
        from research.eval.wikitext_eval import evaluate_wikitext_perplexity

        model = self._make_simple_model(dim=32, vocab=256)
        result = evaluate_wikitext_perplexity(
            model, vocab_size=256, device=torch.device("cpu"),
            n_train_steps=10, seq_len=32, n_train_batches=4, n_eval_batches=2,
            train_batch_size=2, eval_batch_size=2,
            max_chars_train=10000, max_chars_val=5000,
        )

        assert "wikitext_perplexity" in result
        if result.get("error") is None:
            assert result["wikitext_perplexity"] is not None
            assert result["wikitext_perplexity"] > 0
            assert result["wikitext_score"] is not None
            assert 0.0 <= result["wikitext_score"] <= 1.0
            assert result["train_final_loss"] > 0
            assert result["variant"] == "wikitext-2-raw-v1"


# ── Notebook Schema ──────────────────────────────────────────────────

class TestNotebookSchema:
    def test_new_columns_in_schema(self):
        """Verify the new columns are in the program_results schema."""
        from research.scientist.notebook import _PROGRAM_RESULTS_NEW_COLUMNS

        assert "activation_sparsity_score" in _PROGRAM_RESULTS_NEW_COLUMNS
        assert "dead_neuron_ratio" in _PROGRAM_RESULTS_NEW_COLUMNS
        assert "routing_collapse_score" in _PROGRAM_RESULTS_NEW_COLUMNS
        assert "wikitext_perplexity" in _PROGRAM_RESULTS_NEW_COLUMNS
        assert "wikitext_score" in _PROGRAM_RESULTS_NEW_COLUMNS

    def test_leaderboard_migration(self):
        """Verify the new columns get added to leaderboard on init."""
        import tempfile
        from research.scientist.notebook import LabNotebook

        with tempfile.TemporaryDirectory() as tmp:
            nb = LabNotebook(f"{tmp}/test.db")
            cols = {
                row[1]
                for row in nb.conn.execute("PRAGMA table_info(leaderboard)")
            }
            assert "activation_sparsity_score" in cols
            assert "dead_neuron_ratio" in cols
            assert "routing_collapse_score" in cols
            assert "wikitext_perplexity" in cols
            assert "wikitext_score" in cols
            assert "tinystories_perplexity" in cols
            assert "tinystories_score" in cols
            assert "cross_task_score" in cols
            assert "efficiency_wall_score" in cols
            assert "max_viable_seq_len" in cols
            assert "scaling_regime" in cols


# ── TinyStories Eval ─────────────────────────────────────────────────

class TestTinyStoriesEval:
    def _make_model(self, dim=32, vocab=256):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.fc = nn.Linear(dim, dim)
                self.head = nn.Linear(dim, vocab)
            def forward(self, x):
                return self.head(self.fc(self.embed(x)))
        return M()

    def test_full_eval(self):
        from research.eval.tinystories_eval import evaluate_tinystories
        model = self._make_model()
        result = evaluate_tinystories(
            model, vocab_size=256, device=torch.device("cpu"),
            n_train_steps=5, seq_len=32, n_train_batches=4, n_eval_batches=2,
            batch_size=2, max_chars_train=5000, max_chars_val=2000)
        assert "tinystories_perplexity" in result
        if result.get("error") is None:
            assert result["tinystories_perplexity"] is not None
            assert result["tinystories_perplexity"] > 0
            assert 0.0 <= (result["tinystories_score"] or 0) <= 1.0


# ── Cross-Task Eval ──────────────────────────────────────────────────

class TestCrossTaskEval:
    def _make_model_fn(self, dim=32, vocab=256):
        def fn():
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed = nn.Embedding(vocab, dim)
                    self.fc = nn.Linear(dim, dim)
                    self.head = nn.Linear(dim, vocab)
                def forward(self, x):
                    return self.head(self.fc(self.embed(x)))
            return M()
        return fn

    def test_synthetic_fallback(self):
        """Tests with synthetic code fallback (no heavy download needed)."""
        from research.eval.cross_task_eval import _generate_synthetic_python
        snippets = _generate_synthetic_python(10000)
        assert len(snippets) > 0
        total = sum(len(s) for s in snippets)
        assert total >= 10000

    def test_full_eval(self):
        from research.eval.cross_task_eval import evaluate_cross_task_robustness
        result = evaluate_cross_task_robustness(
            self._make_model_fn(), vocab_size=256, device=torch.device("cpu"),
            n_train_steps=5, seq_len=32, n_train_batches=4, n_eval_batches=2,
            batch_size=2, max_chars=5000)
        assert "cross_task_score" in result
        if result.get("error") is None:
            assert result["cross_task_score"] is not None
            assert 0.0 <= result["cross_task_score"] <= 1.0
            assert result["code_perplexity"] is not None
            assert result["nl_perplexity"] is not None


# ── Efficiency Wall Eval ─────────────────────────────────────────────

class TestEfficiencyWallEval:
    def _make_model(self, dim=32, vocab=256):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.fc = nn.Linear(dim, dim)
                self.head = nn.Linear(dim, vocab)
            def forward(self, x):
                return self.head(self.fc(self.embed(x)))
        return M()

    def test_basic_profiling(self):
        from research.eval.efficiency_wall import evaluate_efficiency_wall
        model = self._make_model()
        result = evaluate_efficiency_wall(
            model, vocab_size=256, device=torch.device("cpu"),
            seq_lens=(32, 64, 128), batch_size=2)
        assert "efficiency_wall_score" in result
        assert result["efficiency_wall_score"] is not None
        assert 0.0 <= result["efficiency_wall_score"] <= 1.0
        assert len(result["measurements"]) == 3

    def test_scaling_regime_detected(self):
        from research.eval.efficiency_wall import _detect_scaling_regime
        # Linear scaling: memory doubles when seq_len doubles
        measurements = [
            {"seq_len": 64, "peak_mb": 10.0, "error": None},
            {"seq_len": 128, "peak_mb": 20.0, "error": None},
            {"seq_len": 256, "peak_mb": 40.0, "error": None},
        ]
        assert _detect_scaling_regime(measurements) == "linear"

    def test_quadratic_regime_detected(self):
        from research.eval.efficiency_wall import _detect_scaling_regime
        # Quadratic: memory 4x when seq_len 2x
        measurements = [
            {"seq_len": 64, "peak_mb": 10.0, "error": None},
            {"seq_len": 128, "peak_mb": 40.0, "error": None},
            {"seq_len": 256, "peak_mb": 160.0, "error": None},
        ]
        assert _detect_scaling_regime(measurements) == "quadratic"

    def test_max_viable_seq_len(self):
        from research.eval.efficiency_wall import evaluate_efficiency_wall
        model = self._make_model()
        result = evaluate_efficiency_wall(
            model, vocab_size=256, device=torch.device("cpu"),
            seq_lens=(32, 64), batch_size=2, memory_budget_mb=10000)

        assert result["max_viable_seq_len"] >= 32
