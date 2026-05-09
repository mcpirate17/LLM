"""Tests for associative recall probe internals.

Covers the pure-function components that don't require a model:
- _generate_ar_batch: sequence construction, token uniqueness, shape
- _trapezoidal_auc: AUC computation from learning curves
- _get_special_tokens: SEP/ANS token selection
- ARResult dataclass
"""

import pytest
import torch

from research.eval.associative_recall import (
    ARResult,
    _VOCAB_HI,
    _VOCAB_LO,
    _generate_ar_batch,
    _get_special_tokens,
    _trapezoidal_auc,
)


@pytest.mark.unit
class TestGenerateARBatch:
    """Test the vectorized batch generation for associative recall."""

    def test_output_shape(self):
        batch_size, n_pairs = 8, 20
        sep_token, ans_token = 50256, 50257
        ids, targets = _generate_ar_batch(
            batch_size, n_pairs, sep_token, ans_token, "cpu"
        )
        expected_seq_len = 3 * n_pairs + 4  # pairs + SEP + kQa + kQb + ANS
        assert ids.shape == (batch_size, expected_seq_len)
        assert targets.shape == (batch_size,)

    def test_tokens_in_vocab_range(self):
        ids, targets = _generate_ar_batch(16, 20, 50256, 50257, "cpu")
        # Regular tokens should be in [_VOCAB_LO, _VOCAB_HI)
        pair_region = ids[:, :60]  # first 60 positions = 20 pairs * 3
        assert (pair_region >= _VOCAB_LO).all()
        assert (pair_region < _VOCAB_HI).all()
        # Targets should also be in vocab range
        assert (targets >= _VOCAB_LO).all()
        assert (targets < _VOCAB_HI).all()

    def test_sep_and_ans_tokens_placed_correctly(self):
        n_pairs = 20
        sep_token, ans_token = 50256, 50257
        ids, _ = _generate_ar_batch(4, n_pairs, sep_token, ans_token, "cpu")
        sep_pos = 3 * n_pairs
        assert (ids[:, sep_pos] == sep_token).all()
        assert (ids[:, sep_pos + 3] == ans_token).all()

    def test_query_key_matches_some_pair(self):
        """The query key (positions sep+1, sep+2) should match one of the pairs."""
        n_pairs = 20
        sep_token, ans_token = 50256, 50257
        ids, targets = _generate_ar_batch(1, n_pairs, sep_token, ans_token, "cpu")
        sep_pos = 3 * n_pairs
        query_k0 = ids[0, sep_pos + 1].item()
        query_k1 = ids[0, sep_pos + 2].item()

        # Find the matching pair in the sequence
        found = False
        for i in range(n_pairs):
            k0 = ids[0, i * 3].item()
            k1 = ids[0, i * 3 + 1].item()
            v = ids[0, i * 3 + 2].item()
            if k0 == query_k0 and k1 == query_k1:
                assert targets[0].item() == v, (
                    "Target should be the value of the queried key"
                )
                found = True
                break
        assert found, "Query key not found among pairs"

    def test_no_key_value_collision(self):
        """Keys and values should use distinct tokens within each sample."""
        ids, _ = _generate_ar_batch(1, 20, 50256, 50257, "cpu")
        pair_tokens = ids[0, :60]  # 20 pairs * 3 tokens each
        keys = set()
        values = set()
        for i in range(20):
            keys.add(pair_tokens[i * 3].item())
            keys.add(pair_tokens[i * 3 + 1].item())
            values.add(pair_tokens[i * 3 + 2].item())
        # With 20 pairs we need 60 unique tokens (40 key + 20 value)
        # They're drawn from a single randperm so all 60 should be unique
        all_tokens = [pair_tokens[j].item() for j in range(60)]
        assert len(set(all_tokens)) == 60, "All tokens in pairs should be unique"

    def test_different_batches_are_shuffled(self):
        """Two calls should produce different sequences (random permutation)."""
        ids1, _ = _generate_ar_batch(4, 20, 50256, 50257, "cpu")
        ids2, _ = _generate_ar_batch(4, 20, 50256, 50257, "cpu")
        # Extremely unlikely to be identical
        assert not torch.equal(ids1, ids2)


@pytest.mark.unit
class TestTrapezoidalAUC:
    """Test the AUC computation from learning curves."""

    def test_constant_curve(self):
        """Constant accuracy → AUC equals that accuracy."""
        curve = [(0, 0.5), (100, 0.5), (500, 0.5)]
        auc = _trapezoidal_auc(curve, max_steps=500)
        assert auc == pytest.approx(0.5)

    def test_linear_ramp(self):
        """Linear ramp from 0 to 1 over max_steps → AUC = 0.5."""
        curve = [(0, 0.0), (500, 1.0)]
        auc = _trapezoidal_auc(curve, max_steps=500)
        assert auc == pytest.approx(0.5)

    def test_perfect_from_start(self):
        """Perfect accuracy from step 0 → AUC = 1.0."""
        curve = [(0, 1.0), (500, 1.0)]
        auc = _trapezoidal_auc(curve, max_steps=500)
        assert auc == pytest.approx(1.0)

    def test_zero_everywhere(self):
        curve = [(0, 0.0), (500, 0.0)]
        auc = _trapezoidal_auc(curve, max_steps=500)
        assert auc == pytest.approx(0.0)

    def test_single_point(self):
        curve = [(0, 0.7)]
        auc = _trapezoidal_auc(curve, max_steps=500)
        assert auc == 0.7

    def test_empty_curve(self):
        auc = _trapezoidal_auc([], max_steps=500)
        assert auc == 0.0

    def test_late_improvement(self):
        """Model that only improves at the very end gets low AUC."""
        curve = [(0, 0.0), (400, 0.0), (500, 1.0)]
        auc = _trapezoidal_auc(curve, max_steps=500)
        # Only last 100 steps contribute, triangle: 0.5 * 100 * 1.0 / 500 = 0.1
        assert auc == pytest.approx(0.1)


@pytest.mark.unit
class TestGetSpecialTokens:
    """Test SEP/ANS token selection."""

    def test_large_vocab_model(self):
        """Models with vocab > 50257 should use 50256, 50257."""

        class FakeModel(torch.nn.Module):
            vocab_size = 50304

        sep, ans = _get_special_tokens(FakeModel())
        assert sep == 50256
        assert ans == 50257

    def test_small_vocab_model(self):
        """Smaller vocab uses last two IDs."""

        class FakeModel(torch.nn.Module):
            vocab_size = 1000

        sep, ans = _get_special_tokens(FakeModel())
        assert sep == 998
        assert ans == 999

    def test_embedding_fallback(self):
        """If no vocab_size attr, infer from first Embedding layer."""

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(512, 64)

        sep, ans = _get_special_tokens(FakeModel())
        assert sep == 510
        assert ans == 511


@pytest.mark.unit
class TestARResult:
    """Test the ARResult dataclass."""

    def test_default_values(self):
        r = ARResult()
        assert r.auc == 0.0
        assert r.final_acc == 0.0
        assert r.timed_out is False
        assert r.above_chance is False
        assert r.status == "ok"

    def test_to_dict_keys(self):
        r = ARResult(
            auc=0.5,
            final_acc=0.8,
            timed_out=False,
            above_chance=True,
            learning_curve=[(0, 0.0), (500, 0.8)],
        )
        d = r.to_dict()
        assert d["ar_legacy_auc"] == 0.5
        assert d["ar_legacy_final_acc"] == 0.8
        assert d["ar_legacy_timed_out"] is False
        assert d["ar_legacy_above_chance"] is True
        assert len(d["ar_learning_curve"]) == 2
