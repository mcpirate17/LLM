"""Tests for the graded MQAR scaling probe (research/eval/gmqar.py).

Covers: batch-shape/causality invariants, determinism, an oracle model that
SHOULD score ~1.0 (sanity that the task is solvable and scored correctly), a
random model that should sit at chance, and the AUDC/D50 summary math.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from research.eval.gmqar import (
    GMQARConfig,
    N_SPECIAL,
    QRY_ID,
    SEP_ID,
    default_grid,
    make_gmqar_batch,
    score_cell,
    score_model_gmqar,
)


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def test_batch_shapes_and_answer_positions():
    cfg = GMQARConfig(vocab_size=512, n_pairs=8, n_queries=4, batch_size=5)
    ids, tgt, mask = make_gmqar_batch(cfg, _gen(), "cpu")
    exp_len = 2 * cfg.n_pairs + 1 + cfg.distractor_tokens + 3 * cfg.n_queries
    assert ids.shape == (5, exp_len)
    assert tgt.shape == mask.shape == ids.shape
    # exactly n_queries answer positions per row
    assert mask.sum().item() == 5 * cfg.n_queries
    # targets are -100 everywhere except answer positions
    assert (tgt[~mask] == -100).all()
    assert (tgt[mask] != -100).all()


def test_keys_and_values_disjoint_and_specials_separate():
    cfg = GMQARConfig(vocab_size=512, n_pairs=8, n_queries=4, batch_size=4)
    ids, tgt, mask = make_gmqar_batch(cfg, _gen(), "cpu")
    half = (cfg.vocab_size - N_SPECIAL) // 2
    val_lo = N_SPECIAL + half
    # every target (a value) is in the value pool, above the key pool & specials
    vals = tgt[mask]
    assert (vals >= val_lo).all()
    assert (vals < cfg.vocab_size).all()
    # SEP appears exactly once per row
    assert (ids == SEP_ID).sum().item() == 4
    # QRY appears exactly n_queries times per row
    assert (ids == QRY_ID).sum().item() == 4 * cfg.n_queries


def test_determinism_same_seed():
    cfg = GMQARConfig(vocab_size=512, n_pairs=8, batch_size=4, seed=7)
    a = make_gmqar_batch(cfg, _gen(7), "cpu")
    b = make_gmqar_batch(cfg, _gen(7), "cpu")
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
    # different seed -> different sequences
    c = make_gmqar_batch(cfg, _gen(8), "cpu")
    assert not torch.equal(a[0], c[0])


class _OracleModel(nn.Module):
    """Cheats by reading the context: at each position, if the previous token is
    a key that appeared in a (key,value) pair earlier in THIS row, emit that
    value. This is the ground-truth recall the task asks for, so it must score
    ~1.0 — a check that the task is solvable and the scorer reads the right slot.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        logits = torch.zeros(B, S, self.vocab_size)
        for b in range(B):
            # build key->value map from the KV block (positions before SEP)
            kv: dict[int, int] = {}
            row = input_ids[b].tolist()
            sep = row.index(SEP_ID) if SEP_ID in row else S
            i = 0
            while i + 1 < sep:
                kv[row[i]] = row[i + 1]
                i += 2
            # at each position t, predict token t+1: if token t is a queried key
            # (preceded by QRY), emit its bound value
            for t in range(S):
                if t >= 1 and row[t - 1] == QRY_ID and row[t] in kv:
                    logits[b, t, kv[row[t]]] = 50.0
        return logits


def test_oracle_scores_near_one():
    cfg = GMQARConfig(vocab_size=256, n_pairs=8, n_queries=4, batch_size=16)
    model = _OracleModel(cfg.vocab_size)
    acc = score_cell(model, cfg, "cpu")
    assert acc > 0.99, f"oracle should solve gMQAR, got {acc}"


def test_random_model_at_chance():
    cfg = GMQARConfig(vocab_size=2048, n_pairs=8, n_queries=4, batch_size=32)

    class _Rand(nn.Module):
        def __init__(self, v):
            super().__init__()
            self.lin = nn.Embedding(v, v)

        def forward(self, x):
            return self.lin(x)

    torch.manual_seed(0)
    acc = score_cell(_Rand(cfg.vocab_size), cfg, "cpu")
    # chance ~ 1/(value-pool) ; allow generous slack but must be far below 0.5
    assert acc < 0.1, f"random model should be near chance, got {acc}"


def test_audc_and_d50_summary():
    # small grid for speed
    grid = [
        GMQARConfig(vocab_size=256, n_pairs=2, n_queries=2, batch_size=8),
        GMQARConfig(vocab_size=256, n_pairs=4, n_queries=4, batch_size=8),
    ]
    res = score_model_gmqar(_OracleModel(256), grid=grid, vocab_size=256)
    assert 0.0 <= res.audc <= 1.0
    assert res.audc > 0.99  # oracle solves all cells
    assert res.d50 == 4  # passes >=0.5 at both 2 and 4 pairs -> max is 4
    assert res.chance > 0
    assert len(res.cells) == 2


def test_default_grid_is_valid():
    grid = default_grid(vocab_size=8192)
    assert len(grid) == 10  # 5 pair-counts x 2 distractor settings
    for cfg in grid:
        assert cfg.n_queries <= cfg.n_pairs
        # config __post_init__ would have raised if vocab too small
