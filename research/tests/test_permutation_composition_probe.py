from __future__ import annotations

import torch

from research.eval import permutation_composition_probe as pcp


class _PermutationOracle(torch.nn.Module):
    vocab_size = 32

    def __init__(self, layout: pcp.PermutationLayout) -> None:
        super().__init__()
        self.layout = layout
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        pairs = input_ids[:, 1:-1].reshape(batch, -1, 2)
        targets = pcp._apply_transpositions(
            input_ids[:, 0],
            pairs[:, :, 0],
            pairs[:, :, 1],
        )
        logits = torch.full(
            (batch, seq_len, self.vocab_size),
            -10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, -1, :] = logits[:, -1, :] + self.dummy
        logits[torch.arange(batch, device=input_ids.device), -1, targets] = 10.0
        return logits


class _TinyVocabModel(torch.nn.Module):
    vocab_size = 6

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        return torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)


class _TrainableProbeModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 24) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.emb(input_ids).sum(dim=1)
        pred = self.out(torch.tanh(hidden))
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            device=input_ids.device,
        )
        logits[:, -1, :] = pred
        return logits


class _LazyBufferBlock(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if "centroids" not in self._buffers:
            self.register_buffer("centroids", torch.ones(3, device=input_ids.device))
        return input_ids


class _LazyBufferProbeModel(_TrainableProbeModel):
    def __init__(self) -> None:
        super().__init__()
        self.ops = torch.nn.ModuleList([_LazyBufferBlock()])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(self.ops[0](input_ids))


def test_apply_transpositions_composes_in_order():
    start = torch.tensor([4, 5, 6])
    left = torch.tensor(
        [
            [4, 5],
            [4, 6],
            [6, 5],
        ]
    )
    right = torch.tensor(
        [
            [5, 6],
            [5, 7],
            [7, 4],
        ]
    )

    result = pcp._apply_transpositions(start, left, right)

    assert result.tolist() == [6, 4, 7]


def test_make_batch_targets_match_oracle_application():
    layout = pcp._make_layout(6)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(7)

    input_ids, targets = pcp._make_batch(layout, 64, 3, "cpu", rng)
    pairs = input_ids[:, 1:-1].reshape(64, -1, 2)
    expected = pcp._apply_transpositions(
        input_ids[:, 0], pairs[:, :, 0], pairs[:, :, 1]
    )

    assert torch.equal(targets, expected)
    assert torch.all(input_ids[:, -1] == pcp._QUERY)


def test_public_probe_reports_vocab_too_small():
    result = pcp.permutation_composition_score(
        _TinyVocabModel(),
        n_items=8,
        n_train_steps=0,
        device="cpu",
    )

    payload = result.to_dict()
    assert payload["permutation_composition_status"] == "model_vocab_too_small"
    assert (
        payload["permutation_composition_metric_version"]
        == "permutation_composition_v1"
    )


def test_oracle_scores_perfect_on_train_and_longer_chains():
    layout = pcp._make_layout(8)
    result = pcp.permutation_composition_score(
        _PermutationOracle(layout),
        n_items=layout.n_items,
        train_chain_len=2,
        eval_chain_len=5,
        n_train_steps=0,
        n_eval_batches=2,
        batch_size=32,
        device="cpu",
        seed=11,
    )

    assert result.status == "ok"
    assert result.train_chain_accuracy == 1.0
    assert result.extrapolation_accuracy == 1.0
    assert result.score == 1.0


def test_probe_restores_model_state_and_training_mode():
    model = _TrainableProbeModel()
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    result = pcp.permutation_composition_score(
        model,
        n_items=6,
        train_chain_len=1,
        eval_chain_len=2,
        n_train_steps=1,
        n_eval_batches=1,
        batch_size=8,
        device="cpu",
        seed=5,
    )

    assert result.status == "ok"
    assert not model.training
    after = model.state_dict()
    assert before.keys() == after.keys()
    for key, expected in before.items():
        assert torch.allclose(after[key], expected), key


def test_probe_restores_when_forward_registers_lazy_buffer():
    model = _LazyBufferProbeModel()
    before_keys = set(model.state_dict())

    result = pcp.permutation_composition_score(
        model,
        n_items=6,
        train_chain_len=1,
        eval_chain_len=2,
        n_train_steps=1,
        n_eval_batches=1,
        batch_size=8,
        device="cpu",
        seed=5,
    )

    assert result.status == "ok"
    assert set(model.state_dict()) == before_keys
    assert "ops.0.centroids" not in model.state_dict()


def test_external_eval_result_accepts_permutation_fields():
    from research.scientist.runner._types import ExternalEvalResult

    result = ExternalEvalResult()
    result.permutation_composition_score = 0.5
    result.permutation_composition_metric_version = "permutation_composition_v1"

    assert result.permutation_composition_score == 0.5
    assert result.permutation_composition_metric_version == "permutation_composition_v1"
