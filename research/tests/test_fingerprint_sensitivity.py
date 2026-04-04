import torch


from research.eval.fingerprint import (
    _interaction_metrics,
    _collect_position_sensitivities,
    _analyze_sensitivity,
    _forward_model_from_embed,
    _sensitivity_metrics,
)


def test_sensitivity_metrics_native_returns_valid():
    sens_matrix = torch.tensor(
        [
            [0.5, 0.2, 0.1, 0.0],
            [0.1, 0.7, 0.2, 0.1],
            [0.0, 0.1, 0.6, 0.3],
        ],
        dtype=torch.float32,
    )

    result = _sensitivity_metrics(sens_matrix)

    for key in ("spectral_norm", "uniformity", "effective_rank"):
        assert key in result
        assert isinstance(result[key], float)
    assert result["spectral_norm"] > 0.0
    assert 0.0 <= result["uniformity"] <= 1.0
    assert result["effective_rank"] >= 1.0


def test_interaction_metrics_native_returns_valid():
    influence_matrix = torch.tensor(
        [
            [0.9, 0.4, 0.2, 0.1],
            [0.3, 0.8, 0.4, 0.2],
            [0.1, 0.3, 0.7, 0.5],
            [0.0, 0.1, 0.4, 0.6],
        ],
        dtype=torch.float32,
    )
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    result = _interaction_metrics(influence_matrix, positions)

    for key in ("locality", "sparsity", "symmetry", "hierarchy"):
        assert key in result
        assert isinstance(result[key], float)
        assert 0.0 <= result[key] <= 1.0


def test_sensitivity_metrics_helper_uses_single_path():
    sens_matrix = torch.rand(3, 5, dtype=torch.float32)
    result = _sensitivity_metrics(sens_matrix)
    for key in ("spectral_norm", "uniformity", "effective_rank"):
        assert key in result
        assert isinstance(result[key], float)


def test_sensitivity_collection_returns_correct_shape():
    base = torch.randn(1, 5, 3, dtype=torch.float32)
    embed = base.clone().requires_grad_(True)
    x = embed * 0.5 + embed.roll(shifts=1, dims=1) * 0.25
    positions = torch.tensor([0, 2, 4], dtype=torch.int64)

    result = _collect_position_sensitivities(x, embed, positions)

    assert result is not None
    assert result.shape[0] == 3  # n_positions
    assert result.shape[1] == 5  # seq_len


def test_forward_model_from_embed_prefers_model_fast_path():
    class _FastPathModel:
        def __init__(self):
            self.called = False
            self.pos_enc = lambda x: (_ for _ in ()).throw(
                RuntimeError("fallback used")
            )

        def _fingerprint_forward_from_embed(self, embed_in):
            self.called = True
            return embed_in + 7.0

    model = _FastPathModel()
    embed = torch.randn(1, 4, 3)

    out = _forward_model_from_embed(model, embed)

    assert model.called is True
    assert torch.allclose(out, embed + 7.0)


def test_sensitivity_collection_callable_matches_realized_tensor_path():
    base = torch.randn(1, 5, 3, dtype=torch.float32)
    positions = torch.tensor([0, 2, 4], dtype=torch.int64)

    embed_realized = base.clone().requires_grad_(True)
    x = embed_realized * 0.5 + embed_realized.roll(shifts=1, dims=1) * 0.25
    realized = _collect_position_sensitivities(x, embed_realized, positions)

    embed_callable = base.clone().requires_grad_(True)

    def forward_from_embed(embed_in):
        return embed_in * 0.5 + embed_in.roll(shifts=1, dims=1) * 0.25

    callable_res = _collect_position_sensitivities(
        forward_from_embed,
        embed_callable,
        positions,
    )

    assert callable_res is not None
    assert realized is not None
    assert torch.allclose(callable_res, realized, atol=1e-6, rtol=1e-6)


def test_analyze_sensitivity_uses_callable_collection_path(monkeypatch):
    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(32, 8)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8, bias=False)])

    model = _TinyModel().eval()
    called = {"callable": False}

    def fake_collect(x_or_forward, embed, positions):
        called["callable"] = callable(x_or_forward)
        return torch.ones(positions.numel(), embed.size(1), dtype=torch.float32)

    monkeypatch.setattr(
        "research.eval.fingerprint_sensitivity.collect_position_sensitivities",
        fake_collect,
    )

    result = _analyze_sensitivity(model, torch.device("cpu"), seq_len=8, vocab_size=32)

    assert called["callable"] is True
    assert result["_succeeded"] is True
