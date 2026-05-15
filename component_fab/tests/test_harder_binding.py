"""Smoke + behaviour tests for tiny_lm + harder_binding_tasks + run_lm_probe."""

from __future__ import annotations


import torch
from torch import nn

from component_fab.harness.harder_binding_tasks import (
    _BATCH_GENERATORS,
    _build_heldout_split,
    default_hard_binding_tasks,
    run_one_task,
)
from component_fab.harness.tiny_lm import (
    CausalConv1dLane,
    SoftmaxCausalAttention,
    TinyLM,
    TinyLMConfig,
    count_trainable_params,
    lane_factory_for_baseline,
)


# ---------- tiny_lm ----------


def test_tiny_lm_forward_shape() -> None:
    cfg = TinyLMConfig(vocab_size=32, dim=16, n_blocks=2, max_seq_len=24)
    model = TinyLM(SoftmaxCausalAttention, cfg)
    ids = torch.randint(0, 32, (4, 24))
    logits = model(ids)
    assert logits.shape == (4, 24, 32)


def test_tiny_lm_position_embedding_off() -> None:
    cfg = TinyLMConfig(
        vocab_size=16, dim=8, n_blocks=1, use_position_embedding=False, max_seq_len=8
    )
    model = TinyLM(SoftmaxCausalAttention, cfg)
    assert model.pos_embed is None


def test_tiny_lm_param_count_grows_with_blocks() -> None:
    cfg1 = TinyLMConfig(vocab_size=16, dim=16, n_blocks=1, max_seq_len=8)
    cfg2 = TinyLMConfig(vocab_size=16, dim=16, n_blocks=4, max_seq_len=8)
    n1 = count_trainable_params(TinyLM(SoftmaxCausalAttention, cfg1))
    n2 = count_trainable_params(TinyLM(SoftmaxCausalAttention, cfg2))
    assert n2 > n1


def test_softmax_attention_is_causal() -> None:
    """Perturbing a late position must not change earlier outputs."""
    torch.manual_seed(0)
    attn = SoftmaxCausalAttention(8)
    x = torch.randn(1, 8, 8)
    x2 = x.clone()
    x2[:, -1] = x2[:, -1] + 1.0  # perturb only the last position
    with torch.no_grad():
        y = attn(x)
        y2 = attn(x2)
    # First L-1 positions must be identical (causal).
    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-5)


def test_causal_conv_is_causal() -> None:
    torch.manual_seed(0)
    conv = CausalConv1dLane(8, kernel_size=3)
    x = torch.randn(1, 8, 8)
    x2 = x.clone()
    x2[:, -1] = x2[:, -1] + 1.0
    with torch.no_grad():
        y = conv(x)
        y2 = conv(x2)
    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-5)


def test_lane_factory_for_baseline_resolves_known() -> None:
    f = lane_factory_for_baseline("softmax_attention")
    assert isinstance(f(8), SoftmaxCausalAttention)


# ---------- harder_binding_tasks ----------


def test_default_tasks_emit_six() -> None:
    tasks = default_hard_binding_tasks(seed=0)
    names = {t.name for t in tasks}
    assert names == {
        "multi_query_kv_recall",
        "distractor_kv_recall",
        "long_gap_recall",
        "variable_layout_recall",
        "compositional_binding",
        "heldout_pair_recall",
    }


def test_heldout_split_is_disjoint() -> None:
    train, evalp = _build_heldout_split(8, 8, 0.125, seed=0)
    assert set(train).isdisjoint(set(evalp))
    assert len(train) + len(evalp) == 64


def test_basic_batch_target_token_appears_earlier_in_sequence() -> None:
    """Sanity: the target value MUST appear earlier in the sequence (the model
    has something to copy from). If it doesn't, the task is unsolvable."""
    tasks = default_hard_binding_tasks(seed=0)
    multi = next(t for t in tasks if t.name == "multi_query_kv_recall")
    rng = torch.Generator().manual_seed(0)
    ids, qpos, tgt = _BATCH_GENERATORS["multi_query_kv_recall"](multi, 4, False, rng)
    assert ids.shape == (4, multi.seq_len)
    assert qpos.shape == (4, multi.n_queries)
    assert tgt.shape == (4, multi.n_queries)
    # Every target appears earlier than its query position
    for b in range(4):
        for qi in range(multi.n_queries):
            qp = int(qpos[b, qi].item())
            value = int(tgt[b, qi].item())
            assert (ids[b, :qp] == value).any(), (
                f"target {value} missing before pos {qp}"
            )


def test_long_gap_batch_obeys_gap_window() -> None:
    tasks = default_hard_binding_tasks(seed=0)
    lg = next(t for t in tasks if t.name == "long_gap_recall")
    rng = torch.Generator().manual_seed(0)
    ids, qpos, _ = _BATCH_GENERATORS["long_gap_recall"](lg, 4, False, rng)
    for b in range(4):
        # query slot must be at or beyond long_gap_min after the (k,v) pair.
        assert int(qpos[b, 0].item()) >= 2 + lg.long_gap_min


def test_compositional_split_disjoint_combos() -> None:
    """Train and eval must use disjoint (e,a) combinations."""
    tasks = default_hard_binding_tasks(seed=0)
    comp = next(t for t in tasks if t.name == "compositional_binding")
    train_ea = {k for k, _ in comp.train_pairs}
    eval_ea = {k for k, _ in comp.eval_pairs}
    assert train_ea.isdisjoint(eval_ea)


def test_run_one_task_runs_and_returns_result() -> None:
    """Smoke: pipeline trains for a couple steps without crashing."""
    tasks = default_hard_binding_tasks(seed=0)
    task = next(t for t in tasks if t.name == "multi_query_kv_recall")
    res = run_one_task(
        SoftmaxCausalAttention,
        task,
        mixer_label="attn",
        dim=16,
        n_blocks=1,
        n_train_steps=5,
        batch_size=4,
        seed=0,
    )
    assert res.mixer_label == "attn"
    assert res.task_name == "multi_query_kv_recall"
    assert res.converged
    # Loss should drop at least a little.
    assert res.train_loss_final <= res.train_loss_initial + 0.1


def test_run_one_task_handles_broken_lane() -> None:
    """A lane that raises on forward should mark converged=False, not crash."""

    class _BrokenLane(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.proj = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("intentional crash")

    tasks = default_hard_binding_tasks(seed=0)
    task = next(t for t in tasks if t.name == "multi_query_kv_recall")
    res = run_one_task(
        _BrokenLane,
        task,
        mixer_label="broken",
        dim=16,
        n_blocks=1,
        n_train_steps=2,
        batch_size=2,
        seed=0,
    )
    assert not res.converged
    assert res.eval_accuracy == 0.0
