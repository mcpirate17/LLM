from __future__ import annotations

import math

import torch

from research.eval import trajectory_metrics
from research.eval.icld_velocity import ICLDResult
from research.eval.jacobian_erf import JacobianERFResult
from research.eval.trajectory_metrics import compute_trajectory_metrics
from research.eval.transitive_logit_margin import LogitMarginResult
from research.synthesis.compiler_registry import load_split_op_modules
from research.synthesis.compiled_model import SynthesizedModel
from research.synthesis.graph import ComputationGraph


load_split_op_modules()


def _nm_sparse_graph(model_dim: int = 16) -> ComputationGraph:
    graph = ComputationGraph(model_dim)
    inp = graph.add_input()
    sparse = graph.add_op(
        "nm_sparse_linear",
        [inp],
        {"out_dim": model_dim, "n": 2, "m": 4},
    )
    graph.set_output(sparse)
    return graph


def test_trajectory_metrics_survive_inference_mode_sparse_cache() -> None:
    model = SynthesizedModel(
        [_nm_sparse_graph()],
        vocab_size=64,
        model_dim=16,
        max_seq_len=16,
    )
    model.eval()

    # Screening/eval paths commonly run under inference_mode and populate
    # per-op caches. The subsequent trajectory probes need autograd, so those
    # caches must not poison the backward graph.
    ids = torch.randint(0, 64, (2, 8))
    with torch.inference_mode():
        model(ids)

    result = compute_trajectory_metrics(
        model,
        metric_phase="test",
        device="cpu",
        seq_len=8,
        erf_n_samples=1,
        icld_seq_len=12,
        icld_batch_size=2,
        logit_margin_n_train_steps=4,
        logit_margin_batch_size=2,
        spec_norm_vocab_size=64,
    )

    assert result.spec_norm_status == "ok"
    assert result.jacobian_erf.status == "ok"
    assert result.jacobian_erf.density is not None
    assert result.icld.delta_loss is not None
    assert result.logit_margin.delta_margin is not None
    assert math.isfinite(result.logit_margin.delta_margin)


def test_logit_margin_retries_with_lower_lr_on_nonfinite_result(monkeypatch) -> None:
    calls: list[float] = []

    def fake_sensitivity(*_args, **_kwargs):
        return {
            "_succeeded": True,
            "spectral_norm": 1.0,
            "effective_rank": 1.0,
            "uniformity": 0.5,
        }

    def fake_erf(*_args, **_kwargs):
        return JacobianERFResult(density=1.0, variance=0.0, status="ok")

    def fake_icld(*_args, **_kwargs):
        return ICLDResult(delta_loss=0.1, velocity=0.01, status="ok")

    def fake_margin(*_args, lr: float, **_kwargs):
        calls.append(lr)
        if len(calls) == 1:
            return LogitMarginResult(
                velocity=float("nan"),
                initial_margin=0.0,
                final_margin=float("nan"),
                delta_margin=float("nan"),
                n_steps=60,
                status="ok",
            )
        return LogitMarginResult(
            velocity=0.01,
            initial_margin=-0.1,
            final_margin=0.2,
            delta_margin=0.3,
            n_steps=60,
            status="ok",
        )

    monkeypatch.setattr(trajectory_metrics, "analyze_sensitivity", fake_sensitivity)
    monkeypatch.setattr(trajectory_metrics, "compute_jacobian_erf", fake_erf)
    monkeypatch.setattr(trajectory_metrics, "compute_icld_velocity", fake_icld)
    monkeypatch.setattr(
        trajectory_metrics,
        "compute_transitive_logit_margin",
        fake_margin,
    )

    model = torch.nn.Linear(2, 2)
    result = compute_trajectory_metrics(model, metric_phase="test", device="cpu")

    assert calls == [1e-3, 3e-4]
    assert result.logit_margin.delta_margin == 0.3
    assert result.logit_margin.status == "ok_lr0.0003_fallback"
