import pytest
import torch

from research.eval.binding_curriculum import curriculum_binding_range_profile
from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph
from research.synthesis.reference_architectures import build_reference


def _build_local_only_graph(model_dim: int = 64) -> ComputationGraph:
    g = ComputationGraph(model_dim)
    inp = g.add_input()
    c = g.add_op("conv_only", [inp])
    n = g.add_op("rmsnorm", [c])
    p = g.add_op("linear_proj", [n])
    out = g.add_op("add", [inp, p])
    g.set_output(out)
    return g


def test_curriculum_binding_probe_separates_gpt2_reference_from_local_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1234)
    local_model = compile_model(
        [_build_local_only_graph()], vocab_size=256, max_seq_len=128
    )
    local_model.to(device)
    local = curriculum_binding_range_profile(
        local_model,
        distances=(4, 8, 16, 32),
        n_train_steps=400,
        n_eval=64,
        train_seq_len=128,
        eval_seq_len=128,
        train_batch_size=16,
        eval_batch_size=16,
        device=device,
        seed=123,
    )

    torch.manual_seed(1234)
    gpt2_model = compile_model(
        [build_reference("gpt2", 64)], vocab_size=256, max_seq_len=128
    )
    gpt2_model.to(device)
    gpt2 = curriculum_binding_range_profile(
        gpt2_model,
        distances=(4, 8, 16, 32),
        n_train_steps=400,
        n_eval=64,
        train_seq_len=128,
        eval_seq_len=128,
        train_batch_size=16,
        eval_batch_size=16,
        device=device,
        seed=123,
    )

    assert local.status == "ok"
    assert gpt2.status == "ok"
    assert gpt2.auc > local.auc + 0.01, (local.auc, gpt2.auc)
    assert gpt2.distance_accuracies[4] > local.distance_accuracies[4], (
        local.distance_accuracies,
        gpt2.distance_accuracies,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="seeded binding baseline is calibrated on CUDA",
)
def test_curriculum_binding_probe_seeded_cuda_baseline_values_are_stable():
    torch.manual_seed(1234)
    local_model = compile_model(
        [_build_local_only_graph()], vocab_size=256, max_seq_len=128
    )
    local_model.to("cuda")
    local = curriculum_binding_range_profile(
        local_model,
        distances=(4, 8, 16, 32),
        n_train_steps=400,
        n_eval=64,
        train_seq_len=128,
        eval_seq_len=128,
        train_batch_size=16,
        eval_batch_size=16,
        device="cuda",
        seed=123,
    )

    torch.manual_seed(1234)
    gpt2_model = compile_model(
        [build_reference("gpt2", 64)], vocab_size=256, max_seq_len=128
    )
    gpt2_model.to("cuda")
    gpt2 = curriculum_binding_range_profile(
        gpt2_model,
        distances=(4, 8, 16, 32),
        n_train_steps=400,
        n_eval=64,
        train_seq_len=128,
        eval_seq_len=128,
        train_batch_size=16,
        eval_batch_size=16,
        device="cuda",
        seed=123,
    )

    assert local.status == "ok"
    assert gpt2.status == "ok"
    assert local.train_steps == 400
    assert gpt2.train_steps == 400
    assert local.auc == pytest.approx(0.0049, abs=1e-4)
    assert gpt2.auc == pytest.approx(0.0333, abs=1e-4)
    assert local.distance_accuracies == {
        4: pytest.approx(0.0039, abs=1e-4),
        8: pytest.approx(0.0059, abs=1e-4),
        16: pytest.approx(0.0068, abs=1e-4),
        32: pytest.approx(0.0029, abs=1e-4),
    }
    assert gpt2.distance_accuracies == {
        4: pytest.approx(0.0880, abs=1e-4),
        8: pytest.approx(0.0257, abs=1e-4),
        16: pytest.approx(0.0105, abs=1e-4),
        32: pytest.approx(0.0091, abs=1e-4),
    }
