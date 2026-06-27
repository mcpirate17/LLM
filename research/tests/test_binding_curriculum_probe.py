import pytest
import torch

from research.eval.binding_curriculum import curriculum_binding_range_profile
from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph
from research.synthesis.reference_architectures import build_reference


def _run_probe(model, device: str):
    return curriculum_binding_range_profile(
        model,
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


def _build_local_only_graph(model_dim: int = 64) -> ComputationGraph:
    g = ComputationGraph(model_dim)
    inp = g.add_input()
    c = g.add_op("conv_only", [inp])
    n = g.add_op("rmsnorm", [c])
    p = g.add_op("linear_proj", [n])
    out = g.add_op("add", [inp, p])
    g.set_output(out)
    return g


@pytest.fixture(scope="module")
def curriculum_results():
    """One shared 400-step training per model; both tests assert against it.

    Seeding matches the original per-test setup exactly (manual_seed(1234)
    before each compile, probe seed=123) so the CUDA baseline envelopes
    below stay valid.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1234)
    local_model = compile_model(
        [_build_local_only_graph()], vocab_size=256, max_seq_len=128
    )
    local_model.to(device)
    local = _run_probe(local_model, device)

    torch.manual_seed(1234)
    gpt2_model = compile_model(
        [build_reference("gpt2", 64)], vocab_size=256, max_seq_len=128
    )
    gpt2_model.to(device)
    gpt2 = _run_probe(gpt2_model, device)

    return device, local, gpt2


# Capability experiment (2 models × 400-step curriculum training); times out
# under CPU-only nano probe budgets — run via the slow lane.
@pytest.mark.slow
def test_curriculum_binding_probe_separates_gpt2_reference_from_local_model(
    curriculum_results,
):
    _, local, gpt2 = curriculum_results

    assert local.status == "ok"
    assert gpt2.status == "ok"
    assert gpt2.auc > local.auc + 0.01, (local.auc, gpt2.auc)
    assert gpt2.distance_accuracies[4] > local.distance_accuracies[4], (
        local.distance_accuracies,
        gpt2.distance_accuracies,
    )


@pytest.mark.slow
def test_curriculum_binding_probe_seeded_cuda_baseline_values_are_stable(
    curriculum_results,
):
    device, local, gpt2 = curriculum_results
    if device != "cuda":
        pytest.skip("seeded binding baseline is calibrated on CUDA")

    assert local.status == "ok"
    assert gpt2.status == "ok"
    assert local.train_steps == 400
    assert gpt2.train_steps == 400
    # Exact counts are sensitive to optimizer/kernel numerics across
    # PyTorch/CUDA stacks. Keep calibrated envelopes instead of pinning one
    # fused-AdamW outcome from a single environment.
    assert 0.004 <= local.auc <= 0.005
    assert 0.032 <= gpt2.auc <= 0.036
    assert gpt2.auc > local.auc + 0.025
    assert 0.0 <= local.distance_accuracies[4] <= 0.005
    assert 0.005 <= local.distance_accuracies[8] <= 0.011
    assert 0.003 <= local.distance_accuracies[16] <= 0.007
    assert 0.002 <= local.distance_accuracies[32] <= 0.004
    assert 0.085 <= gpt2.distance_accuracies[4] <= 0.095
    assert 0.024 <= gpt2.distance_accuracies[8] <= 0.028
    assert 0.009 <= gpt2.distance_accuracies[16] <= 0.013
    assert 0.008 <= gpt2.distance_accuracies[32] <= 0.011
    assert gpt2.distance_accuracies[4] > gpt2.distance_accuracies[8]
    assert gpt2.distance_accuracies[8] > gpt2.distance_accuracies[16]
    assert gpt2.distance_accuracies[16] >= gpt2.distance_accuracies[32]
