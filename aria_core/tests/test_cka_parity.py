import aria_core
import torch

from research.eval.fingerprint_native import linear_cka, sequence_self_similarity


def test_cka_parity(monkeypatch):
    n = 64
    X = torch.randn(n, n)
    Y = torch.randn(n, n)

    # Pure-torch reference path (native dispatch disabled)
    monkeypatch.setenv("ARIA_DISABLE_NATIVE_CKA", "1")
    pytorch_res = linear_cka(X, Y)

    native_res = aria_core.linear_cka_f32(X.contiguous(), Y.contiguous())

    assert abs(pytorch_res - native_res) < 1e-5, "CKA parity check failed!"

    id_res = aria_core.linear_cka_f32(X, X)
    assert abs(id_res - 1.0) < 1e-5


def test_sequence_self_similarity_parity():
    reps = torch.randn(4, 12, 16, dtype=torch.float32)
    native = aria_core.sequence_self_similarity_f32(reps.contiguous())
    wrapped = sequence_self_similarity(reps)
    norm = torch.nn.functional.normalize(reps, dim=-1)
    reference = torch.bmm(norm, norm.transpose(1, 2)).mean(dim=0)

    assert native.shape == (12, 12)
    assert torch.allclose(native, reference, atol=1e-5, rtol=1e-5)
    assert torch.allclose(wrapped, reference, atol=1e-5, rtol=1e-5)
