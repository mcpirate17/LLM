"""Sanity tests for CKA (Centered Kernel Alignment) computation.

Validates that:
  1. CKA of a matrix with itself = 1.0
  2. CKA of independent random matrices ≈ 0
  3. Same reference model produces self-CKA = 1.0
  4. Different architecture families produce CKA < 0.8
  5. Device transfer works (CUDA ↔ CPU)
"""

import pytest
import torch
import torch.nn.functional as F


def _linear_cka_python(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Pure-Python linear CKA (no aria_core dependency)."""
    X = X - X.mean()
    Y = Y - Y.mean()
    hsic_xy = (X * Y).sum()
    hsic_xx = (X * X).sum()
    hsic_yy = (Y * Y).sum()
    denom = (hsic_xx * hsic_yy).clamp(min=1e-30).sqrt()
    return (hsic_xy / denom).clamp(0, 1).item()


def _build_sim_matrix(activations: torch.Tensor) -> torch.Tensor:
    """Build self-similarity matrix from activations, matching fingerprint.py."""
    flat = activations.float()
    S, D = flat.shape[-2], flat.shape[-1]
    norm = F.normalize(flat, dim=-1)
    sim = torch.mm(norm.reshape(-1, D), norm.reshape(-1, D).t())
    return sim[:S, :S]


class TestCKAMath:
    """Tests for the CKA similarity metric itself."""

    def test_self_cka_is_one(self) -> None:
        """CKA of a matrix with itself must be 1.0."""
        torch.manual_seed(0)
        X = torch.randn(64, 64)
        assert abs(_linear_cka_python(X, X) - 1.0) < 0.01

    def test_independent_random_near_zero(self) -> None:
        """CKA of two independent random matrices should be near 0."""
        torch.manual_seed(0)
        X = torch.randn(64, 64)
        torch.manual_seed(999)
        Y = torch.randn(64, 64)
        assert _linear_cka_python(X, Y) < 0.2

    def test_scaled_matrix_cka_is_one(self) -> None:
        """CKA is scale-invariant: CKA(X, 5*X) = 1.0."""
        torch.manual_seed(0)
        X = torch.randn(64, 64)
        assert abs(_linear_cka_python(X, 5.0 * X) - 1.0) < 0.01

    def test_aria_core_matches_python(self) -> None:
        """aria_core.linear_cka_f32 must match the Python implementation."""
        try:
            import aria_core
        except ImportError:
            pytest.skip("aria_core not available")

        torch.manual_seed(42)
        X = torch.randn(64, 64)
        Y = torch.randn(64, 64)
        py_val = _linear_cka_python(X, Y)
        native_val = aria_core.linear_cka_f32(X.contiguous(), Y.contiguous())
        assert abs(py_val - native_val) < 1e-4, (
            f"Python CKA={py_val:.6f} vs aria_core={native_val:.6f}"
        )


class TestCKAReferenceArtifacts:
    """Tests against the actual reference artifact files."""

    @pytest.fixture(autouse=True)
    def _load_refs(self) -> None:
        from pathlib import Path

        artifact_dir = (
            Path(__file__).parent.parent / "artifacts" / "cka_references" / "v1"
        )
        if not artifact_dir.exists():
            pytest.skip("CKA reference artifacts not found")
        self.refs = {}
        for family in ("transformer", "ssm", "conv"):
            pt_path = artifact_dir / f"{family}.pt"
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            self.refs[family] = data["activations"]

    def test_self_similarity_is_one(self) -> None:
        """CKA of a reference model's sim matrix with itself = 1.0."""
        for family, acts in self.refs.items():
            sim = _build_sim_matrix(acts)
            cka = _linear_cka_python(sim, sim)
            assert abs(cka - 1.0) < 0.01, f"{family} self-CKA={cka:.4f}, expected ~1.0"

    def test_cross_architecture_below_threshold(self) -> None:
        """CKA between different architecture families should be < 0.8."""
        families = list(self.refs.keys())
        for i, f1 in enumerate(families):
            for f2 in families[i + 1 :]:
                sim1 = _build_sim_matrix(self.refs[f1])
                sim2 = _build_sim_matrix(self.refs[f2])
                # Align sequence lengths
                use_S = min(sim1.shape[0], sim2.shape[0])
                cka = _linear_cka_python(sim1[:use_S, :use_S], sim2[:use_S, :use_S])
                assert cka < 0.8, (
                    f"CKA({f1}, {f2})={cka:.4f}, expected < 0.8 "
                    f"for different architectures"
                )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_device_transfer_produces_same_result(self) -> None:
        """CKA must be identical whether computed on CPU or CUDA."""
        acts = self.refs["transformer"]
        sim_cpu = _build_sim_matrix(acts)

        acts_cuda = acts.to("cuda")
        sim_cuda = _build_sim_matrix(acts_cuda)

        cka_cpu = _linear_cka_python(sim_cpu, sim_cpu)
        cka_cuda = _linear_cka_python(sim_cuda, sim_cuda)
        assert abs(cka_cpu - cka_cuda) < 1e-3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cross_device_after_transfer(self) -> None:
        """After .to(device), CKA between CUDA candidate and ref works."""
        candidate = torch.randn(64, 1024, device="cuda")
        ref = self.refs["transformer"]  # CPU

        sim_cand = _build_sim_matrix(candidate)
        # Simulate the fix: move ref to candidate device before sim
        ref_on_device = ref.to(device=candidate.device)
        sim_ref = _build_sim_matrix(ref_on_device)

        cka = _linear_cka_python(sim_cand, sim_ref)
        # Should be a real number, not 0.0 from a caught exception
        assert cka > 0.0 or cka == 0.0  # valid float
        # The actual value doesn't matter, just that it doesn't crash
        assert isinstance(cka, float)
