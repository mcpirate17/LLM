"""Tests for reference architecture builders and registration pipeline."""

import pytest
import torch

from research.synthesis.reference_architectures import (
    REFERENCE_ARCHITECTURES,
    build_reference,
    list_references,
)
from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph

pytestmark = pytest.mark.unit


class TestReferenceArchitectureBuilders:
    """Test that each reference architecture builds valid graphs."""

    @pytest.fixture(params=list(REFERENCE_ARCHITECTURES.keys()))
    def arch_key(self, request):
        return request.param

    def test_builds_without_error(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        assert isinstance(g, ComputationGraph)
        assert g.n_ops() > 0

    def test_has_valid_output(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        assert g.output_node is not None
        assert g.output_node.output_shape.dim == 64
        assert g.output_node.output_shape.is_standard

    def test_has_gradient_path(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        assert g.has_gradient_path()

    def test_has_metadata(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        assert "architecture" in g.metadata
        assert "reference_name" in g.metadata

    def test_unique_fingerprints(self):
        fps = set()
        for key in REFERENCE_ARCHITECTURES:
            g = build_reference(key, d_model=64)
            fp = g.fingerprint()
            assert fp not in fps, f"Duplicate fingerprint for {key}"
            fps.add(fp)

    def test_different_d_models(self, arch_key):
        for d in [64, 128, 256]:
            g = build_reference(arch_key, d_model=d)
            assert g.output_node.output_shape.dim == d


class TestReferenceCompilation:
    """Test that reference architectures compile to valid torch models."""

    @pytest.fixture(params=list(REFERENCE_ARCHITECTURES.keys()))
    def arch_key(self, request):
        return request.param

    def test_compiles_to_model(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        model = compile_model([g], vocab_size=1000, max_seq_len=32)
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        model = compile_model([g], vocab_size=1000, max_seq_len=32)
        input_ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            output = model(input_ids)
        assert output.shape[0] == 2  # batch
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_backward_pass(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        model = compile_model([g], vocab_size=1000, max_seq_len=32)
        input_ids = torch.randint(0, 1000, (2, 16))
        output = model(input_ids)
        loss = output.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad, f"{arch_key} has no gradients flowing"

    def test_multi_layer_model(self, arch_key):
        layers = [build_reference(arch_key, d_model=64) for _ in range(3)]
        model = compile_model(layers, vocab_size=1000, max_seq_len=32)
        input_ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            output = model(input_ids)
        assert not torch.isnan(output).any()


class TestListReferences:
    def test_lists_all(self):
        refs = list_references()
        assert len(refs) == len(REFERENCE_ARCHITECTURES)
        for ref in refs:
            assert "key" in ref
            assert "name" in ref
            assert "paradigm" in ref

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            build_reference("nonexistent")
