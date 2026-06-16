from __future__ import annotations

import torch

from component_fab.harness.tiny_lm import (
    SoftmaxCausalAttention,
    TinyLM,
    TinyLMConfig,
    lane_factory_for_baseline,
)
from component_fab.intake.model_components import (
    find_component_sites,
    replaceable_component_paths,
)


def test_find_component_sites_discovers_tinylm_mixers() -> None:
    model = TinyLM(
        SoftmaxCausalAttention,
        TinyLMConfig(vocab_size=32, dim=16, n_blocks=2, max_seq_len=16),
    )
    ids = torch.randint(0, 32, (2, 8))

    sites = find_component_sites(model, sample_input=ids)
    by_path = {site.path: site for site in sites}

    assert "blocks.0.lane" in by_path
    assert by_path["blocks.0.lane"].role == "token_mixer"
    assert by_path["blocks.0.lane"].replaceability == "drop_in_sequence_module"
    assert by_path["blocks.0.lane"].input_shape == (2, 8, 16)
    assert by_path["blocks.0.lane"].output_shape == (2, 8, 16)
    assert "embed" not in by_path


def test_replaceable_component_paths_prefers_sequence_modules() -> None:
    model = TinyLM(
        SoftmaxCausalAttention,
        TinyLMConfig(vocab_size=32, dim=16, n_blocks=1, use_ffn=True, max_seq_len=16),
    )
    ids = torch.randint(0, 32, (2, 8))

    paths = replaceable_component_paths(model, sample_input=ids)

    assert paths[0] == "blocks.0.lane"
    assert "blocks.0.mlp" in paths


def test_find_component_sites_handles_standalone_reference_mixer() -> None:
    mixer = lane_factory_for_baseline("gpt2")(16)
    x = torch.randn(2, 8, 16)

    sites = find_component_sites(mixer, sample_input=x, include_fixed=True)
    top = sites[0]

    assert top.path in {"qkv", "proj"}
    assert any(site.role == "projection" for site in sites)
    assert all(site.path for site in sites)
