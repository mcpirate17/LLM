"""Grammar->fab bridge: novel NAS topologies compiled into gradeable lanes."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import component_fab.proposer.nas_bridge as NB
from component_fab.generator.code_generator import (
    generate_module,
    generate_module_from_spec,
)
from component_fab.proposer.nas_bridge import (
    SOURCE_NAS,
    load_cached_graph_json,
    nas_graph_specs,
)
from component_fab.validator.capability import validate_capabilities

_DIM = 32


def test_nas_graph_specs_admit_only_compilable_topologies():
    specs = nas_graph_specs(n_fresh=4, dim=_DIM, seed=100, include_db_winners=False)
    assert specs, "grammar should yield at least one compilable topology"
    for s in specs:
        assert s.math_axes["op_source"] == SOURCE_NAS
        fp = s.math_axes["op_nas_fingerprint"]
        assert load_cached_graph_json(fp) is not None  # self-contained / re-gradeable
        assert s.math_axes["op_nas_ops"] >= 1
        assert s.synthesis_kind == SOURCE_NAS
        assert s.category == "lane"


def test_nas_spec_builds_working_mixer_through_production_path():
    spec = nas_graph_specs(n_fresh=3, dim=_DIM, seed=7, include_db_winners=False)[0]
    module = generate_module_from_spec(spec, dim=_DIM)
    assert not isinstance(module, nn.Linear)  # a real compiled topology, not fallback
    y = module(torch.randn(2, 16, _DIM))
    assert y.shape == (2, 16, _DIM)
    assert torch.isfinite(y).all()
    # the capability gate runs end-to-end on a NAS topology (pass or eliminate)
    sc = validate_capabilities(spec, module, dim=_DIM, seq_len=16)
    assert sc.proposal_id == spec.proposal_id


def test_same_topology_dedupes_by_id():
    a = nas_graph_specs(n_fresh=2, dim=_DIM, seed=3, include_db_winners=False)
    b = nas_graph_specs(n_fresh=2, dim=_DIM, seed=3, include_db_winners=False)
    assert [s.proposal_id for s in a] == [s.proposal_id for s in b]  # deterministic


def test_missing_cache_is_a_hard_error():
    # A nas spec whose graph JSON is not cached must fail loud, not silently fall back.
    axes = {"op_source": SOURCE_NAS, "op_nas_fingerprint": "deadbeef_not_cached"}
    with pytest.raises(RuntimeError, match="cached graph JSON missing"):
        generate_module(axes, dim=_DIM)


def test_non_compilable_graphs_are_dropped(monkeypatch):
    # If nothing compiles at the grading dim, the bridge emits zero specs
    # (never a broken spec that would crash the grade loop).
    monkeypatch.setattr(NB, "_compiles_finite", lambda graph, dim: False)
    assert nas_graph_specs(n_fresh=4, dim=_DIM, seed=1, include_db_winners=False) == []


def test_disabled_returns_empty():
    assert nas_graph_specs(n_fresh=0, include_db_winners=False) == []
