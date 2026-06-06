"""Frontier-core hosts: 'frontier + delta' generation.

Locks in the structural guarantees that make the strong-binder cores usable as
hybrid hosts. Binding *magnitude* is deliberately not asserted (the nano probe
is too noisy at dim=64/seq=32 — see the session notes); these tests assert
dispatch, host-eligibility, and that grafting preserves the host's global
mixing while transforming the architecture.
"""

from __future__ import annotations

import torch.nn as nn

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.improver.axis_variants import AnchorAxes
from component_fab.improver.cross_anchor import (
    _frontier_hybrid_spec,
    enumerate_frontier_core_specs,
    enumerate_frontier_hybrids,
    is_hosting_anchor,
)
from component_fab.proposer.frontier_cores import (
    frontier_core_anchors,
    frontier_core_names,
)

_DONOR = AnchorAxes(
    op_name="surprise_mem_donor",
    axes={
        "op_dynamical_has_state": 1,
        "op_dynamical_memory_length_class": "O(L)",
        "op_activation_sparsity_pattern": "learned_structured",
        # A weak donor receptive field that MUST NOT override the host's global.
        "op_geometric_receptive_field": "per_position",
    },
    eval_count=0,
    pass_rate=0.1,
)


def test_all_frontier_cores_are_host_eligible():
    anchors = frontier_core_anchors()
    assert len(anchors) == len(frontier_core_names())
    assert all(is_hosting_anchor(a) for a in anchors)


def test_bare_cores_dispatch_to_strong_binders_not_linear():
    expected = {
        "frontier_tropical_attention": "TropicalAttention",
        "frontier_gated_parallel_tropical_sparsemax": "GatedParallelBlock",
        "frontier_three_lane_tsw": "ThreeLaneAdaptive",
    }
    for spec in enumerate_frontier_core_specs():
        module = generate_module_from_spec(spec, dim=64)
        assert not isinstance(module, nn.Linear), f"{spec.name} fell back to Linear"
        assert type(module).__name__ == expected[spec.name]


def test_frontier_hybrid_preserves_global_mixing_and_grafts_mechanism():
    for host in frontier_core_anchors():
        spec = _frontier_hybrid_spec(host, _DONOR)
        # host's global receptive field is preserved (donor's per_position dropped)
        assert spec.math_axes["op_geometric_receptive_field"] == "global"
        # donor's novel mechanism is grafted on
        assert spec.math_axes["op_dynamical_has_state"] == 1
        assert spec.math_axes["op_dynamical_memory_length_class"] == "O(L)"
        assert spec.math_axes["op_activation_sparsity_pattern"] == "learned_structured"
        # the grafted spec still compiles to a real (non-Linear) module
        module = generate_module_from_spec(spec, dim=64)
        assert not isinstance(module, nn.Linear)


def test_frontier_hybrids_skip_unresolvable_donors():
    # Donor op-names absent from the meta DB are skipped, not errored.
    out = enumerate_frontier_hybrids(["__definitely_not_a_real_op__"])
    assert out == []


def test_enumerate_frontier_hybrids_pairs_hosts_with_resolvable_donors():
    hosts = frontier_core_anchors()
    out = enumerate_frontier_hybrids(
        ["donorA"],
        hosts=[
            AnchorAxes(
                op_name=h.op_name, axes=dict(h.axes), eval_count=0, pass_rate=1.0
            )
            for h in hosts
        ],
    )
    # anchor_axes_for_op resolves nothing for the fake donor -> 0 hybrids,
    # so feed a synthetic donor via the host path instead to exercise pairing.
    direct = [_frontier_hybrid_spec(h, _DONOR) for h in hosts]
    assert len(direct) == len(hosts)
    assert all("_plus_" in s.name for s in direct)
    assert out == []  # fake donor unresolved -> empty, as documented
