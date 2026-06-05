"""Cross-pollination of component_fab inventions into NAS grammar motifs (M2)."""

from __future__ import annotations

from research.synthesis._motifs_fab import (
    _FAB_INVENTIONS,
    MOTIF_CLASS_FAB,
    register_fab_motifs,
)
from research.synthesis._selection_utils import context_pair_allowed
from research.synthesis.op_roles import OpRole, get_role
from research.synthesis.primitives import PRIMITIVE_REGISTRY


def _fresh() -> tuple[dict, dict]:
    return {}, {}


def test_register_disabled_is_noop() -> None:
    validated, by_class = _fresh()
    assert register_fab_motifs(validated, by_class, enable=False) == []
    assert validated == {} and by_class == {}


def test_register_injects_fab_invention_class() -> None:
    validated, by_class = _fresh()
    names = register_fab_motifs(validated, by_class, enable=True)
    assert len(names) == len(_FAB_INVENTIONS) == 11
    bucket = by_class[MOTIF_CLASS_FAB]
    assert len(bucket) == 11
    for motif in bucket:
        assert motif.motif_class == MOTIF_CLASS_FAB
        ops = [s.op_name for s in motif.steps]
        assert ops[-1] == "linear_proj"  # canonical mixer -> proj shape
        assert len(motif.steps) == 2
        assert motif.lift >= 0.5  # above the MIN_MOTIF_LIFT sampling floor


def test_registered_ops_are_real_and_context_legal() -> None:
    validated, by_class = _fresh()
    register_fab_motifs(validated, by_class, enable=True)
    for motif in by_class[MOTIF_CLASS_FAB]:
        mix_op = motif.steps[0].op_name
        prim = PRIMITIVE_REGISTRY.get(mix_op)
        assert prim is not None, f"{mix_op} missing from registry"
        assert prim.n_inputs == 1
        assert get_role(mix_op) is not OpRole.UNSAFE
        assert context_pair_allowed(mix_op, "linear_proj")


def test_register_is_idempotent_on_repeat() -> None:
    validated, by_class = _fresh()
    register_fab_motifs(validated, by_class, enable=True)
    second = register_fab_motifs(validated, by_class, enable=True)
    assert second == []  # all names already present
    assert len(by_class[MOTIF_CLASS_FAB]) == 11


def test_fab_class_reachable_via_wildcard_path() -> None:
    # The dedicated class is inert unless it is also in the wildcard class list;
    # the M2 wiring adds it to _template_helpers._ALL_CLASSES.
    from research.synthesis._template_helpers import _ALL_CLASSES

    assert MOTIF_CLASS_FAB in _ALL_CLASSES


def test_unknown_op_is_skipped_not_raised() -> None:
    # A bogus invention op must be skipped, never emitted as an illegal motif.
    from research.synthesis import _motifs_fab as fab

    bogus = fab._FabInvention("fab_bogus", "no_such_op_xyz", 1.0, "bogus")
    assert fab._make_fab_motif(bogus) is None


def test_fab_ops_compose_into_validated_graphs() -> None:
    """End-to-end: with fab motifs registered, the grammar composes them into
    valid topologies via the wildcard exploration path (the M2 unlock)."""
    from research.synthesis.grammar import GrammarConfig, generate_layer_graph
    from research.synthesis.motifs import MOTIFS_BY_CLASS, VALIDATED_MOTIFS

    fab_ops = {inv.op for inv in _FAB_INVENTIONS}
    names = register_fab_motifs(VALIDATED_MOTIFS, MOTIFS_BY_CLASS, enable=True)
    try:
        cfg = GrammarConfig.exotic(model_dim=64)
        cfg.wildcard_slot_prob = 0.9  # force exploration to reach the fab class
        hits = 0
        for seed in range(120):
            try:
                graph = generate_layer_graph(cfg, seed)  # validate=True
            except Exception:
                continue
            if {nd.op_name for nd in graph.nodes.values()} & fab_ops:
                hits += 1
        assert hits > 0, "fab ops never composed into a validated graph"
    finally:
        # Restore global catalog state for other tests.
        for name in names:
            VALIDATED_MOTIFS.pop(name, None)
        MOTIFS_BY_CLASS.pop(MOTIF_CLASS_FAB, None)
