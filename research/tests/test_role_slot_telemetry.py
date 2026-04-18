"""Regression: role-slot templates emit ``role:*`` telemetry end-to-end.

Role slots ride on the existing ``template_slot_usage`` channel so the
notebook observability stack can surface them without a schema change.
These tests pin that wiring for the 2026-04-16 capability-first push:

* All six role-slot templates (3 codex originals + 3 v2 variants) must
  emit at least the ``binding_write``, ``global_retrieval`` or
  equivalent, ``binding_read``, and ``merge_policy`` role slots.
* The role names must come through prefixed with ``role:`` so the
  observability aggregator can distinguish them from motif-class slots.
"""

from __future__ import annotations

import random

import pytest

from research.synthesis.graph import ComputationGraph
from research.synthesis.templates import TEMPLATES


ROLE_SLOT_TEMPLATES = (
    "typed_slot_memory_block",
    "sparse_relation_graph_block",
    "token_program_interpreter_block",
    "conv_residual_retrieval_v2",
    "state_space_retrieval_v2",
    "latent_attn_retrieval_v2",
)

# Every capability-first template must at minimum declare a merge policy
# and carry at least one retrieval-family role (``global_retrieval`` or
# ``binding_read``). Some templates describe write/read via a single
# ``global_retrieval`` slot (sparse_relation_graph_block), others use the
# explicit write+read pair — the common contract is "retrieval role
# present + merge policy present".
RETRIEVAL_ROLES = {"role:global_retrieval", "role:binding_read"}
REQUIRED_ROLES_SOFT = {"role:merge_policy"}


@pytest.mark.parametrize("template_name", ROLE_SLOT_TEMPLATES)
def test_role_slot_template_emits_role_slot_telemetry(template_name: str) -> None:
    graph = ComputationGraph(model_dim=128)
    graph.metadata["_active_template"] = template_name
    graph.metadata["_active_template_instance"] = 0
    graph.metadata["_active_template_slot_counter"] = 0

    input_id = graph.add_input()
    out_id = TEMPLATES[template_name](graph, input_id, random.Random(42))
    graph.set_output(out_id)

    slot_usage = graph.metadata.get("template_slot_usage", [])
    assert slot_usage, f"{template_name} wrote no template_slot_usage entries"

    roles_seen: set[str] = set()
    for entry in slot_usage:
        for cls in entry.get("slot_classes") or []:
            if cls.startswith("role:"):
                roles_seen.add(cls)

    missing = REQUIRED_ROLES_SOFT - roles_seen
    assert not missing, (
        f"{template_name} missing required role slots {missing!r}; "
        f"saw {sorted(roles_seen)}"
    )
    assert roles_seen & RETRIEVAL_ROLES, (
        f"{template_name} has no retrieval-family role slot "
        f"({sorted(RETRIEVAL_ROLES)}); saw {sorted(roles_seen)}"
    )


def test_role_slot_templates_contain_retrieval_ops() -> None:
    """Retrieval family (matmul or gather_topk) must appear in each template.

    This is the structural half of the "no retrieval-dead graphs" push —
    the screening gate handles the ones that slip through, but templates
    themselves should never emit a retrieval-dead graph by construction.
    """
    retrieval_ops = {"matmul", "gather_topk", "outer_product", "cosine_similarity"}

    for name in ROLE_SLOT_TEMPLATES:
        graph = ComputationGraph(model_dim=128)
        graph.metadata["_active_template"] = name
        input_id = graph.add_input()
        out_id = TEMPLATES[name](graph, input_id, random.Random(11))
        graph.set_output(out_id)

        op_names = {
            graph.nodes[n].op_name
            for n in graph.nodes
            if not graph.nodes[n].is_input
        }
        assert op_names & retrieval_ops, (
            f"{name} contains no retrieval op ({sorted(retrieval_ops)}); "
            f"op set was {sorted(op_names)}"
        )
