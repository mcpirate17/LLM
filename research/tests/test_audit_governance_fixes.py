"""Regression tests for the 2026-04-17 slot-governance audit fixes.

Pins the four governance changes:
  P0.10 — wildcard breadcrumb on graph metadata; role-slot path never widens
  P0.11 — binding-role legality (binding_read/write/global_retrieval require
          a content-addressed op in the chosen motif's emitted chain)
  P0.12 — broadened allowlists for the two empirically too-restrictive slots
  P1.12 — codex hybrid templates emit explicit slot bindings for their
          named structural slots
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# ── P0.10: wildcard breadcrumb + role-slot strictness ────────────────


def test_role_slot_path_never_sets_wildcard():
    """`pick_role_motif` must never widen to _ALL_CLASSES — role slots are
    contracts, not exploration hints. We assert the source contract directly:
    the function must not import or call _ALL_CLASSES."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "synthesis"
        / "_template_role_slots.py"
    ).read_text()
    # The function body must not reference _ALL_CLASSES (which would widen).
    # The import at the top is fine (it's available for other helpers).
    func_start = src.index("def pick_role_motif(")
    next_def = src.index("\ndef ", func_start + 1)
    body = src[func_start:next_def]
    assert "_ALL_CLASSES" not in body, (
        "pick_role_motif must not widen to _ALL_CLASSES — role slots are "
        "contracts. Use class-bucket fallback instead."
    )


def test_class_slot_wildcard_sets_graph_breadcrumb():
    """When the class-slot picker fires its wildcard fallback, it must leave
    a graph-level breadcrumb so downstream filtering can detect it."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "synthesis"
        / "_template_helpers.py"
    ).read_text()
    assert "_template_wildcard_used" in src
    assert "_template_wildcard_slot_keys" in src


# ── P0.11: binding-role legality ─────────────────────────────────────


def test_binding_legal_roles_defined():
    from research.synthesis._template_role_slots import (
        _BINDING_LEGAL_OPS,
        _BINDING_LEGAL_ROLES,
    )

    # The three role slots flagged "under-observed, broad class bucket without
    # per-role legality checks" in the prior audit, plus the neural-symbolic
    # retrieval sidecar added for the role-slot v2 template.
    assert _BINDING_LEGAL_ROLES == frozenset(
        {"global_retrieval", "binding_read", "binding_write", "neural_symbolic"}
    )
    # Must include the core content-addressed ops; over-narrow legality would
    # silently make every binding role return None.
    for op in ("matmul", "softmax_attention", "gather_topk", "cosine_similarity"):
        assert op in _BINDING_LEGAL_OPS, f"binding-legal op set missing {op}"


def test_motif_is_binding_capable_recognizes_chains():
    from research.synthesis._motif_types import Motif, MotifStep
    from research.synthesis._template_role_slots import _motif_is_binding_capable
    from research.synthesis.op_roles import OpRole

    # A motif that's pure linear projections is NOT binding-capable.
    purely_linear = Motif(
        name="m_linear",
        motif_class="efficient_proj_core",
        steps=(
            MotifStep("linear_proj", OpRole.PROJECT),
            MotifStep("linear_proj_down", OpRole.PROJECT),
        ),
        description="all linear",
        support=1,
        avg_loss_ratio=1.0,
        lift=1.0,
    )
    assert not _motif_is_binding_capable(purely_linear)

    # A motif with matmul IS binding-capable.
    with_matmul = Motif(
        name="m_matmul",
        motif_class="efficient_proj_core",
        steps=(
            MotifStep("linear_proj", OpRole.PROJECT),
            MotifStep("matmul", OpRole.MIX),
        ),
        description="contains matmul",
        support=1,
        avg_loss_ratio=1.0,
        lift=1.0,
    )
    assert _motif_is_binding_capable(with_matmul)


# ── P0.12: broadened allowlists for empirically too-restrictive slots ──


def test_too_restrictive_slot_allowlists_broadened():
    """The two slots that had single-motif allowlists in slots.csv
    (`routed_bottleneck.slot2`, `attn_bottleneck_hybrid.slot2`) must no
    longer be 1-entry restrictive. They can satisfy the audit fix in two
    ways: (a) keep an explicit allowlist with multiple sibling motifs, or
    (b) remove the allowlist entirely so any class-compatible motif is
    admitted. Either form removes the empirical "wildcards beat the
    prescribed motif" bias."""
    from research.synthesis._template_helpers import _SLOT_MOTIF_ALLOWLIST

    for slot_key in ("routed_bottleneck.slot2", "attn_bottleneck_hybrid.slot2"):
        entry = _SLOT_MOTIF_ALLOWLIST.get(slot_key)
        if entry is None:
            # No allowlist = unrestricted within class — strictest fix.
            continue
        assert len(entry) > 1, (
            f"{slot_key} still has single-motif allowlist {entry!r}; "
            "audit empirical data showed this was losing to wildcards"
        )


# ── P1.12: codex hybrids emit named structural slot telemetry ─────────


def test_codex_retention_block_records_named_slots():
    """codex_ssm_retention_block must record telemetry for retention_basis,
    memory_bottleneck, and tail_basis — the named structural slots that
    were previously emitted via raw _add without observability."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "synthesis"
        / "_templates_attention_advanced.py"
    ).read_text()
    # Locate the function body.
    fn_start = src.index("def tpl_codex_ssm_retention_block(")
    fn_end = src.index("\ndef ", fn_start + 1)
    body = src[fn_start:fn_end]
    for slot_name in ("retention_basis", "memory_bottleneck", "tail_basis"):
        assert f'slot_key=f"{{name}}[{{template_instance}}].{slot_name}"' in body, (
            f"codex_ssm_retention_block missing slot telemetry for {slot_name}"
        )


def test_codex_mla_gated_block_records_retention_compress():
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "synthesis"
        / "_templates_attention_advanced.py"
    ).read_text()
    fn_start = src.index("def tpl_codex_ssm_mla_gated_block(")
    fn_end = src.index("\ndef ", fn_start + 1)
    body = src[fn_start:fn_end]
    assert 'slot_key=f"{name}[{template_instance}].retention_compress"' in body, (
        "codex_ssm_mla_gated_block must record telemetry for the "
        "retention_compress slot (was previously a silent _pick_compatible_motif)"
    )
