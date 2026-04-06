from __future__ import annotations

from pathlib import Path

import yaml

from aria_designer.api.app.loader import validate_manifest


_ARIA_ROOT = Path(__file__).resolve().parent.parent
_ROUTING_DIR = _ARIA_ROOT / "components" / "routing"


def _load_manifest(component_id: str) -> dict:
    path = _ROUTING_DIR / component_id / "manifest.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_hybrid_sparse_router_manifests_declare_slots_and_templates():
    expected_slots = {
        "default_path": {"upstream_signal"},
        "hybrid_token_gate": {"default_path", "sparse_route"},
        "sparse_span_builder": {"sparse_route", "fallback_path"},
        "hybrid_sparse_router": {
            "pre_router",
            "default_path",
            "easy_router",
            "medium_router",
            "routed_lane",
            "sparse_spans",
            "difficulty_signal",
            "compression_router",
            "hard_router",
            "token_merge",
            "post_merge",
        },
        "lane_conditioned_block": {"routed_lane", "merge_target"},
    }

    for component_id, slot_names in expected_slots.items():
        manifest = _load_manifest(component_id)
        errors = validate_manifest(manifest)
        assert not errors, f"{component_id} manifest errors: {errors}"

        slots = manifest.get("slots") or []
        templates = manifest.get("templates") or []

        assert {slot["name"] for slot in slots} == slot_names
        assert templates, f"{component_id} should advertise at least one template"
        template_ids = {tpl["id"] for tpl in templates}
        assert "hybrid_sparse_triplet_router" in template_ids
        if component_id == "hybrid_sparse_router":
            assert "multiscale_rich_lane_router" in template_ids
            assert "intelligent_multilane_router" in template_ids
        assert all(tpl["workflow"] == "hybrid_sparse_router" for tpl in templates)


def test_hybrid_sparse_router_manifest_slot_bindings_match_template_variants():
    manifest = _load_manifest("hybrid_sparse_router")
    slot_names = {slot["name"] for slot in manifest.get("slots", [])}
    templates = {
        tpl["id"]: tpl.get("slot_bindings") or {} for tpl in manifest["templates"]
    }

    hybrid_bindings = templates["hybrid_sparse_triplet_router"]
    assert set(hybrid_bindings) == {"default_path", "routed_lane", "sparse_spans"}
    assert set(hybrid_bindings).issubset(slot_names)

    multiscale_bindings = templates["multiscale_difficulty_router"]
    assert set(multiscale_bindings) == {
        "default_path",
        "medium_router",
        "sparse_spans",
        "difficulty_signal",
        "compression_router",
        "hard_router",
    }
    assert set(multiscale_bindings).issubset(slot_names)

    rich_bindings = templates["multiscale_rich_lane_router"]
    assert set(rich_bindings) == {
        "default_path",
        "medium_router",
        "sparse_spans",
        "difficulty_signal",
        "compression_router",
        "hard_router",
    }
    assert set(rich_bindings).issubset(slot_names)

    intelligent_bindings = templates["intelligent_multilane_router"]
    assert set(intelligent_bindings) == {
        "pre_router",
        "easy_router",
        "medium_router",
        "sparse_spans",
        "difficulty_signal",
        "compression_router",
        "hard_router",
        "token_merge",
        "post_merge",
    }
    assert set(intelligent_bindings).issubset(slot_names)


def test_hybrid_sparse_router_manifest_exposes_next_lane_candidate_pools():
    manifest = _load_manifest("hybrid_sparse_router")
    slots = {slot["name"]: slot for slot in manifest.get("slots", [])}

    medium_components = set(slots["medium_router"]["compatible_components"])
    assert medium_components == {
        "routing/hybrid_sparse_router",
        "routing/route_lanes",
        "routing/adaptive_lane_mixer",
        "sparse/semi_structured_2_4_linear",
        "sparse/block_sparse_linear",
        "mixing/rwkv_time_mixing",
        "sparse/nm_sparse_linear",
        "routing/default_path",
        "routing/cheap_verify_blend",
        "linear_algebra/conv1d_seq",
        "mixing/conv_only",
    }

    hard_components = set(slots["hard_router"]["compatible_components"])
    assert hard_components == {
        "routing/compression_mixture_experts",
        "routing/routing_conditioned_compression",
        "routing/dual_compression_blend",
        "routing/route_recursion",
        "routing/adaptive_recursion",
        "routing/mixed_recursion_gate",
        "channel_mixing/moe_topk",
        "channel_mixing/moe_2expert",
        "routing/n_way_sparse_router",
        "mixing/state_space",
    }
