"""Tests for constraint visualization."""

import pytest
import yaml

from aria_designer.runtime.constraints import check_compatibility, compute_palette_constraints


# ── Test fixtures with temp component directories ────────────────────

@pytest.fixture
def components_dir(tmp_path):
    """Create a minimal component directory for testing."""
    # Component A: declares incompatibility with tag "exclusive_b"
    comp_a = tmp_path / "math" / "comp_a"
    comp_a.mkdir(parents=True)
    (comp_a / "manifest.yaml").write_text(yaml.dump({
        "id": "comp_a",
        "category": "math",
        "tags": ["tag_a"],
        "constraints": {
            "incompatible_with": ["exclusive_b"],
            "requires": [],
        },
    }))

    # Component B: has tag "exclusive_b"
    comp_b = tmp_path / "math" / "comp_b"
    comp_b.mkdir(parents=True)
    (comp_b / "manifest.yaml").write_text(yaml.dump({
        "id": "comp_b",
        "category": "math",
        "tags": ["exclusive_b"],
        "constraints": {
            "incompatible_with": [],
            "requires": [],
        },
    }))

    # Component C: requires "tag_a"
    comp_c = tmp_path / "core" / "comp_c"
    comp_c.mkdir(parents=True)
    (comp_c / "manifest.yaml").write_text(yaml.dump({
        "id": "comp_c",
        "category": "core",
        "tags": ["tag_c"],
        "constraints": {
            "incompatible_with": [],
            "requires": ["tag_a"],
        },
    }))

    # Component D: no constraints
    comp_d = tmp_path / "core" / "comp_d"
    comp_d.mkdir(parents=True)
    (comp_d / "manifest.yaml").write_text(yaml.dump({
        "id": "comp_d",
        "category": "core",
        "tags": [],
        "constraints": {
            "incompatible_with": [],
            "requires": [],
        },
    }))

    # graph_input / graph_output (IO)
    for io_id in ("graph_input", "graph_output"):
        io_dir = tmp_path / "io" / io_id
        io_dir.mkdir(parents=True)
        (io_dir / "manifest.yaml").write_text(yaml.dump({
            "id": io_id,
            "category": "io",
            "tags": [],
            "constraints": {"incompatible_with": [], "requires": []},
        }))

    return str(tmp_path)


def _make_workflow(nodes):
    """Helper to build a minimal workflow dict."""
    return {
        "workflow_id": "test",
        "nodes": [{"id": n, "component_type": n, "params": {}} for n in nodes],
        "edges": [],
    }


# ── Tests ────────────────────────────────────────────────────────────

def test_compatible_no_constraints(components_dir):
    wf = _make_workflow(["graph_input", "comp_d"])
    result = check_compatibility(wf, "comp_d", components_dir)
    assert result["compatible"] is True
    assert result["severity"] == "ok"


def test_incompatible_tag(components_dir):
    """comp_a declares incompatible_with=['exclusive_b']. If comp_b (tagged exclusive_b) is in graph, comp_a should be incompatible."""
    wf = _make_workflow(["graph_input", "comp_b"])
    result = check_compatibility(wf, "comp_a", components_dir)
    assert result["compatible"] is False
    assert any("incompatible" in r for r in result["reasons"])


def test_reverse_incompatible(components_dir):
    """If comp_a is in graph and we try to add comp_b, comp_a's constraint should block it."""
    # comp_a declares incompatible_with=['exclusive_b'], comp_b has tag 'exclusive_b'
    wf = _make_workflow(["graph_input", "comp_a"])
    result = check_compatibility(wf, "comp_b", components_dir)
    # comp_a in graph declares incompatible_with=['exclusive_b'],
    # but comp_b doesn't declare incompatible_with comp_a's tags
    # So this is one-way: comp_a blocks comp_b only if comp_a's constraint checks
    # Against comp_b's tags
    # In our implementation, we check existing nodes' incompatible_with against candidate tags
    # comp_a.incompatible_with = ['exclusive_b'], candidate comp_b has tag 'exclusive_b'
    # Wait - the check is: existing node's incompatible_with vs candidate's tags
    # comp_a.incompatible_with = ['exclusive_b'], comp_b.tags = ['exclusive_b']
    # So yes, this should be incompatible
    assert result["compatible"] is False


def test_missing_requirement(components_dir):
    """comp_c requires 'tag_a'. If tag_a is not in graph, it's incompatible."""
    wf = _make_workflow(["graph_input", "comp_d"])
    result = check_compatibility(wf, "comp_c", components_dir)
    assert result["compatible"] is False
    assert any("requires" in r for r in result["reasons"])


def test_requirement_satisfied(components_dir):
    """comp_c requires 'tag_a'. If comp_a (tagged tag_a) is in graph, it's compatible."""
    wf = _make_workflow(["graph_input", "comp_a"])
    result = check_compatibility(wf, "comp_c", components_dir)
    assert result["compatible"] is True


def test_duplicate_io_blocked(components_dir):
    """Only one graph_input should be allowed."""
    wf = _make_workflow(["graph_input", "comp_d"])
    result = check_compatibility(wf, "graph_input", components_dir)
    assert result["compatible"] is False
    assert any("graph_input" in r for r in result["reasons"])


def test_duplicate_output_blocked(components_dir):
    wf = _make_workflow(["graph_input", "graph_output"])
    result = check_compatibility(wf, "graph_output", components_dir)
    assert result["compatible"] is False


def test_unknown_component_is_compatible(components_dir):
    """Unknown components (no manifest) should be allowed."""
    wf = _make_workflow(["graph_input"])
    result = check_compatibility(wf, "nonexistent_op", components_dir)
    assert result["compatible"] is True


def test_compute_palette_constraints(components_dir):
    wf = _make_workflow(["graph_input", "comp_b"])
    all_ids = ["comp_a", "comp_b", "comp_c", "comp_d", "graph_input", "graph_output"]
    palette = compute_palette_constraints(wf, all_ids, components_dir)

    assert len(palette) == len(all_ids)
    # comp_a should be incompatible (comp_b's tag conflicts)
    assert palette["comp_a"]["compatible"] is False
    # comp_d should be compatible
    assert palette["comp_d"]["compatible"] is True
    # graph_input already in graph
    assert palette["graph_input"]["compatible"] is False


def test_empty_workflow(components_dir):
    """Empty workflow: everything should be compatible."""
    wf = {"workflow_id": "test", "nodes": [], "edges": []}
    result = check_compatibility(wf, "comp_a", components_dir)
    assert result["compatible"] is True
