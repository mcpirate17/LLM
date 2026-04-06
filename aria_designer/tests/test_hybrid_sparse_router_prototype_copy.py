from __future__ import annotations

from pathlib import Path


def test_hybrid_sparse_router_prototype_code_is_vendored_into_designer():
    prototype_dir = (
        Path(__file__).resolve().parent.parent
        / "components"
        / "routing"
        / "hybrid_sparse_router"
        / "prototype"
    )
    expected = {
        "README.md",
        "router_distilled.hpp",
        "router_distilled.cpp",
        "sparse_hybrid_router.hpp",
        "sparse_hybrid_router.cpp",
    }
    assert expected <= {path.name for path in prototype_dir.iterdir()}
