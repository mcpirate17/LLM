from __future__ import annotations

from pathlib import Path


def test_hybrid_sparse_router_prototype_code_is_vendored_into_research_native():
    root = Path(__file__).resolve().parent.parent / "runtime" / "native"
    expected = [
        root / "include" / "intelligent_router" / "router_distilled.hpp",
        root / "include" / "intelligent_router" / "sparse_hybrid_router.hpp",
        root / "src" / "intelligent_router_router_distilled.cpp",
        root / "src" / "intelligent_router_sparse_hybrid_router.cpp",
        root / "intelligent_router_proto.md",
    ]
    for path in expected:
        assert path.exists(), f"Missing vendored prototype file: {path}"
