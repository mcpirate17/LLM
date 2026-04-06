# Intelligent Router Prototype

This directory vendors the standalone prototype from `/home/tim/intelligent_router_proto`
into the research-owned native runtime tree.

Files copied here:

- `include/intelligent_router/router_distilled.hpp`
- `include/intelligent_router/sparse_hybrid_router.hpp`
- `src/intelligent_router_router_distilled.cpp`
- `src/intelligent_router_sparse_hybrid_router.cpp`

Purpose:

- keep the promoted hybrid sparse router code inside the monorepo
- make the prototype available to future native integration work
- avoid depending on an external non-repo directory for the reference implementation
