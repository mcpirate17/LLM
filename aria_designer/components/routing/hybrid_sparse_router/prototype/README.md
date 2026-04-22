# Hybrid Sparse Router Prototype

This directory remains as the designer-facing prototype surface, but it no
longer carries a second independent implementation.

The local headers and sources are thin forwarders to the canonical native
implementation under `research/runtime/native/`. That keeps the historical
prototype paths stable without allowing the C++ router logic to drift into
parallel copies.

Forwarded files:

- `router_distilled.hpp`
- `router_distilled.cpp`
- `sparse_hybrid_router.hpp`
- `sparse_hybrid_router.cpp`
