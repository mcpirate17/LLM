# Global Architecture Improvements & Stability Patches
**Date:** March 8, 2026

## 1. Performance & Vectorization 
- Executed a comprehensive performance optimization mapping across the system. 
- Converted standard topological Depth-First Search (DFS) iterations into faster **NumPy vectorizations** where applicable to reduce overhead.
- Profiling and metrics for these adjustments have been recorded in `PERF_FINDINGS.md`.

## 2. Global Dead-Code Elimination
- Unleashed `autoflake` and `vulture` across the entire `research/` directory namespace. 
- Aggressively stripped unused variables, orphaned endpoints, unreachable closures, and unmapped imports. 
- Handled edge cases: Patched syntax interruption errors from multi-line literal string definitions in `antipattern.py` that originally broke the AST parser.

## 3. Resolving Side-Effects of Optimization
- **Slotted Dataclass Crash (`__dict__` error):** Upgrading memory structures to `@dataclass(slots=True)` broke state telemetry because standard mapping relies on the `__dict__` attribute. We resolved this in `research/scientist/runner/_types.py` by converting iterating dictionaries to use `__dataclass_fields__` directly. 
- **Missing JSON Router Constraints (`NameError: _json_safe`):** Global pruning erroneously wiped imported utilities that child routes implicitly relied on. Repaired this by re-injecting direct namespace configurations of local `_json_safe` functions directly into `control.py` and `read.py`.

## 4. Frontend Interactive Rendering Constraints
- Applied strict visual bounds to actionable UI components in React (`Discoveries.js`, `LeaderboardRow.js`, and `DiscoveryRankings.js`).
- `Force Investigate`: Only renders if the fingerprint has not yet begun investigation (i.e. strictly screening).
- `Force Validate`: Only renders if the fingerprint has not yet begun validation. 
- `Delete`: Mutated to be strictly available exclusively on entries that are in the initial `screening` phase or inherently flagged as `failed`/`rejected`. Removed the ability to delete standard Reference Baselines.
