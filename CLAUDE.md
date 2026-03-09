# CLAUDE.md — Code Quality & Performance Standards

## Core Philosophy

- **Performance First:** Prioritize execution speed and memory efficiency. Prefer compiled implementations (Rust, C++, Cython) over pure Python for any computational bottleneck.
- **Zero Duplication:** Before writing new logic, check existing files. If logic exists elsewhere, refactor to a shared utility — never copy-paste.
- **Zero Waste:** Minimize allocations, maximize CPU cache hits, and eliminate dead code (unused imports, variables, functions) immediately.
---
## Architectural Rules

| Rule | Constraint |
|---|---|
| Max function length | 100 lines — decompose into pure, testable sub-functions |
| Max file length | 1250 lines — split into `src/core/`, `src/utils/`, `src/engines/` |
| God objects | Forbidden |
| Side effects | Favor immutable data; use in-place mutations (`numpy out=`) only in hot inner loops |
---
## Python Performance

### Strings
Use `.join()` for concatenation — never `+` in a loop (`O(n)` vs `O(n²)`).

### Iteration
- Prefer **list/dict comprehensions** over `for` loops for speed.
- Prefer **generator expressions** over comprehensions for large datasets (`O(1)` memory).

### Variable Scoping
Localize attribute lookups in tight loops to avoid repeated `LOAD_ATTR` overhead.

```python
# Slow — dot-lookup on every iteration
for x in data: math.sqrt(x)

# Fast — single import, no repeated lookup
from math import sqrt
result = [sqrt(x) for x in data]
```

### Data Structures
- **Set/dict for membership tests** — `x in set_` is `O(1)` vs `x in list_` is `O(n)`. Convert to `set` before any repeated lookups.
- **`collections.deque`** for queue patterns — `O(1)` append/popleft vs list's `O(n)` `pop(0)`.
- **Preallocate lists** — `[None] * n` then assign by index, vs growing with `.append()` in known-size loops.

### Control Flow
- **EAFP over LBYL** — `try/except` is faster than `if key in dict` when misses are rare, due to Python's optimized exception machinery.
- **Dict dispatch over if/elif chains** — constant-time lookup vs linear scanning for op routing (e.g. primitive registry dispatch).

### Imports
- **Lazy imports** for heavy modules (`torch`, `numpy`) in CLI/startup paths — defer with `import` inside function body to speed up cold start.

---

## Memory & Hardware

- **NumPy Contiguity:** Ensure arrays are C-contiguous for row-major access (default) or F-contiguous for column-major, to maximize cache efficiency.
- **Memory Mapping:** Use `mmap` or `numpy.memmap` for multi-gigabyte files — access on-disk data without loading it fully into RAM.
- **Bytecode Inspection:** Use `dis.dis()` on critical paths. Excess `BINARY_SUBSCR` or `LOAD_ATTR` calls are a signal to refactor.
- **Class Memory:** Use `__slots__` in all Python classes to reduce per-instance overhead and accelerate attribute access.
- **`memoryview`** for buffer slicing — avoids copying when passing sub-arrays to C/Cython extensions.
- **Avoid unnecessary `.copy()`** — use NumPy views and slices where mutation safety permits.
---

## Technical Preferences

| Concern | Preferred Approach |
|---|---|
| Computational bottlenecks | Rust (PyO3) or C++ (pybind11) — trigger if core logic exceeds ~10ms/call |
| Vectorization | NumPy / SciPy broadcasting — avoid explicit Python loops |
| JIT compilation | Numba `@njit` for numerical functions that resist full vectorization |
| Expensive deterministic calls | `functools.lru_cache` or `functools.cache` |
| CPU-bound parallelism | `ProcessPoolExecutor` — bypasses GIL for compute-heavy work |
| I/O-bound parallelism | `ThreadPoolExecutor` — lightweight concurrency for network/disk ops |
---

## Implementation Checklist

- [ ] No duplication — checked existing codebase before writing new logic
- [ ] Dead code removed — no unused imports, variables, or functions
- [ ] Type hints applied — all public functions and method signatures annotated
- [ ] Profiling considered — `cProfile` or `line_profiler` recommended for any new hot path
---

## Environment

- **Package manager:** `uv`
- **Python:** 3.12+ (current: 3.12.3)
- **Architecture reference:** NotebookLM via MCP server for specs and benchmark targets
