# aria_core

Native C++/CUDA kernel library for Aria.

`aria_core` builds the `aria_core._C` extension used by `research` when available. Some CPU sources are also recompiled by `aria_designer/runtime` for Designer-side runtime validation.

## Build

```bash
cd /home/tim/Projects/LLM/aria_core
python setup.py build_ext --inplace
```

Restart any dashboard/API process after rebuilding so Python reloads the native extension.

## Structure

- `setup.py` — setuptools extension build definition
- `bindings/` — pybind11 binding code
- `src/cpu/` — CPU kernels and graph helpers
- `src/gpu/` — CUDA kernels
- `include/` — native headers
- `aria_core/` — Python import package and GPU helper modules
- `tests/` — native correctness and equivalence tests

## Development Notes

- Put reusable low-level kernels and native math/graph helpers here.
- Keep Python fallbacks consistent with native behavior when fallbacks exist.
- Test shape inference and finite-value behavior; small native shape bugs cascade into `research` compiler/eval failures.
- Rebuild after C++/CUDA changes before retesting `research` or `aria_designer`.

## Verification

```bash
cd /home/tim/Projects/LLM/aria_core
python -m pytest tests/ -x -q
```
