# Tri-Agent Pytest Execution & Fixing Strategy

Since we just completed a massive "God File Split" refactor (touching nearly every critical system in the `research/scientist` and `synthesis` packages), running `pytest` will yield hundreds of `ModuleNotFoundError`s and API/import mismatch errors. 

To resolve these efficiently, we can parallelize the test-fixing effort across **3 different AI agents** (running in separate chat windows or subagent threads). Here is the strategic division of labor:

---

## 🌩️ Agent 1: The Scientist Core (Highest Complexity)
**Target:** `./research/scientist/tests`
**Mission:** Fix the direct fallout of the God File splits.
**Context to give this agent:**
- Explain that `api.py`, `notebook.py`, `execution.py`, `analytics.py`, `persona.py`, `continuous.py`, `native_runner.py`, and `results.py` were just split into sub-modules.
- Command to run: `pytest research/scientist/tests/`
**Primary Tasks:**
1. Fix all `from research.scientist.X import Y` statements to pull from `research.scientist.X.submodule` or the new `__init__.py` re-exports.
2. Fix broken mocks where test files specifically mocked god-file internal states or monolithic classes.

---

## ⚙️ Agent 2: Engine & Synthesis (Deep Runtime Logic)
**Targets:** `./research/tests/` & `./aria_core/tests/`
**Mission:** Restore the core underlying inference, compiler, and orchestrator tests.
**Context to give this agent:**
- Explain the `synthesis/compiler.py` split.
- Note that C/Rust native runners rely on the new ABI layout (`research/scientist/native/abi.py`).
- Command to run: `pytest research/tests/ aria_core/tests/`
**Primary Tasks:**
1. Ensure the unified graph execution tests pass.
2. Resolve any missing fixture issues now that the autograd and ABI backends were shuffled.
3. Validate that standard core utilities still resolve correctly.

---

## 🎨 Agent 3: Downstream Ecosystems (Hydra, LA3, & Designer)
**Targets:** `./HYDRA/tests`, `./LA3/tests`, `./aria_designer/tests`
**Mission:** Fix external integrations that depended on the old monolithic designs.
**Context to give this agent:**
- Explain that `api.py` is now a set of Flask blueprints, and `notebook.py` uses DAL modules.
- Command to run: `pytest HYDRA/tests/ aria_designer/tests/`
**Primary Tasks:**
1. Fix Hydra trainer integrations (e.g., if Hydra trainers expected `ExperimentRunner` from `runner.py` which was deleted).
2. Fix `aria_designer` UI & workflow tests that relied on the old API payloads.
3. Adjust test requests to target the new blueprint routings.

---

## 🚀 Execution Rules for All Agents:
1. **Fix Imports First:** Agents must grep for `ModuleNotFoundError` before touching logic.
2. **Do Not Revert File Splits:** If an agent can't figure out an import, they must read the new modular files, **not** recreate the old god classes.
3. **Patience & Iteration:** Each agent should run `pytest <their_target> --maxfail=5` to fix issues in small, isolated batches.
