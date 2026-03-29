# Audit Report

Scanned `1038` code files across `758` Python files.

### A. Critical problems
- `aria_core/bindings/bindings.cpp` [critical] File is 1735 lines (>1250).
- `aria_designer/api/app/aria_patch_postprocess.py::_auto_connect_added_nodes` [critical] Function is 129 lines (>100).
- `aria_designer/api/app/conversation.py::_build_difficulty_routed_patch` [critical] Function is 103 lines (>100).
- `aria_designer/api/app/routers/aria.py` [critical] File is 1499 lines (>1250).
- `aria_designer/api/app/routers/aria.py::apply_patch` [critical] Function is 110 lines (>100).
- `aria_designer/api/app/routers/eval.py::evaluate_workflow_stream` [critical] Function is 210 lines (>100).
- `aria_designer/api/app/routers/eval.py::event_stream` [critical] Function is 196 lines (>100).
- `aria_designer/api/app/suggestions.py::_research_score_delta` [critical] Function is 103 lines (>100).
- `aria_designer/api/app/suggestions.py::_suggest_by_name` [critical] Function is 114 lines (>100).
- `aria_designer/api/app/suggestions.py::_suggest_for_leaf_nodes` [critical] Function is 104 lines (>100).
- `aria_designer/runtime/bridge.py::get_component_execution_capability` [critical] Function is 118 lines (>100).
- `aria_designer/runtime/subgraph.py::extract_block` [critical] Function is 112 lines (>100).
- `aria_designer/tests/test_api.py::test_ai_design_refine_evaluate_records_lineage` [critical] Function is 125 lines (>100).
- `aria_designer/tests/test_bridge.py::test_evaluate_workflow_uses_behavioral_fingerprint_for_novelty` [critical] Function is 105 lines (>100).
- `aria_designer/tests/test_compile_all_components.py::_build_workflow` [critical] Function is 111 lines (>100).

### B. Exact targets
- file path: `aria_core/bindings/bindings.cpp`
  symbol/function/class name: `- `
  estimated severity: `critical`
  why it is bad: File is 1735 lines (>1250).
  exact recommendation: Split by responsibility boundaries and isolate orchestration from pure logic.
- file path: `aria_designer/api/app/aria_patch_postprocess.py`
  symbol/function/class name: `_auto_connect_added_nodes `
  estimated severity: `critical`
  why it is bad: Function is 129 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/api/app/conversation.py`
  symbol/function/class name: `_build_difficulty_routed_patch `
  estimated severity: `critical`
  why it is bad: Function is 103 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/api/app/routers/aria.py`
  symbol/function/class name: `- `
  estimated severity: `critical`
  why it is bad: File is 1499 lines (>1250).
  exact recommendation: Split by responsibility boundaries and isolate orchestration from pure logic.
- file path: `aria_designer/api/app/routers/aria.py`
  symbol/function/class name: `apply_patch `
  estimated severity: `critical`
  why it is bad: Function is 110 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/api/app/routers/eval.py`
  symbol/function/class name: `evaluate_workflow_stream `
  estimated severity: `critical`
  why it is bad: Function is 210 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/api/app/routers/eval.py`
  symbol/function/class name: `event_stream `
  estimated severity: `critical`
  why it is bad: Function is 196 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/api/app/suggestions.py`
  symbol/function/class name: `_research_score_delta `
  estimated severity: `critical`
  why it is bad: Function is 103 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/api/app/suggestions.py`
  symbol/function/class name: `_suggest_by_name `
  estimated severity: `critical`
  why it is bad: Function is 114 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/api/app/suggestions.py`
  symbol/function/class name: `_suggest_for_leaf_nodes `
  estimated severity: `critical`
  why it is bad: Function is 104 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/runtime/bridge.py`
  symbol/function/class name: `get_component_execution_capability `
  estimated severity: `critical`
  why it is bad: Function is 118 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/runtime/subgraph.py`
  symbol/function/class name: `extract_block `
  estimated severity: `critical`
  why it is bad: Function is 112 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/tests/test_api.py`
  symbol/function/class name: `test_ai_design_refine_evaluate_records_lineage `
  estimated severity: `critical`
  why it is bad: Function is 125 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/tests/test_bridge.py`
  symbol/function/class name: `test_evaluate_workflow_uses_behavioral_fingerprint_for_novelty `
  estimated severity: `critical`
  why it is bad: Function is 105 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/tests/test_compile_all_components.py`
  symbol/function/class name: `_build_workflow `
  estimated severity: `critical`
  why it is bad: Function is 111 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `aria_designer/tools/bootstrap_components.py`
  symbol/function/class name: `main `
  estimated severity: `critical`
  why it is bad: Function is 117 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `research/arch_builder.py`
  symbol/function/class name: `- `
  estimated severity: `critical`
  why it is bad: File is 1403 lines (>1250).
  exact recommendation: Split by responsibility boundaries and isolate orchestration from pure logic.
- file path: `research/dashboard/src/App.js`
  symbol/function/class name: `- `
  estimated severity: `critical`
  why it is bad: File is 1790 lines (>1250).
  exact recommendation: Split by responsibility boundaries and isolate orchestration from pure logic.
- file path: `research/eval/diagnostic_tasks.py`
  symbol/function/class name: `_train_and_eval_task `
  estimated severity: `critical`
  why it is bad: Function is 133 lines (>100).
  exact recommendation: Split by decision blocks and side-effect boundaries.
- file path: `research/eval/fingerprint.py`
  symbol/function/class name: `- `
  estimated severity: `critical`
  why it is bad: File is 1343 lines (>1250).
  exact recommendation: Split by responsibility boundaries and isolate orchestration from pure logic.

### C. Fast wins
- `aria_designer/api/app/aria_patch_postprocess.py`: Flatten control flow and extract pure helpers.
- `aria_designer/api/app/conversation.py`: Flatten control flow and extract pure helpers.
- `aria_designer/api/app/patcher.py`: Flatten control flow and extract pure helpers.
- `aria_designer/api/app/suggestions.py`: Flatten control flow and extract pure helpers.
- `aria_designer/components/data_io/file_writer/kernel_fallback.py`: Flatten control flow and extract pure helpers.
- `aria_designer/runtime/constraints.py`: Flatten control flow and extract pure helpers.
- `aria_designer/runtime/native_executor.py`: Flatten control flow and extract pure helpers.
- `aria_designer/runtime/native_executor.py`: Flatten control flow and extract pure helpers.
- `aria_designer/runtime/profiler.py`: Flatten control flow and extract pure helpers.
- `aria_designer/tests/test_component_contracts.py`: Flatten control flow and extract pure helpers.

### D. Structural rewrites
- `aria_core/bindings/bindings.cpp`: File is 1735 lines (>1250). Split by responsibility boundaries and isolate orchestration from pure logic.
- `aria_designer/api/app/aria_patch_postprocess.py`: Function is 129 lines (>100). Split by decision blocks and side-effect boundaries.
- `aria_designer/api/app/conversation.py`: Function is 103 lines (>100). Split by decision blocks and side-effect boundaries.
- `aria_designer/api/app/routers/aria.py`: File is 1499 lines (>1250). Split by responsibility boundaries and isolate orchestration from pure logic.
- `aria_designer/api/app/routers/aria.py`: Function is 110 lines (>100). Split by decision blocks and side-effect boundaries.
- `aria_designer/api/app/routers/eval.py`: Function is 210 lines (>100). Split by decision blocks and side-effect boundaries.
- `aria_designer/api/app/routers/eval.py`: Function is 196 lines (>100). Split by decision blocks and side-effect boundaries.
- `aria_designer/api/app/suggestions.py`: Function is 103 lines (>100). Split by decision blocks and side-effect boundaries.
- `aria_designer/api/app/suggestions.py`: Function is 114 lines (>100). Split by decision blocks and side-effect boundaries.
- `aria_designer/api/app/suggestions.py`: Function is 104 lines (>100). Split by decision blocks and side-effect boundaries.

### E. Performance upgrades by language
- Python
  - `aria_designer/api/app/aria_patch_postprocess.py` `_auto_connect_added_nodes`: Flatten control flow and extract pure helpers.
  - `aria_designer/api/app/aria_patch_postprocess.py` `_auto_layout_workflow`: Vectorize with NumPy/PyTorch or move the hotspot into C/C++/Rust/Cython if profiling confirms it.
  - `aria_designer/api/app/benchmark_targets.py` `build_benchmark_analysis`: Vectorize with NumPy/PyTorch or move the hotspot into C/C++/Rust/Cython if profiling confirms it.
  - `aria_designer/api/app/conversation.py` `_build_patch_from_pattern`: Vectorize with NumPy/PyTorch or move the hotspot into C/C++/Rust/Cython if profiling confirms it.
  - `aria_designer/api/app/conversation.py` `_classify_message`: Flatten control flow and extract pure helpers.
  - `aria_designer/api/app/conversation.py` `_respond_resolved_concepts`: Vectorize with NumPy/PyTorch or move the hotspot into C/C++/Rust/Cython if profiling confirms it.
  - `aria_designer/api/app/help_content.py` `_leaf_ids_in_categories`: Vectorize with NumPy/PyTorch or move the hotspot into C/C++/Rust/Cython if profiling confirms it.
  - `aria_designer/api/app/intent_parser.py` `_topological_nodes`: Vectorize with NumPy/PyTorch or move the hotspot into C/C++/Rust/Cython if profiling confirms it.
  - `aria_designer/api/app/mutation.py` `_target_nodes`: Vectorize with NumPy/PyTorch or move the hotspot into C/C++/Rust/Cython if profiling confirms it.
  - `aria_designer/api/app/patcher.py` `apply_patch_ops`: Flatten control flow and extract pure helpers.
- JavaScript/TypeScript
  - Add ESLint/unused-export enforcement next; this pass does not yet scan JS/TS symbol usage deeply.
- Database/SQL
  - No SQL-specific automated audit added in this pass; add query-plan/index checks separately.
- Rust/C/C++/Cython opportunities
  - Prioritize files flagged as `native_hotspot_candidate` after benchmark confirmation.

### F. Proposed patch plan
1. delete dead code
2. split god files
3. split god functions
4. optimize hot paths
5. optimize database access
6. reduce dependency and bundle bloat
7. move justified hotspots to compiled/native code
8. benchmark before/after

### G. Proof
- files scanned: 1038
- dead code hits reported by vulture: 2
- duplicate-code hits reported by pylint: 77
- critical findings: 260
- high findings: 384
