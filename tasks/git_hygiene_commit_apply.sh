#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

run_cmd() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  "$@"
}

confirm() {
  local prompt="$1"
  local reply
  read -r -p "$prompt [y/N] " reply
  [[ "$reply" =~ ^[Yy]([Ee][Ss])?$ ]]
}

stage_block() {
  local label="$1"
  shift
  echo
  echo "Stage block: $label"
  if confirm "Run git add for this block?"; then
    run_cmd git add "$@"
  else
    echo "Skipped staging block: $label"
  fi
}

commit_block() {
  local message="$1"
  echo
  echo "Commit message: $message"
  if confirm "Create this commit now?"; then
    run_cmd git commit -m "$message"
  else
    echo "Skipped commit: $message"
  fi
}

echo "Interactive git hygiene apply script"
echo "Repo: $ROOT_DIR"
echo
echo "This script stages and commits in the planned batches."
echo "It asks before every git add and every git commit."
echo "Special case: research/scientist/intelligence/ml_corpus.py is ambiguous"
echo "and is intentionally not auto-staged here."

stage_block \
  "git hygiene ignore rules" \
  .gitignore
commit_block "chore(git): ignore local db backups and induction probe outputs"

stage_block \
  "runner native-boundary refactors" \
  research/scientist/runner/execution_training.py \
  research/scientist/runner/execution_training_native_boundary.py \
  research/scientist/runner/execution_investigation.py \
  research/scientist/runner/execution_investigation_scoring.py \
  research/scientist/runner/execution_screening.py \
  research/scientist/runner/execution_screening_graphs.py \
  research/scientist/runner/results_auto_escalate_phase7.py \
  research/scientist/runner/auto_escalate_data.py \
  research/scientist/runner/auto_escalate_flow.py \
  research/scientist/runner/_helpers.py \
  research/scientist/runner/execution_experiment_phase3.py \
  research/tests/test_execution_training_native_boundary.py \
  research/tests/test_execution_investigation_scoring.py \
  research/tests/test_execution_screening_graphs.py \
  research/tests/test_execution_screening_imports.py \
  research/tests/test_pipeline_integration.py
commit_block "refactor(runner): split training, investigation, and screening native boundaries"

stage_block \
  "notebook and api ownership splits" \
  research/scientist/notebook/notebook_misc.py \
  research/scientist/notebook/notebook_core.py \
  research/scientist/notebook/notebook_leaderboard.py \
  research/scientist/notebook/notebook_programs.py \
  research/scientist/notebook/_shared.py \
  research/scientist/notebook/program_query_views.py \
  research/scientist/notebook/program_writes.py \
  research/scientist/notebook/program_provenance.py \
  research/scientist/notebook/leaderboard_maintenance.py \
  research/scientist/api_routes/experiments_bp.py \
  research/scientist/api_routes/leaderboard_bp.py \
  research/scientist/api_routes/observability_bp.py \
  research/scientist/api_routes/reporting_bp.py \
  research/scientist/api_routes/strategy_bp.py \
  research/scientist/api_routes/system_bp.py \
  research/scientist/api_routes/_strategy_recommendations.py \
  research/scientist/api_routes/_experiment_launch.py \
  research/scientist/api_routes/_observability_core.py \
  research/scientist/trust_policy.py \
  research/tests/test_notebook.py \
  research/tests/test_api_integration.py \
  research/tests/test_discoveries_api.py \
  research/tests/test_observability_api.py
commit_block "refactor(scientist): split notebook persistence and api route cores"

stage_block \
  "eval native runtime and telemetry" \
  aria_designer/api/app/routers/eval.py \
  aria_designer/runtime/bridge.py \
  research/eval/_eval_native.cpp \
  research/eval/_eval_native.py \
  research/eval/_runner_native.cpp \
  research/eval/_runner_native.py \
  research/eval/cross_task_eval.py \
  research/eval/sandbox.py \
  research/eval/training_core.py \
  research/eval/utils.py \
  research/eval/routing_telemetry.py \
  research/scientist/native/core.py \
  research/scientist/native/dispatch.py \
  research/runtime/native/rust/aria-scheduler/src/corpus.rs \
  research/runtime/native/rust/aria-scheduler/src/executor.rs \
  research/runtime/native/rust/aria-scheduler/src/python_bridge.rs \
  research/tests/test_eval_runner_native.py \
  research/tests/test_native_core_import_retry.py \
  research/tests/test_native_multi_input_graph_dispatch.py \
  research/tests/test_rust_backward.py \
  research/tests/test_subgraph_dispatch.py \
  research/tests/test_interpretability_evals.py
commit_block "perf(eval): consolidate native runner, telemetry, and runtime bridge changes"

echo
echo "Manual decision needed: research/scientist/intelligence/ml_corpus.py"
echo "Choose one:"
echo "  1. Stage it with routing/compiler work"
echo "  2. Stage it with corpus/provenance work"
echo "  3. Leave it unstaged and handle manually"
read -r -p "Selection [1/2/3]: " ml_choice

if [[ "$ml_choice" == "1" ]]; then
  ML_CORPUS_PATHS=(research/scientist/intelligence/ml_corpus.py)
elif [[ "$ml_choice" == "2" ]]; then
  ML_CORPUS_PATHS=(research/scientist/intelligence/ml_corpus.py)
else
  ML_CORPUS_PATHS=()
fi

stage_block \
  "synthesis router compiler changes" \
  "${ML_CORPUS_PATHS[@]}" \
  research/synthesis/_templates_routing.py \
  research/synthesis/compiled_model.py \
  research/synthesis/compiled_op.py \
  research/synthesis/compiled_op_params.py \
  research/synthesis/compiled_op_runtime.py \
  research/synthesis/compiler_ops_routing.py \
  research/synthesis/primitives.py \
  research/synthesis/grammar.py \
  research/tests/test_hybrid_sparse_router_integration.py \
  research/tests/test_multiscale_rich_lane_router_audit.py \
  research/tests/test_comparative_anatomy_routing_templates.py \
  research/tests/test_multiscale_catalogue.py \
  research/tests/test_multiscale_mechanisms.py \
  research/tests/test_multiscale_phase5_schedule.py \
  research/tests/test_observable_three_lane_router.py \
  research/tests/test_routing_template_portfolio_benchmark.py \
  research/tests/test_routing_template_variants.py \
  research/tools/audit_multiscale_rich_lane_router.py \
  research/tools/audit_multiscale_rich_lane_router_phase2.py \
  research/tools/audit_multiscale_rich_lane_router_phase4.py \
  research/tools/audit_multiscale_rich_lane_router_phase5.py \
  research/tools/benchmark_routing_template_portfolio.py \
  research/tools/comparative_anatomy_routing_templates.py \
  research/tools/confirm_multiscale_rich_lane_router_winner.py \
  research/tools/multiscale_catalogue.py \
  research/tools/multiscale_mechanisms.py \
  research/tools/routing_template_variants.py \
  research/tools/run_observable_three_lane_router.py
commit_block "feat(routing): land multiscale, template, and router analysis updates"

CORPUS_EXTRA=()
if [[ "$ml_choice" == "2" ]]; then
  CORPUS_EXTRA=(research/scientist/intelligence/ml_corpus.py)
fi

stage_block \
  "corpus provenance and trust-aware dedup" \
  "${CORPUS_EXTRA[@]}" \
  research/tests/_ml_corpus_test_support.py \
  research/tests/test_ml_corpus_dedup.py \
  research/tests/test_deduped_analytics.py \
  research/tests/test_backfill_template_generation.py \
  research/tools/backfill_templates.py \
  research/tools/explore_under_observed.py \
  research/tools/backfill_provenance_labels.py
commit_block "feat(corpus): add provenance-aware backfill and trust-aware dedup"

stage_block \
  "makefile perf targets" \
  Makefile
commit_block "chore(perf): add screening hotpath make targets"

stage_block \
  "dashboard ui updates" \
  research/dashboard/src/components/ComponentAnalyticsDashboard.js \
  research/dashboard/src/components/ControlPanel.js \
  research/dashboard/src/components/DecisionTraces.js \
  research/dashboard/src/components/Discoveries.js \
  research/dashboard/src/components/TrendCharts.js \
  research/dashboard/src/components/app/AppShellShared.jsx \
  research/dashboard/src/hooks/useAriaData.js \
  research/dashboard/src/hooks/useProgramData.js
commit_block "feat(dashboard): update analytics, traces, and data hooks"

stage_block \
  "docs and induction native probe workspace" \
  research/docs/component_catalogue.csv \
  tasks/induction_native_probe/MIGRATION_CHECKLIST.md \
  tasks/induction_native_probe/README.md \
  tasks/induction_native_probe/__init__.py \
  tasks/induction_native_probe/bench_fast_induction_probe.py \
  tasks/induction_native_probe/fast_induction_probe.py \
  tasks/induction_native_probe/native_induction_probe.cpp
commit_block "docs(tasks): add induction native probe workspace and catalogue data"

echo
echo "Plan complete."
echo "Review remaining status with: git status --short"
