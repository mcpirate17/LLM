# Root Makefile — Build all LLM workspace components
# Automated build pipeline (Phase 4, DRY_HIGH_PERF_TODO.md)

PYTHON ?= python

.PHONY: all aria_core test test-aria_core test-designer test-research test-integration clean clean-junk clean-docs clean-all help guardrails-dry guardrails-dry-report perf-summary governance-check governance-audit profile-hotpaths profile-screening-hotpaths profile-screening-hotpaths-quick

all: aria_core  ## Build everything

# ── aria_core: Unified C++/CUDA kernel library ──────────────────────
aria_core:  ## Build aria_core C++ extension
	@echo "=== Building aria_core ==="
	cd aria_core && $(PYTHON) setup.py build_ext --inplace

# ── Tests ────────────────────────────────────────────────────────────
test: aria_core  ## Run all tests (aria_core + aria_designer + research)
	@echo "=== aria_core equivalence tests ==="
	cd aria_core && $(PYTHON) -m pytest tests/ -x -q
	@echo "=== aria_designer tests ==="
	cd aria_designer && $(PYTHON) -m pytest tests/ --ignore=tests/test_aria_features.py -x -q
	@echo "=== research tests (unit+api) ==="
	cd research && $(PYTHON) -m pytest tests/ -m "unit or api" -x --tb=short

test-aria_core: aria_core  ## Run only aria_core tests
	cd aria_core && $(PYTHON) -m pytest tests/ -x -q

test-designer:  ## Run only aria_designer tests
	cd aria_designer && $(PYTHON) -m pytest tests/ --ignore=tests/test_aria_features.py -x -q

test-research:  ## Run only research tests (unit+api)
	cd research && $(PYTHON) -m pytest tests/ -m "unit or api" -x --tb=short

test-research-all:  ## Run all research test markers
	cd research && $(PYTHON) -m pytest tests/ -m "unit or api" -x --tb=short
	cd research && $(PYTHON) -m pytest tests/ -m pipeline --tb=short
	cd research && $(PYTHON) -m pytest tests/ -m native --tb=short
	cd research && $(PYTHON) -m pytest tests/ -m designer --tb=short

test-integration:  ## Run cross-project observability/bridge contract tests
	$(PYTHON) -m pytest research/tests/test_api_integration.py -k experiment_failures -x --tb=short
	cd aria_designer && $(PYTHON) -m pytest tests/test_api.py -k "structured_error_details or eval_run_store_persists_to_database" -x --tb=short

guardrails-dry:  ## Enforce DRY/language guardrails against baseline
	$(PYTHON) -m research.tools.dry_language_guardrails --strict

guardrails-dry-report:  ## Print DRY/language guardrail metrics
	$(PYTHON) -m research.tools.dry_language_guardrails

perf-summary:  ## Print recent shared performance artifacts
	$(PYTHON) -m research.tools.perf_summary --limit 10

governance-check:  ## Block on guardrail violations from GLOBAL_DEV_PROMPT
	$(PYTHON) conductor/guardrail_audit.py --check --markdown-out tasks/audit/latest_guardrail_report.md --json-out tasks/audit/latest_guardrail_report.json

governance-audit:  ## Generate the full A-G audit report artifact
	$(PYTHON) conductor/guardrail_audit.py --markdown-out tasks/audit/latest_guardrail_report.md --json-out tasks/audit/latest_guardrail_report.json

profile-hotpaths:  ## Run lightweight benchmark/profiling hooks for CI
	$(PYTHON) conductor/profile_hotpaths.py --json-out tasks/audit/profile_hotpaths.json

profile-screening-hotpaths:  ## Run standard targeted experiment-screening hotpath benchmark
	$(PYTHON) -m research.tools.profile_screening_hotpaths --fixture standard --json-out tasks/audit/screening_hotpaths.json

profile-screening-hotpaths-quick:  ## Run quick targeted experiment-screening hotpath benchmark
	$(PYTHON) -m research.tools.profile_screening_hotpaths --fixture quick --json-out tasks/audit/screening_hotpaths_quick.json

dead: ## Standing dead-code detector
	@mkdir -p tasks/audit
	vulture research/ aria_core/ aria_designer/ vulture_whitelist.py \
	  --min-confidence 80 \
	  --exclude "*/.venv/*,*/node_modules/*,*/__pycache__/*,*/.run/*,tests/,migrations/" \
	  | tee tasks/audit/dead_code.txt
	@echo "Dead code candidates: $$(wc -l < tasks/audit/dead_code.txt)"

dupes: ## Duplicate code detector
	@mkdir -p tasks/audit
	pylint research/ aria_core/ aria_designer/ \
	  --disable=all \
	  --enable=duplicate-code \
	  --min-similarity-lines=10 2>&1 | tee tasks/audit/duplication.txt

# ── Clean ────────────────────────────────────────────────────────────
clean:  ## Clean all build artifacts
	cd aria_core && rm -rf build/ dist/ *.egg-info aria_core/_C* aria_core/*.so

clean-junk:  ## Remove all cache files, orphaned DB records and temp logs
	@echo "=== Removing __pycache__ and .pytest_cache ==="
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "=== Cleaning Database Orphans ==="
	$(PYTHON) -m research.tools.db_integrity_cleanup
	@echo "=== Cleaning Designer logs ==="
	rm -f aria_designer/.run/*.log

REPORT_RETENTION_DAYS ?= 14
PERF_RETENTION_DAYS ?= 30

clean-docs:  ## Remove stale docs, old reports, and temp files
	@echo "=== Removing root-level junk ==="
	rm -f "new 2.txt" ":memory:" ":memory:-shm" ":memory:-wal"
	rm -f audit_report.md audit_report.json
	rm -rf audit_results/
	@echo "=== Pruning reports older than $(REPORT_RETENTION_DAYS) days ==="
	find research/reports/ -name "report_*.md" -mtime +$(REPORT_RETENTION_DAYS) -delete 2>/dev/null || true
	find research/reports/ -name "exploration_*" -mtime +$(REPORT_RETENTION_DAYS) -delete 2>/dev/null || true
	@echo "=== Pruning perf artifacts older than $(PERF_RETENTION_DAYS) days ==="
	find research/perf_artifacts/ -mindepth 2 -type d -mtime +$(PERF_RETENTION_DAYS) -exec rm -rf {} + 2>/dev/null || true
	@echo "=== Cleaning logs ==="
	rm -f research/aria_dashboard.log research/aria_dashboard.log.*
	find aria_designer/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
	@echo "Reports remaining: $$(find research/reports/ -name 'report_*.md' 2>/dev/null | wc -l)"

clean-all: clean clean-junk clean-docs  ## Full cleanup: build + cache + docs

# ── Help ─────────────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
