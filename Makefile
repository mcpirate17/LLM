# Root Makefile — Build all LLM workspace components
# Automated build pipeline (Phase 4, DRY_HIGH_PERF_TODO.md)

PYTHON ?= python

.PHONY: all aria_core test clean help guardrails-dry guardrails-dry-report

all: aria_core  ## Build everything

# ── aria_core: Unified C++/CUDA kernel library ──────────────────────
aria_core:  ## Build aria_core C++ extension
	@echo "=== Building aria_core ==="
	cd aria_core && $(PYTHON) setup.py build_ext --inplace

# ── Tests ────────────────────────────────────────────────────────────
test: aria_core  ## Run all tests
	@echo "=== aria_core equivalence tests ==="
	cd aria_core && $(PYTHON) -m pytest tests/ -x -q
	@echo "=== aria_designer tests ==="
	cd aria_designer && $(PYTHON) -m pytest tests/ --ignore=tests/test_aria_features.py -x -q

test-aria_core: aria_core  ## Run only aria_core tests
	cd aria_core && $(PYTHON) -m pytest tests/ -x -q

test-designer:  ## Run only aria_designer tests
	cd aria_designer && $(PYTHON) -m pytest tests/ --ignore=tests/test_aria_features.py -x -q

guardrails-dry:  ## Enforce DRY/language guardrails against baseline
	$(PYTHON) -m research.tools.dry_language_guardrails --strict

guardrails-dry-report:  ## Print DRY/language guardrail metrics
	$(PYTHON) -m research.tools.dry_language_guardrails

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
# ── Help ─────────────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
