# Root Makefile — Build all LLM workspace components
# Automated build pipeline (Phase 4, DRY_HIGH_PERF_TODO.md)

PYTHON ?= python

.PHONY: all aria-core test clean help

all: aria-core  ## Build everything

# ── aria-core: Unified C++/CUDA kernel library ──────────────────────
aria-core:  ## Build aria_core C++ extension
	@echo "=== Building aria-core ==="
	cd aria-core && $(PYTHON) setup.py build_ext --inplace

# ── Tests ────────────────────────────────────────────────────────────
test: aria-core  ## Run all tests
	@echo "=== aria-core equivalence tests ==="
	cd aria-core && $(PYTHON) -m pytest tests/ -x -q
	@echo "=== aria-designer tests ==="
	cd aria-designer && $(PYTHON) -m pytest tests/ --ignore=tests/test_aria_features.py -x -q

test-aria-core: aria-core  ## Run only aria-core tests
	cd aria-core && $(PYTHON) -m pytest tests/ -x -q

test-designer:  ## Run only aria-designer tests
	cd aria-designer && $(PYTHON) -m pytest tests/ --ignore=tests/test_aria_features.py -x -q

# ── Clean ────────────────────────────────────────────────────────────
clean:  ## Clean all build artifacts
	cd aria-core && rm -rf build/ dist/ *.egg-info aria_core/_C* aria_core/*.so

# ── Help ─────────────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
