.PHONY: all setup api ui dev dev-stop test test-api test-manifests test-components test-runtime test-all lint validate bootstrap clean

all: setup bootstrap validate

# ── Setup ────────────────────────────────────────────────────────────

setup: setup-api setup-ui

setup-api:
	cd api && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt -q

setup-ui:
	cd ui && npm install

# ── Development Servers ──────────────────────────────────────────────

api:
	cd api && .venv/bin/uvicorn app.main:app --reload --port 8091

ui:
	cd ui && npm run dev

dev:
	@./tools/dev_up.sh

dev-stop:
	@./tools/dev_down.sh

# ── Bootstrap Components ─────────────────────────────────────────────

bootstrap:
	python -m tools.bootstrap_components

# ── Build ────────────────────────────────────────────────────────────

build-ui:
	cd ui && npm run build

build-native:
	@cmake -S runtime/tests -B runtime/tests/build -DCMAKE_BUILD_TYPE=Release 2>/dev/null
	@cmake --build runtime/tests/build -j$$(nproc) 2>&1 | tail -3

build: build-ui build-native

# ── Test ─────────────────────────────────────────────────────────────

test: test-api test-manifests test-components

test-api:
	python -m pytest tests/ -x --tb=short

test-manifests:
	python tools/validate_manifests.py

test-components:
	python -m pytest components/ -x --tb=short -q

test-runtime:
	python -m pytest runtime/tests/ -x --tb=short

test-native: build-native
	@cd runtime/tests/build && ctest --output-on-failure

test-e2e:
	cd e2e && npx playwright test --config=playwright.config.js

test-all: test test-runtime test-native

# ── Lint ─────────────────────────────────────────────────────────────

lint:
	ruff check api/ runtime/ tools/

# ── Validate ─────────────────────────────────────────────────────────

validate: test-manifests

# ── Clean ────────────────────────────────────────────────────────────

clean:
	rm -rf api/.venv api/aria_designer.db api/__pycache__
	rm -rf ui/node_modules ui/dist
	rm -rf runtime/lib/
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
