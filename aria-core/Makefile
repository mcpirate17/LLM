PYTHON ?= python
VENV := /home/tim/venvs/llm/bin/activate

.PHONY: build clean test install

build:
	@echo "Building aria_core C++ extension..."
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf build/ dist/ *.egg-info aria_core/_C* aria_core/*.so

install: build
	pip install -e .

test: build
	$(PYTHON) -m pytest tests/ -x -q

verify: build
	$(PYTHON) tests/test_equivalence.py
