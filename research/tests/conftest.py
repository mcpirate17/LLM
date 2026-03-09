"""Shared test fixtures and helpers for research/tests/."""
from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

import numpy as np
import pytest

# ── Require marker selection ─────────────────────────────────────────
# Prevents `pytest tests/` from running all 1800+ tests at once.
# Use: pytest -m unit, pytest -m native, pytest -m "unit or api", etc.

_KNOWN_MARKS = {"unit", "native", "api", "pipeline", "e2e", "designer", "slow"}


def pytest_collection_modifyitems(config, items):
    """Abort if no marker filter is specified — forces category selection."""
    marker_expr = config.getoption("-m", default="")
    if not marker_expr:
        # Allow running a single file without -m
        specified_paths = config.args
        if specified_paths and all(
            os.path.isfile(p) or (not os.path.isdir(p)) for p in specified_paths
        ):
            return  # Single file mode — allow
        pytest.exit(
            "ERROR: Running all tests at once is not allowed.\n"
            "Use a marker filter:  pytest -m unit  |  pytest -m native  |  "
            "pytest -m api  |  pytest -m pipeline  |  pytest -m e2e  |  "
            "pytest -m designer\n"
            "Or combine:  pytest -m 'unit or api'\n"
            "Or run a single file:  pytest tests/test_notebook.py",
            returncode=4,
        )

# ── Native library loading ────────────────────────────────────────────

_NATIVE_LIB_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'runtime', 'native', 'build', 'libaria_native_runtime.so'
)

_native_lib = None


def load_native_lib():
    """Load the native runtime shared library, skipping if not built."""
    global _native_lib
    if _native_lib is None:
        if not os.path.exists(_NATIVE_LIB_PATH):
            pytest.skip(f"Native library not built: {_NATIVE_LIB_PATH}")
        _native_lib = ctypes.CDLL(_NATIVE_LIB_PATH)
    return _native_lib


@pytest.fixture
def native_lib():
    """Fixture providing the loaded native runtime library."""
    return load_native_lib()


@pytest.fixture(params=[16, 128, 1024, 4096])
def array_size(request):
    """Various sizes to test vectorization edge cases."""
    return request.param


# ── Common test helpers ───────────────────────────────────────────────

def assert_close(actual: np.ndarray, expected: np.ndarray, label: str = "",
                 atol: float = 1e-5, rtol: float = 1e-5):
    """Assert arrays are close with configurable tolerance."""
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)


def make_fake_graph(op_names):
    """Create a minimal graph-like object for native runner tests."""
    nodes = {}
    for i, name in enumerate(op_names):
        node = SimpleNamespace(op_name=name)
        nodes[f"n{i}"] = node
    return SimpleNamespace(nodes=nodes)
