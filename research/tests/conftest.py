"""Shared test fixtures and helpers for research/tests/."""

from __future__ import annotations

import ctypes
import gc
import os
from types import SimpleNamespace

# ── CPU thread hygiene under xdist ───────────────────────────────────
# Without this every worker initializes an all-core OpenMP/BLAS pool
# (workers × cores threads thrashing shared caches). Must run before
# torch/numpy import anywhere in the worker; explicit env settings win.
_XDIST_WORKERS = os.environ.get("PYTEST_XDIST_WORKER_COUNT")
if _XDIST_WORKERS:
    _threads = str(max(1, (os.cpu_count() or 1) // int(_XDIST_WORKERS)))
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_var, _threads)

import numpy as np
import pytest

# ── Require marker selection ─────────────────────────────────────────
# Prevents `pytest tests/` from running all 1800+ tests at once.
# Use: pytest -m unit, pytest -m native, pytest -m "unit or api", etc.

_KNOWN_MARKS = {"unit", "native", "api", "pipeline", "e2e", "designer", "slow"}
_HEAVY_CLEANUP_MARKS = {"native", "pipeline", "e2e", "slow"}


def _infer_marker_from_nodeid(nodeid: str) -> str:
    """Assign unmarked tests to the category users are forced to select."""
    name = nodeid.lower()
    if "designer" in name:
        return "designer"
    if "e2e" in name or "end_to_end" in name:
        return "e2e"
    if "benchmark" in name or "soak" in name:
        return "slow"
    if "native" in name or "cython" in name or "rust" in name:
        return "native"
    if "api" in name or "dashboard" in name or "sse" in name:
        return "api"
    if "pipeline" in name or "runner" in name:
        return "pipeline"
    return "unit"


def pytest_collection_modifyitems(config, items):
    """Abort if no marker filter is specified — forces category selection.
    Also assign xdist worker groups to avoid duplicate CUDA contexts."""
    marker_expr = config.getoption("-m", default="")
    if not marker_expr:
        # Allow running a single file without -m
        specified_paths = config.args
        if specified_paths and all(
            os.path.isfile(p) or (not os.path.isdir(p)) for p in specified_paths
        ):
            pass  # Single file mode — allow, but still apply xdist groups below
        else:
            pytest.exit(
                "ERROR: Running all tests at once is not allowed.\n"
                "Use a marker filter:  pytest -m unit  |  pytest -m native  |  "
                "pytest -m api  |  pytest -m pipeline  |  pytest -m e2e  |  "
                "pytest -m designer\n"
                "Or combine:  pytest -m 'unit or api'\n"
                "Or run a single file:  pytest tests/test_notebook.py",
                returncode=4,
            )

    # Fill in missing category markers before pytest applies the -m expression.
    # Without this, a large fraction of the tree is silently deselected by the
    # marker gate this file enforces.
    for item in items:
        marker_names = {m.name for m in item.iter_markers()}
        if not (marker_names & _KNOWN_MARKS):
            item.add_marker(
                getattr(pytest.mark, _infer_marker_from_nodeid(item.nodeid))
            )
            marker_names = {m.name for m in item.iter_markers()}

        # xdist worker groups — group heavy tests to avoid duplicate CUDA contexts
        if "native" in marker_names:
            item.add_marker(pytest.mark.xdist_group("native"))
        elif "api" in marker_names:
            item.add_marker(pytest.mark.xdist_group("api"))


# ── Native library loading ────────────────────────────────────────────

_NATIVE_LIB_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "runtime",
    "native",
    "build",
    "libaria_native_runtime.so",
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


def assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str = "",
    atol: float = 1e-5,
    rtol: float = 1e-5,
):
    """Assert arrays are close with configurable tolerance."""
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)


def make_fake_graph(op_names):
    """Create a minimal graph-like object for native runner tests."""
    nodes = {}
    for i, name in enumerate(op_names):
        node = SimpleNamespace(op_name=name)
        nodes[f"n{i}"] = node
    return SimpleNamespace(nodes=nodes)


# ── Autouse cleanup fixture (P0 OOM prevention) ─────────────────────


@pytest.fixture(autouse=True)
def _cleanup_after_test(request):
    """Reclaim memory only for tests likely to allocate heavyweight state."""
    yield
    marker_names = {m.name for m in request.node.iter_markers()}
    nodeid = request.node.nodeid.lower()
    needs_cleanup = (
        os.environ.get("ARIA_TEST_FORCE_CLEANUP") == "1"
        or bool(marker_names & _HEAVY_CLEANUP_MARKS)
        or "native" in nodeid
        or "cuda" in nodeid
        or "benchmark" in nodeid
    )
    if not needs_cleanup:
        return

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ── Shared fixtures ──────────────────────────────────────────────────


@pytest.fixture
def tmp_notebook(tmp_path):
    """Provide a LabNotebook that auto-closes on teardown."""
    from research.scientist.notebook import LabNotebook

    nb = LabNotebook(str(tmp_path / "test.db"))
    yield nb
    nb.close()


@pytest.fixture
def flask_client(tmp_path):
    """Provide a Flask test client with auto-cleanup."""
    from research.scientist.api import create_app

    app = create_app(notebook_path=str(tmp_path / "test_api.db"))
    with app.test_client() as client:
        yield client
