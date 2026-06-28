"""Shared test configuration for aria_designer tests.

Adds the repository-local ``aria_designer/`` package root to ``sys.path`` so
tests can import package modules directly without depending on the shell cwd.
"""

import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

_ARIA_ROOT = str(Path(__file__).resolve().parents[1])
if _ARIA_ROOT not in sys.path:
    sys.path.insert(0, _ARIA_ROOT)


@pytest.fixture(scope="module")
def client() -> Iterator["TestClient"]:
    """Create an API test client backed by a temporary component database."""
    from fastapi.testclient import TestClient

    from aria_designer.api.app import database as db
    from aria_designer.api.app.loader import scan_and_load
    from aria_designer.api.app.main import app

    with tempfile.TemporaryDirectory() as tmpdir:
        db.init_db(Path(tmpdir) / "test.db")
        count = scan_and_load()
        assert count > 0, "No components loaded"

        with TestClient(app) as c:
            yield c
