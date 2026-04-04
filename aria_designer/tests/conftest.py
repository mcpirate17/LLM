"""Shared test configuration for aria_designer tests.

Adds the repository-local ``aria_designer/`` package root to ``sys.path`` so
tests can import package modules directly without depending on the shell cwd.
"""

import sys
from pathlib import Path

_ARIA_ROOT = str(Path(__file__).resolve().parents[1])
if _ARIA_ROOT not in sys.path:
    sys.path.insert(0, _ARIA_ROOT)
