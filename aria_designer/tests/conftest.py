"""Shared test configuration for aria_designer tests.

Adds aria_designer/ to sys.path so that component kernel_fallback.py files
can use bare imports (``from runtime.``, ``from components.``) as they do
in production when the working directory is aria_designer/.
"""

import sys
from pathlib import Path

_ARIA_ROOT = str(Path(__file__).resolve().parents[1])
if _ARIA_ROOT not in sys.path:
    sys.path.insert(0, _ARIA_ROOT)
