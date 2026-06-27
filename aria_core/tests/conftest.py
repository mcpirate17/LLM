"""sys.path repair so `import aria_core` resolves to the real package.

`python -m pytest` run from aria_core/ puts this directory at sys.path[0],
which makes the inner aria_core/aria_core/ extension package shadow the
importable top-level package and break with "attempted relative import
beyond top-level package". Imports must resolve from the repo root.
"""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_PKG_DIR)

sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _PKG_DIR]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
