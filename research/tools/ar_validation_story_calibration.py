"""Compatibility entry point for AR validation story calibration helpers."""

from __future__ import annotations

from research.tools import small_ar_story_calibration as _impl

for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

__all__ = [name for name in globals() if not name.startswith("__")]
