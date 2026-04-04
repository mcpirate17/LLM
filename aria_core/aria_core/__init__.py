"""aria_core — Unified high-performance kernel library for Aria."""

from .._bootstrap import load_native_extension

load_native_extension(globals())


__version__ = "0.1.0"
