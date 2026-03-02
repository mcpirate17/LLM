"""aria_core — Unified high-performance kernel library for Aria."""
import os
import torch  # ensure torch libs are loaded first

from ._C import *  # noqa: F401,F403

__version__ = "0.1.0"
