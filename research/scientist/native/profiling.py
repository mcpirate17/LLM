import logging
from typing import Any, Dict, Optional

from .core import _try_import_rust_scheduler

logger = logging.getLogger(__name__)
_last_profile_data: Optional[Dict[str, Any]] = None

def enable_native_profiling(enable: bool = True) -> bool:
    """Enable or disable native kernel profiling.

    When enabled, subsequent calls to ``dispatch_graph_native()`` will
    record per-node timing data which can be retrieved via
    ``get_native_profile()``.

    Also respects the ``NATIVE_RUNNER_PROFILE=1`` environment variable.

    Returns True if profiling is now enabled, False otherwise.
    """
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "profiler_enable"):
        logger.debug("Rust scheduler profiler_enable not available")
        return False
    rust.profiler_enable(enable)
    return bool(rust.profiler_enabled())

def get_native_profile() -> Optional[Dict[str, Any]]:
    """Return profiling data from the most recent ``dispatch_graph_native()`` call.

    Returns None if the Rust scheduler is unavailable or profiling was
    not enabled. Otherwise returns a dict with:
      - ``node_profiles``: list of dicts with node_id, op_name, duration_us, etc.
      - ``peak_memory_bytes``: peak memory tracked by the profiler
    """
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "profiler_enabled"):
        return None
    if not rust.profiler_enabled():
        return None
    # Profiling data is embedded in execute_graph_with_stats results.
    # This function is a convenience accessor for the last cached result.
    return _last_profile_data
