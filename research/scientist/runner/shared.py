"""Process-wide ExperimentRunner singletons for batch tools.

Batch scripts (rescreen, backpopulate, replay, ...) loop over many results
and historically built a fresh ExperimentRunner per iteration. Each new
runner pays for: CodeHealer init (separate aria-db connection), baseline
transformer load, SSE log handler attach, and a fresh CorpusTokenBatcher
that re-tokenized the corpus (~306 MB). The corpus tokens are now cached
at module level in research.training.data_pipeline, but the rest of the
init still costs ~2-5s per call and accumulates pinned/cached state.

Use get_shared_runner(notebook_path) in batch tools to amortize that cost
across iterations. Call close_shared_runners() at process exit (atexit
handler is registered automatically on first use).
"""

from __future__ import annotations

import atexit
import logging
import threading
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from . import ExperimentRunner

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SHARED: Dict[str, "ExperimentRunner"] = {}
_ATEXIT_REGISTERED = False


def get_shared_runner(notebook_path: str) -> "ExperimentRunner":
    """Return a process-wide singleton runner for a notebook path.

    Caller must NOT call .close() on the returned runner — it lives for the
    process lifetime. close_shared_runners() runs automatically at exit.
    """
    global _ATEXIT_REGISTERED
    key = str(notebook_path)
    with _LOCK:
        runner = _SHARED.get(key)
        if runner is None:
            from . import ExperimentRunner  # lazy: keeps import light

            runner = ExperimentRunner(notebook_path=key)
            _SHARED[key] = runner
            if not _ATEXIT_REGISTERED:
                atexit.register(close_shared_runners)
                _ATEXIT_REGISTERED = True
        return runner


def close_shared_runners() -> None:
    """Close and forget all shared runners. Idempotent."""
    with _LOCK:
        runners = list(_SHARED.values())
        _SHARED.clear()
    for runner in runners:
        close_fn = getattr(runner, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                logger.debug("Shared runner close failed", exc_info=True)
