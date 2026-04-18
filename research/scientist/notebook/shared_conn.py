"""Shared connection accessor for lab_notebook.db.

Any code outside the notebook package that needs to read from
``lab_notebook.db`` should call ``get_notebook_conn(db_path)`` instead
of ``sqlite3.connect(db_path)``.  This returns the process-wide
``NativeConnectionWrapper`` singleton, preventing SHM teardown.

The returned connection must NOT be closed by the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .native_conn import NativeConnectionWrapper


def get_notebook_conn(db_path: str) -> "NativeConnectionWrapper":
    """Return the shared native connection for ``lab_notebook.db``.

    Thread-safe, never closes.  Callers must not call ``.close()``
    (it's a no-op anyway).
    """
    from .native_conn import NativeConnectionWrapper

    return NativeConnectionWrapper(str(db_path))
