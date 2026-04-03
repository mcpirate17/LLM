"""Fast JSON helpers for synthesis hot paths."""

from __future__ import annotations

import json
from typing import Any

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover
    _orjson = None  # type: ignore[assignment]


def dumps_json(value: Any) -> str:
    if _orjson is not None:
        return _orjson.dumps(value).decode("utf-8")
    return json.dumps(value, separators=(",", ":"))


def loads_json(data: str) -> Any:
    if _orjson is not None:
        return _orjson.loads(data)
    return json.loads(data)
