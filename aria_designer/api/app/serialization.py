import numpy as np
from typing import Any
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to python primitives for JSON serialization.
    """
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    
    # Use standard encoder for other types (datetimes, UUIDs, etc.)
    try:
        return jsonable_encoder(obj)
    except Exception:
        return obj

class NumpySafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return super().render(convert_numpy_types(content))
