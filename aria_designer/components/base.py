import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try aria_core for native kernel dispatch
try:
    import aria_core

    _HAS_ARIA_CORE = True
except ImportError:
    _HAS_ARIA_CORE = False


def unsupported_fallback(op_name, *, reason=None):
    """Raise a hard failure for components without an honest Python fallback."""
    detail = f" ({reason})" if reason else ""
    raise NotImplementedError(
        f"Python fallback for {op_name} is unavailable{detail}; "
        "enable the native kernel or remove this component from the workflow."
    )


def _try_native(op_name, *tensors):
    """Try to dispatch a simple op through aria_core. Returns None on failure."""
    if not _HAS_ARIA_CORE:
        return None
    fn = getattr(aria_core, f"{op_name}_f32", None)
    if fn is None:
        return None
    try:
        args = []
        for t in tensors:
            if isinstance(t, torch.Tensor):
                args.append(t.detach().contiguous().float())
            else:
                args.append(t)
        return fn(*args)
    except Exception:
        logger.debug("Native kernel dispatch failed for op %s", op_name, exc_info=True)
        return None


def _result_is_finite(result):
    """Reject invalid native outputs so domain-safe Python fallbacks can take over."""
    return not isinstance(result, torch.Tensor) or torch.isfinite(result).all()


class BaseComponentHandler:
    """Base class for component handlers to reduce boilerplate."""

    def validate_config(self, config):
        return []

    def build(self, config):
        raise NotImplementedError("Subclasses must implement build")

    def forward(self, inputs, config):
        raise NotImplementedError("Subclasses must implement forward")


def make_unary_handler(op_fn, native_op_name=None, native_result_validator=None):
    """Generate a ComponentHandler class for a unary op."""

    class _Module(nn.Module):
        def forward(self, x):
            return op_fn(x)

    class ComponentHandler:
        __slots__ = ()

        def validate_config(self, config):
            return []

        def build(self, config):
            return _Module()

        def forward(self, inputs, config):
            x = inputs.get("x")
            if x is None:
                x = next(iter(inputs.values()))
            if native_op_name is not None:
                result = _try_native(native_op_name, x)
                validator = native_result_validator or _result_is_finite
                if result is not None and validator(result):
                    return {"y": result}
            return {"y": op_fn(x)}

    return ComponentHandler


def make_binary_handler(op_fn, native_op_name=None, native_result_validator=None):
    """Generate a ComponentHandler class for a binary op."""

    class _Module(nn.Module):
        def forward(self, a, b):
            return op_fn(a, b)

    class ComponentHandler:
        __slots__ = ()

        def validate_config(self, config):
            return []

        def build(self, config):
            return _Module()

        def forward(self, inputs, config):
            a = inputs.get("a")
            b = inputs.get("b")
            if a is None or b is None:
                keys = list(inputs.keys())
                if len(keys) >= 2:
                    a = a if a is not None else inputs[keys[0]]
                    b = b if b is not None else inputs[keys[1]]
            if native_op_name is not None:
                result = _try_native(native_op_name, a, b)
                validator = native_result_validator or _result_is_finite
                if result is not None and validator(result):
                    return {"y": result}
            return {"y": op_fn(a, b)}

    return ComponentHandler


def _make_weight(shape, fan_in=None):
    """Create a weight tensor with Kaiming-like init, cached for reuse."""
    scale = (fan_in or shape[-1]) ** -0.5
    return torch.randn(shape) * scale


class NativeComponentHandler(BaseComponentHandler):
    """Base class for components that dispatch to aria_core with config-driven params.

    Subclasses set native_op_name and implement _get_native_args and _fallback.
    Weights are lazily initialized on first forward() and reused thereafter.
    """

    native_op_name = None

    def __init__(self):
        self._weights = {}
        self._initialized = False

    def validate_config(self, config):
        return []

    def build(self, config):
        self._build_config = config
        return nn.Identity()

    def _ensure_weights(self, x, config):
        """Lazily initialize and cache weights based on input shape and config.
        Override in subclass for custom weight shapes. Default: no weights needed."""
        pass

    def _get_native_args(self, inputs, config):
        """Return args tuple for aria_core.{op}_f32(*args). Override in subclass."""
        raise NotImplementedError

    def _fallback(self, inputs, config):
        """Pure PyTorch fallback. Subclasses must provide an honest fallback."""
        unsupported_fallback(
            self.native_op_name or self.__class__.__name__,
            reason="no Python parity implementation",
        )

    def forward(self, inputs, config):
        # Lazy weight init on first call
        x = inputs.get("x")
        if x is None:
            x = next(iter(inputs.values()))
        if not self._initialized:
            self._ensure_weights(x, config)
            self._initialized = True

        if _HAS_ARIA_CORE and self.native_op_name is not None:
            fn = getattr(aria_core, f"{self.native_op_name}_f32", None)
            if fn is not None:
                try:
                    args = self._get_native_args(inputs, config)
                    result = fn(*args)
                    return {"y": result}
                except Exception:
                    logger.debug(
                        "Native kernel failed for %s, using Python fallback",
                        self.native_op_name,
                        exc_info=True,
                    )
        return self._fallback(inputs, config)
