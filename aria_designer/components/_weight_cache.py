"""Shared weight cache for fallback kernel handlers.

All fallback handlers generate deterministic weights via seeded torch.Generator.
Since the same seed always produces the same weights, we cache results by
(shape, seed, dtype) to avoid regenerating on every forward() call.

The cache is module-level and survives across forward passes, compilation
cycles, and profiling runs within the same process.
"""

import torch

_cache: dict[tuple, torch.Tensor] = {}
_MAX_CACHE_SIZE = 256


def cached_randn(
    *shape: int,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    scale: float = 1.0,
) -> torch.Tensor:
    """Return a deterministic random tensor, cached by (shape, seed, dtype).

    The tensor is generated once and stored. Subsequent calls with the same
    key return the cached tensor (moved to the requested device if needed).
    """
    key = (shape, seed, dtype)
    cached = _cache.get(key)
    if cached is not None:
        if cached.device != torch.device(device):
            return cached.to(device)
        return cached

    # Evict oldest entries if cache is full
    if len(_cache) >= _MAX_CACHE_SIZE:
        # Remove first quarter of entries (FIFO-ish)
        keys_to_remove = list(_cache.keys())[: _MAX_CACHE_SIZE // 4]
        for k in keys_to_remove:
            del _cache[k]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    w = torch.randn(*shape, generator=gen, dtype=dtype)
    if scale != 1.0:
        w *= scale
    _cache[key] = w
    return w.to(device) if str(device) != "cpu" else w
