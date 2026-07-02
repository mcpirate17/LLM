"""Numerically-stable chunked causal power-law decay scan (speed + inference VRAM).

Several novel lanes need the causal decayed context

    c[b, t, c] = Σ_{s ≤ t} ρ_c**(t - s) · x[b, s, c]

which is the linear recurrence ``c_t = ρ ⊙ c_{t-1} + x_t``. Materializing the full
``[C, L, L]`` decay matrix is O(L²) time AND memory — fine at nano/probe scale but
the wrong asymptotics for long sequences. This helper computes the SAME result in
chunks of length ``K``: an exact intra-chunk ``[C, K, K]`` matmul plus a carried
state across chunks, so it is O(L·K) time and O(K²) memory. Because the largest
decay power inside a chunk is ``K``, it is also numerically stable (no
``ρ**(-s)`` cumsum trick that overflows for small ρ).

Shared by ``reversible_primitives._CausalDecayMLP`` and
``novel_math_primitives.OctonionicMixerLane`` (per-group decay via
``repeat_interleave``), replacing their duplicated dense implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

_SCAN_TABLE_CACHE_MAX = 128
_SCAN_TABLE_CACHE: dict[
    tuple[int, tuple[str, int | None], torch.dtype],
    tuple[Tensor, Tensor, Tensor],
] = {}


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    dev = torch.device(device)
    return dev.type, dev.index


def _is_inference_tensor(tensor: Tensor) -> bool:
    try:
        return bool(tensor.is_inference())
    except AttributeError:
        return False


def _chunk_decay_tables(
    n: int, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor]:
    key = (int(n), _device_cache_key(device), dtype)
    cached = _SCAN_TABLE_CACHE.get(key)
    if cached is not None:
        idx, exps, causal = cached
        if not (torch.is_grad_enabled() and _is_inference_tensor(idx)):
            return idx, exps, causal
    idx = torch.arange(n, device=device, dtype=dtype)
    exps = (idx[:, None] - idx[None, :]).clamp(min=0)
    causal = (idx[:, None] >= idx[None, :]).to(dtype)
    if len(_SCAN_TABLE_CACHE) >= _SCAN_TABLE_CACHE_MAX:
        _SCAN_TABLE_CACHE.clear()
    _SCAN_TABLE_CACHE[key] = (idx, exps, causal)
    return idx, exps, causal


def causal_decay_context(x: Tensor, decay: Tensor, *, chunk: int = 64) -> Tensor:
    """Causal power-law decayed context, chunked and stable.

    Args:
        x: ``[B, L, C]`` input.
        decay: ``[C]`` per-channel decay ``ρ ∈ (0, 1)``.
        chunk: chunk length ``K`` bounding the intra-chunk matrix and the max
            decay power (so it stays numerically stable).

    Returns:
        ``[B, L, C]`` with ``c[b, t, c] = Σ_{s ≤ t} decay[c]**(t-s) x[b, s, c]``.
    """
    if x.dim() != 3:
        raise ValueError(f"expected [B, L, C]; got {tuple(x.shape)}")
    if decay.shape != (x.shape[-1],):
        raise ValueError(f"decay must be [{x.shape[-1]}]; got {tuple(decay.shape)}")
    if chunk <= 0:
        raise ValueError("chunk must be positive")

    length = x.shape[1]
    log_decay = torch.log(decay)  # [C]
    out = torch.empty_like(x)
    # state = c at the last position of the previous chunk (zeros before the first)
    state = x.new_zeros(x.shape[0], x.shape[-1])
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        xc = x[:, start:end, :]  # [B, n, C]
        n = end - start
        idx, exps, causal = _chunk_decay_tables(n, x.device, x.dtype)
        powmat = torch.exp(exps[None] * log_decay[:, None, None]) * causal[None]
        intra = torch.einsum("cts,bsc->btc", powmat, xc)  # within-chunk
        # carry from earlier chunks: c_{start+j} += decay**(j+1) * state
        carry_factor = torch.exp((idx + 1.0)[:, None] * log_decay[None, :])  # [n, C]
        c = intra + carry_factor[None] * state[:, None, :]
        out[:, start:end, :] = c
        state = c[:, -1, :]
    return out


# ── streaming / autoregressive inference (KV-free, O(1) state) ────────────────
#
# The batched ``causal_decay_context`` is for training / prefill. For token-by-
# token generation, the same recurrence ``c_t = ρ ⊙ c_{t-1} + x_t`` runs with a
# single ``[B, C]`` state — memory independent of context length, so a stack of
# decayed-context lanes decodes with NO growing KV cache (the inference-VRAM
# analog of the reversible training-VRAM lever).


@dataclass(slots=True)
class DecayScanState:
    """Running decay-context state for O(1)-per-token autoregressive inference."""

    context: Tensor  # [B, C] — the current c_t


def init_decay_scan_state(
    batch: int, channels: int, *, device=None, dtype=None
) -> DecayScanState:
    """A zeroed streaming state (``c_{-1} = 0``)."""
    return DecayScanState(
        context=torch.zeros(batch, channels, device=device, dtype=dtype)
    )


def decay_scan_step(
    state: DecayScanState, x_t: Tensor, decay: Tensor
) -> tuple[Tensor, DecayScanState]:
    """One recurrence step ``c_t = decay ⊙ c_{t-1} + x_t``.

    Args:
        state: the running :class:`DecayScanState` (``c_{t-1}``).
        x_t: ``[B, C]`` current token features.
        decay: ``[C]`` per-channel decay ``ρ ∈ (0, 1)``.

    Returns:
        ``(c_t, new_state)`` — ``c_t`` is ``[B, C]``; ``new_state`` carries it.
    """
    if x_t.dim() != 2:
        raise ValueError(f"expected [B, C] token; got {tuple(x_t.shape)}")
    if decay.shape != (x_t.shape[-1],):
        raise ValueError(f"decay must be [{x_t.shape[-1]}]; got {tuple(decay.shape)}")
    c = decay * state.context + x_t
    return c, DecayScanState(context=c)


def causal_decay_context_streaming(x: Tensor, decay: Tensor) -> Tensor:
    """Reference streaming loop — identical result to :func:`causal_decay_context`.

    Exists to pin the streaming recurrence against the batched form in tests and
    to document the O(1)-state inference path; the batched form is preferred for
    training / prefill.
    """
    if x.dim() != 3:
        raise ValueError(f"expected [B, L, C]; got {tuple(x.shape)}")
    state = init_decay_scan_state(
        x.shape[0], x.shape[-1], device=x.device, dtype=x.dtype
    )
    outputs = []
    for t in range(x.shape[1]):
        c_t, state = decay_scan_step(state, x[:, t, :], decay)
        outputs.append(c_t)
    return torch.stack(outputs, dim=1)
