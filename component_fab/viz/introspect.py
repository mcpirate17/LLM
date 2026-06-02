"""Run lanes and extract everything the UI plots.

All measurements are intrinsic (random/structured probe inputs, no training,
no DB). The surprise-memory trace re-runs the lane's OWN ``_addr`` / ``v`` /
gates / ``_delta_step`` / ``_read`` so the captured frames are faithful to the
concrete read algebra (tropical vs learnable semiring), not a re-implementation.
"""

from __future__ import annotations

from typing import Any

import torch

from ..generator.memory_primitives import (
    CausalFastWeightMemoryLane,
    SemiringSurpriseMemoryLane,
    _SurpriseMemoryBase,
)
from ..metrics.mix_speed import measure_mix_speed


def param_count(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def smoke(module: torch.nn.Module, *, dim: int, seq_len: int = 16) -> dict[str, Any]:
    """Forward + backward on random input; check shape + finiteness."""
    out: dict[str, Any] = {
        "forward_ok": False,
        "backward_ok": False,
        "shape_preserved": False,
        "all_finite": False,
    }
    x = torch.randn(2, seq_len, dim, requires_grad=True)
    try:
        y = module(x)
        out["forward_ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"forward: {exc}"
        return out
    out["shape_preserved"] = tuple(y.shape) == tuple(x.shape)
    if not out["shape_preserved"]:
        out["error"] = f"shape {tuple(y.shape)} != {tuple(x.shape)}"
        return out
    try:
        y.sum().backward()
        out["backward_ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"backward: {exc}"
        return out
    grads_finite = all(
        bool(torch.isfinite(p.grad).all())
        for p in module.parameters()
        if p.grad is not None
    )
    out["all_finite"] = bool(torch.isfinite(y).all()) and grads_finite
    return out


def influence_matrix(
    module: torch.nn.Module,
    *,
    dim: int,
    seq_len: int = 24,
    n_trials: int = 4,
    delta_scale: float = 1e-2,
    seed: int = 0,
) -> dict[str, Any]:
    """Token-mixing map: perturb input position i, measure output response at j.

    Returns an ``L x L`` matrix (row i = injection position, col j = response
    position). For a causal lane the matrix is lower-triangular: a perturbation
    can only affect current/future outputs. The headline half-life / global-mix
    numbers come from the shared ``mix_speed`` metric so they match the grader.
    """
    gen = torch.Generator().manual_seed(seed)
    L = seq_len
    accum = torch.zeros(L, L)
    module.eval()
    with torch.no_grad():
        for _ in range(n_trials):
            x = torch.randn(1, L, dim, generator=gen)
            delta = torch.randn(1, dim, generator=gen) * delta_scale
            y = module(x)
            for i in range(L):
                xp = x.clone()
                xp[:, i, :] = xp[:, i, :] + delta
                yp = module(xp)
                resp = (yp - y).pow(2).sum(dim=-1).sqrt()[0]  # [L]
                accum[i] += resp
    matrix = (accum / n_trials).tolist()

    card = measure_mix_speed(module, seq_len=64, feature_dim=dim, n_trials=6)
    return {
        "matrix": matrix,
        "seq_len": L,
        "decay": list(card.response_decay),
        "mix_half_life": (
            None if card.mix_half_life == float("inf") else card.mix_half_life
        ),
        "mixes_globally": bool(card.mixes_globally),
        "is_pure_local": bool(card.is_pure_local),
        "peak_response_at_offset": int(card.peak_response_at_offset),
    }


def _probe_input(
    seq_len: int, dim: int, *, repeat_src: int, repeat_dst: int, seed: int = 7
) -> torch.Tensor:
    """Random sequence with one token repeated, to make 'surprise' legible.

    Token at ``repeat_src`` is copied into ``repeat_dst``. When the memory sees
    the repeated key the second time, its prediction error (surprise) should
    behave differently than for a fresh token — that contrast is the point.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, seq_len, dim, generator=gen)
    x[:, repeat_dst, :] = x[:, repeat_src, :]
    return x


def surprise_trace(
    module: _SurpriseMemoryBase,
    *,
    dim: int,
    seq_len: int = 16,
) -> dict[str, Any]:
    """Step the single-memory delta-rule scan, capturing per-step state.

    Reuses the lane's own ``_addr`` / ``v`` / gates / ``_read`` / ``_delta_step``
    so the memory snapshots reflect the real read algebra. Captures, per step:
    the full memory matrix, the surprise (prediction-error) norm, the memory
    Frobenius norm, and the read vector.
    """
    repeat_src, repeat_dst = 2, max(3, seq_len // 2)
    x = _probe_input(seq_len, dim, repeat_src=repeat_src, repeat_dst=repeat_dst)

    module.eval()
    with torch.no_grad():
        q, k = module._addr(x)
        v = module.v(x)
        write = torch.sigmoid(module.write_gate(x)).squeeze(-1)
        forget = torch.sigmoid(module.forget_gate(x))
        momentum = torch.sigmoid(module.momentum_logit)
        m = module.memory_dim
        memory = x.new_zeros(1, m, m)
        surprise = x.new_zeros(1, m, m)

        frames: list[dict[str, Any]] = []
        for t in range(seq_len):
            # Faithful surprise = ||v_t - read(M_{t-1}, k_t)|| using lane's _read.
            prediction = module._read(memory, k[:, t])
            error_norm = float((v[:, t] - prediction).norm().item())
            memory, surprise, read = module._delta_step(
                memory,
                surprise,
                k_t=k[:, t],
                v_t=v[:, t],
                q_t=q[:, t],
                write=write[:, t],
                forget=forget[:, t],
                momentum=momentum,
            )
            frames.append(
                {
                    "t": t,
                    "memory": memory[0].tolist(),
                    "read": read[0].tolist(),
                    "error_norm": error_norm,
                    "memory_norm": float(memory[0].norm().item()),
                    "write_gate": float(write[0, t].item()),
                    "forget_gate_mean": float(forget[0, t].mean().item()),
                }
            )

    return {
        "frames": frames,
        "memory_dim": m,
        "seq_len": seq_len,
        "repeat_src": repeat_src,
        "repeat_dst": repeat_dst,
        "momentum": float(momentum.item()),
    }


_RECALL_LABELS = (
    "🔑 key",
    "🌙 moon",
    "🐟 fish",
    "⭐ star",
    "🍎 apple",
    "🚗 car",
)


_CLEAN_THRESHOLD = 0.85  # cosine: above this, a crowded recall counts as "clean"


def _delta_memory(
    module: _SurpriseMemoryBase, x: torch.Tensor, idxs: list[int]
) -> torch.Tensor:
    """Run the delta-rule write over the tokens in ``idxs``, return memory."""
    q, k = module._addr(x)
    v = module.v(x)
    write = torch.sigmoid(module.write_gate(x)).squeeze(-1)
    forget = torch.sigmoid(module.forget_gate(x))
    momentum = torch.sigmoid(module.momentum_logit)
    m = module.memory_dim
    memory = x.new_zeros(1, m, m)
    surprise = x.new_zeros(1, m, m)
    for t in idxs:
        memory, surprise, _ = module._delta_step(
            memory,
            surprise,
            k_t=k[:, t],
            v_t=v[:, t],
            q_t=q[:, t],
            write=write[:, t],
            forget=forget[:, t],
            momentum=momentum,
        )
    return memory


def _hebbian_memory(
    module: CausalFastWeightMemoryLane, x: torch.Tensor, idxs: list[int]
) -> torch.Tensor:
    """Run the Hebbian outer-product write over ``idxs``, return memory."""
    k = torch.tanh(module.k(x))
    v = torch.tanh(module.v(x))
    gates = torch.sigmoid(module.write_gate(x)).squeeze(-1)
    decay = torch.sigmoid(module.decay_logit)
    scale = float(module.memory_dim) ** -0.5
    memory = x.new_zeros(1, module.memory_dim, module.memory_dim)
    for t in idxs:
        write = torch.einsum("bi,bj->bij", k[:, t], v[:, t]) * scale
        memory = decay * memory + gates[:, t].view(1, 1, 1) * write
    return memory


def recall_story(
    module: torch.nn.Module, *, dim: int, n_facts: int = 5
) -> dict[str, Any]:
    """Faithful 'watch it remember' demo measuring cross-key interference.

    We file N distinct facts into the memory, then cue each one with its own
    key — twice: once from a memory holding ONLY that fact (the clean target),
    once from the memory holding all N (the crowded reality). ``retention`` is
    the cosine between the two: how much of the clean recall survives the crowd.
    High retention = the other facts didn't smear this one (sharp, winner-take-
    all reads excel here); low retention = cross-key interference (the failure
    mode of a fuzzy summed memory). Everything runs through the lane's OWN write
    + read path, so the contrast is real, not staged. Untrained weights: this is
    the architecture's built-in instinct, not a learned skill.
    """
    n = min(n_facts, len(_RECALL_LABELS))
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(1, n, dim, generator=gen)
    module.eval()
    with torch.no_grad():
        if isinstance(module, _SurpriseMemoryBase):
            keys = module._addr(x)[1]
            build = _delta_memory
            read = lambda mem, key: module._read(mem, key)[0]  # noqa: E731
            if isinstance(module, SemiringSurpriseMemoryLane):
                beta = float(
                    torch.nn.functional.softplus(module.semiring_temp)
                    .clamp(1e-2, 30.0)
                    .item()
                )
                kind = f"learned focus (β={beta:.1f})"
            else:
                kind = "winner-take-all (sharp)"
        elif isinstance(module, CausalFastWeightMemoryLane):
            keys = torch.tanh(module.k(x))
            build = _hebbian_memory
            read = lambda mem, key: torch.einsum("bi,bij->bj", key, mem)[0]  # noqa: E731
            kind = "fuzzy sum (blends notes)"
        else:
            raise ValueError(f"{type(module).__name__} has no recall demo")

        crowded = build(module, x, list(range(n)))
        results: list[dict[str, Any]] = []
        retention_sum = 0.0
        for i in range(n):
            full = read(crowded, keys[:, i])
            clean = read(build(module, x, [i]), keys[:, i])
            retention = float(
                torch.nn.functional.cosine_similarity(full, clean, dim=0).item()
            )
            retention_sum += retention
            results.append(
                {
                    "idx": i,
                    "label": _RECALL_LABELS[i],
                    "retention": retention,
                    "clean": retention >= _CLEAN_THRESHOLD,
                }
            )

    clean_count = sum(1 for r in results if r["clean"])
    return {
        "facts": [{"idx": i, "label": _RECALL_LABELS[i]} for i in range(n)],
        "read_kind": kind,
        "results": results,
        "mean_retention": retention_sum / n,
        "clean_count": clean_count,
        "n_facts": n,
    }


def algebra_spectrum(
    module: _SurpriseMemoryBase,
    *,
    dim: int,
    seq_len: int = 16,
) -> dict[str, Any]:
    """Read the SAME warmed-up memory three ways: mean (β→0), learned β, max (β→∞).

    Shows where a learnable-semiring read sits on the mean↔max axis. The memory
    is warmed by running the probe scan; the query is the final-step query.
    """
    trace = surprise_trace(module, dim=dim, seq_len=seq_len)
    last = trace["frames"][-1]
    memory = torch.tensor(last["memory"]).unsqueeze(0)  # [1, m, m]

    # Reconstruct the final-step query addr to read with.
    x = _probe_input(
        seq_len, dim, repeat_src=trace["repeat_src"], repeat_dst=trace["repeat_dst"]
    )
    with torch.no_grad():
        q, _ = module._addr(x)
        addr = q[:, -1]  # [1, m]
        scores = memory + addr.unsqueeze(-1)  # [1, m_keys, m_vals]
        mean_read = scores.mean(dim=1)[0]
        max_read = scores.amax(dim=1)[0]
        learned_read = module._read(memory, addr)[0]
        beta = None
        if hasattr(module, "semiring_temp"):
            beta = float(
                torch.nn.functional.softplus(module.semiring_temp)
                .clamp(1e-2, 30.0)
                .item()
            )

    return {
        "dims": list(range(memory.shape[2])),
        "mean": mean_read.tolist(),
        "max": max_read.tolist(),
        "learned": learned_read.tolist(),
        "beta": beta,
    }
