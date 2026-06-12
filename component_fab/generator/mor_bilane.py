"""MoR-routed bilane that resumes the native adaptive bilane's trained weights.

The native ``NativeAdaptiveSemiringBiLaneSurpriseMemoryLane`` decides per-token
recursion depth with a frozen, non-differentiable threshold gate that saturates
to max depth. This module subclasses it and swaps ONLY that gate for a
differentiable Mixture-of-Recursions / PonderNet halting router, by overriding
``lane_a._scan`` with a faithful torch port of the native recursion (same
``_scan_params`` projections, semiring read, delta-rule, balanced surprise,
per-key-row decay) in which the integer ``adaptive_steps`` count is replaced by a
learned soft-halting weighting. Every existing parameter is unchanged, so a
checkpoint trained with the native bilane loads directly (``strict=False``); the
only new parameter is ``lane_a.halt_head``.

Faithfulness: with ``force_max_depth=True`` the router puts all mass on the
deepest step, reproducing the native scan at max depth (validated to rel<1e-4
against the C++/CUDA kernel) — so the port is exact and the router only modulates
how much of each refinement is committed.

This is the Phase-1 *torch* path (no CUDA kernel): correct + differentiable, but
slower per step than the native scan — meant for a short validation resume before
investing in the Phase-2 CUDA port. See
``research/notes/mor_native_recursion_router_2026-06-01.md``.
"""

from __future__ import annotations

import torch
from torch import nn

from research.runtime.native.torch_extension_loader import load_local_cuda_extension

from .native_surprise_memory import (
    NativeAdaptiveSemiringBiLaneSurpriseMemoryLane,
    NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane,
)


def _mor_refine_cuda_ext():
    return load_local_cuda_extension(
        __file__,
        "native_mor_refine_cuda.cu",
        "component_fab_native_mor_refine_cuda",
    )


class _NativeMoRRefineScan(torch.autograd.Function):
    """CUDA forward/backward for the refine-each-step MoR scan with the MLP
    halting router. Validated rel<2e-5 (fwd + every input grad) against the torch
    reference ``MoRRefineMLPLaneA._scan`` (see validate_mor_refine_kernel.py).
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        write,
        forget,
        momentum,
        beta,
        balance,
        W1,
        b1,
        W2,
        b2,
        a_coupling,
        max_steps,
    ):
        ext = _mor_refine_cuda_ext()
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        write, forget = write.contiguous(), forget.contiguous()
        W1, b1, W2 = W1.contiguous(), b1.contiguous(), W2.contiguous()
        y, depth, hist, mem_prev, sur_prev = ext.mor_refine_forward(
            q,
            k,
            v,
            write,
            forget,
            float(momentum),
            float(beta),
            float(balance),
            int(max_steps),
            W1,
            b1,
            W2,
            float(b2),
            float(a_coupling),
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            write,
            forget,
            W1,
            b1,
            W2,
            mem_prev,
            sur_prev,
            momentum,
            beta,
            balance,
            b2,
            a_coupling,
        )
        ctx.max_steps = int(max_steps)
        ctx.mark_non_differentiable(hist)  # logging only (per-depth mass)
        # depth IS differentiable: the ponder loss (ponder_weight * mean depth)
        # backprops through depth_acc = Σ_r p_r·r into the halting router.
        return y, depth, hist

    @staticmethod
    def backward(ctx, grad_y, grad_depth, _grad_hist):
        (
            q,
            k,
            v,
            write,
            forget,
            W1,
            b1,
            W2,
            mem_prev,
            sur_prev,
            momentum,
            beta,
            balance,
            b2,
            a_coupling,
        ) = ctx.saved_tensors
        ext = _mor_refine_cuda_ext()
        if grad_depth is None:
            grad_depth = torch.zeros(
                q.shape[0], q.shape[1], device=q.device, dtype=q.dtype
            )
        (gq, gk, gv, gw, gf, gmom, gbeta, gbal, gW1, gb1, gW2, gb2, g_a) = (
            ext.mor_refine_backward(
                q,
                k,
                v,
                write,
                forget,
                float(momentum),
                float(beta),
                float(balance),
                ctx.max_steps,
                W1,
                b1,
                W2,
                float(b2),
                float(a_coupling),
                grad_y.contiguous(),
                grad_depth.contiguous(),
                mem_prev,
                sur_prev,
            )
        )
        return (
            gq,
            gk,
            gv,
            gw,
            gf,
            gmom.reshape(momentum.shape),
            gbeta.reshape(beta.shape),
            gbal.reshape(balance.shape),
            gW1,
            gb1,
            gW2,
            gb2.reshape(b2.shape),
            g_a.reshape(a_coupling.shape),
            None,
        )


class MoRLaneA(NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane):
    """Adaptive semiring MAC lane_a with the threshold gate replaced by a learned
    MoR/PonderNet halting router over the surprise-memory delta-recursion."""

    def __init__(self, *args, ponder_weight: float = 1e-2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Surprise-conditioned halting head: input is [mean|surprise_r|,
        # mean|raw0|] — both prediction-error-driven magnitudes (the same signal
        # the native threshold gate compared, now learned). Zero-init weight +
        # negative bias => starts deep (proven behavior), can only learn to exit.
        self.halt_head: nn.Module = nn.Linear(2, 1)
        self.reset_halt_deep_start()
        self.ponder_weight = float(ponder_weight)
        self.force_max_depth = False
        self.last_ponder_cost: torch.Tensor | None = None
        self.last_mean_depth: float | None = None
        self.last_depth_hist: list[float] | None = None

    def reset_halt_deep_start(self) -> None:
        """Deep-start halt init: bias −2 ⇒ halt prob ~0.12 ⇒ recurse deep, learn to
        exit. For the scalar ``Linear`` gate the weight is zeroed (proven init).
        For a ``Sequential`` MLP router the last weight is kept *small but nonzero*
        — zeroing it would zero the gradient into the hidden layer (dead router)."""
        head = self.halt_head
        if isinstance(head, nn.Sequential):
            last = head[-1]
            nn.init.normal_(last.weight, std=0.02)
        else:
            last = head
            nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, -2.0)

    @staticmethod
    def _bal(raw: torch.Tensor, balance: torch.Tensor) -> torch.Tensor:
        return raw / (1.0 + balance * raw.abs())

    @staticmethod
    def _semiring_read(
        mem: torch.Tensor, addr: torch.Tensor, beta: torch.Tensor, log_m: torch.Tensor
    ) -> torch.Tensor:
        """read[b, j] = (logsumexp_i β(mem[b,i,j]+addr[b,i]) − log m) / β."""
        scores = mem + addr.unsqueeze(-1)  # [B, m_key, m_val]
        lse = torch.logsumexp(beta * scores, dim=1)  # over key axis -> [B, m_val]
        return (lse - log_m) / beta

    def _scan_begin(self, x: torch.Tensor):
        """Shared _scan prologue: projections, shape info, zeroed memory/
        surprise state, and telemetry accumulators. Telemetry stays as device
        tensors — a float()/.item() per (t, r) step would force a GPU->CPU
        sync L*R times per forward; ``_scan_finish`` syncs once at the end."""
        q, k, v, write, forget, momentum, beta, balance = self._scan_params(x)
        b, length, m = v.shape
        scale = float(m) ** -0.5
        log_m = torch.log(torch.tensor(float(m), device=x.device, dtype=x.dtype))
        steps = self.max_recursive_steps
        mem = x.new_zeros(b, m, m)
        sur = x.new_zeros(b, m, m)
        ponder_total = x.new_zeros(())
        depth_total = x.new_zeros(())
        hist_t = x.new_zeros(steps)  # halting mass per depth r=1..steps
        return (
            (q, k, v, write, forget, momentum, beta, balance),
            (b, length, m, scale, log_m, steps),
            (mem, sur, ponder_total, depth_total, hist_t),
        )

    def _scan_finish(
        self,
        outputs: list[torch.Tensor],
        ponder_total: torch.Tensor,
        depth_total: torch.Tensor,
        hist_t: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """Shared _scan epilogue: ponder/depth telemetry (one host sync) and
        the stacked [B, L, m] output."""
        self.last_ponder_cost = self.ponder_weight * (ponder_total / length)
        self.last_mean_depth = float(depth_total) / length
        hist = hist_t.tolist()
        total_mass = sum(hist) or 1.0
        self.last_depth_hist = [round(h / total_mass, 4) for h in hist]
        return torch.stack(outputs, dim=1)  # [B, L, m]

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        params, dims, state = self._scan_begin(x)
        q, k, v, write, forget, momentum, beta, balance = params
        b, length, m, scale, log_m, steps = dims
        mem, sur, ponder_total, depth_total, hist_t = state
        outputs: list[torch.Tensor] = []
        for t in range(length):
            q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
            w_t = write[:, t].view(b, 1, 1)
            read_q = self._semiring_read(mem, q_t, beta, log_m)  # output (pre-write)
            err = v_t - self._semiring_read(mem, k_t, beta, log_m)
            delta = torch.einsum("bi,bj->bij", k_t, err) * scale
            raw0 = momentum * sur + w_t * delta
            raw0_mag = raw0.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)  # [B,1]
            s = self._bal(raw0, balance)
            remainder = x.new_ones(b, 1)
            sur_acc = x.new_zeros(b, m, m)
            depth_acc = x.new_zeros(b, 1)
            for r in range(1, steps + 1):
                if r > 1:
                    s = self._bal(momentum * s + w_t * delta, balance)
                feat = torch.cat(
                    [s.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1), raw0_mag],
                    dim=-1,
                )  # [B, 2]
                halt = torch.sigmoid(self.halt_head(feat))  # [B, 1]
                if self.force_max_depth:
                    # all halting mass on the deepest step (always-max ablation /
                    # faithfulness check): never halt early, force-halt at the end.
                    halt = (
                        torch.ones_like(halt) if r == steps else torch.zeros_like(halt)
                    )
                elif r == steps:
                    halt = torch.ones_like(halt)  # last step must commit
                p_r = remainder * halt
                sur_acc = sur_acc + p_r.unsqueeze(-1) * s
                depth_acc = depth_acc + p_r * float(r)
                hist_t[r - 1] = hist_t[r - 1] + p_r.mean().detach()
                remainder = remainder * (1.0 - halt)
            sur = sur_acc
            decay = (1.0 - forget[:, t]).unsqueeze(-1)  # per key-row i
            mem = decay * mem + sur
            outputs.append(read_q)
            ponder_total = ponder_total + depth_acc.mean()
            depth_total = depth_total + depth_acc.mean().detach()
        return self._scan_finish(outputs, ponder_total, depth_total, hist_t, length)


class MoRAdaptiveSemiringBiLaneSurpriseMemoryLane(
    NativeAdaptiveSemiringBiLaneSurpriseMemoryLane
):
    """Native adaptive bilane with lane_a's depth gate replaced by the MoR router.

    Param-compatible with ``NativeAdaptiveSemiringBiLaneSurpriseMemoryLane`` except
    for the added ``lane_a.halt_head`` — so a native-bilane checkpoint resumes with
    ``strict=False`` and only that head fresh.
    """

    def _make_lane_a(
        self,
        dim,
        memory_dim,
        gate_bias,
        semiring_temp_init,
        recursive_balance_init,
        low_threshold,
        high_threshold,
        max_recursive_steps,
    ) -> nn.Module:
        return MoRLaneA(
            dim,
            memory_dim=memory_dim,
            gate_bias=gate_bias,
            semiring_temp_init=semiring_temp_init,
            recursive_balance_init=recursive_balance_init,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            max_recursive_steps=max_recursive_steps,
        )

    @property
    def last_ponder_cost(self) -> torch.Tensor | None:
        return self.lane_a.last_ponder_cost


class MoRRefineLaneA(MoRLaneA):
    """MoR lane_a with TTT-style **refine-each-step** recursion.

    Unlike ``MoRLaneA`` (which faithfully ports the native *fixed-delta* momentum
    accumulation), here the associative error is **re-measured against the
    progressively refined memory at every inner step** — so surprise genuinely
    shrinks as the memory fits the token, and the router halts on the *current*
    (shrinking) error. Depth therefore means "how many gradient steps to memorize
    this token", which is the principled adaptive-computation reading.

    Output still reads ``M_{t-1}`` (read-before-write), so recursion depth only
    affects *future* recall, never the current token's own readout — no self-read
    shortcut, strictly causal. Not faithful to the native kernel (different
    recursion), so a native checkpoint is a *warm start* (projections load), not
    an exact resume.
    """

    def _halt_features(
        self,
        s_r: torch.Tensor,
        err_r: torch.Tensor,
        r: int,
        steps: int,
    ) -> torch.Tensor:
        """Per-step router input. Base = [mean|surprise_r|, mean|current_err_r|]
        (the two prediction-error magnitudes the native gate compared). Subclasses
        widen this (e.g. add normalized depth) for a higher-capacity router."""
        return torch.cat(
            [
                s_r.abs().mean(dim=(1, 2)).unsqueeze(-1),
                err_r.abs().mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )  # [B, 2]

    def _a_coupling(self) -> torch.Tensor | None:
        """Surprise->depth coupling ``a = a_min + softplus(raw) >= a_min >= 0``,
        or ``None`` when there is no coupling param (the default = pure learned
        router). See ``MoRSurpriseRefineMLPLaneA``."""
        raw = getattr(self, "halt_surprise_coupling", None)
        if raw is None:
            return None
        a_min = getattr(self, "surprise_coupling_min", 0.0)
        return a_min + torch.nn.functional.softplus(raw)

    def _halt_logit_extra(self, err_r: torch.Tensor) -> torch.Tensor | float:
        """Structural surprise floor added to the halt logit: ``-a * mean|err_r|``.
        Makes depth respond to per-token surprise *by construction* (more error ->
        lower halt -> deeper) so the loss cannot collapse the surprise channel.
        Returns 0.0 when there is no coupling (default refine-MLP lane)."""
        a = self._a_coupling()
        if a is None:
            return 0.0
        return -a * err_r.abs().mean(dim=-1, keepdim=True)

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        params, dims, state = self._scan_begin(x)
        q, k, v, write, forget, momentum, beta, balance = params
        b, length, m, scale, log_m, steps = dims
        mem, sur, ponder_total, depth_total, hist_t = state
        outputs: list[torch.Tensor] = []
        for t in range(length):
            q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
            w_t = write[:, t].view(b, 1, 1)
            read_q = self._semiring_read(mem, q_t, beta, log_m)  # output: M_{t-1}
            decay_base = (1.0 - forget[:, t]).unsqueeze(-1) * mem
            mem_r, s_r = mem, sur
            remainder = x.new_ones(b, 1)
            mem_acc = x.new_zeros(b, m, m)
            sur_acc = x.new_zeros(b, m, m)
            depth_acc = x.new_zeros(b, 1)
            for r in range(1, steps + 1):
                err_r = v_t - self._semiring_read(
                    mem_r, k_t, beta, log_m
                )  # re-measured
                delta_r = torch.einsum("bi,bj->bij", k_t, err_r) * scale
                s_r = self._bal(momentum * s_r + w_t * delta_r, balance)
                mem_r = decay_base + s_r
                halt = torch.sigmoid(
                    self.halt_head(self._halt_features(s_r, err_r, r, steps))
                    + self._halt_logit_extra(err_r)
                )
                if self.force_max_depth:
                    halt = (
                        torch.ones_like(halt) if r == steps else torch.zeros_like(halt)
                    )
                elif r == steps:
                    halt = torch.ones_like(halt)
                p_r = remainder * halt
                mem_acc = mem_acc + p_r.unsqueeze(-1) * mem_r
                sur_acc = sur_acc + p_r.unsqueeze(-1) * s_r
                depth_acc = depth_acc + p_r * float(r)
                hist_t[r - 1] = hist_t[r - 1] + p_r.mean().detach()
                remainder = remainder * (1.0 - halt)
            mem, sur = mem_acc, sur_acc
            outputs.append(read_q)
            ponder_total = ponder_total + depth_acc.mean()
            depth_total = depth_total + depth_acc.mean().detach()
        return self._scan_finish(outputs, ponder_total, depth_total, hist_t, length)


class MoRRefineAdaptiveSemiringBiLaneSurpriseMemoryLane(
    MoRAdaptiveSemiringBiLaneSurpriseMemoryLane
):
    """Bilane whose lane_a uses the refine-each-step MoR recursion."""

    def _make_lane_a(
        self,
        dim,
        memory_dim,
        gate_bias,
        semiring_temp_init,
        recursive_balance_init,
        low_threshold,
        high_threshold,
        max_recursive_steps,
    ) -> nn.Module:
        return MoRRefineLaneA(
            dim,
            memory_dim=memory_dim,
            gate_bias=gate_bias,
            semiring_temp_init=semiring_temp_init,
            recursive_balance_init=recursive_balance_init,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            max_recursive_steps=max_recursive_steps,
        )


class MoRRefineMLPLaneA(MoRRefineLaneA):
    """Refine-each-step MoR lane_a with a **substantial MLP halting router**.

    Replaces the 24-param ``Linear(2,1)`` halt gate (which collapsed / under-
    powered the depth policy) with ``Linear(3, H) → GELU → Linear(H, 1)`` over
    ``[mean|surprise_r|, mean|err_r|, r/R]`` — adding normalized depth so the
    router can learn depth-dependent halting (DeepMind-MoR-style learned router
    with real capacity, weight-shared across recursion). Recursion mechanism and
    every other parameter are unchanged, so a native/refine checkpoint warm-starts
    (projections load); only the router is fresh.
    """

    def __init__(self, *args, router_hidden: int = 32, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        h = int(router_hidden)
        self.router_hidden = h
        self.halt_head = nn.Sequential(nn.Linear(3, h), nn.GELU(), nn.Linear(h, 1))
        nn.init.normal_(self.halt_head[0].weight, std=0.02)
        nn.init.zeros_(self.halt_head[0].bias)
        self.reset_halt_deep_start()

    def _halt_features(
        self,
        s_r: torch.Tensor,
        err_r: torch.Tensor,
        r: int,
        steps: int,
    ) -> torch.Tensor:
        base = super()._halt_features(s_r, err_r, r, steps)  # [B, 2]
        r_norm = base.new_full((base.shape[0], 1), float(r) / float(steps))
        return torch.cat([base, r_norm], dim=-1)  # [B, 3]

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        """CUDA kernel on GPU, torch reference on CPU / force-max ablation."""
        if (not x.is_cuda) or self.force_max_depth:
            return super()._scan(x)
        q, k, v, write, forget, momentum, beta, balance = self._scan_params(x)
        a = self._a_coupling()
        a_t = a if a is not None else q.new_zeros(())
        y, depth, hist = _NativeMoRRefineScan.apply(
            q,
            k,
            v,
            write,
            forget,
            momentum,
            beta,
            balance,
            self.halt_head[0].weight,
            self.halt_head[0].bias,
            self.halt_head[2].weight.reshape(-1),
            self.halt_head[2].bias,
            a_t,
            self.max_recursive_steps,
        )
        self.last_mean_depth = float(depth.mean().detach())
        # depth is differentiable, so the ponder loss trains the router toward
        # shallower depth under the compute budget (ponder_weight>0).
        self.last_ponder_cost = self.ponder_weight * depth.mean()
        # hist[b, r] = sum_t p_r; normalize to per-depth fractions for logging.
        h = hist.detach().sum(dim=0)
        self.last_depth_hist = (h / h.sum().clamp_min(1e-9)).tolist()
        return y


class MoRRefineMLPAdaptiveSemiringBiLaneSurpriseMemoryLane(
    MoRRefineAdaptiveSemiringBiLaneSurpriseMemoryLane
):
    """Bilane whose lane_a uses the refine-each-step recursion with the MLP router.

    Router width is taken from the class attribute ``ROUTER_HIDDEN`` (the factory
    sets it from the lane name); ``_make_lane_a`` is invoked during ``super().
    __init__`` so an instance attribute would be too late."""

    ROUTER_HIDDEN: int = 32

    def _make_lane_a(
        self,
        dim,
        memory_dim,
        gate_bias,
        semiring_temp_init,
        recursive_balance_init,
        low_threshold,
        high_threshold,
        max_recursive_steps,
    ) -> nn.Module:
        return MoRRefineMLPLaneA(
            dim,
            memory_dim=memory_dim,
            gate_bias=gate_bias,
            semiring_temp_init=semiring_temp_init,
            recursive_balance_init=recursive_balance_init,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            max_recursive_steps=max_recursive_steps,
            router_hidden=self.ROUTER_HIDDEN,
        )


class MoRSurpriseRefineMLPLaneA(MoRRefineMLPLaneA):
    """Refine MLP-router lane whose depth is driven by BOTH loss AND surprise.

    Adds a structural surprise floor to the halt logit::

        halt = sigmoid( MLP([|s|,|err|,r/R]) − a·mean|err| ),  a = a_min + softplus(raw)

    The learned MLP keeps the loss in control of depth, but the ``−a·mean|err|``
    term (``a ≥ a_min > 0`` so the loss can't zero the surprise channel) makes a
    surprising token recurse deeper *by construction*. The deep-start bias is set
    strongly negative so the run starts near max depth; the loss then erodes the
    *average* depth over training while high-surprise tokens spike depth back up
    token-by-token. ``a`` itself is learned (above its floor), so the loss even
    chooses how strongly surprise weighs in — the purest "both" formulation.
    """

    def __init__(self, *args, surprise_coupling_min: float = 0.05, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # sets MLP halt_head + deep-start (bias −5)
        self.surprise_coupling_min = float(surprise_coupling_min)
        self.halt_surprise_coupling = nn.Parameter(torch.zeros(()))  # a≈a_min+0.69

    def reset_halt_deep_start(self) -> None:
        """Start near *max* depth (not the −2 ~depth-3 of the base): final bias −5
        ⇒ halt≈0 for r<R ⇒ initial depth ≈ max_recursive_steps. Surprise + loss
        take over from there."""
        super().reset_halt_deep_start()
        head = self.halt_head
        if isinstance(head, nn.Sequential):
            nn.init.constant_(head[-1].bias, -5.0)


class MoRSurpriseRefineMLPAdaptiveSemiringBiLaneSurpriseMemoryLane(
    MoRRefineMLPAdaptiveSemiringBiLaneSurpriseMemoryLane
):
    """Bilane whose lane_a is the surprise+loss hybrid-depth refine-MLP lane."""

    SURPRISE_COUPLING_MIN: float = 0.05

    def _make_lane_a(
        self,
        dim,
        memory_dim,
        gate_bias,
        semiring_temp_init,
        recursive_balance_init,
        low_threshold,
        high_threshold,
        max_recursive_steps,
    ) -> nn.Module:
        return MoRSurpriseRefineMLPLaneA(
            dim,
            memory_dim=memory_dim,
            gate_bias=gate_bias,
            semiring_temp_init=semiring_temp_init,
            recursive_balance_init=recursive_balance_init,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            max_recursive_steps=max_recursive_steps,
            router_hidden=self.ROUTER_HIDDEN,
            surprise_coupling_min=self.SURPRISE_COUPLING_MIN,
        )


def set_ponder_weight(model: nn.Module, weight: float) -> int:
    """Override ``ponder_weight`` on every MoR lane (e.g. 0.0 to let LM loss alone
    decide depth, isolating 'does depth help loss' from the compute penalty)."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, MoRLaneA):
            mod.ponder_weight = float(weight)
            n += 1
    return n


def apply_resume_init(model: nn.Module) -> int:
    """Re-apply the deep-start halt init (bias −2, zero weight) to every MoR lane.

    ``_build_tinylm``'s GPT-2 init overwrites ``halt_head`` to ~0.5 halt prob
    (mean depth ~1.9). For resuming a checkpoint trained at deep recursion we want
    the router to *start* near that trained depth so we isolate its learning, not
    a depth-shock. Call after the checkpoint load. Returns the number of lanes hit.
    """
    n = 0
    for mod in model.modules():
        if isinstance(mod, MoRLaneA):
            mod.reset_halt_deep_start()
            n += 1
    return n


def collect_ponder_cost(model: nn.Module) -> torch.Tensor | None:
    """Sum the per-lane MoR ponder costs across a model (None if there are none)."""
    total = None
    for mod in model.modules():
        if isinstance(mod, MoRLaneA) and mod.last_ponder_cost is not None:
            total = (
                mod.last_ponder_cost if total is None else total + mod.last_ponder_cost
            )
    return total
