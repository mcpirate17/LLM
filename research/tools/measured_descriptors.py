#!/usr/bin/env python
"""Closed-book MEASURED descriptors of a computation graph — name-free, label-free.

The anti-cheat representation. Instead of op names or human-*declared* catalog properties
(both are labels — predicting from them is reading the answer key), this probes the graph's
*actual computation* at RANDOM INIT (no training, no capability labels) and measures what it
does to tokens and signal. Open-book on the task (we may build an induction-shaped stimulus);
closed-book on the candidate (its name/category is never consulted).

Core instrument — the **position-Jacobian** J[s] = ‖∂(output[:,q,:]) / ∂(emb[:,s,:])‖: how much
the model's output at query position q depends on the input at each source position s, read off a
single backward pass through `SynthesizedModel._fingerprint_forward_from_embed`. From it:

  - long_range_reach    — fraction of query sensitivity on EARLIER positions (info routes back at
                          all). Necessary for induction; an MLP-class graph scores ~0.
  - content_dependence  — does the *routing distribution* change with input content (data-dependent
                          mixing = attention-class) vs stay fixed (conv/SSM-class). Name-free
                          "attention" signature.
  - content_match_gating— induction-specific: extra routing to the answer-bearing region when the
                          query token MATCHES an earlier token vs not (content-gated copy).
  - causality_violation — query-output mass on FUTURE positions at a mid query (≈0 ⇒ causal).
  - measured_lipschitz / effective_rank / nonlinearity — operator gain, expressivity (singular-value
                          participation ratio), homogeneity deviation.

This is the descriptor-system view (Gemini) meeting training-free NAS proxies (jacob_cov / NTK):
predict capability from the operator's measured behaviour, validated against labels exactly once.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional  # noqa: F401

import numpy as np
import torch

from research.eval.induction_probe import _RESTRICTED_VOCAB, _generate_induction_batch
from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json

_EPS = 1e-9
_DESCRIPTOR_NAMES = (
    "long_range_reach",
    "content_dependence",
    "content_match_gating",
    "causality_violation",
    "measured_lipschitz",
    "effective_rank",
    "nonlinearity",
    "self_dominance",
)

# Weights composing the validated descriptors into ONE capability RANK score (higher = more
# induction/binding-capable). DATA-DERIVED, not guessed: `nas_funnel_ood_eval` (n=600 labeled,
# 2026-06-03) measured each descriptor's single-feature ROC vs induction-capable —
#   content_dependence 0.81, long_range_reach 0.79 (attention-class routing-back),
#   content_match_gating 0.64 (binding-specific content-gated copy — the charter's "won't bind"),
#   causality_violation 0.49 (NOISE at random init; grammar/context rules already guard causality),
#   instability/lipschitz 0.81 but POSITIVE-correlated (expressivity/gain) — so penalizing it was
#   backwards and tanked the composite to ROC 0.22. These positive-only weights restore the
#   composite to ≈ the logistic ceiling (~0.78). Re-fit via `nas_funnel_ood_eval` if descriptors
#   change. causality_violation/instability kept at 0 (non-predictive here; stability is a separate
#   calibrated gate, not the capability rank).
_CAPABILITY_WEIGHTS = {
    "long_range_reach": 1.0,
    "content_dependence": 1.0,
    "content_match_gating": 0.5,
    "causality_violation": 0.0,
    "instability": 0.0,
}
_LIP_STABLE = (
    2.0  # measured_lipschitz beyond this is treated as gain blow-up (instability)
)


class MeasuredDescriptorExtractor:
    """Probe a graph's compiled computation at random init → name-free behavioural descriptors."""

    def __init__(
        self,
        device: Optional[str] = None,
        n_seeds: int = 3,
        gap: int = 8,
        batch: int = 16,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_seeds = n_seeds
        self.gap = gap
        self.batch = batch
        self.seq_len = gap + 3  # _generate_induction_batch layout
        self.query = self.seq_len - 1  # last position (the repeated A)
        self.mid = self.seq_len // 2

    # ── public ──────────────────────────────────────────────────────
    def descriptors(self, graph_json: str) -> Optional[Dict[str, float]]:
        """Return measured descriptors for one graph, or None if it can't be probed."""
        return self.descriptors_from_factory(
            lambda seed: self._build_from_graph(graph_json, seed)
        )

    def descriptors_from_factory(
        self, factory: Callable[[int], Any]
    ) -> Optional[Dict[str, float]]:
        """Measured descriptors for any model built by ``factory(seed)``.

        ``factory(seed)`` must return a model exposing ``embed(ids)`` and
        ``_fingerprint_forward_from_embed(emb)`` (the SynthesizedModel contract).
        This lets callers probe a module they built themselves (e.g. a
        component_fab lane wrapped in a probe adapter), not just a graph_json.
        Returns None if no seed could be probed.
        """
        per_seed: List[Dict[str, float]] = []
        for seed in range(self.n_seeds):
            try:
                d = self._probe_model(factory(seed), seed)
            except Exception:
                d = None
            if d is not None:
                per_seed.append(d)
        if not per_seed:
            return None
        return {k: float(np.mean([d[k] for d in per_seed])) for k in _DESCRIPTOR_NAMES}

    def induction_capable(self, graph_json: str, threshold: float = 0.01) -> bool:
        """Cheap (~0.4s) label-free pre-probe filter: does the graph route info backward at all?

        Validated operating point (n=1102, 2026-05-25): long_range_reach >= 0.01 keeps 99.3% of
        induction-capable graphs while pruning ~55% of incapable ones — necessary-not-sufficient,
        safe to skip the expensive training probe below it. Returns True on probe failure (fail
        open — never silently drop a candidate the probe couldn't measure).
        """
        d = self.descriptors(graph_json)
        return d is None or d["long_range_reach"] >= threshold

    def capability_score(
        self, graph_json: str, weights: Optional[Dict[str, float]] = None
    ) -> Optional[float]:
        """Measured capability RANK score for one graph (higher = more induction/binding-capable).

        Read off the graph's actual computation (OOD-robust, label-free) — the signal that ranked
        novel winners where the declared-feature oracle collapsed. Returns None if unprobeable.
        Reuses ``descriptors()`` (no new probe); composition + weights in
        ``capability_score_from_descriptors``.
        """
        d = self.descriptors(graph_json)
        if d is None:
            return None
        return capability_score_from_descriptors(d, weights)

    # ── one random-init probe ───────────────────────────────────────
    def _build_from_graph(self, graph_json: str, seed: int) -> Any:
        torch.manual_seed(seed)
        graph = graph_from_json(graph_json)
        return compile_model([graph], use_ir=False).to(self.device).eval()

    def _probe_model(self, model: Any, seed: int) -> Optional[Dict[str, float]]:
        gen = torch.Generator(device=self.device).manual_seed(seed)

        ids_match, _ = _generate_induction_batch(self.batch, self.gap, self.device, gen)
        ids_nomatch = _break_query_match(ids_match, self.query, gen)
        ids_randA = _random_ids(self.batch, self.seq_len, self.device, gen)
        ids_randB = _random_ids(self.batch, self.seq_len, self.device, gen)

        j_match = self._pos_jac(model, ids_match, self.query)
        j_nomatch = self._pos_jac(model, ids_nomatch, self.query)
        j_a = self._pos_jac(model, ids_randA, self.query)
        j_b = self._pos_jac(model, ids_randB, self.query)
        j_mid = self._pos_jac(model, ids_randA, self.mid)

        f: Dict[str, float] = {}
        total = float(j_a.sum() + _EPS)
        earlier = j_a[: self.query].sum()  # all source positions before the query
        f["self_dominance"] = float(j_a[self.query] / total)
        f["long_range_reach"] = float(earlier / total)
        f["content_dependence"] = _route_tv(j_a, j_b)
        # induction-specific: routing to the answer-bearing head (pos 0,1) gained by a match
        ans = slice(0, 2)
        gate = _route_frac(j_match, ans) - _route_frac(j_nomatch, ans)
        f["content_match_gating"] = float(max(gate, 0.0))
        # causality at a mid query: sensitivity to FUTURE positions (should be ~0 if causal)
        fut = j_mid[self.mid + 1 :].sum()
        f["causality_violation"] = float(fut / float(j_mid.sum() + _EPS))
        self._signal_descriptors(model, ids_randA, f)
        return f

    def _signal_descriptors(
        self, model: Any, ids: torch.Tensor, f: Dict[str, float]
    ) -> None:
        """Operator gain, expressivity (effective rank), nonlinearity — from forward passes."""
        with torch.no_grad():
            emb = model.embed(ids)
            out = model._fingerprint_forward_from_embed(emb)
            delta = torch.randn_like(emb)
            eps = 1e-2 * emb.norm() / (delta.norm() + _EPS)
            out_pert = model._fingerprint_forward_from_embed(emb + eps * delta)
            f["measured_lipschitz"] = float(
                (out_pert - out).norm() / (eps * delta.norm() + _EPS)
            )
            out2 = model._fingerprint_forward_from_embed(2.0 * emb)
            f["nonlinearity"] = float(
                (out2 - 2.0 * out).norm() / (2.0 * out.norm() + _EPS)
            )
            flat = out.reshape(-1, out.shape[-1])
            f["effective_rank"] = _effective_rank(flat)

    def _pos_jac(self, model: Any, ids: torch.Tensor, query: int) -> torch.Tensor:
        """J[s] = mean_batch ‖∂(output[:,query,:]) / ∂(emb[:,s,:])‖ — query's reliance on each pos."""
        emb = model.embed(ids).detach().requires_grad_(True)
        out = model._fingerprint_forward_from_embed(emb)
        scalar = 0.5 * (out[:, query, :] ** 2).sum()
        (grad,) = torch.autograd.grad(scalar, emb)
        return grad.norm(dim=-1).mean(0).detach()  # [seq_len]


# --------------------------------------------------------------------------- #
# stimulus + measure helpers (pure)
# --------------------------------------------------------------------------- #
def _random_ids(
    batch: int, seq_len: int, device: str, gen: torch.Generator
) -> torch.Tensor:
    return torch.randint(
        1, _RESTRICTED_VOCAB, (batch, seq_len), device=device, generator=gen
    )


def _break_query_match(
    ids_match: torch.Tensor, query: int, gen: torch.Generator
) -> torch.Tensor:
    """Replace the query token (the planted repeat) with one absent earlier ⇒ no content match."""
    ids = ids_match.clone()
    for b in range(ids.shape[0]):
        present = set(ids[b, :query].tolist())
        tok = int(
            torch.randint(1, _RESTRICTED_VOCAB, (1,), device=ids.device, generator=gen)
        )
        while tok in present:
            tok = (tok % (_RESTRICTED_VOCAB - 1)) + 1
        ids[b, query] = tok
    return ids


def _route_frac(j: torch.Tensor, region: slice) -> float:
    return float(j[region].sum() / (j.sum() + _EPS))


def _route_tv(j_a: torch.Tensor, j_b: torch.Tensor) -> float:
    """Total-variation distance between the two routing DISTRIBUTIONS (content-dependence)."""
    pa = j_a / (j_a.sum() + _EPS)
    pb = j_b / (j_b.sum() + _EPS)
    return float(0.5 * (pa - pb).abs().sum())


def _effective_rank(x: torch.Tensor) -> float:
    """exp(entropy of normalized singular values) — soft rank = expressivity of the representation."""
    try:
        s = torch.linalg.svdvals(x.float())
    except Exception:
        return 0.0
    s = s[s > _EPS]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()))


def capability_score_from_descriptors(
    d: Dict[str, float], weights: Optional[Dict[str, float]] = None
) -> float:
    """Compose measured descriptors into one capability RANK score (higher = more capable).

    Pure; see ``_CAPABILITY_WEIGHTS`` for the term rationale. ``instability`` is the gain blow-up
    of ``measured_lipschitz`` past ``_LIP_STABLE``. Missing descriptors/weights default to 0.
    """
    w = weights or _CAPABILITY_WEIGHTS
    instability = max(0.0, float(d.get("measured_lipschitz", 0.0)) - _LIP_STABLE)
    return (
        w.get("long_range_reach", 0.0) * float(d.get("long_range_reach", 0.0))
        + w.get("content_match_gating", 0.0) * float(d.get("content_match_gating", 0.0))
        + w.get("content_dependence", 0.0) * float(d.get("content_dependence", 0.0))
        + w.get("causality_violation", 0.0) * float(d.get("causality_violation", 0.0))
        + w.get("instability", 0.0) * instability
    )


DESCRIPTOR_NAMES = _DESCRIPTOR_NAMES
