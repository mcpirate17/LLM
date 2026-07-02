"""NM-F7: nonabelian group-convolution sequence mixer.

The discrete-group sibling of the nilpotent-Lie signature scan (NM-F3). A token
selects a group element ``g_t`` of a small **nonabelian** group G; the op
transports a state through the sequence by the group's left-regular
representation, ``S_t = M_t S_{t-1} + v_t`` with ``M_t = sum_i pi_i(x_t) P(g_i)``
a convex combination of the fixed ``|G|x|G|`` permutation matrices ``P(g_i)``.

Because the ``P(g_i)`` do **not** commute, the cumulative transport
``prod_t M_t`` is order-sensitive: it encodes the *ordered* group-word product of
the token sequence, which an abelian (ordinary) convolution provably cannot.
This is a genuinely non-QKV mixing law — no query-key similarity, no softmax over
tokens; the mixing weights are a fixed group action and the only learned parts
are the per-token element selector and the value/readout lifts.

Falsifiability (the ~0.3M structured-vs-scrambled probe this op exists for): the
regular representation ``P`` is a fixed buffer. Replace it with ``scramble``d
permutations — same shapes, same doubly-stochastic transport, identity kept at
init — that do **not** form a group (closure broken). If a task that requires
ordered group composition degrades under scrambling, the nonabelian structure is
load-bearing; if not, F7 is falsified cheaply.

Self-contained on purpose — imports only ``torch``. The Python scan over the
sequence is the probe-scale reference; a native scan (NM-F §5) is the production
hot path.
"""

from __future__ import annotations

import torch
from torch import nn


def _dihedral_mult_table(order: int) -> list[list[int]]:
    """Cayley table of the dihedral group of the given (even) order.

    Elements are indexed ``idx = f * n + k`` for reflection flag ``f in {0,1}``
    and rotation ``k in [0, n)`` (``n = order // 2``), i.e. ``s^f r^k``.
    Product law ``s^{f1} r^{k1} . s^{f2} r^{k2} = s^{f1+f2} r^{(-1)^{f2} k1 + k2}``.
    D_n is nonabelian for ``n >= 3`` (order >= 6).
    """
    n = order // 2
    table = [[0] * order for _ in range(order)]
    for i in range(order):
        f1, k1 = divmod(i, n)
        for j in range(order):
            f2, k2 = divmod(j, n)
            f = (f1 + f2) % 2
            k = ((k1 if f2 == 0 else -k1) + k2) % n
            table[i][j] = f * n + k
    return table


def _regular_representation(order: int) -> torch.Tensor:
    """Left-regular representation: ``P[i][a, b] = 1`` iff ``g_i . g_b = g_a``."""
    table = _dihedral_mult_table(order)
    reps = torch.zeros(order, order, order)
    for i in range(order):
        for b in range(order):
            reps[i, table[i][b], b] = 1.0
    return reps


def _scrambled_representation(order: int, seed: int) -> torch.Tensor:
    """A structure-destroying control: keep ``P[0] = I`` (so identity-at-init and
    the doubly-stochastic transport survive) but replace the other ``order - 1``
    permutation matrices with random permutations that do not form a group."""
    gen = torch.Generator().manual_seed(seed)
    reps = torch.zeros(order, order, order)
    reps[0] = torch.eye(order)
    for i in range(1, order):
        perm = torch.randperm(order, generator=gen)
        reps[i, perm, torch.arange(order)] = 1.0
    return reps


class NonabelianGroupConv(nn.Module):
    """Nonabelian group-transport state mixer, ``[B, L, D] -> [B, L, D]``.

    Args:
        dim: model dimension.
        group_order: order of the dihedral group G (even, ``>= 6`` so G is
            nonabelian). Default 8 (the symmetries of a square, D4).
        state_width: value channels carried per group slot. Inner state is
            ``group_order * state_width``.
        scramble_seed: ``None`` uses the true regular representation; an int
            uses the structure-destroying control (for the falsification probe).
    """

    def __init__(
        self,
        dim: int,
        *,
        group_order: int = 8,
        state_width: int = 4,
        scramble_seed: int | None = None,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if group_order < 6 or group_order % 2 != 0:
            raise ValueError(
                f"group_order must be even and >= 6 (nonabelian dihedral), "
                f"got {group_order}"
            )
        if state_width < 1:
            raise ValueError(f"state_width must be >= 1, got {state_width}")

        self.d = dim
        self.g = group_order
        self.m = state_width
        self.inner = group_order * state_width

        reps = (
            _regular_representation(group_order)
            if scramble_seed is None
            else _scrambled_representation(group_order, scramble_seed)
        )
        self.register_buffer("reps", reps)  # (G, G, G), fixed — never trained

        self.select = nn.Linear(dim, group_order, bias=True)
        self.value = nn.Linear(dim, self.inner, bias=False)
        self.readout = nn.Linear(self.inner, dim, bias=False)

        # Identity-at-init: bias the selector hard onto the identity element
        # (index 0) so the transport is ~I, and zero the readout so the whole
        # branch is a no-op at step 0.
        nn.init.zeros_(self.select.weight)
        with torch.no_grad():
            self.select.bias.zero_()
            self.select.bias[0] = 4.0
        nn.init.zeros_(self.readout.weight)

    @property
    def num_parameters(self) -> int:
        # select (G*D + G) + value (inner*D) + readout (D*inner)
        return self.g * self.d + self.g + 2 * self.inner * self.d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.d:
            raise ValueError(f"expected [B, L, {self.d}], got {tuple(x.shape)}")
        n, seq_len, _ = x.shape

        pi = torch.softmax(self.select(x), dim=-1)  # (N, L, G)
        # M_t = sum_i pi_i P(g_i)  -> (N, L, G, G), doubly stochastic. Group axes
        # a, c; batch n; the selected element index i.
        transport = torch.einsum("nli,iac->nlac", pi, self.reps)
        v = self.value(x).view(n, seq_len, self.g, self.m)  # (N, L, G, m)

        state = x.new_zeros(n, self.g, self.m)
        outputs = []
        for t in range(seq_len):
            state = torch.einsum("nac,ncm->nam", transport[:, t], state) + v[:, t]
            outputs.append(state.reshape(n, self.inner))
        features = torch.stack(outputs, dim=1)  # (N, L, inner)
        return x + self.readout(features)
