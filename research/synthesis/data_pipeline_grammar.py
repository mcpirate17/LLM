"""Data-pipeline search grammar — how tokens are folded/packed/routed.

The architecture search already explores *what computes* (mixer math, routing,
recursion). It does **not** search *how data is fed* — packing, ordering, and
folding of the token stream are hard-coded in the batcher. This module makes the
data pipeline a first-class, sampleable genotype (a ``DataRouteSpec``) so a
candidate carries its data route the same way it carries its math axes.

Scope of this increment (start small, expand — see
``tasks/loss_monster_scaffolding_plan.md`` Workstream D):

* ``order`` and ``fold`` are implemented as a **pure, shape-preserving position
  permutation** over ``[..., L]`` token tensors. Fold count, fraction, traversal
  direction, dense/sparse/intermittent pattern, and horizontal/vertical
  orientation are explicit axes. Because each route is a permutation it never
  invents or drops tokens, is deterministic, and is invertible — so it drops
  into any next-token batch path (the train loop's shift-by-one still holds, it
  just changes which token is "next").
* ``pack`` (corpus-window selection) and ``route`` (span-to-submodule) are
  genotype fields. Window packing is owned by
  ``research.training.window_packing``; model-side segment routing is exposed as
  segment ids / gate bias here and must be consumed by a paired route-aware
  block. The sampler emits only post-sample token transforms so generic batcher
  wrappers never silently no-op on corpus- or model-cooperative routes.

Grade every route on capability at a fixed token budget, never on loss. The goal
is to route far-apart evidence into novel mixers sooner, then keep the routes
that lift induction/binding/state capability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

# Genotype value sets. The identity value is always first.
DATA_PACKS: tuple[str, ...] = (
    "contiguous",  # wired
    "doc_boundary",  # follow-up (needs corpus internals)
    "length_bucketed",  # follow-up
    "best_fit",  # follow-up
)
DATA_ORDERS: tuple[str, ...] = (
    "natural",  # identity
    "reverse",  # flip the whole window
    "bidirectional",  # first half forward, second half reversed
)
SEQ_FOLDS: tuple[int, ...] = (1, 8, 16, 32)  # 1 == no fold; else serpentine factor
SEQ_FOLD_FRACTIONS: tuple[float, ...] = (
    0.25,
    0.5,
    0.75,
    1.0,
)
SEQ_FOLD_DIRECTIONS: tuple[str, ...] = (
    "alternate",  # boustrophedon: every other fold traverses backward
    "forward",  # every fold traverses left-to-right / top-to-bottom
    "backward",  # every fold traverses right-to-left / bottom-to-top
)
SEQ_FOLD_PATTERNS: tuple[str, ...] = (
    "dense",  # fold every selected position
    "sparse",  # fold every other selected position, leave gaps in place
    "intermittent",  # fold alternating blocks, leave cooldown blocks in place
)
SEQ_FOLD_ORIENTATIONS: tuple[str, ...] = (
    "horizontal",  # segment-wise within the sequence axis
    "vertical",  # column-wise across fold segments, maximizes distant adjacency
)
DATA_ROUTES: tuple[str, ...] = (
    "none",  # all positions to one path
    "local_global_split",  # positional: first half -> local lane, rest -> carrier
    "surprisal_split",  # data-driven: high monster-surprisal tokens -> carrier
)

# Axis keys this genotype contributes to a candidate's logged spec.
AXIS_PACK = "op_data_pack"
AXIS_ORDER = "op_data_order"
AXIS_FOLD = "op_seq_fold"
AXIS_FOLD_FRACTION = "op_seq_fold_fraction"
AXIS_FOLD_DIRECTION = "op_seq_fold_direction"
AXIS_FOLD_PATTERN = "op_seq_fold_pattern"
AXIS_FOLD_ORIENTATION = "op_seq_fold_orientation"
AXIS_ROUTE = "op_data_route"


@dataclass(frozen=True, slots=True)
class DataRouteSpec:
    """How a token window is packed, ordered, folded, and routed."""

    pack: str = "contiguous"
    order: str = "natural"
    fold: int = 1
    fold_fraction: float = 1.0
    fold_direction: str = "alternate"
    fold_pattern: str = "dense"
    fold_orientation: str = "horizontal"
    route: str = "none"
    # Fraction of positions sent to the carrier (long-range) lane when a split
    # route is active. 0.3 = the 01:00Z surprisal scan's actionable "hard ~30%".
    carrier_fraction: float = 0.3

    def __post_init__(self) -> None:
        if self.pack not in DATA_PACKS:
            raise ValueError(f"unknown data pack {self.pack!r}; valid={DATA_PACKS}")
        if self.order not in DATA_ORDERS:
            raise ValueError(f"unknown data order {self.order!r}; valid={DATA_ORDERS}")
        if int(self.fold) not in SEQ_FOLDS:
            raise ValueError(f"unknown seq fold {self.fold!r}; valid={SEQ_FOLDS}")
        if not _is_supported_fraction(float(self.fold_fraction)):
            raise ValueError(
                f"unknown seq fold fraction {self.fold_fraction!r}; "
                f"valid={SEQ_FOLD_FRACTIONS}"
            )
        if self.fold_direction not in SEQ_FOLD_DIRECTIONS:
            raise ValueError(
                f"unknown seq fold direction {self.fold_direction!r}; "
                f"valid={SEQ_FOLD_DIRECTIONS}"
            )
        if self.fold_pattern not in SEQ_FOLD_PATTERNS:
            raise ValueError(
                f"unknown seq fold pattern {self.fold_pattern!r}; "
                f"valid={SEQ_FOLD_PATTERNS}"
            )
        if self.fold_orientation not in SEQ_FOLD_ORIENTATIONS:
            raise ValueError(
                f"unknown seq fold orientation {self.fold_orientation!r}; "
                f"valid={SEQ_FOLD_ORIENTATIONS}"
            )
        if self.route not in DATA_ROUTES:
            raise ValueError(f"unknown data route {self.route!r}; valid={DATA_ROUTES}")
        if not 0.0 <= float(self.carrier_fraction) <= 1.0:
            raise ValueError(
                f"carrier_fraction must be in [0, 1], got {self.carrier_fraction}"
            )

    @property
    def key(self) -> str:
        return (
            f"{self.pack}/{self.order}/fold{self.fold}"
            f"x{self.fold_fraction:.2f}/{self.fold_orientation}/"
            f"{self.fold_direction}/{self.fold_pattern}/"
            f"{self.route}@{self.carrier_fraction:.2f}"
        )

    @property
    def is_token_identity(self) -> bool:
        """True when pack/order/fold leave the token *stream* untouched.

        ``route`` is orthogonal — it assigns positions to submodules (a segment
        map the model consumes), it does not permute tokens — so it does not
        affect ``apply_data_route``.
        """
        return (
            self.pack == "contiguous"
            and self.order == "natural"
            and int(self.fold) == 1
        )

    @property
    def is_identity(self) -> bool:
        """True when the whole route (incl. submodule assignment) is a no-op."""
        return self.is_token_identity and self.route == "none"


AXIS_CARRIER_FRACTION = "op_data_carrier_fraction"


def data_route_to_axes(spec: DataRouteSpec) -> dict[str, Any]:
    """Flatten the genotype into ``op_data_*`` axes for the candidate's spec."""
    return {
        AXIS_PACK: spec.pack,
        AXIS_ORDER: spec.order,
        AXIS_FOLD: int(spec.fold),
        AXIS_FOLD_FRACTION: float(spec.fold_fraction),
        AXIS_FOLD_DIRECTION: spec.fold_direction,
        AXIS_FOLD_PATTERN: spec.fold_pattern,
        AXIS_FOLD_ORIENTATION: spec.fold_orientation,
        AXIS_ROUTE: spec.route,
        AXIS_CARRIER_FRACTION: float(spec.carrier_fraction),
    }


def data_route_from_axes(axes: dict[str, Any]) -> DataRouteSpec:
    """Rebuild a ``DataRouteSpec`` from a candidate's axes (round-trips)."""
    raw_fraction = axes.get(AXIS_CARRIER_FRACTION)
    return DataRouteSpec(
        pack=str(axes.get(AXIS_PACK) or "contiguous"),
        order=str(axes.get(AXIS_ORDER) or "natural"),
        fold=int(axes.get(AXIS_FOLD) or 1),
        fold_fraction=float(axes.get(AXIS_FOLD_FRACTION) or 1.0),
        fold_direction=str(axes.get(AXIS_FOLD_DIRECTION) or "alternate"),
        fold_pattern=str(axes.get(AXIS_FOLD_PATTERN) or "dense"),
        fold_orientation=str(axes.get(AXIS_FOLD_ORIENTATION) or "horizontal"),
        route=str(axes.get(AXIS_ROUTE) or "none"),
        carrier_fraction=0.3 if raw_fraction is None else float(raw_fraction),
    )


def _is_supported_fraction(value: float) -> bool:
    return any(math.isclose(value, allowed) for allowed in SEQ_FOLD_FRACTIONS)


def _order_permutation(length: int, order: str) -> torch.Tensor:
    """Position indices implementing the global ``order`` over ``length``."""
    base = torch.arange(length)
    if order == "natural":
        return base
    if order == "reverse":
        return base.flip(0)
    if order == "bidirectional":
        half = length // 2
        return torch.cat([base[:half], base[half:].flip(0)])
    raise ValueError(f"unimplemented data order {order!r}")


def _active_fold_length(length: int, fold: int, fraction: float) -> int:
    """How many leading positions participate in folding."""

    if fold <= 1:
        return length
    active = int(round(length * float(fraction)))
    active = max(fold, active)
    return min(length, active)


def _fold_selection(active_len: int, fold: int, pattern: str) -> torch.Tensor:
    """Positions within the active prefix that the fold transform touches."""

    positions = torch.arange(active_len)
    if pattern == "dense":
        return positions
    if pattern == "sparse":
        return positions[::2]
    if pattern == "intermittent":
        block = max(1, int(fold))
        keep = (positions // block) % 2 == 0
        return positions[keep]
    raise ValueError(f"unimplemented seq fold pattern {pattern!r}")


def _split_segments(values: torch.Tensor, fold: int) -> list[torch.Tensor]:
    length = values.shape[0]
    if fold > length:
        raise ValueError(f"seq fold {fold} exceeds sequence length {length}")
    seg_len = length // fold
    segments: list[torch.Tensor] = []
    for s in range(fold):
        start = s * seg_len
        stop = length if s == fold - 1 else start + seg_len
        segments.append(values[start:stop])
    return segments


def _fold_horizontal(values: torch.Tensor, fold: int, direction: str) -> torch.Tensor:
    segments = _split_segments(values, fold)
    out: list[torch.Tensor] = []
    for index, segment in enumerate(segments):
        reverse = direction == "backward" or (
            direction == "alternate" and index % 2 == 1
        )
        out.append(segment.flip(0) if reverse else segment)
    return torch.cat(out)


def _fold_vertical(values: torch.Tensor, fold: int, direction: str) -> torch.Tensor:
    segments = _split_segments(values, fold)
    max_len = max(segment.shape[0] for segment in segments)
    out: list[torch.Tensor] = []
    for column in range(max_len):
        row_order = range(len(segments))
        if direction == "backward" or (direction == "alternate" and column % 2 == 1):
            row_order = reversed(range(len(segments)))
        for row in row_order:
            segment = segments[row]
            if column < segment.shape[0]:
                out.append(segment[column : column + 1])
    return torch.cat(out)


def _fold_values(
    values: torch.Tensor, *, fold: int, direction: str, orientation: str
) -> torch.Tensor:
    if values.shape[0] <= 1 or fold <= 1:
        return values
    if orientation == "horizontal":
        return _fold_horizontal(values, fold, direction)
    if orientation == "vertical":
        return _fold_vertical(values, fold, direction)
    raise ValueError(f"unimplemented seq fold orientation {orientation!r}")


def _apply_fold(perm: torch.Tensor, spec: DataRouteSpec) -> torch.Tensor:
    """Fold selected positions. Always returns a pure permutation of ``perm``."""
    fold = int(spec.fold)
    fold = int(fold)
    if fold <= 1:
        return perm
    length = perm.shape[0]
    if fold > length:
        raise ValueError(f"seq fold {fold} exceeds sequence length {length}")
    active_len = _active_fold_length(length, fold, spec.fold_fraction)
    selected = _fold_selection(active_len, fold, spec.fold_pattern)
    effective_fold = max(1, min(fold, int(selected.shape[0])))
    out = perm.clone()
    out[selected] = _fold_values(
        perm[selected],
        fold=effective_fold,
        direction=spec.fold_direction,
        orientation=spec.fold_orientation,
    )
    return out


def route_permutation(length: int, spec: DataRouteSpec) -> torch.Tensor:
    """Composed position permutation (order then fold) for a window of ``length``.

    Returns ``arange(length)`` for the identity route. Always a permutation, so
    ``tokens.index_select(-1, perm)`` is shape-preserving and token-preserving.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    perm = _order_permutation(length, spec.order)
    return _apply_fold(perm, spec)


def apply_data_route(tokens: torch.Tensor, spec: DataRouteSpec) -> torch.Tensor:
    """Apply a ``DataRouteSpec``'s token *permutation* to a ``[..., L]`` tensor.

    Pure and deterministic. Only ``order``/``fold`` permute tokens; ``route`` is
    a submodule-assignment (see :func:`route_segment_ids`) and does NOT affect
    the token stream, so it is ignored here. ``pack`` non-identity values need
    corpus-level window selection and fail loud.
    """
    if spec.pack != "contiguous":
        raise NotImplementedError(
            f"data pack {spec.pack!r} needs corpus-level window selection "
            "(CorpusTokenBatcher); not wired in apply_data_route yet"
        )
    if spec.is_token_identity:
        return tokens
    if tokens.ndim < 1 or tokens.shape[-1] == 0:
        raise ValueError(
            f"expected a non-empty [..., L] tensor, got {tuple(tokens.shape)}"
        )
    perm = route_permutation(tokens.shape[-1], spec).to(tokens.device)
    return tokens.index_select(-1, perm)


# ---------------- surprisal-driven span routing (signal -> segments) ----------
#
# Bridges the 01:00Z monster-surprisal scorer to the route axis: the scorer
# emits per-token surprisal (bits); we turn the hardest ``carrier_fraction`` of
# positions into carrier-bound segments and the rest into cheap-lane segments.
# ``LossMonsterPairedBlock`` (Workstream B) consumes the gate bias so the
# carrier (long-range/induction) lane handles the hard tokens and the local loss
# specialist handles the predictable remainder.

CARRIER_SEGMENT = 1  # long-range / induction carrier lane
LOCAL_SEGMENT = 0  # cheap local (loss-monster) lane


def route_segments_from_surprisal(
    surprisal: torch.Tensor, carrier_fraction: float
) -> torch.Tensor:
    """Per-row top-``carrier_fraction`` surprisal positions -> ``CARRIER_SEGMENT``.

    ``surprisal`` is ``[..., L]`` (e.g. the monster scorer's bits/token). Returns
    a same-shape int64 segment map. Exact top-k (not a threshold) so ties never
    push more than ``round(L * carrier_fraction)`` tokens to the carrier.
    """
    if surprisal.ndim < 1 or surprisal.shape[-1] == 0:
        raise ValueError(
            f"surprisal must be a non-empty [..., L] tensor, got {tuple(surprisal.shape)}"
        )
    length = surprisal.shape[-1]
    k = int(round(length * float(carrier_fraction)))
    segments = torch.full_like(surprisal, LOCAL_SEGMENT, dtype=torch.long)
    if k <= 0:
        return segments
    if k >= length:
        return torch.full_like(segments, CARRIER_SEGMENT)
    carrier_idx = surprisal.topk(k, dim=-1).indices
    segments.scatter_(-1, carrier_idx, CARRIER_SEGMENT)
    return segments


def route_segment_ids(
    spec: DataRouteSpec,
    *,
    length: int | None = None,
    surprisal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-position submodule assignment for ``spec.route`` (the model consumes it).

    * ``none`` -> all ``LOCAL_SEGMENT``.
    * ``local_global_split`` -> first half local, second half carrier (positional).
    * ``surprisal_split`` -> hardest ``carrier_fraction`` by ``surprisal`` -> carrier.
    """
    if spec.route == "surprisal_split":
        if surprisal is None:
            raise ValueError("surprisal_split route requires per-token surprisal")
        return route_segments_from_surprisal(surprisal, spec.carrier_fraction)
    if length is None or length <= 0:
        raise ValueError(f"route_segment_ids needs a positive length, got {length}")
    if spec.route == "none":
        return torch.full((length,), LOCAL_SEGMENT, dtype=torch.long)
    if spec.route == "local_global_split":
        cut = int(round(length * (1.0 - spec.carrier_fraction)))
        segments = torch.full((length,), LOCAL_SEGMENT, dtype=torch.long)
        segments[cut:] = CARRIER_SEGMENT
        return segments
    raise ValueError(f"unimplemented data route {spec.route!r}")


def gate_bias_from_segments(
    segments: torch.Tensor, *, strength: float = 4.0
) -> torch.Tensor:
    """Additive gate-logit bias: ``+strength`` for carrier, ``-strength`` else.

    Shaped ``[..., 1]`` to add onto ``LossMonsterPairedBlock``'s ``[..., 1]``
    gate logit so a positive bias drives the partner (carrier) weight up. A
    large ``strength`` makes routing near-hard; a small one nudges the learned
    gate with the surprisal prior.
    """
    sign = segments.to(torch.float32) * 2.0 - 1.0  # carrier(1)->+1, local(0)->-1
    return (sign * float(strength)).unsqueeze(-1)


def sample_data_route_spec(gen: torch.Generator) -> DataRouteSpec:
    """Sample an implemented data route (order x fold controls; pack/route identity).

    Deterministic given ``gen``. Only the wired axes vary so the search never
    emits a spec ``apply_data_route`` would reject.
    """
    order = DATA_ORDERS[int(torch.randint(len(DATA_ORDERS), (1,), generator=gen))]
    fold = int(SEQ_FOLDS[int(torch.randint(len(SEQ_FOLDS), (1,), generator=gen))])
    fraction = float(
        SEQ_FOLD_FRACTIONS[
            int(torch.randint(len(SEQ_FOLD_FRACTIONS), (1,), generator=gen))
        ]
    )
    direction = SEQ_FOLD_DIRECTIONS[
        int(torch.randint(len(SEQ_FOLD_DIRECTIONS), (1,), generator=gen))
    ]
    pattern = SEQ_FOLD_PATTERNS[
        int(torch.randint(len(SEQ_FOLD_PATTERNS), (1,), generator=gen))
    ]
    orientation = SEQ_FOLD_ORIENTATIONS[
        int(torch.randint(len(SEQ_FOLD_ORIENTATIONS), (1,), generator=gen))
    ]
    return DataRouteSpec(
        pack="contiguous",
        order=order,
        fold=fold,
        fold_fraction=fraction,
        fold_direction=direction,
        fold_pattern=pattern,
        fold_orientation=orientation,
        route="none",
    )


DEFAULT_DATA_ROUTE_VARIANT_NAMES: tuple[str, ...] = (
    "reverse",
    "bidirectional",
    "fold8_horizontal_alternate",
    "fold16_vertical_alternate",
    "fold32_vertical_alternate",
    "fold16_vertical_half",
    "fold16_sparse_vertical",
    "fold16_intermittent_horizontal",
    "fold8_backward_horizontal",
    "fold16_bidirectional_vertical",
    "doc_boundary",
    "surprisal_split_30",
    "local_global_30",
)


def implemented_data_route_specs() -> dict[str, DataRouteSpec]:
    """Named route genotypes that are safe to carry as candidate axes."""

    return {
        "reverse": DataRouteSpec(order="reverse"),
        "bidirectional": DataRouteSpec(order="bidirectional"),
        "fold8_horizontal_alternate": DataRouteSpec(fold=8),
        "fold16_vertical_alternate": DataRouteSpec(
            fold=16,
            fold_orientation="vertical",
            fold_direction="alternate",
        ),
        "fold32_vertical_alternate": DataRouteSpec(
            fold=32,
            fold_orientation="vertical",
            fold_direction="alternate",
        ),
        "fold16_vertical_half": DataRouteSpec(
            fold=16,
            fold_fraction=0.5,
            fold_orientation="vertical",
            fold_direction="alternate",
        ),
        "fold16_sparse_vertical": DataRouteSpec(
            fold=16,
            fold_pattern="sparse",
            fold_orientation="vertical",
            fold_direction="alternate",
        ),
        "fold16_intermittent_horizontal": DataRouteSpec(
            fold=16,
            fold_pattern="intermittent",
            fold_orientation="horizontal",
            fold_direction="backward",
        ),
        "fold8_backward_horizontal": DataRouteSpec(
            fold=8,
            fold_direction="backward",
        ),
        "fold16_bidirectional_vertical": DataRouteSpec(
            order="bidirectional",
            fold=16,
            fold_orientation="vertical",
            fold_direction="alternate",
        ),
        "doc_boundary": DataRouteSpec(pack="doc_boundary"),
        "surprisal_split_30": DataRouteSpec(
            route="surprisal_split",
            carrier_fraction=0.3,
        ),
        "local_global_30": DataRouteSpec(
            route="local_global_split",
            carrier_fraction=0.3,
        ),
    }


def batchable_data_route_specs() -> dict[str, DataRouteSpec]:
    """Named routes a corpus A/B runner can apply without model-side routing.

    ``surprisal_split`` and ``local_global_split`` are implemented as segment
    maps, but they require a model that consumes ``route_prior``. Keep them
    search-visible via :func:`implemented_data_route_specs` and out of generic
    next-token batch A/Bs where they would otherwise be no-ops.
    """

    return {
        name: spec
        for name, spec in implemented_data_route_specs().items()
        if spec.route == "none"
    }
