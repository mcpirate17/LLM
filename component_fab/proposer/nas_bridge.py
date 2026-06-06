"""Bridge: NAS synthesis graphs -> gradeable component_fab lanes.

The fab generator is a recombination engine over registered primitives; the NAS
grammar (``research.synthesis``) generates genuinely novel op-DAG TOPOLOGIES that
fab's fixed templates cannot express (arbitrary split/fuse/route/recurse
arrangements). This bridge compiles such graphs into fab lanes so
never-before-tried structures flow through the same capability -> Tier-2 -> BLiMP
funnel as every other candidate.

Robustness contract: every graph is compiled and forward-checked AT THE FAB
GRADING DIM up front. Anything that fails to generate / compile / forward-finite
at that dim is dropped here (fail loud at the bridge — log + skip), never emitted
as a spec. The grading loop has no per-spec exception guard, so this up-front
filter is what keeps the autonomous cycle alive. It also correctly drops DB
graphs trained at a larger dim that don't re-dimension cleanly.

The graph JSON is cached under ``catalog/nas_graphs/<fingerprint>.json`` so a spec
is self-contained and re-gradeable; ``code_generator._dispatch_nas_graph``
reloads it by fingerprint.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import torch

from component_fab.proposer.spec_generator import ProposalSpec, make_proposal_id

_LOG = logging.getLogger(__name__)
_REPO = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO / "component_fab" / "catalog" / "nas_graphs"

AXIS_SOURCE = "op_source"
AXIS_FINGERPRINT = "op_nas_fingerprint"
SOURCE_NAS = "nas_graph"


def _cache_path(fingerprint: str) -> Path:
    return _CACHE_DIR / f"{fingerprint}.json"


@lru_cache(maxsize=4096)
def load_cached_graph_json(fingerprint: str) -> str | None:
    """Return the cached graph JSON for ``fingerprint`` (None if absent).

    Cached because ``code_generator._dispatch_nas_graph`` calls this once per
    NAS-guided spec — typically hundreds of times per fab cycle. The function
    is pure (same fingerprint ⇒ same content as long as the cache file is not
    rewritten underneath us). If the cache file is updated externally, callers
    can invoke ``load_cached_graph_json.cache_clear()``.
    """
    path = _cache_path(fingerprint)
    return path.read_text(encoding="utf-8") if path.exists() else None


def _compiles_finite(graph: Any, dim: int) -> bool:
    """True iff ``graph`` compiles and forwards a finite (B,L,D) tensor at ``dim``."""
    from research.synthesis.compiler import compile_graph

    try:
        module = compile_graph(graph, use_ir=True)
        with torch.no_grad():
            y = module(torch.randn(2, 16, dim))
        return bool(y.shape[-1] == dim and torch.isfinite(y).all().item())
    except Exception as exc:  # noqa: BLE001 — a bad graph must not reach grading
        _LOG.debug("nas graph rejected at dim=%d: %s", dim, exc)
        return False


def build_graph_spec(
    graph: Any,
    *,
    dim: int,
    origin: str,
    desc: str = "",
) -> ProposalSpec | None:
    """Compile-test ``graph`` at ``dim``, cache it, and wrap it as a ProposalSpec.

    Returns None when the graph does not compile/forward-finite at ``dim``.
    """
    from research.synthesis.serializer import graph_from_json, graph_to_json

    if graph.model_dim != dim:
        # Re-dimension to the grading dim; graphs with a baked larger dim fail the
        # compile-test below and are dropped.
        graph = graph_from_json(graph_to_json(graph), model_dim=dim)
    if not _compiles_finite(graph, dim):
        return None

    fingerprint = graph.fingerprint()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(fingerprint).write_text(graph_to_json(graph), encoding="utf-8")

    name = f"nas_{origin}_{fingerprint[:12]}"
    axes: dict[str, Any] = {
        AXIS_SOURCE: SOURCE_NAS,
        AXIS_FINGERPRINT: fingerprint,
        "op_nas_origin": origin,
        "op_nas_ops": int(graph.n_ops()),
        "op_nas_depth": int(graph.depth()),
    }
    return ProposalSpec(
        proposal_id=make_proposal_id(name, axes),
        name=name,
        category="lane",
        synthesis_kind=SOURCE_NAS,
        math_axes=axes,
        anchor_witness_op=origin,
        anchor_witnesses_all=(origin,),
        declared_property_row={},
        predicted_lift=0.5,
        rationale=f"NAS-synthesized topology ({origin}) {desc}".strip(),
        notes=(
            f"nas_graph fp={fingerprint}",
            f"ops={graph.n_ops()} depth={graph.depth()}",
            *((desc,) if desc else ()),
        ),
    )


def _fresh_grammar_specs(
    n_fresh: int,
    dim: int,
    seed: int,
    seen: set[str],
    cfg: Any = None,
) -> list[ProposalSpec]:
    from research.synthesis.grammar import GrammarConfig, generate_layer_graph

    if cfg is None:
        cfg = GrammarConfig(model_dim=dim)
    out: list[ProposalSpec] = []
    attempts = 0
    max_attempts = max(4, n_fresh * 5)
    s = seed
    while len(out) < n_fresh and attempts < max_attempts:
        attempts += 1
        s += 1
        try:
            graph = generate_layer_graph(config=cfg, seed=s)
        except Exception as exc:  # noqa: BLE001 — grammar may reject a sample
            _LOG.debug("grammar gen seed=%d rejected: %s", s, exc)
            continue
        spec = build_graph_spec(graph, dim=dim, origin="grammar")
        if spec is None or spec.proposal_id in seen:
            continue
        seen.add(spec.proposal_id)
        out.append(spec)
    return out


def _db_winner_specs(
    fingerprints: Sequence[tuple[str, str, float]], dim: int, seen: set[str]
) -> list[ProposalSpec]:
    """Best-effort ingest of curated novel-winner graphs by fingerprint.

    Many are trained at a larger dim and won't re-dimension to ``dim`` — those are
    dropped by the compile-test. Whatever survives is a real, proven topology.
    """
    try:
        from research.tools.ensemble_screening import _load_graphs_by_fingerprint

        loaded = _load_graphs_by_fingerprint(tuple(fingerprints))
    except Exception as exc:  # noqa: BLE001 — DB optional; never break the cycle
        _LOG.debug("DB winner load failed: %s", exc)
        return []
    out: list[ProposalSpec] = []
    for _fp, desc, auc, graph in loaded:
        spec = build_graph_spec(
            graph, dim=dim, origin="db", desc=f"{desc} auc={auc:.3f}"
        )
        if spec is not None and spec.proposal_id not in seen:
            seen.add(spec.proposal_id)
            out.append(spec)
    return out


def build_nas_archive(dim: int, *, max_graphs: int = 64) -> Any:
    """MAP-Elites archive over the cached NAS population, measured at init.

    The bridge caches every admitted topology under ``catalog/nas_graphs/``; this
    reads the ``max_graphs`` most-recently-cached graphs (so per-cycle cost stays
    flat as the cache grows), measures each one's label-free behaviour descriptors
    on CPU, and bins them. ``empty_niches`` on the result is what steers fresh
    sampling toward the regions the NAS population does not yet cover (anti-collapse
    — see ``research/synthesis/quality_diversity``). Returns ``None`` when nothing
    measurable is cached yet (e.g. the first cycle) so the caller falls back to
    plain random sampling.
    """
    from research.synthesis.quality_diversity import MapElitesArchive
    from research.tools.measured_descriptors import (
        MeasuredDescriptorExtractor,
        capability_score_from_descriptors,
    )

    if not _CACHE_DIR.exists():
        return None
    paths = sorted(
        _CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )[: max(1, max_graphs)]
    if not paths:
        return None
    extractor = MeasuredDescriptorExtractor(device="cpu")
    archive = MapElitesArchive()
    n_added = 0
    for path in paths:
        try:
            descriptors = extractor.descriptors(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — measurement must not break the cycle
            _LOG.debug("descriptor measure failed for %s: %s", path.name, exc)
            continue
        if not descriptors:
            continue
        archive.add(
            path.stem, descriptors, capability_score_from_descriptors(descriptors)
        )
        n_added += 1
    return archive if n_added else None


def _exploration_config(dim: int, max_graphs: int) -> Any:
    """Archive-derived exploration ``GrammarConfig`` (None → keep base grammar)."""
    from research.synthesis.archive_guided import exploration_config_from_archive

    archive = build_nas_archive(dim, max_graphs=max_graphs)
    if archive is None:
        _LOG.info("nas_bridge: no cached NAS population yet — random sampling")
        return None
    cfg, guidance = exploration_config_from_archive(archive, model_dim=dim)
    _LOG.info(
        "nas_bridge: archive coverage %.2f, %d reachable-empty niches → "
        "%d exploration target ops",
        guidance.coverage,
        guidance.reachable_empty,
        len(guidance.target_ops),
    )
    return cfg


def nas_graph_specs(
    *,
    n_fresh: int = 6,
    dim: int = 32,
    seed: int = 0,
    include_db_winners: bool = True,
    archive_guided: bool = False,
    archive_max_graphs: int = 64,
) -> list[ProposalSpec]:
    """Novel NAS topologies as gradeable fab specs.

    ``n_fresh`` graphs are sampled fresh from the grammar at ``dim`` (the workhorse
    — genuinely new structures). When ``include_db_winners`` is set, curated
    proven-winner graphs are also attempted (best-effort; dim-incompatible ones are
    dropped). When ``archive_guided`` is set, the grammar is biased toward the ops
    that fill empty niches in the cached NAS population's behaviour space, instead
    of sampling uniformly at random — the anti-collapse lever. It fails safe to
    random sampling when the archive is empty or the measurement path errors.
    """
    if n_fresh <= 0 and not include_db_winners:
        return []
    cfg: Any = None
    if archive_guided and n_fresh > 0:
        try:
            cfg = _exploration_config(dim, archive_max_graphs)
        except Exception as exc:  # noqa: BLE001 — guidance is best-effort, never fatal
            _LOG.warning("nas_bridge: archive guidance failed, using random: %s", exc)
            cfg = None
    seen: set[str] = set()
    specs = _fresh_grammar_specs(max(0, n_fresh), dim, seed, seen, cfg)
    if include_db_winners:
        try:
            from research.tools.ensemble_screening import TOP_AR_FPS

            specs += _db_winner_specs(TOP_AR_FPS, dim, seen)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("curated winner fingerprints unavailable: %s", exc)
    if specs:
        _LOG.info("nas_bridge: %d novel topologies admitted at dim=%d", len(specs), dim)
    return specs
