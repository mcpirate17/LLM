"""Registry of lanes the visualizer can instantiate + explain.

Each entry knows how to build the ``nn.Module`` from a single ``dim`` and
carries human-readable explainer metadata (the read/write equations, the
mechanism description, complexity). The surprise-memory family is described
explicitly; a handful of attention lanes are added best-effort (skipped if
their constructor needs more than ``dim``) so the gallery has breadth without
becoming fragile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from torch import nn

from ..generator.memory_primitives import (
    CausalFastWeightMemoryLane,
    CausalSlotRouterMemoryLane,
    HierarchicalResidualCompressorLane,
    PadicSurpriseMemoryLane,
    SemiringSurpriseMemoryLane,
    TropicalSurpriseMemoryLane,
    _SurpriseMemoryBase,
)


@dataclass(frozen=True, slots=True)
class LaneInfo:
    lane_id: str
    title: str
    family: str
    category: str
    complexity: str
    summary: str
    write_eq: str
    read_eq: str
    builder: Callable[[int], nn.Module]
    cls: type[nn.Module]
    notes: tuple[str, ...] = field(default_factory=tuple)
    plain: str = ""  # jargon-free analogy for the "🔰 plain English" toggle

    def is_surprise_memory(self) -> bool:
        return issubclass(self.cls, _SurpriseMemoryBase)

    def supports_trace(self) -> bool:
        # The base single-memory scan is traceable; the p-adic multi-level
        # forward has a different layout, so we only deep-trace single-memory.
        return self.is_surprise_memory() and not issubclass(
            self.cls, PadicSurpriseMemoryLane
        )

    def supports_spectrum(self) -> bool:
        return issubclass(self.cls, SemiringSurpriseMemoryLane)

    def supports_recall(self) -> bool:
        # The 'watch it remember' demo needs the single-memory delta-rule path
        # (tropical / semiring) or the Hebbian fast-weight baseline. The p-adic
        # multi-level layout and the stateless attention lanes are excluded.
        return self.supports_trace() or issubclass(self.cls, CausalFastWeightMemoryLane)


_DELTA_WRITE = (
    "eₜ = vₜ − read(Mₜ₋₁, kₜ)          (surprise = prediction error)\n"
    "Sₜ = μ·Sₜ₋₁ + gₜ·(kₜ ⊗ eₜ)        (momentum on the surprise stream)\n"
    "Mₜ = (1 − αₜ)·Mₜ₋₁ + Sₜ            (data-dependent forget gate αₜ)"
)


_REGISTRY: dict[str, LaneInfo] = {}


def _add(info: LaneInfo) -> None:
    _REGISTRY[info.lane_id] = info


_add(
    LaneInfo(
        lane_id="tropical_surprise_memory",
        title="Tropical Surprise Memory",
        family="surprise-memory",
        category="lane",
        complexity="O(L · M²),  M ≤ D",
        summary=(
            "Titans/TTT test-time memory whose write is the SURPRISE (the "
            "associative prediction error), read by MAX-PLUS retrieval: the "
            "single strongest stored key→value association wins. Winner-take-all "
            "retrieval sharpens exact recall and kills the cross-key "
            "interference that sum-based linear memory hits on dense binding."
        ),
        write_eq=_DELTA_WRITE,
        read_eq="read[j] = maxᵢ ( M[i, j] + addrᵢ )          (max-plus / tropical)",
        builder=lambda dim: TropicalSurpriseMemoryLane(dim),
        cls=TropicalSurpriseMemoryLane,
        notes=("baseline: causal_fast_weight_memory", "winner-take-all read"),
        plain=(
            "Imagine a notebook that only bothered to write something down when "
            "it was actually surprised by what it saw. 📝 As it reads a sentence, "
            "it's constantly whispering a guess for the next word. When it's wrong, "
            "BAM! That surprise is exactly what it jots down—a little note linking "
            "the clue to the answer. 💥 When it needs to remember later, it "
            "doesn't just average all its notes into a fuzzy mess; it grabs the "
            "single best-matching note (Winner-Take-All!). This makes it "
            "incredibly good at exact recall, like remembering a specific "
            "password without it getting mixed up with everything else."
        ),
    )
)
_add(
    LaneInfo(
        lane_id="semiring_surprise_memory",
        title="Semiring Surprise Memory (learnable read)",
        family="surprise-memory",
        category="lane",
        complexity="O(L · M²),  M ≤ D",
        summary=(
            "Same delta-rule write, but the retrieval ALGEBRA is learnable: a "
            "tempered log-sum-exp with inverse-temperature β = softplus(param). "
            "β→∞ recovers the proven tropical max-plus read; β→0 becomes the "
            "soft arithmetic mean. β starts ≈4 (sharp, near tropical) and only "
            "softens if the data prefers it. Strictly generalizes the family."
        ),
        write_eq=_DELTA_WRITE,
        read_eq=(
            "read[j] = (1/β)·( logsumexpᵢ ( β·(M[i, j] + addrᵢ) ) − log m )\n"
            "β = softplus(θ) ∈ [1e-2, 30]      β→∞: max-plus · β→0: mean"
        ),
        builder=lambda dim: SemiringSurpriseMemoryLane(dim),
        cls=SemiringSurpriseMemoryLane,
        notes=("baseline: tropical_surprise_memory", "learns mean↔max sharpness"),
        plain=(
            "This is the 'Surprise Notebook' but with a magic <b>Focus Knob</b> "
            "added. 🎛️ Crank the knob up high, and it grabs only the single "
            "best-matching note (sharp and precise). Turn it down, and it blends "
            "all its notes into a fuzzy average. The clever part? Nobody tells it "
            "where to set the knob—it <i>learns</i> the best setting from the "
            "data! 🧠 It starts with a sharp focus and only softens its vision "
            "if that actually helps it understand the patterns better."
        ),
    )
)
_add(
    LaneInfo(
        lane_id="padic_surprise_memory",
        title="p-adic Surprise Memory (ultrametric)",
        family="surprise-memory",
        category="lane",
        complexity="O(L · K · M²),  K levels",
        summary=(
            "A hierarchy of K delta-rule memories addressed with p-adic "
            "block-pooled keys. The finest level (block size 1) behaves like "
            "TropicalSurpriseMemory so exact recall is preserved; coarse levels "
            "share capacity across p-adically-near keys to generalize across "
            "long gaps. Output is a learned gated sum across levels."
        ),
        write_eq=_DELTA_WRITE + "\n(applied per ultrametric level ℓ, gated sum read)",
        read_eq="read = Σℓ gateℓ · maxᵢ ( Mℓ[i, j] + pool_ℓ(addr)ᵢ )",
        builder=lambda dim: PadicSurpriseMemoryLane(dim),
        cls=PadicSurpriseMemoryLane,
        notes=("multi-scale generalization of tropical", "p=2, up to 3 levels"),
        plain=(
            "Imagine keeping a stack of notebooks at different 'zoom levels' at "
            "the same time. 📚 One notebook captures every tiny, exact detail. "
            "Another notebook groups similar things together in big batches. "
            "When it reads, it blends the info from all these levels. 🕵️ This "
            "way, it can remember exact facts (like a specific name) <i>and</i> "
            "spot big-picture patterns (like the general topic) at the same time. "
            "It's like having a microscope and a telescope working together!"
        ),
    )
)
_add(
    LaneInfo(
        lane_id="causal_fast_weight_memory",
        title="Causal Fast-Weight Memory",
        family="surprise-memory",
        category="lane",
        complexity="O(L · D · M),  M ≤ D",
        summary=(
            "The baseline the surprise family generalizes: a pure HEBBIAN "
            "fast-weight write (k ⊗ v outer product) with scalar decay, read by "
            "the current query. No prediction error, no softmax over positions — "
            "linear-attention memory. Useful to contrast against the delta-rule "
            "(surprise) write."
        ),
        write_eq="Mₜ = decay·Mₜ₋₁ + gₜ·(kₜ ⊗ vₜ)·scale      (Hebbian, scalar decay)",
        read_eq="read = qₜ · Mₜ          (Euclidean inner product)",
        builder=lambda dim: CausalFastWeightMemoryLane(dim),
        cls=CausalFastWeightMemoryLane,
        notes=("Hebbian write (no surprise)", "linear-attention memory"),
        plain=(
            "This is the 'Old School' notebook. 📖 It writes down <i>everything</i> "
            "it sees, not just the surprising parts. Because it's constantly "
            "scribbling, its notes tend to smear together. When it tries to "
            "remember something, it gets a fuzzy blend of many different facts. "
            "It's simpler and faster, but not nearly as sharp as the 'Surprise' "
            "notebooks. We keep it here as a baseline so you can see why "
            "focusing on surprises is such a big deal! 📉"
        ),
    )
)
_add(
    LaneInfo(
        lane_id="causal_slot_router_memory",
        title="Causal Slot-Router Memory",
        family="routing",
        category="routing",
        complexity="O(L · S · D),  S slots",
        summary=(
            "Routing-as-memory: each token softly routes to one of a few "
            "persistent slots, writes a gated candidate into the selected "
            "slot(s), then reads a route-weighted mixture. Tests whether routing "
            "over a tiny persistent state can hold key-specific information."
        ),
        write_eq="slots = slots·(1 − route·g) + (route·g)·tanh(W x)",
        read_eq="read = Σ_s routeₛ · slotₛ",
        builder=lambda dim: CausalSlotRouterMemoryLane(dim),
        cls=CausalSlotRouterMemoryLane,
        notes=("4 slots", "routing collapse is the failure mode"),
        plain=(
            "Picture a small set of labeled drawers in a desk. 🗄️ As each word "
            "arrives, it has to pick which drawer to hide in. Some words might "
            "try to squeeze into the same drawer, while others find their own "
            "spot. To remember, the model just opens the drawers it's pointing "
            "at. 🗝️ The big challenge: can just 4 or 8 drawers hold enough "
            "distinct facts, or will everything end up in one big messy pile? "
            "That's called 'Routing Collapse'!"
        ),
    )
)
_add(
    LaneInfo(
        lane_id="hierarchical_residual_compressor",
        title="Hierarchical Residual Compressor",
        family="compression",
        category="compression",
        complexity="O(L · K · D),  K levels",
        summary=(
            "A fixed stack of learned summaries updated at powers-of-two periods "
            "(level ℓ updates every 2^ℓ tokens). The readout gates over all "
            "levels. State budget is fixed in the number of levels, not the "
            "sequence length — a long-gap recall candidate under a small budget."
        ),
        write_eq="summaryₗ = (1 − gate)·summaryₗ + gate·tanh(Wₗ[summaryₗ; xₜ])   every 2^ℓ steps",
        read_eq="read = W_read [summary₀; …; summary_{K-1}]",
        builder=lambda dim: HierarchicalResidualCompressorLane(dim),
        cls=HierarchicalResidualCompressorLane,
        notes=("4 levels", "O(log L) state"),
        plain=(
            "Imagine a team of note-takers working at different speeds. 🏃‍♂️ "
            "The first one scribbles a note for every single word. The second "
            "one only writes a summary every 2 words. The third one every 4, "
            "and so on. Instead of remembering the whole book word-for-word, the "
            "model keeps this small stack of summaries. 🏢 The fast ones catch the "
            "details, while the slow ones preserve the 'big picture' from a long "
            "time ago without costing much memory!"
        ),
    )
)


def _try_add_attention() -> None:
    """Best-effort registration of attention lanes that build from ``dim``."""
    try:
        from ..generator.primitive_templates import _lanes_a, _lanes_b
    except Exception:  # noqa: BLE001 - optional breadth, never fatal
        return

    candidates: tuple[tuple[str, str, str, str, str], ...] = (
        # (module attr, title, read_eq, summary, plain)
        (
            "TropicalAttention",
            "Tropical Attention",
            "out_i = max_{j≤i} ( scale·qᵢ·kⱼ + vⱼ )",
            "Max-plus (tropical) attention: each position takes the single best "
            "scoring past token instead of a softmax mixture. Sparse, "
            "winner-take-all addressing — the attention analogue of the tropical "
            "memory read.",
            "Normal attention lets each word blend together a fuzzy mix of all "
            "the earlier words. Tropical attention is much more decisive! 🥊 "
            "Instead of a blend, each word copies from the <b>single most "
            "relevant</b> earlier word. No compromising—just the best match "
            "wins. It's sharp, sparse, and very direct. 🎯",
        ),
        (
            "SparsemaxAttention",
            "Sparsemax Attention",
            "out_i = Σⱼ sparsemax(qᵢ·k)ⱼ · vⱼ",
            "Sparsemax replaces softmax: weights are a Euclidean projection onto "
            "the simplex, so most become exactly zero. Sits between dense softmax "
            "and hard tropical.",
            "This is a middle ground between 'blend everything' and 'pick just "
            "one.' ⚖️ Instead of giving every earlier word a tiny sliver of "
            "attention, it zeroes out the irrelevant ones entirely and only "
            "keeps a handful of useful ones. It's like having a VIP list for "
            "which words get to speak! 🎟️",
        ),
        (
            "TemperedTropicalAttention",
            "Tempered Tropical Attention",
            "out_i = (1/β)·logsumexp_{j≤i} ( β·(qᵢ·kⱼ + vⱼ) )   per head",
            "Learnable-temperature tropical attention: per-head β slides between "
            "hard max (β→∞) and soft mean (β→0) — the attention cousin of the "
            "semiring memory read.",
            "Attention with that same magic <b>Focus Knob</b>! 🎛️ Each "
            "'attention head' gets its own knob to decide if it should be "
            "sharp (picking one word) or soft (blending many). The model "
            "fiddles with these knobs during training to find the perfect "
            "balance for every head. 🎼",
        ),
    )
    for attr, title, read_eq, summary, plain in candidates:
        cls = getattr(_lanes_a, attr, None) or getattr(_lanes_b, attr, None)
        if cls is None:
            continue
        try:
            cls(16)  # probe: does it build from dim alone?
        except Exception:  # noqa: BLE001
            continue
        _add(
            LaneInfo(
                lane_id=_snake(attr),
                title=title,
                family="attention",
                category="lane",
                complexity="O(L² · D)",
                summary=summary,
                write_eq="(stateless: q/k/v projections + causal mask)",
                read_eq=read_eq,
                builder=(lambda c: lambda dim: c(dim))(cls),
                cls=cls,
                notes=("stateless attention", "no memory trace"),
                plain=plain,
            )
        )


def _snake(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


_try_add_attention()


_FACTS_CACHE: dict[str, dict[str, str]] | None = None

# math_axes key  ->  (friendly label, emoji) for the "how it works" facts panel
_FACT_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("op_information_flow", "How information flows", "🌊"),
    ("op_forgetting_rule", "What it chooses to forget", "🧹"),
    ("op_target_failure_mode", "The hard problem it's built to beat", "🎯"),
    ("op_causality_argument", "Why it can't peek at the future", "⏪"),
)


def _load_invention_facts() -> dict[str, dict[str, str]]:
    """Lazily read the per-mechanism descriptor prose from the invention run.

    The grader records a plain(ish)-language ``math_axes`` row per invented
    mechanism (information flow, forgetting rule, target failure mode, causality
    argument). We reuse that text rather than re-authoring it so the explainer
    can never drift from what was actually graded.
    """
    global _FACTS_CACHE
    if _FACTS_CACHE is not None:
        return _FACTS_CACHE
    import json
    from pathlib import Path

    facts: dict[str, dict[str, str]] = {}
    path = Path(__file__).resolve().parents[1] / "catalog" / "invention_run_latest.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        _FACTS_CACHE = facts
        return facts
    for result in data.get("results", []):
        axes = result.get("spec", {}).get("math_axes", {})
        mech = axes.get("op_invention_mechanism")
        if not mech:
            continue
        facts[mech] = {
            "information_flow": axes.get("op_information_flow", ""),
            "forgetting_rule": axes.get("op_forgetting_rule", ""),
            "target_failure_mode": axes.get("op_target_failure_mode", ""),
            "causality_argument": axes.get("op_causality_argument", ""),
            "complexity": axes.get("op_complexity", ""),
            "expected_baseline": axes.get("op_expected_baseline", ""),
        }
    _FACTS_CACHE = facts
    return facts


def flow_facts(lane_id: str) -> list[dict[str, str]] | None:
    """Labeled how-it-works facts for a lane, or ``None`` if undescribed.

    Returns a list of ``{label, emoji, text}`` ready for the UI. Lanes added
    best-effort (e.g. the attention variants) have no invention record and
    return ``None`` so the panel is simply skipped.
    """
    record = _load_invention_facts().get(lane_id)
    if not record:
        return None
    out: list[dict[str, str]] = []
    for key, label, emoji in _FACT_FIELDS:
        field_key = key.removeprefix("op_")
        text = record.get(field_key, "")
        if text:
            out.append({"label": label, "emoji": emoji, "text": text})
    return out or None


def all_lanes() -> list[LaneInfo]:
    return list(_REGISTRY.values())


def get_lane(lane_id: str) -> LaneInfo:
    if lane_id not in _REGISTRY:
        raise KeyError(lane_id)
    return _REGISTRY[lane_id]


def lane_metadata(info: LaneInfo) -> dict[str, Any]:
    return {
        "lane_id": info.lane_id,
        "title": info.title,
        "family": info.family,
        "category": info.category,
        "complexity": info.complexity,
        "summary": info.summary,
        "write_eq": info.write_eq,
        "read_eq": info.read_eq,
        "plain": info.plain,
        "notes": list(info.notes),
        "supports_trace": info.supports_trace(),
        "supports_spectrum": info.supports_spectrum(),
        "supports_recall": info.supports_recall(),
        "is_surprise_memory": info.is_surprise_memory(),
        "class_name": info.cls.__name__,
    }
