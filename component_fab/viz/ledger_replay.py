"""Read the invention catalog for the Hall of Fame view.

The Hall of Fame replays ``invention_ledger.jsonl`` — the append-only record of
the handful of *named, explainable* lanes this site teaches (tropical / semiring
/ p-adic surprise memory, the fast-weight baseline, the slot router, the
compressor, …). Each ``grade`` event carries a composite score and a plain
``metadata.mechanism`` tag; ``promote`` events mark which graduated. We group
grades by proposal into a score history, attach the registry's human title /
family / analogy for that mechanism, and surface a plain-language score
breakdown so the leaderboard reads like a trophy shelf, not a hash dump.

(The older ``ledger.jsonl`` holds 800+ machine-bred hybrids with unreadable
names; it is intentionally NOT the source here.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CATALOG = Path(__file__).resolve().parents[1] / "catalog"
_LEDGER = _CATALOG / "invention_ledger.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _humanize(mechanism: str) -> str:
    return mechanism.replace("_", " ").title() if mechanism else "Unknown design"


def _registry_lookup(mechanism: str) -> dict[str, Any]:
    """Title / family / analogy for a mechanism, from the lane registry.

    The 6 memory/routing/compression lanes share their ``lane_id`` with the
    grade event's ``mechanism`` tag, so we can borrow the same human-readable
    metadata the explainer pages use. Anything not in the registry (e.g. the
    symplectic mixer) falls back to a titleized name.
    """
    from . import lane_registry  # lazy: avoids import cost when unused

    try:
        info = lane_registry.get_lane(mechanism)
    except KeyError:
        return {
            "title": _humanize(mechanism),
            "family": "other",
            "plain": "",
            "in_explainer": False,
        }
    return {
        "title": info.title,
        "family": info.family,
        "plain": info.plain,
        "in_explainer": True,
    }


def _score_story(
    meta: dict[str, Any], smoke: bool, learns: bool
) -> list[dict[str, Any]]:
    """Plain-language breakdown of why a design earned its score."""
    nb = meta.get("nb_max_accuracy")
    can_bind = meta.get("can_bind")
    story = [
        {
            "label": "Doesn't crash",
            "ok": bool(smoke),
            "plain": "Builds, runs a sentence forward and back, and never blows up to NaN/∞.",
        },
        {
            "label": "Actually learns",
            "ok": bool(learns),
            "plain": "Its outputs respond to the input in a way a learner could improve on — not a dead constant.",
        },
        {
            "label": "Can bind facts",
            "ok": bool(can_bind),
            "plain": "Can it pin one specific fact to one specific cue and pull it back later? The hard part.",
        },
    ]
    if nb is not None:
        story.append(
            {
                "label": f"Needle recall {nb:.0%}",
                "ok": float(nb) >= 0.5,
                "plain": "Hide a value behind a key, ask for it later — how often does it return the right one?",
            }
        )
    return story


def load_ledger() -> dict[str, Any]:
    events = _read_jsonl(_LEDGER)

    by_id: dict[str, dict[str, Any]] = {}
    promotes: list[dict[str, Any]] = []
    for ev in events:
        kind = ev.get("event")
        pid = ev.get("proposal_id")
        if pid is None:
            continue
        if kind == "promote":
            promotes.append(
                {
                    "proposal_id": pid,
                    "status": ev.get("status"),
                    "timestamp": ev.get("timestamp"),
                }
            )
            entry = by_id.get(pid)
            if entry is not None:
                entry["promotion_status"] = ev.get("status")
            continue
        if kind != "grade":
            continue
        entry = by_id.setdefault(
            pid,
            {
                "proposal_id": pid,
                "name": ev.get("name"),
                "category": ev.get("category"),
                "synthesis_kind": ev.get("synthesis_kind"),
                "history": [],
                "cycles": [],
                "promotion_status": "pending",
                "last_metadata": {},
                "smoke_pass": False,
                "learned_signal": False,
            },
        )
        entry["history"].append(float(ev.get("composite_score") or 0.0))
        entry["cycles"].append(ev.get("cycle"))
        entry["last_metadata"] = ev.get("metadata") or {}
        entry["name"] = ev.get("name") or entry["name"]
        entry["smoke_pass"] = bool(ev.get("smoke_pass"))
        entry["learned_signal"] = bool(ev.get("learned_signal"))

    rows = list(by_id.values())
    for r in rows:
        r["last_score"] = r["history"][-1] if r["history"] else 0.0
        r["best_score"] = max(r["history"]) if r["history"] else 0.0
        r["grades"] = len(r["history"])
        mechanism = r["last_metadata"].get("mechanism") or ""
        r["mechanism"] = mechanism
        reg = _registry_lookup(mechanism)
        r["title"] = reg["title"]
        r["family"] = reg["family"]
        r["plain"] = reg["plain"]
        r["in_explainer"] = reg["in_explainer"]
        # the explainer's lane page is keyed by mechanism (== lane_id)
        r["lane_id"] = mechanism if reg["in_explainer"] else None
        r["score_story"] = _score_story(
            r["last_metadata"], r["smoke_pass"], r["learned_signal"]
        )
    rows.sort(key=lambda r: r["best_score"], reverse=True)
    return {
        "proposals": rows,
        "promotes": promotes,
        "promoted_count": sum(1 for r in rows if r["promotion_status"] == "promoted"),
        "total_events": len(events),
        "catalog_dir": str(_CATALOG),
    }
