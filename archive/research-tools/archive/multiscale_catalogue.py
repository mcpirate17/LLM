from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from research.synthesis.compiler import _OP_DISPATCH
from research.synthesis.primitives import OP_NAME_ALIASES


COMPONENTS_ROOT = Path(__file__).resolve().parents[2] / "aria_designer" / "components"
HYBRID_ROUTER_MANIFEST = (
    COMPONENTS_ROOT / "routing" / "hybrid_sparse_router" / "manifest.yaml"
)


@dataclass(slots=True)
class ManifestEntry:
    manifest_id: str
    manifest_name: str
    path: str
    path_category: str
    manifest_category: str
    status: str
    runtime_name: str
    is_alias_manifest: bool
    has_dispatch: bool


def _load_manifest(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_manifest_entries(root: Path = COMPONENTS_ROOT) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    for path in sorted(root.rglob("manifest.yaml")):
        manifest = _load_manifest(path)
        manifest_id = str(manifest.get("id") or "")
        if not manifest_id:
            continue
        entries.append(
            ManifestEntry(
                manifest_id=manifest_id,
                manifest_name=str(manifest.get("name") or manifest_id),
                path=str(path),
                path_category=path.parent.parent.name,
                manifest_category=str(manifest.get("category") or ""),
                status=str(manifest.get("status") or "unknown"),
                runtime_name=OP_NAME_ALIASES.get(manifest_id, manifest_id),
                is_alias_manifest=manifest_id in OP_NAME_ALIASES,
                has_dispatch=manifest_id in _OP_DISPATCH,
            )
        )
    return entries


def _duplicate_id_table(entries: list[ManifestEntry]) -> list[dict[str, Any]]:
    by_id: dict[str, list[ManifestEntry]] = {}
    for entry in entries:
        by_id.setdefault(entry.manifest_id, []).append(entry)
    rows: list[dict[str, Any]] = []
    for manifest_id, group in sorted(by_id.items()):
        if len(group) <= 1:
            continue
        rows.append(
            {
                "manifest_id": manifest_id,
                "count": len(group),
                "paths": [entry.path for entry in group],
                "path_categories": [entry.path_category for entry in group],
            }
        )
    return rows


def _path_category_mismatches(entries: list[ManifestEntry]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry.manifest_category and entry.manifest_category != entry.path_category:
            rows.append(
                {
                    "manifest_id": entry.manifest_id,
                    "path": entry.path,
                    "path_category": entry.path_category,
                    "manifest_category": entry.manifest_category,
                }
            )
    return rows


def _slot_refs() -> dict[str, list[str]]:
    manifest = _load_manifest(HYBRID_ROUTER_MANIFEST)
    slots = {slot["name"]: slot for slot in manifest.get("slots") or []}
    return {
        "medium_router": list(slots["medium_router"]["compatible_components"]),
        "hard_router": list(slots["hard_router"]["compatible_components"]),
    }


def resolve_component_ref(
    component_ref: str,
    entries: list[ManifestEntry],
) -> dict[str, Any]:
    _, leaf = (
        component_ref.split("/", 1) if "/" in component_ref else ("", component_ref)
    )
    by_id = {entry.manifest_id: entry for entry in entries}
    runtime_name = OP_NAME_ALIASES.get(leaf, leaf)
    exact = by_id.get(leaf)
    runtime_entry = by_id.get(runtime_name)
    representative = exact or runtime_entry
    if representative is None:
        raise KeyError(f"Unresolved component ref: {component_ref}")
    return {
        "slot_ref": component_ref,
        "manifest_id": representative.manifest_id,
        "manifest_name": representative.manifest_name,
        "manifest_path": representative.path,
        "manifest_path_category": representative.path_category,
        "manifest_category": representative.manifest_category,
        "runtime_name": runtime_name,
        "canonical_name": runtime_name,
        "is_alias_ref": leaf in OP_NAME_ALIASES,
        "dispatch_name": leaf if leaf in _OP_DISPATCH else runtime_name,
        "has_dispatch": (leaf in _OP_DISPATCH) or (runtime_name in _OP_DISPATCH),
    }


def build_multiscale_registry(root: Path = COMPONENTS_ROOT) -> dict[str, Any]:
    entries = load_manifest_entries(root)
    slot_refs = _slot_refs()

    medium_rows = [
        resolve_component_ref(ref, entries) for ref in slot_refs["medium_router"]
    ]
    hard_rows = [
        resolve_component_ref(ref, entries) for ref in slot_refs["hard_router"]
    ]

    support_ids = [
        "default_path",
        "hybrid_token_gate",
        "hybrid_sparse_router",
        "sparse_span_builder",
        "lane_conditioned_block",
        "token_class_proj",
        "signal_conditioned_compression",
    ]
    by_id = {entry.manifest_id: entry for entry in entries}
    support_rows = []
    for manifest_id in support_ids:
        entry = by_id[manifest_id]
        support_rows.append(
            {
                "manifest_id": entry.manifest_id,
                "manifest_name": entry.manifest_name,
                "manifest_path": entry.path,
                "runtime_name": entry.runtime_name,
                "canonical_name": entry.runtime_name,
            }
        )

    canonical_all = {entry.runtime_name for entry in entries}
    reachable_canonical = {
        *[row["canonical_name"] for row in medium_rows],
        *[row["canonical_name"] for row in hard_rows],
        *[row["canonical_name"] for row in support_rows],
    }
    routing_count = sum(1 for entry in entries if entry.path_category == "routing")

    return {
        "entries": [asdict(entry) for entry in entries],
        "duplicate_manifest_ids": _duplicate_id_table(entries),
        "path_category_mismatches": _path_category_mismatches(entries),
        "alias_mapping": [
            {
                "manifest_or_slot_name": src,
                "runtime_name": dst,
            }
            for src, dst in sorted(OP_NAME_ALIASES.items())
        ],
        "slot_refs": slot_refs,
        "medium_candidates": medium_rows,
        "hard_candidates": hard_rows,
        "support_components": support_rows,
        "summary": {
            "total_catalogue_size": len(entries),
            "canonical_component_count": len(canonical_all),
            "routing_component_count": routing_count,
            "reachable_for_template_count": len(reachable_canonical),
            "medium_candidate_count": len(
                {row["canonical_name"] for row in medium_rows}
            ),
            "hard_candidate_count": len({row["canonical_name"] for row in hard_rows}),
        },
    }


def assert_no_duplicate_logical_candidates(
    rows: list[dict[str, Any]], label: str
) -> None:
    seen: dict[str, list[str]] = {}
    for row in rows:
        seen.setdefault(row["canonical_name"], []).append(row["slot_ref"])
    dupes = {name: refs for name, refs in seen.items() if len(refs) > 1}
    if dupes:
        raise ValueError(
            f"{label} candidate pool has duplicate logical components: {dupes}"
        )
