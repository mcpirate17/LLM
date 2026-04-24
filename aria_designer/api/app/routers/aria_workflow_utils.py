from __future__ import annotations

from typing import Any, Dict

from aria_designer.component_identity import canonicalize_workflow_ids

from ..workflow_support import get_approved_registry_ids


def canonicalize_workflow_payload(
    workflow: Dict[str, Any], *, preserve_raw_ids: bool = False
) -> Dict[str, Any]:
    canonicalize_workflow_ids(
        workflow,
        get_approved_registry_ids(),
        preserve_raw_ids=preserve_raw_ids,
    )
    return workflow
