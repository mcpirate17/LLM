"""Help and component tips API router."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from ..help_content import get_component_tips, get_patterns_summary
from ..research_signals import fetch_research_recommendation_signals

router = APIRouter(prefix="/api/v1/help", tags=["help"])


@router.get("/component/{component_id}/tips")
def component_tips(component_id: str) -> Dict[str, Any]:
    """Get compatibility tips, usage patterns, and research warnings for a component."""
    signals = fetch_research_recommendation_signals(force=False)
    return get_component_tips(component_id, research_signals=signals)


@router.get("/patterns")
def patterns_summary() -> Dict[str, Any]:
    """Get common architecture patterns and general tips for the help panel."""
    return get_patterns_summary()
