"""Chat API router for conversational Aria interactions."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from ..conversation import ConversationManager
from ..models import (
    ChatMessageRequest,
    utc_now_iso as _utc_now,
)
from ..research_signals import fetch_research_recommendation_signals

router = APIRouter(prefix="/api/v1/aria/chat", tags=["chat"])


@router.post("")
def send_message(req: ChatMessageRequest) -> Dict[str, Any]:
    """Send a message to Aria and get a response.

    If session_id is not provided, a new session is created automatically.
    """
    session_id = req.session_id
    workflow = req.workflow.model_dump() if req.workflow else None

    if not session_id:
        session_id = ConversationManager.start_session(workflow)

    conv = ConversationManager.get_session(session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if conv.get("status") == "ended":
        raise HTTPException(status_code=410, detail="Session has ended")

    signals = fetch_research_recommendation_signals(force=False)
    result = ConversationManager.process_message(
        session_id, req.message, workflow, research_signals=signals,
    )

    return {
        "session_id": session_id,
        "role": "aria",
        "content": result.get("content", ""),
        "patch_proposal": result.get("patch_proposal"),
        "suggestions": result.get("suggestions", []),
        "needs_clarification": result.get("needs_clarification", False),
    }


@router.get("/{session_id}/history")
def get_history(session_id: str) -> List[Dict[str, Any]]:
    """Get conversation history for a session."""
    conv = ConversationManager.get_session(session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return ConversationManager.get_history(session_id)


@router.delete("/{session_id}")
def end_session(session_id: str) -> Dict[str, str]:
    """End a conversation session."""
    if not ConversationManager.end_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ended", "session_id": session_id}
