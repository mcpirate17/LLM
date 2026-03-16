"""Aria Designer API — FastAPI application entry point.

All route handlers live in api/app/routers/. This module handles only:
- App creation, middleware, lifespan
- WebSocket collaboration endpoint
- Health check
- Router registration
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running from api/.venv
# (needed for aria_designer.runtime.importer and research.* imports)
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import database as db
from .loader import scan_and_load
from .collaboration import collab_manager

logger = logging.getLogger(__name__)

# Re-export shared symbols so tests that monkeypatch app.main still work.
from .shared_api import (  # noqa: F401
    HAS_BRIDGE,
    bridge_evaluate,
    _sync_lineage_to_research,
)
from .config import settings  # noqa: F401


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + load components from disk."""
    db.init_db()

    cleaned = db.cleanup_orphaned_workflows(max_age_hours=48)
    if cleaned > 0:
        logger.info("Startup cleanup: %d abandoned workflows removed", cleaned)

    count = scan_and_load()
    logger.info("Startup complete: %d components loaded", count)
    yield


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(title="Aria Designer API", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────

from .routers.components import router as components_router
from .routers.workflows import router as workflows_router
from .routers.eval import router as eval_router
from .routers.aria import router as aria_router
from .routers.blocks import router as blocks_router
from .routers.import_export import router as import_export_router
from .routers.help import router as help_router
from .routers.chat import router as chat_router

app.include_router(components_router)
app.include_router(workflows_router)
app.include_router(eval_router)
app.include_router(aria_router)
app.include_router(blocks_router)
app.include_router(import_export_router)
app.include_router(help_router)
app.include_router(chat_router)


# ── Health ────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    counts = db.count_components()
    return {"status": "ok", "components": counts}


# ── WebSocket Collaboration ───────────────────────────────────────────

@app.websocket("/api/v1/collaboration/{workflow_id}")
async def collaboration_endpoint(websocket: WebSocket, workflow_id: str):
    await collab_manager.connect(workflow_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await collab_manager.broadcast(workflow_id, data, sender=websocket)
    except WebSocketDisconnect:
        collab_manager.disconnect(workflow_id, websocket)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        collab_manager.disconnect(workflow_id, websocket)
