"""FastAPI app for the component_fab visual explainer.

Endpoints:
- ``GET /``                       single-page app (Plotly, no build step)
- ``GET /api/lanes``              gallery metadata for every registered lane
- ``GET /api/lanes/{id}``         full explainer card (equations, params, smoke)
- ``GET /api/lanes/{id}/mixing``  L×L token-mixing influence map + decay curve
- ``GET /api/lanes/{id}/trace``   surprise-memory state, frame per token
- ``GET /api/lanes/{id}/spectrum``learnable-semiring read: mean / learned / max
- ``GET /api/ledger``             replay of catalog grade/promote history
- ``GET /api/run/stream``         SSE: grade every lane live, one event each
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import introspect, ledger_replay, lane_registry

_STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="component_fab visual explainer", version="0.1.0")

_DIM = 16  # the fab's grading dim; keeps memory matrices small + legible
_SEED = 0  # fixed init so every probe/page-load is reproducible (no flicker)


def _build(info: lane_registry.LaneInfo) -> torch.nn.Module:
    """Build a lane with a fixed seed so its random init is reproducible.

    The lanes are untrained, so without this every request would re-roll the
    weights and the plots/recall bars would jitter between page loads. A fixed
    seed pins one representative init per lane (the recall contrast is robust
    across seeds, so the choice of seed is not cherry-picking).
    """
    torch.manual_seed(_SEED)
    return info.builder(_DIM)


@app.get("/api/lanes")
def list_lanes() -> dict[str, Any]:
    return {
        "lanes": [lane_registry.lane_metadata(i) for i in lane_registry.all_lanes()]
    }


@app.get("/api/lanes/{lane_id}")
def lane_card(lane_id: str) -> dict[str, Any]:
    info = _get(lane_id)
    module = _build(info)
    meta = lane_registry.lane_metadata(info)
    meta["params"] = introspect.param_count(module)
    meta["smoke"] = introspect.smoke(module, dim=_DIM)
    meta["dim"] = _DIM
    meta["facts"] = lane_registry.flow_facts(lane_id)
    return meta


@app.get("/api/lanes/{lane_id}/mixing")
def lane_mixing(lane_id: str) -> dict[str, Any]:
    info = _get(lane_id)
    module = _build(info)
    return introspect.influence_matrix(module, dim=_DIM)


@app.get("/api/lanes/{lane_id}/trace")
def lane_trace(lane_id: str) -> dict[str, Any]:
    info = _get(lane_id)
    if not info.supports_trace():
        raise HTTPException(400, f"{lane_id} has no single-memory trace")
    module = _build(info)
    return introspect.surprise_trace(module, dim=_DIM)


@app.get("/api/lanes/{lane_id}/recall")
def lane_recall(lane_id: str) -> dict[str, Any]:
    info = _get(lane_id)
    if not info.supports_recall():
        raise HTTPException(400, f"{lane_id} has no recall demo")
    module = _build(info)
    return introspect.recall_story(module, dim=_DIM)


@app.get("/api/lanes/{lane_id}/spectrum")
def lane_spectrum(lane_id: str) -> dict[str, Any]:
    info = _get(lane_id)
    if not info.supports_spectrum():
        raise HTTPException(400, f"{lane_id} has no learnable-semiring read")
    module = _build(info)
    return introspect.algebra_spectrum(module, dim=_DIM)


@app.get("/api/ledger")
def ledger() -> dict[str, Any]:
    return ledger_replay.load_ledger()


@app.get("/api/run/stream")
async def run_stream() -> StreamingResponse:
    """Grade every registered lane live, emitting an SSE event per lane.

    This re-runs the same intrinsic checks the fab grader uses (smoke +
    mix-speed) so the stream is a faithful 'watch it being tested' view.
    """

    async def gen():
        lanes = lane_registry.all_lanes()
        yield _sse("start", {"total": len(lanes)})
        for idx, info in enumerate(lanes):
            yield _sse(
                "grading", {"lane_id": info.lane_id, "title": info.title, "index": idx}
            )
            # Run the (CPU-bound) grade off the event loop.
            payload = await asyncio.to_thread(_grade_one, info)
            yield _sse("graded", payload)
            await asyncio.sleep(0)  # flush
        yield _sse("done", {"total": len(lanes)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


def _grade_one(info: lane_registry.LaneInfo) -> dict[str, Any]:
    try:
        module = _build(info)
        sm = introspect.smoke(module, dim=_DIM)
        params = introspect.param_count(module)
        mix = introspect.influence_matrix(module, dim=_DIM, seq_len=16, n_trials=2)
        smoke_score = 1.0 if sm.get("all_finite") else 0.0
        return {
            "lane_id": info.lane_id,
            "title": info.title,
            "family": info.family,
            "params": params,
            "smoke_pass": bool(sm.get("all_finite")),
            "smoke_score": smoke_score,
            "mix_half_life": mix["mix_half_life"],
            "mixes_globally": mix["mixes_globally"],
            "error": sm.get("error"),
        }
    except Exception as exc:  # noqa: BLE001 - report, never crash the stream
        return {
            "lane_id": info.lane_id,
            "title": info.title,
            "error": str(exc),
            "smoke_pass": False,
            "smoke_score": 0.0,
        }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _get(lane_id: str) -> lane_registry.LaneInfo:
    try:
        return lane_registry.get_lane(lane_id)
    except KeyError:
        raise HTTPException(404, f"unknown lane {lane_id!r}") from None


# Static assets (app.js, style.css). Mounted last so it doesn't shadow routes.
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
