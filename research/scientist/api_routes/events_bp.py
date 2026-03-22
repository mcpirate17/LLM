"""events API route registration."""

from __future__ import annotations

import logging
from flask import jsonify, request, Response
from ..json_utils import fast_dumps as _json_dumps, fast_loads as _json_loads
from ._helpers import get_runner, get_sse_timeout_seconds
from ._utils import with_notebook_context
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_events_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    @app.route("/api/live-feed")
    @wnb
    def api_live_feed(nb=None):
        """List persisted live-feed events for replay in the dashboard."""
        exp_id = request.args.get("experiment_id")
        n = request.args.get("n", 100, type=int)
        query_limit = max(n, 1000)
        entries = nb.get_entries(
            experiment_id=exp_id,
            entry_type="live_feed",
            limit=query_limit,
        )

        if not exp_id:
            latest_exp_id = next(
                (
                    entry.get("experiment_id")
                    for entry in entries
                    if entry.get("experiment_id")
                ),
                None,
            )
            if latest_exp_id:
                entries = [
                    entry
                    for entry in entries
                    if entry.get("experiment_id") == latest_exp_id
                ]

        events = []
        for entry in reversed(entries):
            evt = _entry_to_live_feed_event(entry)
            if evt is not None:
                events.append(evt)
        if len(events) > n:
            events = events[-n:]
        return jsonify(events)

    @app.route("/api/live-loss-curve")
    def api_live_loss_curve():
        """Return the in-memory training loss curve for the live chart."""
        runner = get_runner(notebook_path)
        try:
            return jsonify(runner.get_live_loss_curve())
        except Exception as e:
            logger.error("Error in /api/live-loss-curve: %s", e)
            return jsonify([])

    @app.route("/api/events")
    def api_events():
        """SSE endpoint for real-time experiment events."""
        runner = get_runner(notebook_path)
        sse_timeout = get_sse_timeout_seconds()

        def event_stream():
            while True:
                for event in runner.get_events(timeout=sse_timeout):
                    data = _json_dumps(event.get("data", {}), safe=True)
                    yield f"event: {event['type']}\ndata: {data}\n\n"
                yield "event: keepalive\ndata: {}\n\n"

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )


def _entry_to_live_feed_event(entry: dict):
    """Convert a notebook entry to a live-feed event dict, or None if not applicable."""
    if not isinstance(entry, dict):
        return None
    content = entry.get("content", "")
    metadata = entry.get("metadata") or entry.get("metadata_json") or {}
    if isinstance(metadata, str):
        try:
            metadata = _json_loads(metadata)
        except Exception:
            metadata = {}
    ret = {
        "type": metadata.get("event_type") or metadata.get("live_feed_type", "info"),
        "content": content,
        "timestamp": entry.get("timestamp"),
        "experiment_id": entry.get("experiment_id"),
        "metadata": metadata,
    }
    payload = metadata.get("payload")
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k not in ret:
                ret[k] = v
    return ret
