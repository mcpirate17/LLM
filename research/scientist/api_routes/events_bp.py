"""events API route registration."""
from __future__ import annotations

import functools
import time
import datetime
from flask import jsonify, request, Response
from ..json_utils import json_safe as _json_safe
from ..notebook import LabNotebook
from .deps import ApiRouteContext, install_legacy_symbols

def register_events_routes(app, context: ApiRouteContext):
    install_legacy_symbols(globals(), context)

    @app.route("/api/live-feed")
    def api_live_feed():
        """List persisted live-feed events for replay in the dashboard."""
        exp_id = request.args.get("experiment_id")
        n = request.args.get("n", 100, type=int)
        nb = LabNotebook(notebook_path)
        try:
            query_limit = max(n, 1000)
            entries = nb.get_entries(
                experiment_id=exp_id,
                entry_type="live_feed",
                limit=query_limit,
            )

            # Default behavior should show a coherent experiment stream.
            # Without this, mixed cross-experiment rows can look like broken
            # generation timelines (e.g., Gen 3 -> Gen 13 with unrelated runs).
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
        except Exception as e:
            logger.error(f"Error in /api/live-feed: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/live-loss-curve")
    def api_live_loss_curve():
        """Return the in-memory training loss curve for the live chart."""
        if _runner is None:
            return jsonify([])
        try:
            return jsonify(_runner.get_live_loss_curve())
        except Exception as e:
            logger.error("Error in /api/live-loss-curve: %s", e)
            return jsonify([])


    @app.route("/api/events")
    def api_events():
        """SSE endpoint for real-time experiment events."""
        runner = _get_runner(notebook_path)
        sse_timeout = _get_sse_timeout_seconds()

        def event_stream():
            while True:
                for event in runner.get_events(timeout=sse_timeout):
                    data = json.dumps(
                        _json_safe(event.get("data", {})),
                        allow_nan=False,
                    )
                    yield f"event: {event['type']}\ndata: {data}\n\n"
                # After timeout, check if client is still connected
                yield f"event: keepalive\ndata: {{}}\n\n"

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )


