"""Tests for Aria Designer Collaboration (WebSockets)."""

from fastapi.testclient import TestClient


def test_websocket_broadcast():
    from aria_designer.api.app.main import app

    client = TestClient(app)

    with client.websocket_connect("/api/v1/collaboration/wf_collab") as ws1:
        with client.websocket_connect("/api/v1/collaboration/wf_collab") as ws2:
            # ws1 sends a move event
            msg = {
                "type": "node_moved",
                "node_id": "n1",
                "position": {"x": 100, "y": 200},
            }
            ws1.send_json(msg)

            # ws2 should receive it
            received = ws2.receive_json()
            assert received == msg


def test_websocket_isolation():
    """Verify that messages aren't leaked between workflows."""
    from aria_designer.api.app.main import app

    client = TestClient(app)

    with client.websocket_connect("/api/v1/collaboration/wf_a") as ws_a1:
        with client.websocket_connect("/api/v1/collaboration/wf_a") as ws_a2:
            with client.websocket_connect("/api/v1/collaboration/wf_b") as ws_b1:
                with client.websocket_connect("/api/v1/collaboration/wf_b") as ws_b2:
                    # Send message from ws_a1
                    ws_a1.send_json({"msg": "for_a"})

                    # ws_a2 (same workflow) should receive it
                    received_a = ws_a2.receive_json()
                    assert received_a["msg"] == "for_a"

                    # Send message from ws_b1
                    ws_b1.send_json({"msg": "for_b"})

                    # ws_b2 (same workflow) should receive its own workflow's message,
                    # proving the message for wf_a did not leak to wf_b.
                    received_b = ws_b2.receive_json()
                    assert received_b["msg"] == "for_b"
