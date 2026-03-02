from typing import List, Dict, Any
from fastapi import WebSocket

class CollaborationManager:
    """Manages active WebSocket connections for real-time collaboration."""
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, workflow_id: str, websocket: WebSocket):
        await websocket.accept()
        if workflow_id not in self.active_connections:
            self.active_connections[workflow_id] = []
        self.active_connections[workflow_id].append(websocket)

    def disconnect(self, workflow_id: str, websocket: WebSocket):
        if workflow_id in self.active_connections:
            self.active_connections[workflow_id].remove(websocket)

    async def broadcast(self, workflow_id: str, message: Dict[str, Any], sender: WebSocket = None):
        """Send message to all users editing the same workflow."""
        if workflow_id in self.active_connections:
            for connection in self.active_connections[workflow_id]:
                if connection != sender:
                    await connection.send_json(message)

collab_manager = CollaborationManager()
