from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import json
import time
import uuid
from typing import Dict, List, Optional


class _ChatMixin:
    """Chat operations for the Lab Notebook."""

    __slots__ = ()

    # ── Aria Chat Persistence ──────────────────────────────────────

    def save_chat_message(
        self,
        session_id: str,
        role: str,
        text: str,
        label: Optional[str] = None,
        message_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Persist a single chat message and return its message_id."""
        mid = message_id or str(uuid.uuid4())
        self.conn.execute(
            """INSERT OR REPLACE INTO aria_chat
               (message_id, session_id, timestamp, role, text, label, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                mid,
                session_id,
                time.time(),
                role,
                text,
                label,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self._maybe_commit()
        return mid

    def get_chat_history(
        self,
        session_id: str,
        limit: int = 50,
        include_compacted: bool = False,
    ) -> List[Dict]:
        """Return chat messages for a session, newest last."""
        if include_compacted:
            rows = self.conn.execute(
                """SELECT * FROM aria_chat
                   WHERE session_id = ?
                   ORDER BY timestamp ASC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM aria_chat
                   WHERE session_id = ? AND compacted = 0
                   ORDER BY timestamp ASC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_messages_compacted(
        self, message_ids: List[str], summary_message_id: str
    ) -> None:
        """Mark messages as compacted (replaced by a summary)."""
        if not message_ids:
            return
        placeholders = ",".join("?" for _ in message_ids)
        self.conn.execute(
            f"UPDATE aria_chat SET compacted = 1 WHERE message_id IN ({placeholders})",
            message_ids,
        )
        # Update the summary message to reference what it summarizes
        self.conn.execute(
            "UPDATE aria_chat SET summary_of = ? WHERE message_id = ?",
            (json.dumps(message_ids), summary_message_id),
        )
        self._maybe_commit()

    def compact_old_chat(self, max_chars: int = 160) -> int:
        """Truncate and purge old chat messages to keep the DB lean.

        - Truncates all compacted messages to max_chars
        - Deletes compacted messages older than 7 days
        Returns number of rows affected.
        """
        cutoff = time.time() - 7 * 86400
        # Delete old compacted messages
        n1 = self.conn.execute(
            "DELETE FROM aria_chat WHERE compacted = 1 AND timestamp < ?",
            (cutoff,),
        ).rowcount
        # Truncate long messages that are already compacted
        self.conn.execute(
            """UPDATE aria_chat SET text = SUBSTR(text, 1, ?) || '...'
               WHERE compacted = 1 AND LENGTH(text) > ?""",
            (max_chars, max_chars),
        )
        self._maybe_commit()
        return n1
