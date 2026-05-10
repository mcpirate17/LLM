from __future__ import annotations

"""Leaderboard repair helpers for orphaned program rows."""

from typing import Any, Dict, List, Optional

from ..thresholds import TIER_RANK


def _empty_orphan_repair_summary() -> Dict[str, Any]:
    return {
        "rebound_rows": 0,
        "fingerprints_repaired": 0,
        "deleted_duplicate_rows": 0,
        "fingerprints": [],
    }


class _ProgramLeaderboardRepairMixin:
    @staticmethod
    def _group_orphan_leaderboard_rows(
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            fp = str(row.get("graph_fingerprint") or "").strip()
            canonical_rid = str(row.get("canonical_result_id") or "").strip()
            if not fp or not canonical_rid:
                continue
            bucket = grouped.setdefault(
                fp,
                {
                    "canonical_result_id": canonical_rid,
                    "entry_ids": [],
                    "orphan_result_ids": [],
                },
            )
            bucket["entry_ids"].append(str(row["entry_id"]))
            bucket["orphan_result_ids"].append(str(row["orphan_result_id"]))
        return grouped

    @staticmethod
    def _leaderboard_row_to_keep(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return max(
            rows,
            key=lambda row: (
                int(TIER_RANK.get(str(row.get("tier") or "").lower(), -1)),
                float(row.get("timestamp") or 0.0),
                float(row.get("composite_score") or -1e9),
            ),
        )

    def _delete_duplicate_leaderboard_rows(self, canonical_rid: str) -> int:
        dup_rows = [
            dict(row)
            for row in self.conn.execute(
                "SELECT entry_id, tier, timestamp, composite_score "
                "FROM leaderboard WHERE result_id = ?",
                (canonical_rid,),
            ).fetchall()
        ]
        if not dup_rows:
            return 0
        keep = self._leaderboard_row_to_keep(dup_rows)
        delete_ids = [
            str(row["entry_id"])
            for row in dup_rows
            if str(row["entry_id"]) != str(keep["entry_id"])
        ]
        if not delete_ids:
            return 0
        delete_placeholders = ",".join("?" for _ in delete_ids)
        self.conn.execute(
            f"DELETE FROM leaderboard WHERE entry_id IN ({delete_placeholders})",
            delete_ids,
        )
        return len(delete_ids)

    def repair_rebindable_orphan_leaderboard_rows(
        self,
        *,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Rebind orphan leaderboard rows to canonical program rows by fingerprint."""
        query = """
            SELECT l.entry_id,
                   l.result_id AS orphan_result_id,
                   l.architecture_desc AS graph_fingerprint,
                   pr.result_id AS canonical_result_id
            FROM leaderboard l
            LEFT JOIN program_results_compat pr0 ON pr0.result_id = l.result_id
            JOIN program_results_compat pr ON pr.graph_fingerprint = l.architecture_desc
            WHERE pr0.result_id IS NULL
              AND l.architecture_desc IS NOT NULL
              AND TRIM(l.architecture_desc) != ''
            ORDER BY l.timestamp DESC
        """
        params: List[Any] = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        rows = [dict(row) for row in self.conn.execute(query, params).fetchall()]
        if not rows:
            return _empty_orphan_repair_summary()

        rebound_rows = 0
        deleted_duplicate_rows = 0
        repaired_fps: List[str] = []
        for graph_fingerprint, bucket in self._group_orphan_leaderboard_rows(
            rows
        ).items():
            canonical_rid = bucket["canonical_result_id"]
            orphan_entry_ids = bucket["entry_ids"]
            repaired_fps.append(graph_fingerprint)
            rebound_rows += len(orphan_entry_ids)
            placeholders = ",".join("?" for _ in orphan_entry_ids)
            self.conn.execute(
                f"UPDATE leaderboard SET result_id = ? WHERE entry_id IN ({placeholders})",
                [canonical_rid, *orphan_entry_ids],
            )
            self._sync_fingerprint_leaderboard(canonical_rid)
            deleted_duplicate_rows += self._delete_duplicate_leaderboard_rows(
                canonical_rid
            )

        self._maybe_commit()
        return {
            "rebound_rows": rebound_rows,
            "fingerprints_repaired": len(repaired_fps),
            "deleted_duplicate_rows": deleted_duplicate_rows,
            "fingerprints": repaired_fps,
        }
