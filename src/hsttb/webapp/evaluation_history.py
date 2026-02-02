"""
Evaluation History Storage.

Stores evaluation inputs and results in SQLite for later visualization.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path.home() / ".hsttb" / "evaluation_history.db"


@dataclass
class EvaluationEntry:
    """A single evaluation history entry."""

    id: str
    timestamp: str

    # Input
    ground_truth: str | None
    predicted: str
    mode: str  # "reference_based", "quality_only", "combined"

    # Configuration
    config: dict = field(default_factory=dict)

    # Results
    overall_score: float | None = None
    results: dict = field(default_factory=dict)

    # Metadata
    label: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "mode": self.mode,
            "config": self.config,
            "overall_score": self.overall_score,
            "results": self.results,
            "label": self.label,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvaluationEntry:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            ground_truth=data.get("ground_truth"),
            predicted=data["predicted"],
            mode=data["mode"],
            config=data.get("config", {}),
            overall_score=data.get("overall_score"),
            results=data.get("results", {}),
            label=data.get("label", ""),
            notes=data.get("notes", ""),
        )


class EvaluationHistoryStore:
    """SQLite-backed evaluation history storage."""

    def __init__(self, db_path: Path | None = None):
        """Initialize the store."""
        self._db_path = db_path or DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_history (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    ground_truth TEXT,
                    predicted TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    config TEXT NOT NULL DEFAULT '{}',
                    overall_score REAL,
                    results TEXT NOT NULL DEFAULT '{}',
                    label TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index on timestamp for faster sorting
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON evaluation_history(timestamp DESC)
            """)

            conn.commit()
            logger.info(f"Evaluation history database initialized at {self._db_path}")
        finally:
            conn.close()

    def add_entry(
        self,
        ground_truth: str | None,
        predicted: str,
        mode: str,
        config: dict,
        overall_score: float | None,
        results: dict,
        label: str = "",
        notes: str = "",
    ) -> EvaluationEntry:
        """Add a new evaluation entry."""
        entry_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        entry = EvaluationEntry(
            id=entry_id,
            timestamp=timestamp,
            ground_truth=ground_truth,
            predicted=predicted,
            mode=mode,
            config=config,
            overall_score=overall_score,
            results=results,
            label=label,
            notes=notes,
        )

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO evaluation_history
                (id, timestamp, ground_truth, predicted, mode, config, overall_score, results, label, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.timestamp,
                    entry.ground_truth,
                    entry.predicted,
                    entry.mode,
                    json.dumps(entry.config),
                    entry.overall_score,
                    json.dumps(entry.results),
                    entry.label,
                    entry.notes,
                ),
            )
            conn.commit()
            logger.info(f"Added evaluation entry: {entry_id}")
        finally:
            conn.close()

        return entry

    def get_entry(self, entry_id: str) -> EvaluationEntry | None:
        """Get a specific entry by ID."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM evaluation_history WHERE id = ?",
                (entry_id,),
            ).fetchone()

            if row:
                return self._row_to_entry(row)
            return None
        finally:
            conn.close()

    def list_entries(
        self,
        limit: int | None = 50,
        offset: int = 0,
        mode: str | None = None,
    ) -> list[EvaluationEntry]:
        """List evaluation entries."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM evaluation_history"
            params: list = []

            if mode:
                query += " WHERE mode = ?"
                params.append(mode)

            query += " ORDER BY timestamp DESC"

            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_entry(row) for row in rows]
        finally:
            conn.close()

    def update_entry(
        self,
        entry_id: str,
        label: str | None = None,
        notes: str | None = None,
    ) -> bool:
        """Update entry label or notes."""
        conn = self._get_connection()
        try:
            updates = []
            params = []

            if label is not None:
                updates.append("label = ?")
                params.append(label)
            if notes is not None:
                updates.append("notes = ?")
                params.append(notes)

            if not updates:
                return False

            params.append(entry_id)
            query = f"UPDATE evaluation_history SET {', '.join(updates)} WHERE id = ?"

            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM evaluation_history WHERE id = ?",
                (entry_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def clear_all(self) -> int:
        """Clear all entries."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM evaluation_history")
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Get history statistics."""
        conn = self._get_connection()
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM evaluation_history"
            ).fetchone()[0]

            by_mode = {}
            for row in conn.execute(
                "SELECT mode, COUNT(*) as count FROM evaluation_history GROUP BY mode"
            ).fetchall():
                by_mode[row["mode"]] = row["count"]

            avg_score = conn.execute(
                "SELECT AVG(overall_score) FROM evaluation_history WHERE overall_score IS NOT NULL"
            ).fetchone()[0]

            recent = conn.execute(
                "SELECT timestamp FROM evaluation_history ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()

            return {
                "total_entries": total,
                "by_mode": by_mode,
                "average_score": round(avg_score, 4) if avg_score else None,
                "last_evaluation": recent["timestamp"] if recent else None,
            }
        finally:
            conn.close()

    def search(self, query: str, limit: int = 20) -> list[EvaluationEntry]:
        """Search entries by text content."""
        conn = self._get_connection()
        try:
            search_term = f"%{query}%"
            rows = conn.execute(
                """
                SELECT * FROM evaluation_history
                WHERE ground_truth LIKE ?
                   OR predicted LIKE ?
                   OR label LIKE ?
                   OR notes LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (search_term, search_term, search_term, search_term, limit),
            ).fetchall()
            return [self._row_to_entry(row) for row in rows]
        finally:
            conn.close()

    def _row_to_entry(self, row: sqlite3.Row) -> EvaluationEntry:
        """Convert database row to EvaluationEntry."""
        return EvaluationEntry(
            id=row["id"],
            timestamp=row["timestamp"],
            ground_truth=row["ground_truth"],
            predicted=row["predicted"],
            mode=row["mode"],
            config=json.loads(row["config"]) if row["config"] else {},
            overall_score=row["overall_score"],
            results=json.loads(row["results"]) if row["results"] else {},
            label=row["label"] or "",
            notes=row["notes"] or "",
        )


# Singleton instance
_store: EvaluationHistoryStore | None = None


def get_evaluation_history() -> EvaluationHistoryStore:
    """Get singleton history store."""
    global _store
    if _store is None:
        _store = EvaluationHistoryStore()
    return _store
