"""
TTS history storage for persistent audio management.

Stores generated TTS audio files and their metadata for later reuse
in STT evaluation workflows.

Example:
    >>> history = TTSHistory()
    >>> entry = history.add_entry(
    ...     file_id="abc123",
    ...     text="Patient takes metformin 500mg",
    ...     voice="professional",
    ...     file_path=Path("/path/to/audio.mp3")
    ... )
    >>> print(entry.created_at)
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default storage directory (user's home or app data)
DEFAULT_STORAGE_DIR = Path.home() / ".hsttb" / "tts_history"


@dataclass
class TTSHistoryEntry:
    """A single TTS history entry."""

    file_id: str
    text: str
    voice: str
    file_path: str
    created_at: str
    file_size: int = 0
    duration_seconds: float | None = None
    model: str = "eleven_turbo_v2"
    label: str = ""  # User-defined label

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TTSHistoryEntry:
        """Create from dictionary."""
        return cls(
            file_id=data["file_id"],
            text=data["text"],
            voice=data["voice"],
            file_path=data["file_path"],
            created_at=data["created_at"],
            file_size=data.get("file_size", 0),
            duration_seconds=data.get("duration_seconds"),
            model=data.get("model", "eleven_turbo_v2"),
            label=data.get("label", ""),
        )

    @property
    def text_preview(self) -> str:
        """Get a short preview of the text."""
        if len(self.text) <= 50:
            return self.text
        return self.text[:47] + "..."


class TTSHistory:
    """Manages TTS history storage and retrieval."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        """Initialize TTS history.

        Args:
            storage_dir: Directory for storing TTS files and metadata.
                        Defaults to ~/.hsttb/tts_history
        """
        self._storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._audio_dir = self._storage_dir / "audio"
        self._audio_dir.mkdir(exist_ok=True)
        self._index_file = self._storage_dir / "index.json"
        self._entries: dict[str, TTSHistoryEntry] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the index file."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    data = json.load(f)
                    self._entries = {
                        entry_data["file_id"]: TTSHistoryEntry.from_dict(entry_data)
                        for entry_data in data.get("entries", [])
                    }
                logger.info(f"Loaded {len(self._entries)} TTS history entries")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load TTS history index: {e}")
                self._entries = {}
        else:
            self._entries = {}

    def _save_index(self) -> None:
        """Save the index file."""
        data = {
            "version": 1,
            "entries": [entry.to_dict() for entry in self._entries.values()],
        }
        with open(self._index_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_entry(
        self,
        file_id: str,
        text: str,
        voice: str,
        file_path: Path,
        model: str = "eleven_turbo_v2",
        label: str = "",
        duration_seconds: float | None = None,
    ) -> TTSHistoryEntry:
        """Add a new TTS entry to history.

        Args:
            file_id: Unique identifier for the file
            text: The text that was converted to speech
            voice: Voice used for generation
            file_path: Path to the source audio file
            model: TTS model used
            label: Optional user-defined label
            duration_seconds: Audio duration in seconds

        Returns:
            The created TTSHistoryEntry
        """
        # Copy file to history storage
        dest_path = self._audio_dir / f"{file_id}.mp3"
        if file_path != dest_path:
            shutil.copy2(file_path, dest_path)

        file_size = dest_path.stat().st_size

        entry = TTSHistoryEntry(
            file_id=file_id,
            text=text,
            voice=voice,
            file_path=str(dest_path),
            created_at=datetime.now().isoformat(),
            file_size=file_size,
            duration_seconds=duration_seconds,
            model=model,
            label=label,
        )

        self._entries[file_id] = entry
        self._save_index()
        logger.info(f"Added TTS history entry: {file_id}")

        return entry

    def get_entry(self, file_id: str) -> TTSHistoryEntry | None:
        """Get a TTS entry by ID."""
        return self._entries.get(file_id)

    def get_file_path(self, file_id: str) -> Path | None:
        """Get the audio file path for an entry."""
        entry = self._entries.get(file_id)
        if entry:
            path = Path(entry.file_path)
            if path.exists():
                return path
        return None

    def list_entries(
        self,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TTSHistoryEntry]:
        """List all TTS history entries.

        Args:
            limit: Maximum number of entries to return
            offset: Number of entries to skip

        Returns:
            List of TTSHistoryEntry sorted by creation date (newest first)
        """
        entries = sorted(
            self._entries.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )

        if offset:
            entries = entries[offset:]
        if limit:
            entries = entries[:limit]

        return entries

    def delete_entry(self, file_id: str) -> bool:
        """Delete a TTS entry and its audio file.

        Args:
            file_id: The entry ID to delete

        Returns:
            True if deleted, False if not found
        """
        entry = self._entries.get(file_id)
        if not entry:
            return False

        # Delete the audio file
        file_path = Path(entry.file_path)
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Deleted TTS audio file: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to delete audio file: {e}")

        # Remove from index
        del self._entries[file_id]
        self._save_index()
        logger.info(f"Deleted TTS history entry: {file_id}")

        return True

    def update_label(self, file_id: str, label: str) -> bool:
        """Update the label for an entry.

        Args:
            file_id: The entry ID
            label: New label

        Returns:
            True if updated, False if not found
        """
        entry = self._entries.get(file_id)
        if not entry:
            return False

        entry.label = label
        self._save_index()
        return True

    def clear_all(self) -> int:
        """Delete all TTS history entries.

        Returns:
            Number of entries deleted
        """
        count = len(self._entries)

        # Delete all audio files
        for entry in self._entries.values():
            file_path = Path(entry.file_path)
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    pass

        self._entries = {}
        self._save_index()
        logger.info(f"Cleared {count} TTS history entries")

        return count

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the TTS history.

        Returns:
            Dictionary with stats (count, total_size, etc.)
        """
        total_size = sum(e.file_size for e in self._entries.values())
        voices = {}
        for entry in self._entries.values():
            voices[entry.voice] = voices.get(entry.voice, 0) + 1

        return {
            "count": len(self._entries),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "voices_used": voices,
            "storage_dir": str(self._storage_dir),
        }


# Singleton instance
_tts_history: TTSHistory | None = None


def get_tts_history() -> TTSHistory:
    """Get the global TTS history instance."""
    global _tts_history
    if _tts_history is None:
        _tts_history = TTSHistory()
    return _tts_history
