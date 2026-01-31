"""
Audio upload and processing handler.

Handles audio file uploads, validation, metadata extraction,
and temporary file management for the HSTTB webapp.

Example:
    >>> handler = AudioHandler()
    >>> result = await handler.save_upload(upload_file)
    >>> print(result.duration_seconds)
    15.5
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import UploadFile

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
    ".opus": "audio/opus",
}

# Maximum file size (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

# Audio storage directory
AUDIO_STORAGE_DIR = Path(tempfile.gettempdir()) / "hsttb_audio"


@dataclass
class AudioMetadata:
    """
    Metadata for an uploaded audio file.

    Attributes:
        file_id: Unique identifier for the file.
        filename: Original filename.
        file_path: Path to stored file.
        format: Audio format (e.g., "wav").
        mime_type: MIME type.
        file_size: Size in bytes.
        duration_seconds: Duration in seconds (if available).
        sample_rate: Sample rate in Hz (if available).
        channels: Number of audio channels (if available).
        created_at: Upload timestamp.
    """

    file_id: str
    filename: str
    file_path: Path
    format: str
    mime_type: str
    file_size: int
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "file_path": str(self.file_path),
            "format": self.format,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AudioValidationError(Exception):
    """Raised when audio validation fails."""

    pass


class AudioHandler:
    """
    Handles audio file uploads and processing.

    Provides methods for saving, validating, and extracting
    metadata from uploaded audio files.

    Attributes:
        storage_dir: Directory for storing uploaded files.
        max_file_size: Maximum allowed file size in bytes.

    Example:
        >>> handler = AudioHandler()
        >>> metadata = await handler.save_upload(file)
        >>> print(f"Saved: {metadata.file_path}")
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        max_file_size: int = MAX_FILE_SIZE,
    ) -> None:
        """
        Initialize the audio handler.

        Args:
            storage_dir: Directory for storing files.
            max_file_size: Maximum file size in bytes.
        """
        self._storage_dir = storage_dir or AUDIO_STORAGE_DIR
        self._max_file_size = max_file_size
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile) -> AudioMetadata:
        """
        Save an uploaded audio file.

        Args:
            file: FastAPI UploadFile object.

        Returns:
            AudioMetadata with file information.

        Raises:
            AudioValidationError: If file is invalid.
        """
        # Validate filename and format
        if not file.filename:
            raise AudioValidationError("No filename provided")

        original_filename = file.filename
        file_ext = Path(original_filename).suffix.lower()

        if file_ext not in SUPPORTED_FORMATS:
            raise AudioValidationError(
                f"Unsupported format: {file_ext}. "
                f"Supported: {', '.join(SUPPORTED_FORMATS.keys())}"
            )

        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Validate file size
        if file_size > self._max_file_size:
            raise AudioValidationError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB. "
                f"Maximum: {self._max_file_size / 1024 / 1024:.1f}MB"
            )

        if file_size == 0:
            raise AudioValidationError("Empty file")

        # Generate storage path
        file_path = self._storage_dir / f"{file_id}{file_ext}"

        # Write file
        with open(file_path, "wb") as f:
            f.write(content)

        # Extract metadata
        metadata = AudioMetadata(
            file_id=file_id,
            filename=original_filename,
            file_path=file_path,
            format=file_ext[1:],  # Remove leading dot
            mime_type=SUPPORTED_FORMATS[file_ext],
            file_size=file_size,
            created_at=datetime.now(),
        )

        # Try to extract audio properties
        try:
            audio_info = self._get_audio_info(file_path)
            metadata.duration_seconds = audio_info.get("duration")
            metadata.sample_rate = audio_info.get("sample_rate")
            metadata.channels = audio_info.get("channels")
        except Exception as e:
            logger.warning(f"Could not extract audio info: {e}")

        logger.info(f"Saved audio file: {file_id} ({file_size} bytes)")
        return metadata

    def _get_audio_info(self, file_path: Path) -> dict[str, Any]:
        """
        Extract audio file properties.

        Args:
            file_path: Path to audio file.

        Returns:
            Dictionary with audio properties.
        """
        try:
            import soundfile as sf

            with sf.SoundFile(str(file_path)) as f:
                return {
                    "duration": len(f) / f.samplerate,
                    "sample_rate": f.samplerate,
                    "channels": f.channels,
                    "format": f.format,
                    "subtype": f.subtype,
                }
        except Exception:
            # Try with wave module for WAV files
            if file_path.suffix.lower() == ".wav":
                return self._get_wav_info(file_path)
            raise

    def _get_wav_info(self, file_path: Path) -> dict[str, Any]:
        """Extract info from WAV file using standard library."""
        import wave

        with wave.open(str(file_path), "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            channels = f.getnchannels()
            return {
                "duration": frames / rate,
                "sample_rate": rate,
                "channels": channels,
            }

    def get_file(self, file_id: str) -> Path | None:
        """
        Get the path to a stored file.

        Args:
            file_id: The file identifier.

        Returns:
            Path to file or None if not found.
        """
        # Search for file with any supported extension
        for ext in SUPPORTED_FORMATS:
            file_path = self._storage_dir / f"{file_id}{ext}"
            if file_path.exists():
                return file_path
        return None

    def delete_file(self, file_id: str) -> bool:
        """
        Delete a stored file.

        Args:
            file_id: The file identifier.

        Returns:
            True if deleted, False if not found.
        """
        file_path = self.get_file(file_id)
        if file_path and file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted audio file: {file_id}")
            return True
        return False

    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Delete files older than specified age.

        Args:
            max_age_hours: Maximum file age in hours.

        Returns:
            Number of files deleted.
        """
        import time

        deleted = 0
        cutoff = time.time() - (max_age_hours * 3600)

        for file_path in self._storage_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                deleted += 1

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old audio files")

        return deleted

    def validate_audio(self, file_path: Path) -> bool:
        """
        Validate that a file is a valid audio file.

        Args:
            file_path: Path to file to validate.

        Returns:
            True if valid, False otherwise.
        """
        try:
            info = self._get_audio_info(file_path)
            return info.get("duration", 0) > 0
        except Exception:
            return False

    def get_file_hash(self, file_path: Path) -> str:
        """
        Get SHA256 hash of a file.

        Args:
            file_path: Path to file.

        Returns:
            Hex digest of file hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def convert_to_wav(
        self,
        input_path: Path,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> Path:
        """
        Convert audio file to WAV format.

        Args:
            input_path: Input audio file path.
            sample_rate: Target sample rate.
            channels: Target number of channels.

        Returns:
            Path to converted WAV file.

        Raises:
            RuntimeError: If conversion fails.
        """
        output_path = input_path.with_suffix(".converted.wav")

        try:
            import soundfile as sf
            import numpy as np

            # Read input file
            data, orig_rate = sf.read(str(input_path))

            # Convert to mono if needed
            if len(data.shape) > 1 and data.shape[1] > channels:
                data = np.mean(data, axis=1)

            # Resample if needed
            if orig_rate != sample_rate:
                try:
                    from scipy import signal
                    num_samples = int(len(data) * sample_rate / orig_rate)
                    data = signal.resample(data, num_samples)
                except ImportError:
                    logger.warning("scipy not available, skipping resample")

            # Write output
            sf.write(str(output_path), data, sample_rate)
            return output_path

        except Exception as e:
            raise RuntimeError(f"Audio conversion failed: {e}") from e


# Global handler instance
_audio_handler: AudioHandler | None = None


def get_audio_handler() -> AudioHandler:
    """Get the global audio handler instance."""
    global _audio_handler
    if _audio_handler is None:
        _audio_handler = AudioHandler()
    return _audio_handler
