"""
Audio file loading utilities.

This module provides functions for loading audio files from disk,
converting them to the required format (16kHz mono), and generating
checksums for reproducibility verification.

Example:
    >>> from hsttb.audio.loader import AudioLoader
    >>> loader = AudioLoader()
    >>> audio_data, sample_rate = loader.load("recording.wav")
    >>> print(f"Loaded {len(audio_data)} samples at {sample_rate}Hz")
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from hsttb.core.exceptions import AudioFormatError, AudioLoadError

if TYPE_CHECKING:
    pass


# Supported audio formats
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


class AudioLoader:
    """
    Audio file loader with format conversion.

    Loads audio files and converts them to the standard format
    used by the framework (16kHz mono float32).

    Attributes:
        target_sample_rate: Target sample rate for output.
        target_channels: Target number of channels (1 for mono).

    Example:
        >>> loader = AudioLoader(target_sample_rate=16000)
        >>> audio, sr = loader.load("path/to/audio.wav")
        >>> assert sr == 16000
        >>> assert audio.ndim == 1  # mono
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
    ) -> None:
        """
        Initialize the audio loader.

        Args:
            target_sample_rate: Desired sample rate (default: 16000).
            target_channels: Desired channels (default: 1 for mono).
        """
        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if target_channels not in (1, 2):
            raise ValueError("target_channels must be 1 or 2")

        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels

    def load(self, file_path: Path | str) -> tuple[np.ndarray, int]:
        """
        Load an audio file and convert to target format.

        Args:
            file_path: Path to the audio file.

        Returns:
            Tuple of (audio_data, sample_rate) where audio_data
            is a numpy array of float32 samples.

        Raises:
            AudioLoadError: If file cannot be loaded.
            AudioFormatError: If file format is not supported.
        """
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            raise AudioLoadError(f"Audio file not found: {path}", file_path=str(path))

        # Validate extension
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise AudioFormatError(
                f"Unsupported audio format: {path.suffix}",
                expected_format=", ".join(SUPPORTED_EXTENSIONS),
                actual_format=path.suffix,
            )

        try:
            # Load audio file
            audio_data, sample_rate = sf.read(path, dtype="float32")
        except Exception as e:
            raise AudioLoadError(
                f"Failed to load audio file: {e}", file_path=str(path)
            ) from e

        # Convert to mono if needed
        if audio_data.ndim > 1 and self.target_channels == 1:
            audio_data = self._to_mono(audio_data)

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            audio_data = self._resample(
                audio_data, sample_rate, self.target_sample_rate
            )

        return audio_data, self.target_sample_rate

    def load_raw(self, file_path: Path | str) -> tuple[np.ndarray, int]:
        """
        Load audio file without format conversion.

        Args:
            file_path: Path to the audio file.

        Returns:
            Tuple of (audio_data, sample_rate) as-is from file.

        Raises:
            AudioLoadError: If file cannot be loaded.
        """
        path = Path(file_path)

        if not path.exists():
            raise AudioLoadError(f"Audio file not found: {path}", file_path=str(path))

        try:
            audio_data, sample_rate = sf.read(path, dtype="float32")
            return audio_data, sample_rate
        except Exception as e:
            raise AudioLoadError(
                f"Failed to load audio file: {e}", file_path=str(path)
            ) from e

    def get_info(self, file_path: Path | str) -> dict[str, object]:
        """
        Get information about an audio file without loading it fully.

        Args:
            file_path: Path to the audio file.

        Returns:
            Dictionary with file information.

        Raises:
            AudioLoadError: If file info cannot be read.
        """
        path = Path(file_path)

        if not path.exists():
            raise AudioLoadError(f"Audio file not found: {path}", file_path=str(path))

        try:
            info = sf.info(path)
            return {
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration_seconds": info.duration,
                "frames": info.frames,
                "format": info.format,
                "subtype": info.subtype,
            }
        except Exception as e:
            raise AudioLoadError(
                f"Failed to read audio info: {e}", file_path=str(path)
            ) from e

    def get_checksum(self, file_path: Path | str) -> str:
        """
        Generate a checksum for an audio file.

        This is useful for verifying that the same audio file
        is used across benchmark runs.

        Args:
            file_path: Path to the audio file.

        Returns:
            SHA-256 checksum hex string.

        Raises:
            AudioLoadError: If file cannot be read.
        """
        path = Path(file_path)

        if not path.exists():
            raise AudioLoadError(f"Audio file not found: {path}", file_path=str(path))

        try:
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            raise AudioLoadError(
                f"Failed to compute checksum: {e}", file_path=str(path)
            ) from e

    def _to_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono by averaging channels.

        Args:
            audio_data: Multi-channel audio array.

        Returns:
            Mono audio array.
        """
        if audio_data.ndim == 1:
            return audio_data
        # Average across channels
        result: np.ndarray = np.mean(audio_data, axis=1).astype(np.float32)
        return result

    def _resample(
        self,
        audio_data: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Uses simple linear interpolation. For production use,
        consider using scipy.signal.resample or librosa.

        Args:
            audio_data: Input audio samples.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio array.
        """
        if orig_sr == target_sr:
            return audio_data

        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        target_length = int(len(audio_data) * ratio)

        # Simple linear interpolation resampling
        # For better quality, use scipy.signal.resample
        indices = np.linspace(0, len(audio_data) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)

        result: np.ndarray = resampled.astype(np.float32)
        return result


def get_audio_duration(file_path: Path | str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        file_path: Path to the audio file.

    Returns:
        Duration in seconds.

    Raises:
        AudioLoadError: If file info cannot be read.
    """
    loader = AudioLoader()
    info = loader.get_info(file_path)
    duration = info["duration_seconds"]
    if not isinstance(duration, (int, float)):
        raise ValueError("Invalid duration value")
    return float(duration)


def validate_audio_file(file_path: Path | str) -> bool:
    """
    Validate that a file is a readable audio file.

    Args:
        file_path: Path to the audio file.

    Returns:
        True if file is valid, False otherwise.
    """
    path = Path(file_path)

    if not path.exists():
        return False

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False

    try:
        sf.info(path)
        return True
    except Exception:
        return False
