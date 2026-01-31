"""
Tests for audio upload and processing handler.

Tests file upload, validation, metadata extraction, and cleanup.
"""
from __future__ import annotations

import io
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

# Import the audio_handler module directly using importlib.util
# This avoids the webapp.__init__.py -> app.py import chain
# which requires python-multipart
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "audio_handler",
        Path(__file__).parent.parent / "src" / "hsttb" / "webapp" / "audio_handler.py"
    )
    if spec and spec.loader:
        audio_handler_module: ModuleType = importlib.util.module_from_spec(spec)
        sys.modules["audio_handler"] = audio_handler_module
        spec.loader.exec_module(audio_handler_module)

        SUPPORTED_FORMATS = audio_handler_module.SUPPORTED_FORMATS
        AudioHandler = audio_handler_module.AudioHandler
        AudioMetadata = audio_handler_module.AudioMetadata
        AudioValidationError = audio_handler_module.AudioValidationError
        get_audio_handler = audio_handler_module.get_audio_handler
    else:
        pytest.skip("Could not load audio_handler module", allow_module_level=True)
except Exception as e:
    pytest.skip(f"Could not import audio_handler: {e}", allow_module_level=True)


# Mock UploadFile for testing
@dataclass
class MockUploadFile:
    """Mock FastAPI UploadFile for testing."""

    filename: str | None
    _content: bytes
    _position: int = 0

    async def read(self) -> bytes:
        """Read file content."""
        return self._content

    async def seek(self, position: int) -> None:
        """Seek to position."""
        self._position = position


def create_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Create a valid WAV file in memory."""
    num_samples = int(duration_s * sample_rate)
    samples = np.random.randint(-32768, 32767, num_samples, dtype=np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())

    return buffer.getvalue()


class TestAudioMetadata:
    """Tests for AudioMetadata dataclass."""

    def test_creation(self) -> None:
        """Create AudioMetadata with basic attributes."""
        metadata = AudioMetadata(
            file_id="test-123",
            filename="test.wav",
            file_path=Path("/tmp/test.wav"),
            format="wav",
            mime_type="audio/wav",
            file_size=1024,
        )

        assert metadata.file_id == "test-123"
        assert metadata.filename == "test.wav"
        assert metadata.format == "wav"

    def test_to_dict(self) -> None:
        """to_dict returns serializable dictionary."""
        metadata = AudioMetadata(
            file_id="test-456",
            filename="audio.mp3",
            file_path=Path("/tmp/audio.mp3"),
            format="mp3",
            mime_type="audio/mpeg",
            file_size=2048,
            duration_seconds=5.5,
            sample_rate=44100,
            channels=2,
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert result["file_id"] == "test-456"
        assert result["duration_seconds"] == 5.5
        assert result["sample_rate"] == 44100

    def test_optional_fields(self) -> None:
        """Optional fields default to None."""
        metadata = AudioMetadata(
            file_id="test",
            filename="test.wav",
            file_path=Path("/tmp/test.wav"),
            format="wav",
            mime_type="audio/wav",
            file_size=100,
        )

        assert metadata.duration_seconds is None
        assert metadata.sample_rate is None
        assert metadata.channels is None


class TestAudioHandler:
    """Tests for AudioHandler class."""

    @pytest.fixture
    def handler(self, tmp_path: Path) -> AudioHandler:
        """Create AudioHandler with temp directory."""
        return AudioHandler(storage_dir=tmp_path)

    @pytest.fixture
    def valid_wav_upload(self) -> MockUploadFile:
        """Create a valid WAV file upload."""
        return MockUploadFile(
            filename="test_audio.wav",
            _content=create_wav_bytes(duration_s=1.0),
        )

    @pytest.mark.asyncio
    async def test_save_upload_valid_wav(
        self, handler: AudioHandler, valid_wav_upload: MockUploadFile
    ) -> None:
        """Save a valid WAV file upload."""
        metadata = await handler.save_upload(valid_wav_upload)

        assert metadata.filename == "test_audio.wav"
        assert metadata.format == "wav"
        assert metadata.mime_type == "audio/wav"
        assert metadata.file_size > 0
        assert metadata.file_path.exists()

    @pytest.mark.asyncio
    async def test_save_upload_extracts_metadata(
        self, handler: AudioHandler, valid_wav_upload: MockUploadFile
    ) -> None:
        """Save upload extracts audio metadata."""
        metadata = await handler.save_upload(valid_wav_upload)

        # Should extract duration and sample rate from WAV
        assert metadata.duration_seconds is not None
        assert metadata.duration_seconds > 0
        assert metadata.sample_rate == 16000
        assert metadata.channels == 1

    @pytest.mark.asyncio
    async def test_save_upload_no_filename(self, handler: AudioHandler) -> None:
        """Save upload raises error for missing filename."""
        upload = MockUploadFile(filename=None, _content=b"data")

        with pytest.raises(AudioValidationError, match="No filename"):
            await handler.save_upload(upload)

    @pytest.mark.asyncio
    async def test_save_upload_unsupported_format(self, handler: AudioHandler) -> None:
        """Save upload raises error for unsupported format."""
        upload = MockUploadFile(filename="test.xyz", _content=b"data")

        with pytest.raises(AudioValidationError, match="Unsupported format"):
            await handler.save_upload(upload)

    @pytest.mark.asyncio
    async def test_save_upload_empty_file(self, handler: AudioHandler) -> None:
        """Save upload raises error for empty file."""
        upload = MockUploadFile(filename="empty.wav", _content=b"")

        with pytest.raises(AudioValidationError, match="Empty file"):
            await handler.save_upload(upload)

    @pytest.mark.asyncio
    async def test_save_upload_file_too_large(self, tmp_path: Path) -> None:
        """Save upload raises error for file exceeding max size."""
        handler = AudioHandler(storage_dir=tmp_path, max_file_size=100)
        upload = MockUploadFile(filename="large.wav", _content=b"x" * 200)

        with pytest.raises(AudioValidationError, match="File too large"):
            await handler.save_upload(upload)

    def test_get_file_exists(self, handler: AudioHandler, tmp_path: Path) -> None:
        """Get file returns path for existing file."""
        # Create a file manually
        file_id = "test-get-file"
        file_path = tmp_path / f"{file_id}.wav"
        file_path.write_bytes(b"audio data")

        result = handler.get_file(file_id)

        assert result == file_path

    def test_get_file_not_exists(self, handler: AudioHandler) -> None:
        """Get file returns None for nonexistent file."""
        result = handler.get_file("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_file(
        self, handler: AudioHandler, valid_wav_upload: MockUploadFile
    ) -> None:
        """Delete file removes the file."""
        metadata = await handler.save_upload(valid_wav_upload)
        assert metadata.file_path.exists()

        result = handler.delete_file(metadata.file_id)

        assert result is True
        assert not metadata.file_path.exists()

    def test_delete_file_not_exists(self, handler: AudioHandler) -> None:
        """Delete file returns False for nonexistent file."""
        result = handler.delete_file("nonexistent")
        assert result is False

    def test_get_file_hash(self, handler: AudioHandler, tmp_path: Path) -> None:
        """Get file hash returns SHA256 hash."""
        file_path = tmp_path / "hash_test.wav"
        file_path.write_bytes(b"test content for hashing")

        hash_result = handler.get_file_hash(file_path)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex digest


class TestAudioHandlerValidation:
    """Tests for audio validation in AudioHandler."""

    @pytest.fixture
    def handler(self, tmp_path: Path) -> AudioHandler:
        """Create AudioHandler with temp directory."""
        return AudioHandler(storage_dir=tmp_path)

    def test_validate_audio_valid_wav(
        self, handler: AudioHandler, tmp_path: Path
    ) -> None:
        """Validate audio returns True for valid WAV."""
        file_path = tmp_path / "valid.wav"
        file_path.write_bytes(create_wav_bytes())

        result = handler.validate_audio(file_path)

        assert result is True

    def test_validate_audio_invalid(
        self, handler: AudioHandler, tmp_path: Path
    ) -> None:
        """Validate audio returns False for invalid file."""
        file_path = tmp_path / "invalid.wav"
        file_path.write_bytes(b"not audio data")

        result = handler.validate_audio(file_path)

        assert result is False


class TestAudioHandlerConversion:
    """Tests for audio conversion in AudioHandler."""

    @pytest.fixture
    def handler(self, tmp_path: Path) -> AudioHandler:
        """Create AudioHandler with temp directory."""
        return AudioHandler(storage_dir=tmp_path)

    def test_convert_to_wav(self, handler: AudioHandler, tmp_path: Path) -> None:
        """Convert audio to WAV format."""
        # Create input file
        input_path = tmp_path / "input.wav"
        input_path.write_bytes(create_wav_bytes(sample_rate=44100))

        # Convert
        try:
            output_path = handler.convert_to_wav(
                input_path, sample_rate=16000, channels=1
            )
            assert output_path.exists()
        except RuntimeError:
            # May fail if soundfile/scipy not available
            pytest.skip("Audio conversion dependencies not available")


class TestAudioHandlerCleanup:
    """Tests for file cleanup in AudioHandler."""

    @pytest.fixture
    def handler(self, tmp_path: Path) -> AudioHandler:
        """Create AudioHandler with temp directory."""
        return AudioHandler(storage_dir=tmp_path)

    def test_cleanup_old_files(self, handler: AudioHandler, tmp_path: Path) -> None:
        """Cleanup removes old files."""
        import time

        # Create an "old" file by modifying its mtime
        old_file = tmp_path / "old.wav"
        old_file.write_bytes(b"old audio")

        # Set mtime to 2 days ago
        import os
        old_time = time.time() - (48 * 3600)
        os.utime(old_file, (old_time, old_time))

        deleted = handler.cleanup_old_files(max_age_hours=24)

        assert deleted == 1
        assert not old_file.exists()

    def test_cleanup_keeps_recent_files(
        self, handler: AudioHandler, tmp_path: Path
    ) -> None:
        """Cleanup keeps recent files."""
        # Create a recent file
        recent_file = tmp_path / "recent.wav"
        recent_file.write_bytes(b"recent audio")

        deleted = handler.cleanup_old_files(max_age_hours=24)

        assert deleted == 0
        assert recent_file.exists()


class TestSupportedFormats:
    """Tests for supported audio formats."""

    def test_wav_supported(self) -> None:
        """WAV format is supported."""
        assert ".wav" in SUPPORTED_FORMATS
        assert SUPPORTED_FORMATS[".wav"] == "audio/wav"

    def test_mp3_supported(self) -> None:
        """MP3 format is supported."""
        assert ".mp3" in SUPPORTED_FORMATS
        assert SUPPORTED_FORMATS[".mp3"] == "audio/mpeg"

    def test_flac_supported(self) -> None:
        """FLAC format is supported."""
        assert ".flac" in SUPPORTED_FORMATS

    def test_ogg_supported(self) -> None:
        """OGG format is supported."""
        assert ".ogg" in SUPPORTED_FORMATS

    def test_webm_supported(self) -> None:
        """WebM format is supported."""
        assert ".webm" in SUPPORTED_FORMATS


class TestGetAudioHandler:
    """Tests for global audio handler."""

    def test_get_audio_handler_returns_instance(self) -> None:
        """get_audio_handler returns AudioHandler instance."""
        handler = get_audio_handler()
        assert isinstance(handler, AudioHandler)

    def test_get_audio_handler_singleton(self) -> None:
        """get_audio_handler returns same instance."""
        handler1 = get_audio_handler()
        handler2 = get_audio_handler()
        assert handler1 is handler2
