"""
Tests for audio module.

Tests the audio loader and streaming chunker implementations.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hsttb.audio.chunker import StreamingChunker, collect_chunks, create_test_chunks
from hsttb.audio.loader import (
    SUPPORTED_EXTENSIONS,
    AudioLoader,
    get_audio_duration,
    validate_audio_file,
)
from hsttb.core.config import ChunkingConfig, NetworkConfig, StreamingProfile
from hsttb.core.exceptions import AudioFormatError, AudioLoadError


class TestAudioLoader:
    """Tests for AudioLoader class."""

    @pytest.fixture
    def loader(self) -> AudioLoader:
        """Create a standard audio loader."""
        return AudioLoader(target_sample_rate=16000, target_channels=1)

    @pytest.fixture
    def temp_wav_file(self) -> Path:
        """Create a temporary WAV file for testing."""
        # Generate 1 second of audio at 44100Hz stereo
        sample_rate = 44100
        duration = 1.0
        samples = int(sample_rate * duration)
        # Create stereo audio with simple sine waves
        t = np.linspace(0, duration, samples, dtype=np.float32)
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)  # 880Hz
        audio = np.column_stack([left, right])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            yield Path(f.name)

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def temp_mono_wav(self) -> Path:
        """Create a temporary mono WAV file."""
        sample_rate = 16000
        duration = 0.5
        samples = int(sample_rate * duration)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)).astype(
            np.float32
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            yield Path(f.name)

        Path(f.name).unlink(missing_ok=True)

    def test_initialization_defaults(self) -> None:
        """AudioLoader initializes with correct defaults."""
        loader = AudioLoader()
        assert loader.target_sample_rate == 16000
        assert loader.target_channels == 1

    def test_initialization_custom(self) -> None:
        """AudioLoader accepts custom parameters."""
        loader = AudioLoader(target_sample_rate=22050, target_channels=2)
        assert loader.target_sample_rate == 22050
        assert loader.target_channels == 2

    def test_initialization_invalid_sample_rate(self) -> None:
        """Raises error for invalid sample rate."""
        with pytest.raises(ValueError, match="positive"):
            AudioLoader(target_sample_rate=0)

        with pytest.raises(ValueError, match="positive"):
            AudioLoader(target_sample_rate=-16000)

    def test_initialization_invalid_channels(self) -> None:
        """Raises error for invalid channels."""
        with pytest.raises(ValueError, match="1 or 2"):
            AudioLoader(target_channels=3)

    def test_load_and_convert(self, loader: AudioLoader, temp_wav_file: Path) -> None:
        """Load converts audio to target format."""
        audio, sample_rate = loader.load(temp_wav_file)

        # Should be converted to 16kHz mono
        assert sample_rate == 16000
        assert audio.ndim == 1  # mono
        assert audio.dtype == np.float32
        # Duration should be preserved (approximately 1 second)
        expected_samples = 16000 * 1  # 1 second at 16kHz
        assert abs(len(audio) - expected_samples) < 100  # Allow small variation

    def test_load_mono_no_conversion(
        self, loader: AudioLoader, temp_mono_wav: Path
    ) -> None:
        """Load handles already-mono files correctly."""
        audio, sample_rate = loader.load(temp_mono_wav)

        assert sample_rate == 16000
        assert audio.ndim == 1
        # Duration: 0.5 seconds at 16kHz
        expected_samples = 8000
        assert abs(len(audio) - expected_samples) < 50

    def test_load_file_not_found(self, loader: AudioLoader) -> None:
        """Load raises AudioLoadError for missing file."""
        with pytest.raises(AudioLoadError, match="not found"):
            loader.load("/nonexistent/audio.wav")

    def test_load_unsupported_format(self, loader: AudioLoader) -> None:
        """Load raises AudioFormatError for unsupported formats."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not audio")
            temp_path = Path(f.name)

        try:
            with pytest.raises(AudioFormatError, match="Unsupported"):
                loader.load(temp_path)
        finally:
            temp_path.unlink()

    def test_load_raw(self, loader: AudioLoader, temp_wav_file: Path) -> None:
        """Load raw returns original format without conversion."""
        audio, sample_rate = loader.load_raw(temp_wav_file)

        # Should be original 44100Hz stereo
        assert sample_rate == 44100
        assert audio.ndim == 2  # stereo
        assert audio.shape[1] == 2  # 2 channels

    def test_load_raw_file_not_found(self, loader: AudioLoader) -> None:
        """Load raw raises error for missing file."""
        with pytest.raises(AudioLoadError, match="not found"):
            loader.load_raw("/nonexistent/file.wav")

    def test_get_info(self, loader: AudioLoader, temp_wav_file: Path) -> None:
        """Get info returns correct file information."""
        info = loader.get_info(temp_wav_file)

        assert info["sample_rate"] == 44100
        assert info["channels"] == 2
        assert isinstance(info["duration_seconds"], float)
        assert info["duration_seconds"] > 0.9  # approximately 1 second
        assert info["frames"] == 44100  # 1 second at 44100Hz
        assert "format" in info
        assert "subtype" in info

    def test_get_info_file_not_found(self, loader: AudioLoader) -> None:
        """Get info raises error for missing file."""
        with pytest.raises(AudioLoadError, match="not found"):
            loader.get_info("/nonexistent/file.wav")

    def test_get_checksum(self, loader: AudioLoader, temp_wav_file: Path) -> None:
        """Get checksum returns consistent hash."""
        checksum1 = loader.get_checksum(temp_wav_file)
        checksum2 = loader.get_checksum(temp_wav_file)

        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA-256 hex string
        assert all(c in "0123456789abcdef" for c in checksum1)

    def test_get_checksum_file_not_found(self, loader: AudioLoader) -> None:
        """Get checksum raises error for missing file."""
        with pytest.raises(AudioLoadError, match="not found"):
            loader.get_checksum("/nonexistent/file.wav")

    def test_to_mono(self, loader: AudioLoader) -> None:
        """Internal mono conversion works correctly."""
        stereo = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32
        )  # stereo
        mono = loader._to_mono(stereo)

        assert mono.ndim == 1
        assert len(mono) == 3
        np.testing.assert_array_almost_equal(mono, [0.5, 0.5, 0.5])

    def test_to_mono_already_mono(self, loader: AudioLoader) -> None:
        """Mono input returns unchanged."""
        mono_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = loader._to_mono(mono_input)

        np.testing.assert_array_equal(result, mono_input)

    def test_resample(self, loader: AudioLoader) -> None:
        """Internal resampling works correctly."""
        # Create 1000 samples at original rate
        original = np.linspace(0, 1, 1000, dtype=np.float32)
        # Resample from 44100 to 16000
        resampled = loader._resample(original, 44100, 16000)

        # Check approximate length ratio
        expected_length = int(1000 * 16000 / 44100)
        assert abs(len(resampled) - expected_length) < 10

    def test_resample_same_rate(self, loader: AudioLoader) -> None:
        """Resampling with same rate returns original."""
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = loader._resample(original, 16000, 16000)

        np.testing.assert_array_equal(result, original)


class TestAudioLoaderUtilities:
    """Tests for audio loader utility functions."""

    @pytest.fixture
    def temp_wav_file(self) -> Path:
        """Create a temporary WAV file."""
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        audio = np.zeros(samples, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            yield Path(f.name)

        Path(f.name).unlink(missing_ok=True)

    def test_get_audio_duration(self, temp_wav_file: Path) -> None:
        """get_audio_duration returns correct duration."""
        duration = get_audio_duration(temp_wav_file)
        assert abs(duration - 2.0) < 0.01  # approximately 2 seconds

    def test_validate_audio_file_valid(self, temp_wav_file: Path) -> None:
        """validate_audio_file returns True for valid files."""
        assert validate_audio_file(temp_wav_file) is True

    def test_validate_audio_file_nonexistent(self) -> None:
        """validate_audio_file returns False for missing files."""
        assert validate_audio_file("/nonexistent/file.wav") is False

    def test_validate_audio_file_unsupported_extension(self) -> None:
        """validate_audio_file returns False for unsupported extensions."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not audio")
            temp_path = Path(f.name)

        try:
            assert validate_audio_file(temp_path) is False
        finally:
            temp_path.unlink()

    def test_supported_extensions(self) -> None:
        """Supported extensions include common audio formats."""
        assert ".wav" in SUPPORTED_EXTENSIONS
        assert ".flac" in SUPPORTED_EXTENSIONS
        assert ".ogg" in SUPPORTED_EXTENSIONS
        assert ".mp3" in SUPPORTED_EXTENSIONS


class TestStreamingChunker:
    """Tests for StreamingChunker class."""

    @pytest.fixture
    def profile(self) -> StreamingProfile:
        """Create a test streaming profile."""
        return StreamingProfile(
            profile_name="test",
            chunking=ChunkingConfig(
                chunk_size_ms=1000,
                chunk_jitter_ms=0,  # No jitter for deterministic tests
                overlap_ms=0,
            ),
            network=NetworkConfig(
                delay_ms=0,  # No delay for faster tests
                jitter_ms=0,
            ),
        )

    @pytest.fixture
    def profile_with_overlap(self) -> StreamingProfile:
        """Create a profile with overlap."""
        return StreamingProfile(
            profile_name="overlap_test",
            chunking=ChunkingConfig(
                chunk_size_ms=1000,
                chunk_jitter_ms=0,
                overlap_ms=200,  # 200ms overlap
            ),
            network=NetworkConfig(delay_ms=0, jitter_ms=0),
        )

    @pytest.fixture
    def audio_data(self) -> np.ndarray:
        """Create test audio data (3 seconds at 16kHz)."""
        duration = 3.0
        sample_rate = 16000
        samples = int(duration * sample_rate)
        return np.sin(
            2 * np.pi * 440 * np.linspace(0, duration, samples)
        ).astype(np.float32)

    def test_initialization(self, profile: StreamingProfile) -> None:
        """StreamingChunker initializes correctly."""
        chunker = StreamingChunker(profile, seed=42)
        assert chunker.profile == profile
        assert chunker.seed == 42

    def test_reset(self, profile: StreamingProfile) -> None:
        """Reset reinitializes RNG state."""
        chunker = StreamingChunker(profile, seed=42)

        # Generate some random numbers
        val1 = chunker._rng.random()

        # Reset with same seed
        chunker.reset(seed=42)
        val2 = chunker._rng.random()

        # Should get same value
        assert val1 == val2

    def test_reset_with_new_seed(self, profile: StreamingProfile) -> None:
        """Reset with new seed changes RNG."""
        chunker = StreamingChunker(profile, seed=42)

        val1 = chunker._rng.random()

        chunker.reset(seed=123)
        val2 = chunker._rng.random()

        # Should get different value (with high probability)
        assert val1 != val2

    @pytest.mark.asyncio
    async def test_stream_audio_basic(
        self, profile: StreamingProfile, audio_data: np.ndarray
    ) -> None:
        """Stream audio produces correct chunks."""
        chunker = StreamingChunker(profile, seed=42)
        chunks = await collect_chunks(chunker, audio_data, 16000)

        # 3 seconds of audio with 1 second chunks = 3 chunks
        assert len(chunks) == 3

        # Check sequence IDs
        assert [c.sequence_id for c in chunks] == [0, 1, 2]

        # Check timestamps
        assert chunks[0].timestamp_ms == 0
        assert chunks[1].timestamp_ms == 1000
        assert chunks[2].timestamp_ms == 2000

        # Check durations
        for chunk in chunks:
            assert chunk.duration_ms == 1000

        # Check final flag
        assert chunks[0].is_final is False
        assert chunks[1].is_final is False
        assert chunks[2].is_final is True

    @pytest.mark.asyncio
    async def test_stream_audio_with_overlap(
        self, profile_with_overlap: StreamingProfile, audio_data: np.ndarray
    ) -> None:
        """Stream audio with overlap produces correct chunks."""
        chunker = StreamingChunker(profile_with_overlap, seed=42)
        chunks = await collect_chunks(chunker, audio_data, 16000)

        # With 1000ms chunks and 200ms overlap, step is 800ms
        # 3000ms audio / 800ms step = 4 chunks (ceiling)
        assert len(chunks) >= 4

        # Verify overlap in timestamps
        # Each chunk starts 800ms after the previous
        for i in range(1, len(chunks)):
            expected_start = i * 800
            assert chunks[i].timestamp_ms == expected_start

    @pytest.mark.asyncio
    async def test_stream_audio_empty_raises(
        self, profile: StreamingProfile
    ) -> None:
        """Empty audio raises ValueError."""
        chunker = StreamingChunker(profile)
        empty_audio = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="empty"):
            async for _ in chunker.stream_audio(empty_audio, 16000):
                pass

    @pytest.mark.asyncio
    async def test_stream_audio_invalid_sample_rate(
        self, profile: StreamingProfile, audio_data: np.ndarray
    ) -> None:
        """Invalid sample rate raises ValueError."""
        chunker = StreamingChunker(profile)

        with pytest.raises(ValueError, match="positive"):
            async for _ in chunker.stream_audio(audio_data, 0):
                pass

        with pytest.raises(ValueError, match="positive"):
            async for _ in chunker.stream_audio(audio_data, -16000):
                pass

    @pytest.mark.asyncio
    async def test_stream_audio_deterministic(
        self, profile: StreamingProfile, audio_data: np.ndarray
    ) -> None:
        """Same seed produces identical results."""
        chunker1 = StreamingChunker(profile, seed=42)
        chunker2 = StreamingChunker(profile, seed=42)

        chunks1 = await collect_chunks(chunker1, audio_data, 16000)
        chunks2 = await collect_chunks(chunker2, audio_data, 16000)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.sequence_id == c2.sequence_id
            assert c1.timestamp_ms == c2.timestamp_ms
            assert c1.duration_ms == c2.duration_ms
            assert c1.is_final == c2.is_final
            assert c1.data == c2.data

    def test_get_expected_chunks(self, profile: StreamingProfile) -> None:
        """Calculate expected chunk count correctly."""
        chunker = StreamingChunker(profile)

        # 3000ms audio with 1000ms chunks, no overlap = 3 chunks
        assert chunker.get_expected_chunks(3000) == 3

        # 3500ms audio = 4 chunks (ceiling)
        assert chunker.get_expected_chunks(3500) == 4

        # 1000ms = 1 chunk
        assert chunker.get_expected_chunks(1000) == 1

    def test_get_expected_chunks_with_overlap(
        self, profile_with_overlap: StreamingProfile
    ) -> None:
        """Calculate expected chunks with overlap."""
        chunker = StreamingChunker(profile_with_overlap)

        # 3000ms with 1000ms chunk, 200ms overlap (800ms step)
        # 3000 / 800 = 3.75 -> 4 chunks
        assert chunker.get_expected_chunks(3000) == 4

    def test_get_expected_chunks_invalid_config(self) -> None:
        """Invalid config raises error at profile creation."""
        # The StreamingProfile validation catches overlap > chunk_size
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="overlap_ms"):
            StreamingProfile(
                profile_name="bad",
                chunking=ChunkingConfig(
                    chunk_size_ms=100,
                    overlap_ms=200,  # Overlap > chunk size
                ),
            )

    def test_get_chunk_boundaries(self, profile: StreamingProfile) -> None:
        """Get chunk boundaries returns correct ranges."""
        chunker = StreamingChunker(profile)
        boundaries = chunker.get_chunk_boundaries(3000)

        assert len(boundaries) == 3
        assert boundaries[0] == (0, 1000)
        assert boundaries[1] == (1000, 2000)
        assert boundaries[2] == (2000, 3000)

    def test_get_chunk_boundaries_with_overlap(
        self, profile_with_overlap: StreamingProfile
    ) -> None:
        """Get chunk boundaries with overlap shows overlapping ranges."""
        chunker = StreamingChunker(profile_with_overlap)
        boundaries = chunker.get_chunk_boundaries(3000)

        # Chunks overlap, so ranges will overlap
        # First chunk: 0-1000
        # Second chunk: 800-1800
        # Third chunk: 1600-2600
        # Fourth chunk: 2400-3000
        assert len(boundaries) >= 4
        assert boundaries[0] == (0, 1000)
        assert boundaries[1] == (800, 1800)
        assert boundaries[2] == (1600, 2600)


class TestChunkerUtilities:
    """Tests for chunker utility functions."""

    def test_create_test_chunks(self) -> None:
        """create_test_chunks creates correct number of chunks."""
        chunks = create_test_chunks(num_chunks=5, chunk_duration_ms=500)

        assert len(chunks) == 5

        for i, chunk in enumerate(chunks):
            assert chunk.sequence_id == i
            assert chunk.timestamp_ms == i * 500
            assert chunk.duration_ms == 500
            assert chunk.is_final == (i == 4)
            assert len(chunk.data) > 0

    def test_create_test_chunks_deterministic(self) -> None:
        """create_test_chunks is deterministic."""
        chunks1 = create_test_chunks(num_chunks=3)
        chunks2 = create_test_chunks(num_chunks=3)

        for c1, c2 in zip(chunks1, chunks2):
            assert c1.data == c2.data

    @pytest.mark.asyncio
    async def test_collect_chunks(self) -> None:
        """collect_chunks collects all chunks."""
        profile = StreamingProfile(
            profile_name="test",
            chunking=ChunkingConfig(chunk_size_ms=500, chunk_jitter_ms=0),
            network=NetworkConfig(delay_ms=0),
        )
        chunker = StreamingChunker(profile)
        audio = np.zeros(16000, dtype=np.float32)  # 1 second

        chunks = await collect_chunks(chunker, audio, 16000)

        assert len(chunks) == 2  # 1000ms / 500ms = 2 chunks


class TestChunkerTiming:
    """Tests for chunker timing behavior."""

    @pytest.fixture
    def timed_profile(self) -> StreamingProfile:
        """Create a profile with small delays for testing."""
        return StreamingProfile(
            profile_name="timed",
            chunking=ChunkingConfig(
                chunk_size_ms=100,
                chunk_jitter_ms=5,  # Small jitter
            ),
            network=NetworkConfig(
                delay_ms=10,
                jitter_ms=2,
            ),
        )

    @pytest.mark.asyncio
    async def test_timing_with_delays(self, timed_profile: StreamingProfile) -> None:
        """Chunker applies timing delays."""
        chunker = StreamingChunker(timed_profile, seed=42)
        audio = np.zeros(1600, dtype=np.float32)  # 100ms at 16kHz

        import time

        start = time.time()
        chunks = await collect_chunks(chunker, audio, 16000)
        elapsed = time.time() - start

        # Should have some delay applied (but not too much)
        # At least 1 chunk worth of delay
        assert elapsed > 0.005  # At least 5ms
        assert len(chunks) >= 1
