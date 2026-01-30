"""
Tests for STT adapter module.

Tests the adapter interface, registry, and mock implementations.
"""
from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from hsttb.adapters import (
    FailingMockAdapter,
    MockSTTAdapter,
    STTAdapter,
    get_adapter,
    is_adapter_registered,
    list_adapters,
    register_adapter,
    unregister_adapter,
)
from hsttb.core.exceptions import STTTranscriptionError
from hsttb.core.types import AudioChunk, TranscriptSegment


class TestSTTAdapterInterface:
    """Tests for STTAdapter abstract interface."""

    def test_cannot_instantiate_abstract(self) -> None:
        """Cannot instantiate abstract STTAdapter directly."""
        with pytest.raises(TypeError):
            STTAdapter()  # type: ignore[abstract]

    def test_concrete_implementation_required(self) -> None:
        """Must implement all abstract methods."""

        class IncompleteAdapter(STTAdapter):
            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore[abstract]


class TestAdapterRegistry:
    """Tests for adapter registry functions."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        # Save existing adapters
        self._original_adapters = list_adapters()

    def teardown_method(self) -> None:
        """Restore registry after each test."""
        # Registry persists due to module-level state, so we need
        # to ensure test adapters are removed
        for name in list_adapters():
            if name not in self._original_adapters:
                unregister_adapter(name)

    def test_list_adapters(self) -> None:
        """list_adapters returns registered adapters."""
        adapters = list_adapters()
        assert isinstance(adapters, list)
        # Mock adapters should be registered
        assert "mock" in adapters
        assert "failing_mock" in adapters

    def test_is_adapter_registered(self) -> None:
        """is_adapter_registered checks registration status."""
        assert is_adapter_registered("mock") is True
        assert is_adapter_registered("nonexistent") is False

    def test_get_adapter_valid(self) -> None:
        """get_adapter returns instance for registered adapter."""
        adapter = get_adapter("mock")
        assert isinstance(adapter, MockSTTAdapter)
        assert adapter.name == "mock"

    def test_get_adapter_with_kwargs(self) -> None:
        """get_adapter passes kwargs to constructor."""
        adapter = get_adapter("mock", responses=["custom response"], delay_ms=100)
        assert isinstance(adapter, MockSTTAdapter)
        assert adapter.responses == ["custom response"]
        assert adapter.delay_ms == 100

    def test_get_adapter_invalid(self) -> None:
        """get_adapter raises ValueError for unknown adapter."""
        with pytest.raises(ValueError, match="Unknown adapter: 'nonexistent'"):
            get_adapter("nonexistent")

    def test_register_adapter_decorator(self) -> None:
        """register_adapter decorator registers a class."""

        @register_adapter("test_adapter")
        class TestAdapter(STTAdapter):
            @property
            def name(self) -> str:
                return "test_adapter"

            async def initialize(self) -> None:
                pass

            async def transcribe_stream(
                self,
                audio_stream: AsyncIterator[AudioChunk],
            ) -> AsyncIterator[TranscriptSegment]:
                async for chunk in audio_stream:
                    if chunk.is_final:
                        yield TranscriptSegment(
                            text="test",
                            is_partial=False,
                            is_final=True,
                            confidence=1.0,
                            start_time_ms=0,
                            end_time_ms=100,
                        )

            async def transcribe_file(
                self, file_path: Path | str  # noqa: ARG002
            ) -> str:
                return "test"

        assert is_adapter_registered("test_adapter") is True
        adapter = get_adapter("test_adapter")
        assert adapter.name == "test_adapter"

        # Cleanup
        unregister_adapter("test_adapter")

    def test_unregister_adapter_existing(self) -> None:
        """unregister_adapter removes registered adapter."""

        @register_adapter("temp_adapter")
        class TempAdapter(STTAdapter):
            @property
            def name(self) -> str:
                return "temp"

            async def initialize(self) -> None:
                pass

            async def transcribe_stream(
                self,
                audio_stream: AsyncIterator[AudioChunk],  # noqa: ARG002
            ) -> AsyncIterator[TranscriptSegment]:
                if False:  # pragma: no cover
                    yield  # type: ignore[misc]

            async def transcribe_file(
                self, file_path: Path | str  # noqa: ARG002
            ) -> str:
                return ""

        assert is_adapter_registered("temp_adapter") is True
        result = unregister_adapter("temp_adapter")
        assert result is True
        assert is_adapter_registered("temp_adapter") is False

    def test_unregister_adapter_nonexistent(self) -> None:
        """unregister_adapter returns False for unknown adapter."""
        result = unregister_adapter("definitely_not_registered")
        assert result is False


class TestMockSTTAdapter:
    """Tests for MockSTTAdapter implementation."""

    @pytest.fixture
    def adapter(self) -> MockSTTAdapter:
        """Create a mock adapter for testing."""
        return MockSTTAdapter(
            responses=["Hello world", "Test response", "Third response"],
            delay_ms=0,
            confidence=0.95,
        )

    @pytest.fixture
    def temp_audio_file(self) -> Path:
        """Create a temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            return Path(f.name)

    def test_initialization(self, adapter: MockSTTAdapter) -> None:
        """MockSTTAdapter initializes with correct defaults."""
        assert adapter.name == "mock"
        assert adapter.responses == ["Hello world", "Test response", "Third response"]
        assert adapter.delay_ms == 0
        assert adapter.confidence == 0.95
        assert adapter.call_count == 0

    def test_default_responses(self) -> None:
        """MockSTTAdapter has default response when none provided."""
        adapter = MockSTTAdapter()
        assert adapter.responses == ["mock transcript"]

    @pytest.mark.asyncio
    async def test_initialize(self, adapter: MockSTTAdapter) -> None:
        """initialize() prepares adapter for use."""
        await adapter.initialize()
        # Can verify internal state
        assert adapter._initialized is True

    @pytest.mark.asyncio
    async def test_context_manager(self, adapter: MockSTTAdapter) -> None:
        """Adapter works as async context manager."""
        async with adapter as a:
            assert a._initialized is True
        assert a._initialized is False

    @pytest.mark.asyncio
    async def test_transcribe_file(
        self, adapter: MockSTTAdapter, temp_audio_file: Path
    ) -> None:
        """transcribe_file returns responses in sequence."""
        await adapter.initialize()

        result1 = await adapter.transcribe_file(temp_audio_file)
        assert result1 == "Hello world"
        assert adapter.call_count == 1

        result2 = await adapter.transcribe_file(temp_audio_file)
        assert result2 == "Test response"
        assert adapter.call_count == 2

        result3 = await adapter.transcribe_file(temp_audio_file)
        assert result3 == "Third response"
        assert adapter.call_count == 3

        # Cycles back
        result4 = await adapter.transcribe_file(temp_audio_file)
        assert result4 == "Hello world"
        assert adapter.call_count == 4

        # Cleanup
        temp_audio_file.unlink()

    @pytest.mark.asyncio
    async def test_transcribe_file_not_initialized(
        self, adapter: MockSTTAdapter, temp_audio_file: Path
    ) -> None:
        """transcribe_file raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.transcribe_file(temp_audio_file)
        temp_audio_file.unlink()

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, adapter: MockSTTAdapter) -> None:
        """transcribe_file raises FileNotFoundError for missing file."""
        await adapter.initialize()
        with pytest.raises(FileNotFoundError):
            await adapter.transcribe_file("/nonexistent/file.wav")

    @pytest.mark.asyncio
    async def test_transcribe_stream(self, adapter: MockSTTAdapter) -> None:
        """transcribe_stream yields segments for final chunks."""
        await adapter.initialize()

        async def audio_generator() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(
                data=b"chunk1",
                sequence_id=0,
                timestamp_ms=0,
                duration_ms=100,
                is_final=False,
            )
            yield AudioChunk(
                data=b"chunk2",
                sequence_id=1,
                timestamp_ms=100,
                duration_ms=100,
                is_final=True,
            )
            yield AudioChunk(
                data=b"chunk3",
                sequence_id=2,
                timestamp_ms=200,
                duration_ms=100,
                is_final=True,
            )

        segments = []
        async for segment in adapter.transcribe_stream(audio_generator()):
            segments.append(segment)

        assert len(segments) == 2
        assert segments[0].text == "Hello world"
        assert segments[0].is_final is True
        assert segments[0].start_time_ms == 0
        assert segments[0].end_time_ms == 200

        assert segments[1].text == "Test response"
        assert segments[1].start_time_ms == 200
        assert segments[1].end_time_ms == 300

    @pytest.mark.asyncio
    async def test_transcribe_stream_not_initialized(
        self, adapter: MockSTTAdapter
    ) -> None:
        """transcribe_stream raises if not initialized."""

        async def audio_generator() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(
                data=b"chunk",
                sequence_id=0,
                timestamp_ms=0,
                duration_ms=100,
                is_final=True,
            )

        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in adapter.transcribe_stream(audio_generator()):
                pass

    def test_reset(self, adapter: MockSTTAdapter) -> None:
        """reset() clears call counter."""
        adapter._call_count = 5
        adapter.reset()
        assert adapter.call_count == 0

    def test_repr(self, adapter: MockSTTAdapter) -> None:
        """__repr__ shows adapter info."""
        repr_str = repr(adapter)
        assert "MockSTTAdapter" in repr_str
        assert "mock" in repr_str


class TestFailingMockAdapter:
    """Tests for FailingMockAdapter implementation."""

    @pytest.fixture
    def adapter(self) -> FailingMockAdapter:
        """Create a failing adapter for testing."""
        return FailingMockAdapter(
            fail_on_calls={1, 3},
            error_message="Test failure",
        )

    @pytest.fixture
    def temp_audio_file(self) -> Path:
        """Create a temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            return Path(f.name)

    def test_initialization(self, adapter: FailingMockAdapter) -> None:
        """FailingMockAdapter initializes with correct values."""
        assert adapter.name == "failing_mock"
        assert adapter.fail_on_calls == {1, 3}
        assert adapter.error_message == "Test failure"

    def test_default_fail_on_calls(self) -> None:
        """Default fails on second call (index 1)."""
        adapter = FailingMockAdapter()
        assert adapter.fail_on_calls == {1}

    @pytest.mark.asyncio
    async def test_transcribe_file_succeeds_then_fails(
        self, adapter: FailingMockAdapter, temp_audio_file: Path
    ) -> None:
        """Fails on specified call numbers."""
        await adapter.initialize()

        # Call 0 succeeds
        result = await adapter.transcribe_file(temp_audio_file)
        assert result == "transcript_1"

        # Call 1 fails
        with pytest.raises(STTTranscriptionError, match="Test failure"):
            await adapter.transcribe_file(temp_audio_file)

        # Call 2 succeeds
        result = await adapter.transcribe_file(temp_audio_file)
        assert result == "transcript_3"

        # Call 3 fails
        with pytest.raises(STTTranscriptionError, match="Test failure"):
            await adapter.transcribe_file(temp_audio_file)

        # Call 4 succeeds
        result = await adapter.transcribe_file(temp_audio_file)
        assert result == "transcript_5"

        temp_audio_file.unlink()

    @pytest.mark.asyncio
    async def test_transcribe_file_not_initialized(
        self, adapter: FailingMockAdapter, temp_audio_file: Path
    ) -> None:
        """Raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.transcribe_file(temp_audio_file)
        temp_audio_file.unlink()

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(
        self, adapter: FailingMockAdapter
    ) -> None:
        """Raises FileNotFoundError for missing file."""
        await adapter.initialize()
        with pytest.raises(FileNotFoundError):
            await adapter.transcribe_file("/nonexistent/file.wav")

    @pytest.mark.asyncio
    async def test_transcribe_stream_fails(
        self, adapter: FailingMockAdapter
    ) -> None:
        """Stream transcription fails on specified calls."""
        adapter = FailingMockAdapter(fail_on_calls={0})
        await adapter.initialize()

        async def audio_generator() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(
                data=b"chunk",
                sequence_id=0,
                timestamp_ms=0,
                duration_ms=100,
                is_final=True,
            )

        with pytest.raises(STTTranscriptionError):
            async for _ in adapter.transcribe_stream(audio_generator()):
                pass

    @pytest.mark.asyncio
    async def test_cleanup(self, adapter: FailingMockAdapter) -> None:
        """Cleanup resets initialization state."""
        await adapter.initialize()
        assert adapter._initialized is True
        await adapter.cleanup()
        assert adapter._initialized is False


class TestAdapterFactoryIntegration:
    """Integration tests for adapter factory."""

    @pytest.mark.asyncio
    async def test_get_and_use_mock_adapter(self) -> None:
        """Full workflow using factory to get mock adapter."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio")
            temp_path = Path(f.name)

        adapter = get_adapter(
            "mock",
            responses=["Hello from factory"],
            confidence=0.99,
        )

        async with adapter as a:
            result = await a.transcribe_file(temp_path)
            assert result == "Hello from factory"

        temp_path.unlink()

    @pytest.mark.asyncio
    async def test_get_and_use_failing_adapter(self) -> None:
        """Full workflow using factory to get failing adapter."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio")
            temp_path = Path(f.name)

        adapter = get_adapter(
            "failing_mock",
            fail_on_calls={0},
            error_message="Factory error",
        )

        async with adapter as a:
            with pytest.raises(STTTranscriptionError, match="Factory error"):
                await a.transcribe_file(temp_path)

        temp_path.unlink()
