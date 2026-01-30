"""
Mock STT adapter for testing.

This module provides a mock adapter that returns predefined responses,
useful for testing the benchmark framework without actual STT services.

Example:
    >>> adapter = MockSTTAdapter(responses=["Hello world", "Test response"])
    >>> await adapter.initialize()
    >>> result = await adapter.transcribe_file("test.wav")
    >>> print(result)  # "Hello world"
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from hsttb.adapters.base import STTAdapter
from hsttb.adapters.registry import register_adapter
from hsttb.core.types import AudioChunk, TranscriptSegment


@register_adapter("mock")
class MockSTTAdapter(STTAdapter):
    """
    Mock STT adapter for testing purposes.

    Returns predefined responses in sequence. Useful for testing
    the benchmark framework without requiring actual STT services.

    Attributes:
        responses: List of transcript responses to return.
        delay_ms: Artificial delay per transcription (milliseconds).
        fail_rate: Probability of simulated failure (0.0 to 1.0).

    Example:
        >>> adapter = MockSTTAdapter(
        ...     responses=["Patient has diabetes", "No chest pain"],
        ...     delay_ms=100
        ... )
        >>> async with adapter as a:
        ...     result = await a.transcribe_file("audio.wav")
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        delay_ms: int = 0,
        fail_rate: float = 0.0,
        confidence: float = 0.95,
    ) -> None:
        """
        Initialize the mock adapter.

        Args:
            responses: List of responses to return (cycles through).
            delay_ms: Artificial delay per transcription.
            fail_rate: Probability of failure (0.0 to 1.0).
            confidence: Confidence score for transcripts.
        """
        self.responses = responses or ["mock transcript"]
        self.delay_ms = delay_ms
        self.fail_rate = fail_rate
        self.confidence = confidence
        self._call_count = 0
        self._initialized = False

    @property
    def name(self) -> str:
        """Return adapter name."""
        return "mock"

    async def initialize(self) -> None:
        """Initialize the mock adapter."""
        self._initialized = True
        self._call_count = 0

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[TranscriptSegment]:
        """
        Transcribe streaming audio with mock responses.

        Yields one TranscriptSegment per final chunk received.

        Args:
            audio_stream: Async iterator of audio chunks.

        Yields:
            TranscriptSegment with mock transcription.
        """
        if not self._initialized:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

        start_time = 0
        async for chunk in audio_stream:
            # Simulate processing delay
            if self.delay_ms > 0:
                await asyncio.sleep(self.delay_ms / 1000)

            # Only yield on final chunks
            if chunk.is_final:
                response = self._get_next_response()
                end_time = chunk.timestamp_ms + chunk.duration_ms
                yield TranscriptSegment(
                    text=response,
                    is_partial=False,
                    is_final=True,
                    confidence=self.confidence,
                    start_time_ms=start_time,
                    end_time_ms=end_time,
                )
                start_time = end_time

    async def transcribe_file(
        self,
        file_path: Path | str,
    ) -> str:
        """
        Transcribe a file with mock response.

        Args:
            file_path: Path to the audio file (not actually read).

        Returns:
            Next mock response in sequence.
        """
        if not self._initialized:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

        # Validate file exists (for realistic behavior)
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Simulate processing delay
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)

        return self._get_next_response()

    async def cleanup(self) -> None:
        """Clean up mock adapter."""
        self._initialized = False

    def _get_next_response(self) -> str:
        """Get the next response in the cycle."""
        response = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        return response

    @property
    def call_count(self) -> int:
        """Number of transcriptions performed."""
        return self._call_count

    def reset(self) -> None:
        """Reset the call counter."""
        self._call_count = 0


@register_adapter("failing_mock")
class FailingMockAdapter(STTAdapter):
    """
    Mock adapter that fails on specific calls.

    Useful for testing error handling in the benchmark framework.

    Attributes:
        fail_on_calls: Set of call numbers that should fail (0-indexed).
        error_message: Message for the simulated error.
    """

    def __init__(
        self,
        fail_on_calls: set[int] | None = None,
        error_message: str = "Simulated transcription failure",
    ) -> None:
        """
        Initialize the failing mock adapter.

        Args:
            fail_on_calls: Set of call indices that should fail.
            error_message: Error message for failures.
        """
        self.fail_on_calls = fail_on_calls or {1}  # Fail on second call by default
        self.error_message = error_message
        self._call_count = 0
        self._initialized = False

    @property
    def name(self) -> str:
        """Return adapter name."""
        return "failing_mock"

    async def initialize(self) -> None:
        """Initialize the adapter."""
        self._initialized = True
        self._call_count = 0

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[TranscriptSegment]:
        """Transcribe with potential failures."""
        from hsttb.core.exceptions import STTTranscriptionError

        if not self._initialized:
            raise RuntimeError("Adapter not initialized")

        async for chunk in audio_stream:
            if chunk.is_final:
                if self._call_count in self.fail_on_calls:
                    self._call_count += 1
                    raise STTTranscriptionError(self.error_message, adapter_name=self.name)

                self._call_count += 1
                yield TranscriptSegment(
                    text=f"transcript_{self._call_count}",
                    is_partial=False,
                    is_final=True,
                    confidence=0.9,
                    start_time_ms=0,
                    end_time_ms=chunk.timestamp_ms + chunk.duration_ms,
                )

    async def transcribe_file(self, file_path: Path | str) -> str:
        """Transcribe file with potential failure."""
        from hsttb.core.exceptions import STTTranscriptionError

        if not self._initialized:
            raise RuntimeError("Adapter not initialized")

        # Validate file exists (for realistic behavior)
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if self._call_count in self.fail_on_calls:
            self._call_count += 1
            raise STTTranscriptionError(self.error_message, adapter_name=self.name)

        self._call_count += 1
        return f"transcript_{self._call_count}"

    async def cleanup(self) -> None:
        """Clean up adapter."""
        self._initialized = False
