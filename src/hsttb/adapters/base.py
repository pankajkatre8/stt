"""
Abstract base class for STT adapters.

This module defines the interface that all STT adapters must implement.
The interface is designed to be minimal, async-first, and streaming-aware.

Example:
    >>> class MyAdapter(STTAdapter):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my-adapter"
    ...
    ...     async def initialize(self) -> None:
    ...         pass
    ...
    ...     async def transcribe_stream(self, audio_stream):
    ...         async for chunk in audio_stream:
    ...             yield TranscriptSegment(...)
    ...
    ...     async def transcribe_file(self, file_path):
    ...         return "transcribed text"
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hsttb.core.types import AudioChunk, TranscriptSegment


class STTAdapter(ABC):
    """
    Abstract base class for Speech-to-Text adapters.

    All STT providers must implement this interface to be used
    with the HSTTB benchmark framework. The interface supports
    both streaming and file-based transcription.

    Example:
        >>> adapter = WhisperAdapter(model="base")
        >>> await adapter.initialize()
        >>> async for segment in adapter.transcribe_stream(audio_stream):
        ...     print(segment.text)
        >>> await adapter.cleanup()

    As context manager:
        >>> async with WhisperAdapter() as adapter:
        ...     result = await adapter.transcribe_file("audio.wav")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this adapter.

        Returns:
            String identifier (e.g., "whisper-base", "deepgram-nova").
        """
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the adapter.

        This method should load models, establish connections,
        and prepare the adapter for transcription. It is called
        once before any transcription operations.

        Raises:
            STTAdapterError: If initialization fails.
        """
        ...

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[TranscriptSegment]:
        """
        Transcribe streaming audio.

        Process audio chunks as they arrive and yield transcript
        segments. May yield partial (in-progress) and final results.

        Args:
            audio_stream: Async iterator of audio chunks.

        Yields:
            TranscriptSegment for each partial or final result.

        Raises:
            STTAdapterError: If transcription fails.
        """
        # This yield is needed to make it an async generator
        # Concrete implementations will override this completely
        if False:  # pragma: no cover
            yield

    @abstractmethod
    async def transcribe_file(
        self,
        file_path: Path | str,
    ) -> str:
        """
        Transcribe a complete audio file.

        This method is for non-streaming transcription, typically
        used for comparison or when streaming is not needed.

        Args:
            file_path: Path to the audio file.

        Returns:
            Complete transcript text.

        Raises:
            FileNotFoundError: If file doesn't exist.
            STTAdapterError: If transcription fails.
        """
        ...

    async def cleanup(self) -> None:  # noqa: B027
        """
        Clean up resources.

        Override this method to release resources, close connections,
        or unload models. Default implementation does nothing.
        """

    async def __aenter__(self) -> STTAdapter:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.cleanup()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name!r})>"
