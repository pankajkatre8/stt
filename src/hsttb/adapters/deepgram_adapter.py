"""
Deepgram STT adapter.

Provides integration with Deepgram's speech-to-text API with
support for medical vocabulary through the Nova-2 Medical model.

Example:
    >>> adapter = DeepgramAdapter(api_key="your-key")
    >>> await adapter.initialize()
    >>> text = await adapter.transcribe_file("audio.wav")
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hsttb.adapters.base import STTAdapter
from hsttb.adapters.registry import register_adapter
from hsttb.core.exceptions import STTAdapterError, STTConnectionError, STTTranscriptionError

if TYPE_CHECKING:
    from hsttb.core.types import AudioChunk, TranscriptSegment

logger = logging.getLogger(__name__)

# Deepgram models
DEEPGRAM_MODELS = {
    "nova-2": "nova-2",  # Latest general model
    "nova-2-medical": "nova-2-medical",  # Medical specialized
    "nova-2-phonecall": "nova-2-phonecall",  # Phone call optimized
    "nova-2-meeting": "nova-2-meeting",  # Meeting optimized
    "nova": "nova",  # Previous generation
    "enhanced": "enhanced",  # Enhanced accuracy
    "base": "base",  # Base model
}


@register_adapter("deepgram")
class DeepgramAdapter(STTAdapter):
    """
    STT adapter using Deepgram API.

    Deepgram provides fast, accurate speech recognition with
    specialized models for medical transcription.

    Attributes:
        api_key: Deepgram API key.
        model: Deepgram model to use.
        language: Language code.
        punctuate: Enable punctuation.
        diarize: Enable speaker diarization.
        smart_format: Enable smart formatting.

    Example:
        >>> async with DeepgramAdapter(model="nova-2-medical") as adapter:
        ...     text = await adapter.transcribe_file("audio.wav")
        ...     print(text)

        >>> # Streaming with medical vocabulary
        >>> async for segment in adapter.transcribe_stream(audio_chunks):
        ...     if segment.is_final:
        ...         print(segment.text)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "nova-2-medical",
        language: str = "en-US",
        punctuate: bool = True,
        diarize: bool = False,
        smart_format: bool = True,
        utterances: bool = False,
        keywords: list[str] | None = None,
        sample_rate: int = 16000,
    ) -> None:
        """
        Initialize the Deepgram adapter.

        Args:
            api_key: Deepgram API key (or DEEPGRAM_API_KEY env var).
            model: Deepgram model to use.
            language: Language code for transcription.
            punctuate: Enable automatic punctuation.
            diarize: Enable speaker diarization.
            smart_format: Enable smart formatting (dates, numbers, etc.).
            utterances: Enable utterance detection.
            keywords: Custom keywords to boost.
            sample_rate: Expected audio sample rate.
        """
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        self._model = model
        self._language = language
        self._punctuate = punctuate
        self._diarize = diarize
        self._smart_format = smart_format
        self._utterances = utterances
        self._keywords = keywords or []
        self._sample_rate = sample_rate

        self._client: Any = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Return unique adapter identifier."""
        return f"deepgram-{self._model}"

    async def initialize(self) -> None:
        """
        Initialize the Deepgram client.

        Raises:
            STTAdapterError: If deepgram-sdk is not installed.
            STTConnectionError: If authentication fails.
        """
        if self._initialized:
            return

        try:
            from deepgram import DeepgramClient
        except ImportError as e:
            raise STTAdapterError(
                "deepgram-sdk is not installed. Install with: "
                "pip install deepgram-sdk"
            ) from e

        if not self._api_key:
            raise STTConnectionError(
                "Deepgram API key not provided. Set DEEPGRAM_API_KEY "
                "environment variable or pass api_key parameter."
            )

        try:
            self._client = DeepgramClient(self._api_key)
            self._initialized = True
            logger.info(f"Initialized Deepgram client: {self._model}")
        except Exception as e:
            raise STTConnectionError(
                f"Failed to initialize Deepgram client: {e}"
            ) from e

    async def transcribe_file(self, file_path: Path | str) -> str:
        """
        Transcribe an audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.

        Raises:
            FileNotFoundError: If file doesn't exist.
            STTTranscriptionError: If transcription fails.
        """
        if not self._initialized:
            await self.initialize()

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            from deepgram import FileSource, PrerecordedOptions

            # Read audio file
            with open(file_path, "rb") as f:
                buffer_data = f.read()

            # Create options
            options = self._create_prerecorded_options()

            # Create file source
            payload: FileSource = {"buffer": buffer_data}

            # Transcribe
            response = await asyncio.to_thread(
                lambda: self._client.listen.prerecorded.v("1").transcribe_file(
                    payload, options
                )
            )

            # Extract transcript
            if response.results and response.results.channels:
                channel = response.results.channels[0]
                if channel.alternatives:
                    return channel.alternatives[0].transcript

            return ""

        except Exception as e:
            raise STTTranscriptionError(
                f"Deepgram transcription failed: {e}"
            ) from e

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[TranscriptSegment]:
        """
        Transcribe streaming audio using Deepgram's live API.

        Args:
            audio_stream: Async iterator of audio chunks.

        Yields:
            TranscriptSegment for each recognized result.
        """
        from hsttb.core.types import TranscriptSegment

        if not self._initialized:
            await self.initialize()

        try:
            from deepgram import LiveOptions, LiveTranscriptionEvents

            # Create live options
            options = self._create_live_options()

            # Results queue
            results_queue: asyncio.Queue[TranscriptSegment | None] = asyncio.Queue()

            # Create live connection
            connection = self._client.listen.live.v("1")

            # Event handlers
            def on_message(self: Any, result: Any, **kwargs: Any) -> None:
                if result.channel and result.channel.alternatives:
                    alt = result.channel.alternatives[0]
                    segment = TranscriptSegment(
                        text=alt.transcript,
                        is_partial=not result.is_final,
                        is_final=result.is_final,
                        confidence=alt.confidence if hasattr(alt, "confidence") else 0.9,
                        start_time_ms=int(result.start * 1000) if hasattr(result, "start") else 0,
                        end_time_ms=int((result.start + result.duration) * 1000)
                        if hasattr(result, "start") and hasattr(result, "duration")
                        else 0,
                    )
                    asyncio.get_event_loop().call_soon_threadsafe(
                        results_queue.put_nowait, segment
                    )

            def on_close(self: Any, close: Any, **kwargs: Any) -> None:
                asyncio.get_event_loop().call_soon_threadsafe(
                    results_queue.put_nowait, None
                )

            def on_error(self: Any, error: Any, **kwargs: Any) -> None:
                logger.error(f"Deepgram streaming error: {error}")

            # Register handlers
            connection.on(LiveTranscriptionEvents.Transcript, on_message)
            connection.on(LiveTranscriptionEvents.Close, on_close)
            connection.on(LiveTranscriptionEvents.Error, on_error)

            # Start connection
            await asyncio.to_thread(lambda: connection.start(options))

            # Send audio task
            async def send_audio() -> None:
                async for chunk in audio_stream:
                    await asyncio.to_thread(lambda c=chunk: connection.send(c.data))
                    if chunk.is_final:
                        break
                await asyncio.to_thread(connection.finish)

            # Start sending audio in background
            send_task = asyncio.create_task(send_audio())

            # Yield results as they come
            try:
                while True:
                    segment = await results_queue.get()
                    if segment is None:
                        break
                    yield segment
            finally:
                if not send_task.done():
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass

        except ImportError:
            # Fallback for older SDK versions
            logger.warning("Live transcription requires deepgram-sdk>=3.0")
            async for segment in self._transcribe_stream_fallback(audio_stream):
                yield segment

        except Exception as e:
            logger.error(f"Deepgram streaming error: {e}")
            raise STTTranscriptionError(
                f"Deepgram streaming failed: {e}"
            ) from e

    async def _transcribe_stream_fallback(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[TranscriptSegment]:
        """
        Fallback streaming using buffered file transcription.

        Used when live API is not available.
        """
        from hsttb.core.types import TranscriptSegment

        import tempfile

        # Buffer audio chunks
        audio_buffer = bytearray()
        last_timestamp = 0

        async for chunk in audio_stream:
            audio_buffer.extend(chunk.data)
            last_timestamp = chunk.timestamp_ms + chunk.duration_ms

            # Transcribe every 5 seconds of audio
            if len(audio_buffer) > self._sample_rate * 2 * 5:  # 5 seconds of 16-bit audio
                # Write to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    import wave

                    with wave.open(f.name, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(self._sample_rate)
                        wav.writeframes(bytes(audio_buffer))

                    # Transcribe
                    text = await self.transcribe_file(f.name)

                    if text:
                        yield TranscriptSegment(
                            text=text,
                            is_partial=not chunk.is_final,
                            is_final=chunk.is_final,
                            confidence=0.9,
                            start_time_ms=0,
                            end_time_ms=last_timestamp,
                        )

                    # Clear buffer
                    audio_buffer.clear()

                    # Clean up
                    Path(f.name).unlink()

    def _create_prerecorded_options(self) -> Any:
        """Create options for pre-recorded transcription."""
        from deepgram import PrerecordedOptions

        options = PrerecordedOptions(
            model=self._model,
            language=self._language,
            punctuate=self._punctuate,
            diarize=self._diarize,
            smart_format=self._smart_format,
            utterances=self._utterances,
        )

        # Add keywords if specified
        if self._keywords:
            options.keywords = self._keywords

        return options

    def _create_live_options(self) -> Any:
        """Create options for live transcription."""
        from deepgram import LiveOptions

        options = LiveOptions(
            model=self._model,
            language=self._language,
            punctuate=self._punctuate,
            smart_format=self._smart_format,
            encoding="linear16",
            sample_rate=self._sample_rate,
            channels=1,
            interim_results=True,
        )

        return options

    async def cleanup(self) -> None:
        """Release client resources."""
        if self._client is not None:
            self._client = None
            self._initialized = False
            logger.info("Cleaned up Deepgram client")

    def get_available_models(self) -> list[str]:
        """
        Get list of available Deepgram models.

        Returns:
            List of model names.
        """
        return list(DEEPGRAM_MODELS.keys())


# Convenience function
async def transcribe_with_deepgram(
    file_path: Path | str,
    model: str = "nova-2-medical",
    language: str = "en-US",
) -> str:
    """
    Transcribe an audio file with Deepgram.

    Args:
        file_path: Path to the audio file.
        model: Deepgram model to use.
        language: Language code.

    Returns:
        Transcribed text.
    """
    async with DeepgramAdapter(
        model=model,
        language=language,
    ) as adapter:
        return await adapter.transcribe_file(file_path)
