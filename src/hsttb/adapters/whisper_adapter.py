"""
Whisper STT adapter for local transcription.

Uses OpenAI's Whisper model for offline speech-to-text transcription.
Supports multiple model sizes and both file and streaming transcription.

Example:
    >>> adapter = WhisperAdapter(model_size="base")
    >>> await adapter.initialize()
    >>> text = await adapter.transcribe_file("audio.wav")
    >>> print(text)
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hsttb.adapters.base import STTAdapter
from hsttb.adapters.registry import register_adapter
from hsttb.core.exceptions import STTAdapterError, STTTranscriptionError

if TYPE_CHECKING:
    from hsttb.core.types import AudioChunk, TranscriptSegment

logger = logging.getLogger(__name__)

# Valid Whisper model sizes
WHISPER_MODEL_SIZES = ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3")


@register_adapter("whisper")
class WhisperAdapter(STTAdapter):
    """
    STT adapter using OpenAI's Whisper model.

    Whisper provides high-quality offline transcription with support
    for multiple languages and model sizes. Larger models are more
    accurate but require more compute resources.

    Model sizes and approximate requirements:
        - tiny: ~1GB RAM, fastest
        - base: ~1GB RAM
        - small: ~2GB RAM
        - medium: ~5GB RAM
        - large: ~10GB RAM, most accurate
        - large-v2: ~10GB RAM, improved accuracy
        - large-v3: ~10GB RAM, latest version

    Attributes:
        model_size: The Whisper model size to use.
        language: Optional language code (e.g., "en" for English).
        device: Compute device ("cuda", "cpu", or None for auto).
        compute_type: Compute precision ("float16", "float32", "int8").

    Example:
        >>> async with WhisperAdapter(model_size="base") as adapter:
        ...     text = await adapter.transcribe_file("recording.wav")
        ...     print(text)

        >>> # Streaming transcription
        >>> async for segment in adapter.transcribe_stream(audio_chunks):
        ...     if segment.is_final:
        ...         print(segment.text)
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        fp16: bool = True,
    ) -> None:
        """
        Initialize the Whisper adapter.

        Args:
            model_size: Whisper model size (tiny/base/small/medium/large).
            language: Optional language code for transcription.
            device: Compute device ("cuda", "cpu", or None for auto).
            compute_type: Compute precision type.
            fp16: Use half-precision (FP16) for GPU inference.

        Raises:
            ValueError: If model_size is invalid.
        """
        if model_size not in WHISPER_MODEL_SIZES:
            raise ValueError(
                f"Invalid model_size: {model_size!r}. "
                f"Must be one of: {WHISPER_MODEL_SIZES}"
            )

        self._model_size = model_size
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._fp16 = fp16
        self._model: Any = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Return unique adapter identifier."""
        return f"whisper-{self._model_size}"

    async def initialize(self) -> None:
        """
        Load the Whisper model.

        Raises:
            STTAdapterError: If whisper is not installed or model fails to load.
        """
        if self._initialized:
            return

        try:
            import whisper
        except ImportError as e:
            raise STTAdapterError(
                "Whisper is not installed. Install with: pip install openai-whisper"
            ) from e

        try:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: whisper.load_model(
                    self._model_size,
                    device=self._device,
                ),
            )
            self._initialized = True
            logger.info(f"Loaded Whisper model: {self._model_size}")
        except Exception as e:
            raise STTAdapterError(
                f"Failed to load Whisper model {self._model_size!r}: {e}"
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
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    str(file_path),
                    language=self._language,
                    fp16=self._fp16 and self._device != "cpu",
                ),
            )
            return str(result.get("text", "")).strip()
        except Exception as e:
            raise STTTranscriptionError(
                f"Whisper transcription failed for {file_path}: {e}"
            ) from e

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[TranscriptSegment]:
        """
        Transcribe streaming audio.

        Whisper doesn't natively support streaming, so we buffer chunks
        and perform batch transcription. Returns intermediate results
        at configurable intervals.

        Args:
            audio_stream: Async iterator of audio chunks.

        Yields:
            TranscriptSegment for each processed chunk.
        """
        from hsttb.core.types import TranscriptSegment

        if not self._initialized:
            await self.initialize()

        # Buffer to accumulate audio chunks
        audio_buffer = bytearray()
        chunk_duration_ms = 0
        sequence_id = 0
        last_transcript = ""

        # Configuration for streaming simulation
        buffer_threshold_ms = 3000  # Process every 3 seconds
        min_buffer_ms = 500  # Minimum buffer before processing

        async for chunk in audio_stream:
            audio_buffer.extend(chunk.data)
            chunk_duration_ms += chunk.duration_ms

            # Process when buffer threshold is reached or on final chunk
            should_process = (
                chunk_duration_ms >= buffer_threshold_ms
                or chunk.is_final
            )

            if should_process and chunk_duration_ms >= min_buffer_ms:
                # Write buffer to temp file for Whisper
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    self._write_wav(tmp_path, bytes(audio_buffer))

                try:
                    # Transcribe the buffer
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: self._model.transcribe(
                            str(tmp_path),
                            language=self._language,
                            fp16=self._fp16 and self._device != "cpu",
                        ),
                    )
                    transcript = str(result.get("text", "")).strip()

                    # Yield segment
                    yield TranscriptSegment(
                        text=transcript,
                        is_partial=not chunk.is_final,
                        is_final=chunk.is_final,
                        confidence=0.9,  # Whisper doesn't provide per-segment confidence
                        start_time_ms=chunk.timestamp_ms - chunk_duration_ms,
                        end_time_ms=chunk.timestamp_ms,
                    )

                    last_transcript = transcript
                    sequence_id += 1

                finally:
                    # Clean up temp file
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

                # Reset buffer if not final (keep accumulating for final)
                if not chunk.is_final:
                    # Keep some overlap for context
                    audio_buffer = bytearray()
                    chunk_duration_ms = 0

    def _write_wav(self, path: Path, audio_data: bytes) -> None:
        """
        Write raw audio data to a WAV file.

        Args:
            path: Output file path.
            audio_data: Raw PCM audio bytes.
        """
        import wave

        # Assume 16kHz, 16-bit mono audio (Whisper's expected format)
        sample_rate = 16000
        sample_width = 2  # 16-bit
        channels = 1

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

    async def cleanup(self) -> None:
        """Release model resources."""
        if self._model is not None:
            # Help garbage collection
            self._model = None
            self._initialized = False
            logger.info(f"Cleaned up Whisper model: {self._model_size}")

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata.
        """
        return {
            "model_size": self._model_size,
            "language": self._language,
            "device": self._device,
            "initialized": self._initialized,
            "fp16": self._fp16,
        }


# Convenience function for quick transcription
async def transcribe_with_whisper(
    file_path: Path | str,
    model_size: str = "base",
    language: str | None = None,
) -> str:
    """
    Transcribe an audio file with Whisper.

    Convenience function that handles adapter lifecycle.

    Args:
        file_path: Path to the audio file.
        model_size: Whisper model size.
        language: Optional language code.

    Returns:
        Transcribed text.

    Example:
        >>> text = await transcribe_with_whisper("recording.wav")
        >>> print(text)
    """
    async with WhisperAdapter(
        model_size=model_size,
        language=language,
    ) as adapter:
        return await adapter.transcribe_file(file_path)
