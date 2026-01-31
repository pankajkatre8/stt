"""
Google Cloud Speech-to-Text adapter.

Provides integration with Google Cloud Speech API for high-accuracy
transcription with support for medical vocabulary customization.

Example:
    >>> adapter = GeminiAdapter(api_key="your-key")
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

# Google Cloud Speech models
SPEECH_MODELS = {
    "latest_short": "latest_short",  # For audio < 1 minute
    "latest_long": "latest_long",  # For audio > 1 minute
    "telephony": "telephony",  # Optimized for phone audio
    "medical_dictation": "medical_dictation",  # Medical transcription
    "medical_conversation": "medical_conversation",  # Medical conversations
}


@register_adapter("gemini")
@register_adapter("google-cloud-speech")
class GeminiAdapter(STTAdapter):
    """
    STT adapter using Google Cloud Speech-to-Text API.

    Supports both synchronous and streaming transcription with
    options for medical vocabulary enhancement.

    Attributes:
        api_key: Google Cloud API key or path to credentials.
        model: Speech model to use.
        language_code: Language code (e.g., "en-US").
        enable_punctuation: Auto-punctuate transcription.
        enable_word_timestamps: Include word-level timestamps.

    Example:
        >>> async with GeminiAdapter() as adapter:
        ...     text = await adapter.transcribe_file("audio.wav")
        ...     print(text)

        >>> # Streaming transcription
        >>> async for segment in adapter.transcribe_stream(audio_chunks):
        ...     print(segment.text)
    """

    def __init__(
        self,
        api_key: str | None = None,
        credentials_path: str | None = None,
        model: str = "latest_long",
        language_code: str = "en-US",
        enable_punctuation: bool = True,
        enable_word_timestamps: bool = False,
        sample_rate: int = 16000,
        medical_mode: bool = False,
    ) -> None:
        """
        Initialize the Google Cloud Speech adapter.

        Args:
            api_key: API key for authentication.
            credentials_path: Path to service account credentials JSON.
            model: Speech recognition model to use.
            language_code: BCP-47 language code.
            enable_punctuation: Enable automatic punctuation.
            enable_word_timestamps: Include word-level timestamps.
            sample_rate: Expected audio sample rate.
            medical_mode: Enable medical transcription model.
        """
        self._api_key = api_key or os.environ.get("GOOGLE_CLOUD_API_KEY")
        self._credentials_path = credentials_path or os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

        # Use medical model if enabled
        if medical_mode and model not in ("medical_dictation", "medical_conversation"):
            model = "medical_dictation"

        self._model = model
        self._language_code = language_code
        self._enable_punctuation = enable_punctuation
        self._enable_word_timestamps = enable_word_timestamps
        self._sample_rate = sample_rate
        self._medical_mode = medical_mode

        self._client: Any = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Return unique adapter identifier."""
        return f"google-cloud-speech-{self._model}"

    async def initialize(self) -> None:
        """
        Initialize the Google Cloud Speech client.

        Raises:
            STTAdapterError: If google-cloud-speech is not installed.
            STTConnectionError: If authentication fails.
        """
        if self._initialized:
            return

        try:
            from google.cloud import speech
        except ImportError as e:
            raise STTAdapterError(
                "google-cloud-speech is not installed. Install with: "
                "pip install google-cloud-speech"
            ) from e

        try:
            # Set credentials if provided
            if self._credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self._credentials_path

            # Create client
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(
                None,
                speech.SpeechClient,
            )

            self._initialized = True
            logger.info(f"Initialized Google Cloud Speech client: {self._model}")

        except Exception as e:
            raise STTConnectionError(
                f"Failed to initialize Google Cloud Speech client: {e}"
            ) from e

    async def transcribe_file(self, file_path: Path | str) -> str:
        """
        Transcribe an audio file.

        Uses synchronous recognition for short audio (<1 min)
        or async recognition for longer audio.

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
            from google.cloud import speech

            # Read audio file
            with open(file_path, "rb") as f:
                content = f.read()

            # Create recognition config
            config = self._create_recognition_config()

            # Create audio object
            audio = speech.RecognitionAudio(content=content)

            # Determine file size to choose sync vs async
            file_size = file_path.stat().st_size

            loop = asyncio.get_event_loop()

            if file_size < 10 * 1024 * 1024:  # < 10MB, use sync
                response = await loop.run_in_executor(
                    None,
                    lambda: self._client.recognize(config=config, audio=audio),
                )
            else:
                # Use long-running recognition for larger files
                operation = await loop.run_in_executor(
                    None,
                    lambda: self._client.long_running_recognize(
                        config=config, audio=audio
                    ),
                )
                response = await loop.run_in_executor(
                    None,
                    lambda: operation.result(timeout=300),
                )

            # Extract text from response
            transcripts = []
            for result in response.results:
                if result.alternatives:
                    transcripts.append(result.alternatives[0].transcript)

            return " ".join(transcripts).strip()

        except Exception as e:
            raise STTTranscriptionError(
                f"Google Cloud Speech transcription failed: {e}"
            ) from e

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[TranscriptSegment]:
        """
        Transcribe streaming audio.

        Uses Google Cloud Speech streaming recognition for
        real-time transcription.

        Args:
            audio_stream: Async iterator of audio chunks.

        Yields:
            TranscriptSegment for each recognized result.
        """
        from hsttb.core.types import TranscriptSegment

        if not self._initialized:
            await self.initialize()

        try:
            from google.cloud import speech

            # Create streaming config
            config = self._create_recognition_config()
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
            )

            # Generator for streaming requests
            async def request_generator() -> AsyncIterator[Any]:
                # First request with config
                yield speech.StreamingRecognizeRequest(
                    streaming_config=streaming_config
                )

                # Subsequent requests with audio
                async for chunk in audio_stream:
                    yield speech.StreamingRecognizeRequest(audio_content=chunk.data)

            # Run streaming recognition in thread pool
            loop = asyncio.get_event_loop()

            # Collect requests first (needed for sync API)
            requests = []
            async for req in request_generator():
                requests.append(req)

            # Execute streaming recognition
            responses = await loop.run_in_executor(
                None,
                lambda: list(self._client.streaming_recognize(requests=iter(requests))),
            )

            # Process responses
            for response in responses:
                for result in response.results:
                    if result.alternatives:
                        alt = result.alternatives[0]
                        yield TranscriptSegment(
                            text=alt.transcript,
                            is_partial=not result.is_final,
                            is_final=result.is_final,
                            confidence=alt.confidence if alt.confidence else 0.9,
                            start_time_ms=0,
                            end_time_ms=0,
                        )

        except Exception as e:
            logger.error(f"Streaming transcription error: {e}")
            raise STTTranscriptionError(
                f"Google Cloud Speech streaming failed: {e}"
            ) from e

    def _create_recognition_config(self) -> Any:
        """Create recognition configuration."""
        from google.cloud import speech

        # Base config
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self._sample_rate,
            language_code=self._language_code,
            enable_automatic_punctuation=self._enable_punctuation,
            model=self._model,
        )

        # Add word timestamps if requested
        if self._enable_word_timestamps:
            config.enable_word_time_offsets = True

        # Add speech contexts for medical terms if in medical mode
        if self._medical_mode:
            medical_phrases = [
                "metformin", "lisinopril", "atorvastatin", "amlodipine",
                "hydrochlorothiazide", "omeprazole", "levothyroxine",
                "hypertension", "diabetes mellitus", "hyperlipidemia",
                "myocardial infarction", "congestive heart failure",
            ]
            config.speech_contexts = [
                speech.SpeechContext(phrases=medical_phrases, boost=10.0)
            ]

        return config

    async def cleanup(self) -> None:
        """Release client resources."""
        if self._client is not None:
            self._client = None
            self._initialized = False
            logger.info("Cleaned up Google Cloud Speech client")

    def get_available_models(self) -> list[str]:
        """
        Get list of available speech models.

        Returns:
            List of model names.
        """
        return list(SPEECH_MODELS.keys())


# Convenience function
async def transcribe_with_google_cloud(
    file_path: Path | str,
    language_code: str = "en-US",
    medical_mode: bool = False,
) -> str:
    """
    Transcribe an audio file with Google Cloud Speech.

    Args:
        file_path: Path to the audio file.
        language_code: BCP-47 language code.
        medical_mode: Enable medical transcription.

    Returns:
        Transcribed text.
    """
    async with GeminiAdapter(
        language_code=language_code,
        medical_mode=medical_mode,
    ) as adapter:
        return await adapter.transcribe_file(file_path)
