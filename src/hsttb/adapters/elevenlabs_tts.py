"""
ElevenLabs TTS generator for test audio generation.

Generates high-quality audio from text using ElevenLabs API.
Useful for creating test audio files from ground truth transcripts.

Example:
    >>> generator = ElevenLabsTTSGenerator(api_key="your-key")
    >>> await generator.generate_audio("Patient has diabetes", Path("output.wav"))
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hsttb.core.exceptions import HSSTBError

logger = logging.getLogger(__name__)


class TTSGenerationError(HSSTBError):
    """Raised when TTS generation fails."""

    pass


# ElevenLabs voice presets
VOICE_PRESETS = {
    # Default voices
    "rachel": "21m00Tcm4TlvDq8ikWAM",  # Female, American
    "drew": "29vD33N1CtxCmqQRPOHJ",  # Male, American
    "clyde": "2EiwWnXFnvU5JabPnv8n",  # Male, American, deep
    "paul": "5Q0t7uMcjvnagumLfvZi",  # Male, American
    "domi": "AZnzlk1XvdvUeBnXmlld",  # Female, American
    "nicole": "piTKgcLEGmPE4e6mEKli",  # Female, American, soft
    "adam": "pNInz6obpgDQGcFmaJgB",  # Male, American, deep
    "sam": "yoZ06aMxZJJ28mfd3POQ",  # Male, American
    # Medical/professional voices
    "professional": "21m00Tcm4TlvDq8ikWAM",  # Rachel - clear, professional
    "clinical": "5Q0t7uMcjvnagumLfvZi",  # Paul - authoritative
}


@dataclass
class VoiceSettings:
    """
    Voice customization settings.

    Attributes:
        stability: Voice stability (0.0-1.0). Lower = more expressive.
        similarity_boost: Voice similarity (0.0-1.0). Higher = closer to original.
        style: Style exaggeration (0.0-1.0).
        use_speaker_boost: Enable speaker boost for clarity.
    """

    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to API-compatible dictionary."""
        return {
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost,
        }


class ElevenLabsTTSGenerator:
    """
    Text-to-speech generator using ElevenLabs API.

    Generates high-quality speech audio from text, useful for
    creating test audio files from ground truth transcripts.

    Attributes:
        api_key: ElevenLabs API key.
        voice_id: Voice to use for generation.
        model_id: TTS model to use.
        voice_settings: Voice customization settings.

    Example:
        >>> generator = ElevenLabsTTSGenerator()
        >>> await generator.generate_audio(
        ...     "Patient takes metformin for diabetes",
        ...     Path("test_audio.wav")
        ... )
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice: str = "professional",
        model: str = "eleven_turbo_v2",
        voice_settings: VoiceSettings | None = None,
    ) -> None:
        """
        Initialize the ElevenLabs TTS generator.

        Args:
            api_key: ElevenLabs API key (or ELEVENLABS_API_KEY env var).
            voice: Voice name or ID to use.
            model: TTS model to use.
            voice_settings: Custom voice settings.
        """
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self._voice_id = VOICE_PRESETS.get(voice, voice)
        self._model_id = model
        self._voice_settings = voice_settings or VoiceSettings()
        self._client: Any = None

    async def _ensure_client(self) -> Any:
        """Ensure client is initialized."""
        if self._client is not None:
            return self._client

        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as e:
            raise TTSGenerationError(
                "elevenlabs is not installed. Install with: pip install elevenlabs"
            ) from e

        if not self._api_key:
            raise TTSGenerationError(
                "ElevenLabs API key not provided. Set ELEVENLABS_API_KEY "
                "environment variable or pass api_key parameter."
            )

        self._client = ElevenLabs(api_key=self._api_key)
        return self._client

    async def generate_audio(
        self,
        text: str,
        output_path: Path | str,
        output_format: str = "mp3_44100_128",
    ) -> Path:
        """
        Generate audio from text.

        Args:
            text: Text to convert to speech.
            output_path: Output file path.
            output_format: Audio output format.

        Returns:
            Path to generated audio file.

        Raises:
            TTSGenerationError: If generation fails.
        """
        if not text.strip():
            raise TTSGenerationError("Empty text provided")

        output_path = Path(output_path)

        try:
            client = await self._ensure_client()

            # Generate audio in thread pool
            audio_bytes = await asyncio.to_thread(
                self._generate_sync,
                client,
                text,
                output_format,
            )

            # Write to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            logger.info(f"Generated audio: {output_path} ({len(audio_bytes)} bytes)")
            return output_path

        except Exception as e:
            if "elevenlabs" in str(type(e).__module__).lower():
                raise TTSGenerationError(f"ElevenLabs API error: {e}") from e
            raise TTSGenerationError(f"Audio generation failed: {e}") from e

    def _generate_sync(
        self,
        client: Any,
        text: str,
        output_format: str,
    ) -> bytes:
        """Synchronous audio generation."""
        from elevenlabs import VoiceSettings as ELVoiceSettings

        voice_settings = ELVoiceSettings(
            stability=self._voice_settings.stability,
            similarity_boost=self._voice_settings.similarity_boost,
            style=self._voice_settings.style,
            use_speaker_boost=self._voice_settings.use_speaker_boost,
        )

        # Generate audio
        audio = client.generate(
            text=text,
            voice=self._voice_id,
            model=self._model_id,
            voice_settings=voice_settings,
            output_format=output_format,
        )

        # Collect audio bytes
        if hasattr(audio, "__iter__"):
            return b"".join(audio)
        return audio

    async def generate_batch(
        self,
        texts: list[str],
        output_dir: Path | str,
        prefix: str = "audio",
    ) -> list[Path]:
        """
        Generate audio for multiple texts.

        Args:
            texts: List of texts to convert.
            output_dir: Directory for output files.
            prefix: Filename prefix.

        Returns:
            List of generated audio file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"{prefix}_{i:04d}.mp3"
            try:
                path = await self.generate_audio(text, output_path)
                results.append(path)
            except Exception as e:
                logger.error(f"Failed to generate audio {i}: {e}")

        return results

    async def get_available_voices(self) -> list[dict[str, Any]]:
        """
        Get list of available voices.

        Returns:
            List of voice dictionaries with id and name.
        """
        try:
            client = await self._ensure_client()

            voices = await asyncio.to_thread(
                lambda: client.voices.get_all()
            )

            return [
                {
                    "voice_id": v.voice_id,
                    "name": v.name,
                    "category": getattr(v, "category", "custom"),
                }
                for v in voices.voices
            ]

        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return []

    async def get_available_models(self) -> list[dict[str, Any]]:
        """
        Get list of available TTS models.

        Returns:
            List of model dictionaries.
        """
        # Static list of commonly available models
        return [
            {
                "model_id": "eleven_turbo_v2",
                "name": "Turbo v2",
                "description": "Fast, high-quality generation",
            },
            {
                "model_id": "eleven_multilingual_v2",
                "name": "Multilingual v2",
                "description": "Multi-language support",
            },
            {
                "model_id": "eleven_monolingual_v1",
                "name": "Monolingual v1",
                "description": "English-only, older model",
            },
        ]


class AudioTestGenerator:
    """
    Generator for creating test audio from ground truth transcripts.

    Combines TTS generation with metadata for benchmark testing.

    Example:
        >>> generator = AudioTestGenerator()
        >>> result = await generator.generate_test_case(
        ...     "Patient takes metformin 500mg twice daily",
        ...     Path("test_cases/case_001")
        ... )
    """

    def __init__(
        self,
        tts_generator: ElevenLabsTTSGenerator | None = None,
    ) -> None:
        """
        Initialize the test generator.

        Args:
            tts_generator: TTS generator to use.
        """
        self._tts = tts_generator or ElevenLabsTTSGenerator()

    async def generate_test_case(
        self,
        ground_truth: str,
        output_dir: Path | str,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a complete test case.

        Creates audio file and metadata for benchmark testing.

        Args:
            ground_truth: Ground truth transcript.
            output_dir: Output directory.
            case_id: Optional case identifier.

        Returns:
            Dictionary with test case metadata.
        """
        import json
        import uuid

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        case_id = case_id or str(uuid.uuid4())[:8]

        # Generate audio
        audio_path = output_dir / f"{case_id}.mp3"
        await self._tts.generate_audio(ground_truth, audio_path)

        # Create metadata
        metadata = {
            "case_id": case_id,
            "ground_truth": ground_truth,
            "audio_file": audio_path.name,
            "format": "mp3",
        }

        # Write metadata
        metadata_path = output_dir / f"{case_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    async def generate_test_suite(
        self,
        cases: list[dict[str, str]],
        output_dir: Path | str,
        suite_name: str = "test_suite",
    ) -> dict[str, Any]:
        """
        Generate a suite of test cases.

        Args:
            cases: List of dicts with "id" and "text" keys.
            output_dir: Output directory.
            suite_name: Suite name.

        Returns:
            Suite metadata including all case IDs.
        """
        import json

        output_dir = Path(output_dir)
        suite_dir = output_dir / suite_name
        suite_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for case in cases:
            case_id = case.get("id", None)
            text = case.get("text", "")

            if text:
                try:
                    result = await self.generate_test_case(
                        text, suite_dir, case_id
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to generate case {case_id}: {e}")

        # Write suite manifest
        manifest = {
            "suite_name": suite_name,
            "total_cases": len(results),
            "cases": results,
        }

        manifest_path = suite_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest
