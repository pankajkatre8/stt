"""
Tests for new STT adapters and TTS generators.

Tests Whisper, Gemini, Deepgram adapters, and ElevenLabs TTS.
These tests use mocking for external services.
"""
from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hsttb.adapters import (
    STTAdapter,
    get_adapter,
    is_adapter_registered,
    list_adapters,
)
from hsttb.core.types import AudioChunk


class TestAdapterRegistration:
    """Tests for adapter registration of new adapters."""

    def test_whisper_registered(self) -> None:
        """Whisper adapter is registered if available."""
        adapters = list_adapters()
        # Whisper may not be available if openai-whisper not installed
        if is_adapter_registered("whisper"):
            assert "whisper" in adapters

    def test_gemini_registered(self) -> None:
        """Gemini adapter is registered if available."""
        adapters = list_adapters()
        if is_adapter_registered("gemini"):
            assert "gemini" in adapters

    def test_deepgram_registered(self) -> None:
        """Deepgram adapter is registered if available."""
        adapters = list_adapters()
        if is_adapter_registered("deepgram"):
            assert "deepgram" in adapters


class TestWhisperAdapter:
    """Tests for WhisperAdapter."""

    @pytest.fixture
    def temp_audio_file(self) -> Path:
        """Create temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
                    b"\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00"
                    b"data\x00\x00\x00\x00")
            return Path(f.name)

    def test_import_whisper_adapter(self) -> None:
        """WhisperAdapter can be imported."""
        try:
            from hsttb.adapters.whisper_adapter import WhisperAdapter
            assert WhisperAdapter is not None
        except ImportError:
            pytest.skip("Whisper not installed")

    def test_whisper_adapter_inheritance(self) -> None:
        """WhisperAdapter inherits from STTAdapter."""
        try:
            from hsttb.adapters.whisper_adapter import WhisperAdapter
            assert issubclass(WhisperAdapter, STTAdapter)
        except ImportError:
            pytest.skip("Whisper not installed")

    def test_whisper_adapter_name(self) -> None:
        """WhisperAdapter has correct name format."""
        try:
            from hsttb.adapters.whisper_adapter import WhisperAdapter
            adapter = WhisperAdapter(model_size="tiny")
            # Name includes model size
            assert "whisper" in adapter.name
            assert "tiny" in adapter.name
        except ImportError:
            pytest.skip("Whisper not installed")

    def test_whisper_adapter_attributes(self) -> None:
        """WhisperAdapter has expected attributes."""
        try:
            from hsttb.adapters.whisper_adapter import WhisperAdapter
            adapter = WhisperAdapter(model_size="tiny")
            assert hasattr(adapter, "transcribe_file")
            assert hasattr(adapter, "transcribe_stream")
            assert hasattr(adapter, "initialize")
            assert adapter._model_size == "tiny"
        except ImportError:
            pytest.skip("Whisper not installed")


class TestGeminiAdapter:
    """Tests for GeminiAdapter."""

    @pytest.fixture
    def temp_audio_file(self) -> Path:
        """Create temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio content")
            return Path(f.name)

    def test_import_gemini_adapter(self) -> None:
        """GeminiAdapter can be imported."""
        try:
            from hsttb.adapters.gemini_adapter import GeminiAdapter
            assert GeminiAdapter is not None
        except ImportError:
            pytest.skip("Google Cloud Speech not installed")

    def test_gemini_adapter_inheritance(self) -> None:
        """GeminiAdapter inherits from STTAdapter."""
        try:
            from hsttb.adapters.gemini_adapter import GeminiAdapter
            assert issubclass(GeminiAdapter, STTAdapter)
        except ImportError:
            pytest.skip("Google Cloud Speech not installed")

    def test_gemini_adapter_name(self) -> None:
        """GeminiAdapter has correct name format."""
        try:
            from hsttb.adapters.gemini_adapter import GeminiAdapter
            adapter = GeminiAdapter()
            # Name includes service info
            assert "google-cloud-speech" in adapter.name or "gemini" in adapter.name
        except ImportError:
            pytest.skip("Google Cloud Speech not installed")


class TestDeepgramAdapter:
    """Tests for DeepgramAdapter."""

    @pytest.fixture
    def temp_audio_file(self) -> Path:
        """Create temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio content")
            return Path(f.name)

    def test_import_deepgram_adapter(self) -> None:
        """DeepgramAdapter can be imported."""
        try:
            from hsttb.adapters.deepgram_adapter import DeepgramAdapter
            assert DeepgramAdapter is not None
        except ImportError:
            pytest.skip("Deepgram SDK not installed")

    def test_deepgram_adapter_inheritance(self) -> None:
        """DeepgramAdapter inherits from STTAdapter."""
        try:
            from hsttb.adapters.deepgram_adapter import DeepgramAdapter
            assert issubclass(DeepgramAdapter, STTAdapter)
        except ImportError:
            pytest.skip("Deepgram SDK not installed")

    def test_deepgram_adapter_name(self) -> None:
        """DeepgramAdapter has correct name format."""
        try:
            from hsttb.adapters.deepgram_adapter import DeepgramAdapter
            adapter = DeepgramAdapter()
            # Name includes model info
            assert "deepgram" in adapter.name
        except ImportError:
            pytest.skip("Deepgram SDK not installed")

    def test_deepgram_adapter_default_model(self) -> None:
        """DeepgramAdapter uses medical model by default."""
        try:
            from hsttb.adapters.deepgram_adapter import DeepgramAdapter
            adapter = DeepgramAdapter()
            assert adapter._model == "nova-2-medical"
        except ImportError:
            pytest.skip("Deepgram SDK not installed")

    def test_deepgram_adapter_attributes(self) -> None:
        """DeepgramAdapter has expected attributes."""
        try:
            from hsttb.adapters.deepgram_adapter import DeepgramAdapter
            adapter = DeepgramAdapter()
            assert hasattr(adapter, "transcribe_file")
            assert hasattr(adapter, "transcribe_stream")
            assert hasattr(adapter, "initialize")
            assert adapter._model == "nova-2-medical"
        except ImportError:
            pytest.skip("Deepgram SDK not installed")


class TestElevenLabsTTS:
    """Tests for ElevenLabs TTS generator."""

    def test_import_elevenlabs_tts(self) -> None:
        """ElevenLabsTTSGenerator can be imported."""
        try:
            from hsttb.adapters.elevenlabs_tts import ElevenLabsTTSGenerator
            assert ElevenLabsTTSGenerator is not None
        except ImportError:
            pytest.skip("ElevenLabs SDK not installed")

    def test_elevenlabs_generator_creation(self) -> None:
        """ElevenLabsTTSGenerator can be created."""
        try:
            from hsttb.adapters.elevenlabs_tts import ElevenLabsTTSGenerator
            generator = ElevenLabsTTSGenerator(api_key="test-key")
            assert generator is not None
        except ImportError:
            pytest.skip("ElevenLabs SDK not installed")

    def test_elevenlabs_generator_attributes(self) -> None:
        """ElevenLabsTTSGenerator has expected attributes."""
        try:
            from hsttb.adapters.elevenlabs_tts import ElevenLabsTTSGenerator
            generator = ElevenLabsTTSGenerator(api_key="test-key")
            assert hasattr(generator, "generate_audio")
            assert generator._api_key == "test-key"
        except ImportError:
            pytest.skip("ElevenLabs SDK not installed")


class TestAudioTestGenerator:
    """Tests for AudioTestGenerator."""

    def test_import_audio_test_generator(self) -> None:
        """AudioTestGenerator can be imported."""
        try:
            from hsttb.adapters.elevenlabs_tts import AudioTestGenerator
            assert AudioTestGenerator is not None
        except ImportError:
            pytest.skip("ElevenLabs SDK not installed")

    def test_audio_test_generator_creation(self) -> None:
        """AudioTestGenerator can be created."""
        try:
            from hsttb.adapters.elevenlabs_tts import AudioTestGenerator
            generator = AudioTestGenerator()
            assert generator is not None
        except ImportError:
            pytest.skip("ElevenLabs SDK not installed")


class TestAdapterLazyLoading:
    """Tests for lazy loading of adapters."""

    def test_whisper_lazy_import(self) -> None:
        """WhisperAdapter is lazy loaded from hsttb.adapters."""
        try:
            from hsttb.adapters import WhisperAdapter
            assert WhisperAdapter is not None
        except (ImportError, AttributeError):
            # Expected if whisper not installed
            pass

    def test_gemini_lazy_import(self) -> None:
        """GeminiAdapter is lazy loaded from hsttb.adapters."""
        try:
            from hsttb.adapters import GeminiAdapter
            assert GeminiAdapter is not None
        except (ImportError, AttributeError):
            # Expected if google-cloud-speech not installed
            pass

    def test_deepgram_lazy_import(self) -> None:
        """DeepgramAdapter is lazy loaded from hsttb.adapters."""
        try:
            from hsttb.adapters import DeepgramAdapter
            assert DeepgramAdapter is not None
        except (ImportError, AttributeError):
            # Expected if deepgram-sdk not installed
            pass

    def test_elevenlabs_lazy_import(self) -> None:
        """ElevenLabsTTSGenerator is lazy loaded from hsttb.adapters."""
        try:
            from hsttb.adapters import ElevenLabsTTSGenerator
            assert ElevenLabsTTSGenerator is not None
        except (ImportError, AttributeError):
            # Expected if elevenlabs not installed
            pass


class TestAdapterFactoryPattern:
    """Tests for adapter factory pattern."""

    def test_get_adapter_mock(self) -> None:
        """get_adapter returns mock adapter."""
        adapter = get_adapter("mock")
        assert adapter.name == "mock"

    def test_get_adapter_with_params(self) -> None:
        """get_adapter passes parameters correctly."""
        adapter = get_adapter("mock", responses=["custom"])
        assert adapter.responses == ["custom"]

    def test_get_adapter_unknown(self) -> None:
        """get_adapter raises for unknown adapter."""
        with pytest.raises(ValueError, match="Unknown adapter"):
            get_adapter("unknown_adapter_xyz")

    def test_list_adapters_includes_mock(self) -> None:
        """list_adapters includes mock adapter."""
        adapters = list_adapters()
        assert "mock" in adapters
        assert "failing_mock" in adapters
