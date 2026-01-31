"""
STT adapters for model-agnostic evaluation.

This module provides the adapter interface and implementations
for various STT providers including local (Whisper) and cloud
(Google Cloud, Deepgram) options.

Example:
    >>> from hsttb.adapters import get_adapter, list_adapters
    >>> # List available adapters
    >>> print(list_adapters())
    ['mock', 'whisper', 'gemini', 'google-cloud-speech', 'deepgram']
    >>> # Using factory
    >>> adapter = get_adapter("whisper", model_size="base")
    >>> await adapter.initialize()
    >>> text = await adapter.transcribe_file("audio.wav")
"""
from __future__ import annotations

from hsttb.adapters.base import STTAdapter
from hsttb.adapters.mock_adapter import FailingMockAdapter, MockSTTAdapter
from hsttb.adapters.registry import (
    clear_registry,
    get_adapter,
    is_adapter_registered,
    list_adapters,
    register_adapter,
    unregister_adapter,
)

# Import adapters to trigger registration
# These are lazy-loaded to avoid requiring all dependencies
def __getattr__(name: str) -> type:
    """Lazy import adapters."""
    if name == "WhisperAdapter":
        from hsttb.adapters.whisper_adapter import WhisperAdapter
        return WhisperAdapter

    if name == "GeminiAdapter":
        from hsttb.adapters.gemini_adapter import GeminiAdapter
        return GeminiAdapter

    if name == "DeepgramAdapter":
        from hsttb.adapters.deepgram_adapter import DeepgramAdapter
        return DeepgramAdapter

    if name == "ElevenLabsTTSGenerator":
        from hsttb.adapters.elevenlabs_tts import ElevenLabsTTSGenerator
        return ElevenLabsTTSGenerator

    if name == "AudioTestGenerator":
        from hsttb.adapters.elevenlabs_tts import AudioTestGenerator
        return AudioTestGenerator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Trigger registration of adapters that are available
def _register_available_adapters() -> None:
    """Register adapters that have their dependencies installed."""
    # Whisper adapter
    try:
        from hsttb.adapters.whisper_adapter import WhisperAdapter  # noqa: F401
    except ImportError:
        pass

    # Gemini/Google Cloud adapter
    try:
        from hsttb.adapters.gemini_adapter import GeminiAdapter  # noqa: F401
    except ImportError:
        pass

    # Deepgram adapter
    try:
        from hsttb.adapters.deepgram_adapter import DeepgramAdapter  # noqa: F401
    except ImportError:
        pass


# Register available adapters on import
_register_available_adapters()


__all__ = [
    # Base class
    "STTAdapter",
    # Registry functions
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "is_adapter_registered",
    "unregister_adapter",
    "clear_registry",
    # Mock adapters (always available)
    "MockSTTAdapter",
    "FailingMockAdapter",
    # Production adapters (lazy-loaded)
    "WhisperAdapter",
    "GeminiAdapter",
    "DeepgramAdapter",
    # TTS generators
    "ElevenLabsTTSGenerator",
    "AudioTestGenerator",
]
