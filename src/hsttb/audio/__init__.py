"""
Audio loading and streaming simulation.

This module provides audio file loading, streaming chunking,
and profile-based streaming simulation for reproducible benchmarks.
"""

from __future__ import annotations

from hsttb.audio.chunker import (
    StreamingChunker,
    collect_chunks,
    create_test_chunks,
)
from hsttb.audio.loader import (
    SUPPORTED_EXTENSIONS,
    AudioLoader,
    get_audio_duration,
    validate_audio_file,
)

__all__ = [
    # Loader
    "AudioLoader",
    "SUPPORTED_EXTENSIONS",
    "get_audio_duration",
    "validate_audio_file",
    # Chunker
    "StreamingChunker",
    "create_test_chunks",
    "collect_chunks",
]
