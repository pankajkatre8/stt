# TASK-1S01: STT Adapter Interface

## Metadata
- **Status**: pending
- **Complexity**: Medium (1-2 hours)
- **Blocked By**: TASK-1C02, TASK-1A03
- **Blocks**: TASK-1S02, TASK-1S03

## Objective

Define the abstract STT adapter interface that all STT model adapters must implement, ensuring model-agnostic evaluation.

## Context

This interface is fundamental to the model-agnostic design. Every STT provider (Whisper, Deepgram, AWS, custom) will implement this interface. The evaluation logic never knows which specific adapter it's using - it only works with this interface.

## Requirements

- [ ] Define `STTAdapter` abstract base class
- [ ] Define async `initialize()` method
- [ ] Define async `transcribe_stream()` method
- [ ] Define async `transcribe_file()` method
- [ ] Define async `cleanup()` method
- [ ] Define `name` property for identification
- [ ] Create adapter registry/factory function
- [ ] Ensure interface is minimal and clean

## Acceptance Criteria

- [ ] AC1: Abstract class cannot be instantiated
- [ ] AC2: Concrete classes must implement all methods
- [ ] AC3: Type hints are complete
- [ ] AC4: Factory function returns correct adapter
- [ ] AC5: Interface supports async context manager

## Files to Create/Modify

- Create: `src/hsttb/adapters/base.py`
- Create: `src/hsttb/adapters/registry.py`
- Modify: `src/hsttb/adapters/__init__.py`

## Implementation Details

### src/hsttb/adapters/base.py

```python
"""
Abstract base class for STT adapters.

This module defines the interface that all STT adapters must implement.
The interface is designed to be minimal, async-first, and streaming-aware.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from pathlib import Path
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
        ...
        # This is needed to make it an async generator
        yield  # type: ignore

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

    async def cleanup(self) -> None:
        """
        Clean up resources.

        Override this method to release resources, close connections,
        or unload models. Default implementation does nothing.
        """
        pass

    async def __aenter__(self) -> STTAdapter:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name!r})>"
```

### src/hsttb/adapters/registry.py

```python
"""
Adapter registry and factory functions.

This module provides a registry of available STT adapters and
factory functions to instantiate them by name.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from hsttb.adapters.base import STTAdapter

# Registry of available adapters
_ADAPTER_REGISTRY: dict[str, Type[STTAdapter]] = {}


def register_adapter(name: str):
    """
    Decorator to register an adapter class.

    Args:
        name: The name to register the adapter under.

    Example:
        >>> @register_adapter("my_stt")
        ... class MySTTAdapter(STTAdapter):
        ...     pass
    """
    def decorator(cls: Type[STTAdapter]) -> Type[STTAdapter]:
        _ADAPTER_REGISTRY[name] = cls
        return cls
    return decorator


def get_adapter(name: str, **kwargs) -> STTAdapter:
    """
    Get an adapter instance by name.

    Args:
        name: The registered adapter name.
        **kwargs: Arguments to pass to the adapter constructor.

    Returns:
        An instance of the requested adapter.

    Raises:
        ValueError: If adapter name is not registered.

    Example:
        >>> adapter = get_adapter("whisper", model_size="base")
        >>> await adapter.initialize()
    """
    if name not in _ADAPTER_REGISTRY:
        available = ", ".join(_ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown adapter: {name!r}. Available: {available}"
        )
    return _ADAPTER_REGISTRY[name](**kwargs)


def list_adapters() -> list[str]:
    """
    List all registered adapter names.

    Returns:
        List of registered adapter names.
    """
    return list(_ADAPTER_REGISTRY.keys())


def is_adapter_registered(name: str) -> bool:
    """
    Check if an adapter is registered.

    Args:
        name: The adapter name to check.

    Returns:
        True if registered, False otherwise.
    """
    return name in _ADAPTER_REGISTRY
```

### src/hsttb/adapters/__init__.py

```python
"""
STT adapters for HSTTB.

This module provides the STT adapter interface and registry.
"""
from hsttb.adapters.base import STTAdapter
from hsttb.adapters.registry import (
    get_adapter,
    list_adapters,
    register_adapter,
    is_adapter_registered,
)

__all__ = [
    "STTAdapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "is_adapter_registered",
]
```

## Testing Requirements

- Unit tests required: Yes
- Test file: `tests/unit/adapters/test_base.py`
- Test cases:
  - [ ] Abstract class cannot be instantiated
  - [ ] Concrete subclass must implement all methods
  - [ ] Registry registers adapters
  - [ ] Factory returns correct adapter
  - [ ] Unknown adapter raises ValueError
  - [ ] Context manager works

## Notes

- Keep interface minimal - only essential methods
- Use async throughout for consistency
- Context manager support is convenient
- Registry enables dynamic adapter loading
- Don't add adapter-specific methods to base class
