# TASK-1A02: Streaming Chunker

## Metadata
- **Status**: pending
- **Complexity**: Large (2-4 hours)
- **Blocked By**: TASK-1A01, TASK-1C03
- **Blocks**: TASK-1A03, TASK-1S01

## Objective

Implement a streaming audio chunker that splits audio into chunks according to configurable streaming profiles, enabling reproducible benchmark streaming simulation.

## Context

This is a critical component for benchmark validity. The chunker must:
1. Simulate real-world streaming conditions
2. Be deterministic for reproducibility
3. Support configurable chunk sizes and jitter
4. Enable the SRS metric to work correctly

## Requirements

- [ ] Implement `StreamingChunker` class
- [ ] Accept audio data and sample rate
- [ ] Chunk according to `StreamingProfile` configuration
- [ ] Support configurable chunk size (ms)
- [ ] Support configurable jitter (timing variation)
- [ ] Support configurable overlap
- [ ] Use seeded random for deterministic behavior
- [ ] Yield `AudioChunk` objects asynchronously
- [ ] Track timestamps and sequence IDs correctly

## Acceptance Criteria

- [ ] AC1: Same audio + same seed = identical chunk sequence
- [ ] AC2: Different seeds produce different jitter patterns
- [ ] AC3: Chunk timestamps are monotonically increasing
- [ ] AC4: Final chunk has `is_final=True`
- [ ] AC5: Async iteration works correctly
- [ ] AC6: No audio data is lost or duplicated

## Files to Create/Modify

- Create: `src/hsttb/audio/chunker.py`
- Modify: `src/hsttb/audio/__init__.py`

## Implementation Details

### src/hsttb/audio/chunker.py

```python
"""
Audio streaming chunker for benchmark simulation.

This module provides deterministic audio chunking to simulate
streaming conditions for reproducible benchmarking.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator

import numpy as np

from hsttb.core.types import AudioChunk

if TYPE_CHECKING:
    from hsttb.core.config import StreamingProfile


class StreamingChunker:
    """
    Chunks audio data into streaming segments.

    The chunker simulates real-world streaming conditions with
    configurable chunk sizes, jitter, and delays. Using the same
    seed produces identical chunk sequences for reproducibility.

    Attributes:
        profile: The streaming profile configuration.
        seed: Random seed for deterministic behavior.

    Example:
        >>> profile = StreamingProfile(chunk_size_ms=1000)
        >>> chunker = StreamingChunker(profile, seed=42)
        >>> async for chunk in chunker.stream_audio(audio_data, 16000):
        ...     process(chunk)
    """

    def __init__(self, profile: StreamingProfile, seed: int = 42) -> None:
        """
        Initialize the chunker.

        Args:
            profile: Streaming profile with chunking configuration.
            seed: Random seed for reproducibility.
        """
        self.profile = profile
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        """
        Reset the random state.

        Args:
            seed: New seed, or use original if None.
        """
        self._rng = np.random.default_rng(seed if seed is not None else self.seed)

    async def stream_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> AsyncIterator[AudioChunk]:
        """
        Stream audio data as chunks.

        Args:
            audio_data: Audio samples as numpy array.
            sample_rate: Sample rate in Hz.

        Yields:
            AudioChunk objects in sequence.

        Raises:
            ValueError: If audio_data is empty.
        """
        if len(audio_data) == 0:
            raise ValueError("audio_data cannot be empty")

        # Calculate samples per chunk
        samples_per_ms = sample_rate / 1000
        chunk_samples = int(self.profile.chunking.chunk_size_ms * samples_per_ms)
        overlap_samples = int(self.profile.chunking.overlap_ms * samples_per_ms)

        # Ensure we have valid chunk size
        if chunk_samples <= 0:
            raise ValueError("chunk_size_ms must result in positive samples")

        position = 0
        sequence_id = 0
        total_samples = len(audio_data)

        while position < total_samples:
            # Calculate chunk boundaries
            end_position = min(position + chunk_samples, total_samples)
            is_final = end_position >= total_samples

            # Extract chunk data
            chunk_data = audio_data[position:end_position]

            # Calculate timing
            timestamp_ms = int(position / samples_per_ms)
            duration_ms = int(len(chunk_data) / samples_per_ms)

            # Create chunk
            chunk = AudioChunk(
                data=chunk_data.tobytes(),
                sequence_id=sequence_id,
                timestamp_ms=timestamp_ms,
                duration_ms=duration_ms,
                is_final=is_final,
            )

            yield chunk

            # Apply jitter delay if configured
            jitter_ms = self._get_jitter()
            if jitter_ms > 0:
                await asyncio.sleep(jitter_ms / 1000)

            # Apply network delay if configured
            delay_ms = self._get_network_delay()
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)

            # Move to next position (accounting for overlap)
            step = chunk_samples - overlap_samples
            position += max(step, 1)  # Ensure progress
            sequence_id += 1

    def _get_jitter(self) -> int:
        """Get jitter value in milliseconds."""
        jitter_ms = self.profile.chunking.chunk_jitter_ms
        if jitter_ms <= 0:
            return 0
        return int(self._rng.uniform(-jitter_ms, jitter_ms))

    def _get_network_delay(self) -> int:
        """Get network delay in milliseconds."""
        if not hasattr(self.profile, 'network'):
            return 0
        delay_ms = getattr(self.profile.network, 'delay_ms', 0)
        jitter_ms = getattr(self.profile.network, 'jitter_ms', 0)
        if delay_ms <= 0:
            return 0
        if jitter_ms > 0:
            return int(delay_ms + self._rng.uniform(-jitter_ms, jitter_ms))
        return delay_ms

    def get_expected_chunks(
        self,
        audio_duration_ms: int,
    ) -> int:
        """
        Calculate expected number of chunks.

        Args:
            audio_duration_ms: Audio duration in milliseconds.

        Returns:
            Expected number of chunks.
        """
        chunk_ms = self.profile.chunking.chunk_size_ms
        overlap_ms = self.profile.chunking.overlap_ms
        step_ms = chunk_ms - overlap_ms

        if step_ms <= 0:
            raise ValueError("chunk_size_ms must be greater than overlap_ms")

        return (audio_duration_ms + step_ms - 1) // step_ms


def create_test_chunks(
    num_chunks: int,
    chunk_duration_ms: int = 1000,
    sample_rate: int = 16000,
) -> list[AudioChunk]:
    """
    Create test chunks for unit testing.

    Args:
        num_chunks: Number of chunks to create.
        chunk_duration_ms: Duration of each chunk.
        sample_rate: Sample rate in Hz.

    Returns:
        List of AudioChunk objects.
    """
    samples_per_chunk = int(chunk_duration_ms * sample_rate / 1000)
    chunks = []

    for i in range(num_chunks):
        # Generate silent audio with small noise
        audio_data = np.random.randn(samples_per_chunk).astype(np.float32) * 0.01

        chunks.append(AudioChunk(
            data=audio_data.tobytes(),
            sequence_id=i,
            timestamp_ms=i * chunk_duration_ms,
            duration_ms=chunk_duration_ms,
            is_final=(i == num_chunks - 1),
        ))

    return chunks
```

## Testing Requirements

- Unit tests required: Yes
- Test file: `tests/unit/audio/test_chunker.py`
- Test cases:
  - [ ] Same seed produces identical chunks
  - [ ] Different seeds produce different jitter
  - [ ] Timestamps are monotonically increasing
  - [ ] Final chunk has is_final=True
  - [ ] All audio data is included (no loss)
  - [ ] Overlap works correctly
  - [ ] Empty audio raises ValueError
  - [ ] Expected chunk count is accurate

## Healthcare Considerations

- Chunking boundaries can affect STT quality
- This is why we need reproducible chunking
- SRS metric compares same audio across profiles
- Boundary issues must be detectable

## Notes

- Use numpy for audio data (efficient)
- Async generator for streaming pattern
- Seeded random for reproducibility
- Test thoroughly - this affects benchmark validity
