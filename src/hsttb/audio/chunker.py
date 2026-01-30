"""
Audio streaming chunker for benchmark simulation.

This module provides deterministic audio chunking to simulate
streaming conditions for reproducible benchmarking.

Example:
    >>> from hsttb.audio.chunker import StreamingChunker
    >>> from hsttb.core.config import load_profile
    >>> profile = load_profile("realtime_mobile")
    >>> chunker = StreamingChunker(profile, seed=42)
    >>> async for chunk in chunker.stream_audio(audio_data, 16000):
    ...     process(chunk)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

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
        >>> profile = StreamingProfile(
        ...     profile_name="test",
        ...     chunking=ChunkingConfig(chunk_size_ms=1000)
        ... )
        >>> chunker = StreamingChunker(profile, seed=42)
        >>> async for chunk in chunker.stream_audio(audio_data, 16000):
        ...     print(f"Chunk {chunk.sequence_id}: {len(chunk.data)} bytes")
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
            audio_data: Audio samples as numpy array (float32).
            sample_rate: Sample rate in Hz.

        Yields:
            AudioChunk objects in sequence.

        Raises:
            ValueError: If audio_data is empty or sample_rate is invalid.
        """
        if len(audio_data) == 0:
            raise ValueError("audio_data cannot be empty")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        # Calculate samples per chunk
        samples_per_ms = sample_rate / 1000
        chunk_samples = int(self.profile.chunking.chunk_size_ms * samples_per_ms)
        overlap_samples = int(self.profile.chunking.overlap_ms * samples_per_ms)

        # Ensure valid chunk size
        if chunk_samples <= 0:
            raise ValueError("chunk_size_ms must result in positive samples")

        # Calculate step size (chunk - overlap)
        step_samples = max(chunk_samples - overlap_samples, 1)

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

            # Ensure minimum duration
            if duration_ms <= 0:
                duration_ms = 1

            # Create chunk
            chunk = AudioChunk(
                data=chunk_data.tobytes(),
                sequence_id=sequence_id,
                timestamp_ms=timestamp_ms,
                duration_ms=duration_ms,
                is_final=is_final,
            )

            yield chunk

            # Apply timing simulation (jitter + network delay)
            await self._apply_timing_delays()

            # Move to next position
            position += step_samples
            sequence_id += 1

    async def _apply_timing_delays(self) -> None:
        """Apply jitter and network delay if configured."""
        total_delay_ms = 0

        # Add chunk jitter
        jitter_ms = self.profile.chunking.chunk_jitter_ms
        if jitter_ms > 0:
            jitter = int(self._rng.uniform(-jitter_ms, jitter_ms))
            total_delay_ms += max(0, jitter)  # Only positive delays

        # Add network delay
        if hasattr(self.profile, "network"):
            base_delay = self.profile.network.delay_ms
            network_jitter = self.profile.network.jitter_ms

            if base_delay > 0:
                if network_jitter > 0:
                    delay = base_delay + int(
                        self._rng.uniform(-network_jitter, network_jitter)
                    )
                    total_delay_ms += max(0, delay)
                else:
                    total_delay_ms += base_delay

        # Apply delay
        if total_delay_ms > 0:
            await asyncio.sleep(total_delay_ms / 1000)

    def get_expected_chunks(self, audio_duration_ms: int) -> int:
        """
        Calculate expected number of chunks.

        Args:
            audio_duration_ms: Audio duration in milliseconds.

        Returns:
            Expected number of chunks.

        Raises:
            ValueError: If configuration is invalid.
        """
        chunk_ms = self.profile.chunking.chunk_size_ms
        overlap_ms = self.profile.chunking.overlap_ms
        step_ms = chunk_ms - overlap_ms

        if step_ms <= 0:
            raise ValueError("chunk_size_ms must be greater than overlap_ms")

        # Ceiling division
        return (audio_duration_ms + step_ms - 1) // step_ms

    def get_chunk_boundaries(self, audio_duration_ms: int) -> list[tuple[int, int]]:
        """
        Get chunk boundaries without creating actual chunks.

        Useful for planning or testing.

        Args:
            audio_duration_ms: Audio duration in milliseconds.

        Returns:
            List of (start_ms, end_ms) tuples for each chunk.
        """
        chunk_ms = self.profile.chunking.chunk_size_ms
        overlap_ms = self.profile.chunking.overlap_ms
        step_ms = max(chunk_ms - overlap_ms, 1)

        boundaries = []
        position = 0

        while position < audio_duration_ms:
            end = min(position + chunk_ms, audio_duration_ms)
            boundaries.append((position, end))
            position += step_ms

        return boundaries


def create_test_chunks(
    num_chunks: int,
    chunk_duration_ms: int = 1000,
    sample_rate: int = 16000,
) -> list[AudioChunk]:
    """
    Create test chunks for unit testing.

    Args:
        num_chunks: Number of chunks to create.
        chunk_duration_ms: Duration of each chunk in milliseconds.
        sample_rate: Sample rate in Hz.

    Returns:
        List of AudioChunk objects.
    """
    samples_per_chunk = int(chunk_duration_ms * sample_rate / 1000)
    chunks = []

    for i in range(num_chunks):
        # Generate silent audio with small noise
        rng = np.random.default_rng(seed=i)
        audio_data = rng.standard_normal(samples_per_chunk).astype(np.float32) * 0.01

        chunks.append(
            AudioChunk(
                data=audio_data.tobytes(),
                sequence_id=i,
                timestamp_ms=i * chunk_duration_ms,
                duration_ms=chunk_duration_ms,
                is_final=(i == num_chunks - 1),
            )
        )

    return chunks


async def collect_chunks(
    chunker: StreamingChunker,
    audio_data: np.ndarray,
    sample_rate: int,
) -> list[AudioChunk]:
    """
    Collect all chunks from a chunker into a list.

    Utility function for testing.

    Args:
        chunker: The streaming chunker.
        audio_data: Audio samples.
        sample_rate: Sample rate.

    Returns:
        List of all chunks.
    """
    chunks = []
    async for chunk in chunker.stream_audio(audio_data, sample_rate):
        chunks.append(chunk)
    return chunks
