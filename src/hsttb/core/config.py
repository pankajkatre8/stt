"""
Configuration models for HSTTB.

This module provides Pydantic models for configuration management,
including streaming profiles, evaluation settings, and YAML loading.

Example:
    >>> from hsttb.core.config import StreamingProfile, load_profile
    >>> profile = load_profile("ideal")
    >>> print(profile.chunking.chunk_size_ms)
    1000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

# ==============================================================================
# Audio Configuration
# ==============================================================================


class AudioConfig(BaseModel):
    """
    Audio format configuration.

    Attributes:
        sample_rate: Sample rate in Hz (default: 16000).
        channels: Number of audio channels (default: 1 for mono).
        bit_depth: Bits per sample (default: 16).
    """

    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=2)
    bit_depth: int = Field(default=16, ge=8, le=32)

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        """Validate sample rate is a common value."""
        common_rates = {8000, 11025, 16000, 22050, 32000, 44100, 48000}
        if v not in common_rates:
            # Allow but warn about uncommon rates
            pass
        return v


# ==============================================================================
# Chunking Configuration
# ==============================================================================


class ChunkingConfig(BaseModel):
    """
    Audio chunking configuration for streaming simulation.

    Attributes:
        chunk_size_ms: Size of each chunk in milliseconds.
        chunk_jitter_ms: Random jitter added to chunk timing (±ms).
        overlap_ms: Overlap between consecutive chunks.
    """

    chunk_size_ms: int = Field(default=1000, ge=100, le=10000)
    chunk_jitter_ms: int = Field(default=0, ge=0, le=500)
    overlap_ms: int = Field(default=0, ge=0, le=500)


# ==============================================================================
# Network Simulation Configuration
# ==============================================================================


class NetworkConfig(BaseModel):
    """
    Network simulation configuration.

    Attributes:
        delay_ms: Base network delay in milliseconds.
        jitter_ms: Random jitter in network delay (±ms).
        packet_loss_rate: Probability of packet loss (0.0 to 1.0).
    """

    delay_ms: int = Field(default=0, ge=0, le=5000)
    jitter_ms: int = Field(default=0, ge=0, le=1000)
    packet_loss_rate: float = Field(default=0.0, ge=0.0, le=1.0)


# ==============================================================================
# VAD Configuration
# ==============================================================================


class VADConfig(BaseModel):
    """
    Voice Activity Detection configuration.

    Attributes:
        enabled: Whether VAD is enabled.
        silence_threshold_ms: Minimum silence duration to detect pause.
        aggressiveness: VAD aggressiveness level (0-3).
    """

    enabled: bool = Field(default=False)
    silence_threshold_ms: int = Field(default=300, ge=100, le=2000)
    aggressiveness: int = Field(default=1, ge=0, le=3)


# ==============================================================================
# Streaming Profile
# ==============================================================================


class StreamingProfile(BaseModel):
    """
    Complete streaming profile configuration.

    A streaming profile defines how audio is chunked and delivered
    for streaming simulation. Using consistent profiles ensures
    reproducible benchmarks.

    Attributes:
        profile_name: Unique name for this profile.
        description: Human-readable description.
        audio: Audio format configuration.
        chunking: Chunking configuration.
        network: Network simulation configuration.
        vad: VAD configuration.

    Example:
        >>> profile = StreamingProfile(
        ...     profile_name="realtime_mobile",
        ...     description="Simulates mobile network conditions",
        ...     chunking=ChunkingConfig(chunk_size_ms=1000, chunk_jitter_ms=50)
        ... )
    """

    profile_name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="")
    audio: AudioConfig = Field(default_factory=AudioConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    vad: VADConfig = Field(default_factory=VADConfig)

    @field_validator("profile_name")
    @classmethod
    def validate_profile_name(cls, v: str) -> str:
        """Validate profile name is alphanumeric with underscores."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "profile_name must be alphanumeric with underscores/hyphens"
            )
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate cross-field constraints."""
        if self.chunking.overlap_ms >= self.chunking.chunk_size_ms:
            raise ValueError("overlap_ms must be less than chunk_size_ms")


# ==============================================================================
# Evaluation Configuration
# ==============================================================================


class MetricWeights(BaseModel):
    """
    Weights for composite metric scoring.

    Attributes:
        ter: Weight for Term Error Rate (default: 0.4).
        ner: Weight for NER F1 score (default: 0.3).
        crs: Weight for Context Retention Score (default: 0.3).
    """

    ter: float = Field(default=0.4, ge=0.0, le=1.0)
    ner: float = Field(default=0.3, ge=0.0, le=1.0)
    crs: float = Field(default=0.3, ge=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        """Validate weights sum to 1.0."""
        total = self.ter + self.ner + self.crs
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Metric weights must sum to 1.0, got {total}")


class TERConfig(BaseModel):
    """
    TER computation configuration.

    Attributes:
        fuzzy_threshold: Minimum similarity for fuzzy matching.
        case_sensitive: Whether matching is case-sensitive.
        expand_abbreviations: Whether to expand medical abbreviations.
    """

    fuzzy_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    case_sensitive: bool = Field(default=False)
    expand_abbreviations: bool = Field(default=True)


class NERConfig(BaseModel):
    """
    NER computation configuration.

    Attributes:
        span_tolerance: Character tolerance for span matching.
        text_threshold: Minimum text similarity for matching.
        model_name: Name of the NER model to use.
    """

    span_tolerance: int = Field(default=5, ge=0, le=50)
    text_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    model_name: str = Field(default="en_ner_bc5cdr_md")


class CRSConfig(BaseModel):
    """
    CRS computation configuration.

    Attributes:
        semantic_weight: Weight for semantic similarity.
        entity_weight: Weight for entity continuity.
        negation_weight: Weight for negation consistency.
        embedding_model: Sentence embedding model to use.
    """

    semantic_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    entity_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    negation_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    def model_post_init(self, __context: Any) -> None:
        """Validate weights sum to 1.0."""
        total = self.semantic_weight + self.entity_weight + self.negation_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"CRS weights must sum to 1.0, got {total}")


class EvaluationConfig(BaseModel):
    """
    Complete evaluation configuration.

    Attributes:
        weights: Weights for composite scoring.
        ter: TER computation settings.
        ner: NER computation settings.
        crs: CRS computation settings.
    """

    weights: MetricWeights = Field(default_factory=MetricWeights)
    ter: TERConfig = Field(default_factory=TERConfig)
    ner: NERConfig = Field(default_factory=NERConfig)
    crs: CRSConfig = Field(default_factory=CRSConfig)


# ==============================================================================
# Built-in Profiles
# ==============================================================================


BUILTIN_PROFILES: dict[str, StreamingProfile] = {
    "ideal": StreamingProfile(
        profile_name="ideal",
        description="Ideal conditions for baseline measurement",
        chunking=ChunkingConfig(chunk_size_ms=1000, chunk_jitter_ms=0, overlap_ms=0),
        network=NetworkConfig(delay_ms=0, jitter_ms=0, packet_loss_rate=0.0),
        vad=VADConfig(enabled=False),
    ),
    "realtime_mobile": StreamingProfile(
        profile_name="realtime_mobile",
        description="Simulates mobile network conditions",
        chunking=ChunkingConfig(chunk_size_ms=1000, chunk_jitter_ms=50, overlap_ms=100),
        network=NetworkConfig(delay_ms=50, jitter_ms=30, packet_loss_rate=0.01),
        vad=VADConfig(enabled=True, silence_threshold_ms=300),
    ),
    "realtime_clinical": StreamingProfile(
        profile_name="realtime_clinical",
        description="Simulates clinical environment",
        chunking=ChunkingConfig(chunk_size_ms=500, chunk_jitter_ms=20, overlap_ms=50),
        network=NetworkConfig(delay_ms=20, jitter_ms=10, packet_loss_rate=0.0),
        vad=VADConfig(enabled=True, silence_threshold_ms=500),
    ),
    "high_latency": StreamingProfile(
        profile_name="high_latency",
        description="High latency conditions for stress testing",
        chunking=ChunkingConfig(
            chunk_size_ms=2000, chunk_jitter_ms=100, overlap_ms=200
        ),
        network=NetworkConfig(delay_ms=200, jitter_ms=100, packet_loss_rate=0.02),
        vad=VADConfig(enabled=True, silence_threshold_ms=300),
    ),
}


# ==============================================================================
# Profile Loading Functions
# ==============================================================================


def load_profile(name_or_path: str) -> StreamingProfile:
    """
    Load a streaming profile by name or from a YAML file.

    Args:
        name_or_path: Built-in profile name or path to YAML file.

    Returns:
        StreamingProfile instance.

    Raises:
        ValueError: If profile name is unknown and path doesn't exist.
        ConfigurationError: If YAML file is invalid.

    Example:
        >>> profile = load_profile("ideal")
        >>> profile = load_profile("configs/custom_profile.yaml")
    """
    from hsttb.core.exceptions import ConfigurationError

    # Check if it's a built-in profile
    if name_or_path in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name_or_path]

    # Try to load from file
    path = Path(name_or_path)
    if path.exists():
        try:
            return load_profile_from_yaml(path)
        except Exception as e:
            raise ConfigurationError(f"Failed to load profile from {path}: {e}") from e

    # Unknown profile
    available = ", ".join(BUILTIN_PROFILES.keys())
    raise ValueError(f"Unknown profile: {name_or_path!r}. Available: {available}")


def load_profile_from_yaml(path: Path) -> StreamingProfile:
    """
    Load a streaming profile from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        StreamingProfile instance.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ConfigurationError: If YAML is invalid.
    """
    from hsttb.core.exceptions import ConfigurationError

    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return StreamingProfile(**data)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Failed to parse profile {path}: {e}") from e


def save_profile_to_yaml(profile: StreamingProfile, path: Path) -> None:
    """
    Save a streaming profile to a YAML file.

    Args:
        profile: The profile to save.
        path: Path to save the YAML file.
    """
    with open(path, "w") as f:
        yaml.dump(profile.model_dump(), f, default_flow_style=False, sort_keys=False)


def list_builtin_profiles() -> list[str]:
    """
    List all built-in profile names.

    Returns:
        List of built-in profile names.
    """
    return list(BUILTIN_PROFILES.keys())


def get_builtin_profile(name: str) -> StreamingProfile:
    """
    Get a built-in profile by name.

    Args:
        name: Profile name.

    Returns:
        StreamingProfile instance.

    Raises:
        ValueError: If profile name is unknown.
    """
    if name not in BUILTIN_PROFILES:
        available = ", ".join(BUILTIN_PROFILES.keys())
        raise ValueError(f"Unknown built-in profile: {name!r}. Available: {available}")
    return BUILTIN_PROFILES[name]
