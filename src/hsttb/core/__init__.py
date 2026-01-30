"""
Core types and configuration for HSTTB.

This module provides the foundational types and configuration
classes used throughout the framework.
"""

from __future__ import annotations

from hsttb.core.config import (
    BUILTIN_PROFILES,
    AudioConfig,
    ChunkingConfig,
    CRSConfig,
    EvaluationConfig,
    MetricWeights,
    NERConfig,
    NetworkConfig,
    StreamingProfile,
    TERConfig,
    VADConfig,
    get_builtin_profile,
    list_builtin_profiles,
    load_profile,
    load_profile_from_yaml,
    save_profile_to_yaml,
)
from hsttb.core.exceptions import (
    AudioError,
    AudioFormatError,
    AudioLoadError,
    BenchmarkError,
    ConfigurationError,
    CRSComputationError,
    EvaluationError,
    HSSTBError,
    LexiconError,
    LexiconLoadError,
    LexiconLookupError,
    MetricComputationError,
    NERComputationError,
    ReportGenerationError,
    STTAdapterError,
    STTConnectionError,
    STTTranscriptionError,
    TERComputationError,
)
from hsttb.core.types import (
    AudioChunk,
    BenchmarkResult,
    BenchmarkSummary,
    CRSResult,
    Entity,
    EntityLabel,
    EntityMatch,
    ErrorType,
    MatchType,
    MedicalTerm,
    MedicalTermCategory,
    NERResult,
    SegmentCRSScore,
    SRSResult,
    TermError,
    TERResult,
    TranscriptSegment,
)

__all__ = [
    # Configuration
    "AudioConfig",
    "ChunkingConfig",
    "NetworkConfig",
    "VADConfig",
    "StreamingProfile",
    "MetricWeights",
    "TERConfig",
    "NERConfig",
    "CRSConfig",
    "EvaluationConfig",
    "BUILTIN_PROFILES",
    "load_profile",
    "load_profile_from_yaml",
    "save_profile_to_yaml",
    "list_builtin_profiles",
    "get_builtin_profile",
    # Exceptions
    "HSSTBError",
    "ConfigurationError",
    "AudioError",
    "AudioLoadError",
    "AudioFormatError",
    "STTAdapterError",
    "STTConnectionError",
    "STTTranscriptionError",
    "LexiconError",
    "LexiconLoadError",
    "LexiconLookupError",
    "MetricComputationError",
    "TERComputationError",
    "NERComputationError",
    "CRSComputationError",
    "EvaluationError",
    "BenchmarkError",
    "ReportGenerationError",
    # Enums
    "EntityLabel",
    "MedicalTermCategory",
    "MatchType",
    "ErrorType",
    # Audio types
    "AudioChunk",
    # Transcript types
    "TranscriptSegment",
    # Entity types
    "Entity",
    "MedicalTerm",
    # Error/Match types
    "TermError",
    "EntityMatch",
    # Result types
    "TERResult",
    "NERResult",
    "CRSResult",
    "SRSResult",
    "SegmentCRSScore",
    # Benchmark types
    "BenchmarkResult",
    "BenchmarkSummary",
]
