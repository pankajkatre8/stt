# TASK-1C02: Core Types

## Metadata
- **Status**: pending
- **Complexity**: Medium (1-2 hours)
- **Blocked By**: TASK-1C01
- **Blocks**: TASK-1C03, TASK-1C04, TASK-1A01

## Objective

Define all core data types used throughout the HSTTB framework with complete type hints and validation.

## Context

These types form the contract between all components. They must be well-designed because changing them later affects many files. Healthcare accuracy depends on these types correctly representing medical data.

## Requirements

- [ ] Define `AudioChunk` dataclass for streaming audio
- [ ] Define `TranscriptSegment` dataclass for STT output
- [ ] Define `Entity` dataclass for NER results
- [ ] Define `MedicalTerm` dataclass for lexicon entries
- [ ] Define enums for entity labels and term categories
- [ ] Define result types for each metric (TER, NER, CRS)
- [ ] All types must have complete type hints
- [ ] All types must be immutable where appropriate
- [ ] Add validation in `__post_init__` where needed

## Acceptance Criteria

- [ ] AC1: All types can be instantiated with valid data
- [ ] AC2: Invalid data raises appropriate errors
- [ ] AC3: Types are properly exported from `hsttb.core.types`
- [ ] AC4: mypy --strict passes
- [ ] AC5: All types have docstrings

## Files to Create/Modify

- Create: `src/hsttb/core/types.py`
- Modify: `src/hsttb/core/__init__.py` (exports)

## Implementation Details

### src/hsttb/core/types.py

```python
"""
Core type definitions for HSTTB.

This module defines all data types used throughout the framework.
All types are designed to be immutable and validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EntityLabel(str, Enum):
    """Labels for medical entities."""
    DRUG = "DRUG"
    DOSAGE = "DOSAGE"
    SYMPTOM = "SYMPTOM"
    DIAGNOSIS = "DIAGNOSIS"
    ANATOMY = "ANATOMY"
    LAB_VALUE = "LAB_VALUE"
    PROCEDURE = "PROCEDURE"


class MedicalTermCategory(str, Enum):
    """Categories for medical terms."""
    DRUG = "drug"
    DIAGNOSIS = "diagnosis"
    DOSAGE = "dosage"
    ANATOMY = "anatomy"
    PROCEDURE = "procedure"
    SYMPTOM = "symptom"


class MatchType(str, Enum):
    """Types of entity/term matches."""
    EXACT = "exact"
    PARTIAL = "partial"
    DISTORTED = "distorted"
    MISSING = "missing"
    HALLUCINATED = "hallucinated"


@dataclass(frozen=True)
class AudioChunk:
    """
    A chunk of audio data for streaming processing.

    Attributes:
        data: Raw audio bytes (PCM format).
        sequence_id: Sequential identifier for ordering.
        timestamp_ms: Start time in milliseconds.
        duration_ms: Duration in milliseconds.
        is_final: Whether this is the last chunk.
    """
    data: bytes
    sequence_id: int
    timestamp_ms: int
    duration_ms: int
    is_final: bool = False

    def __post_init__(self) -> None:
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")


@dataclass(frozen=True)
class TranscriptSegment:
    """
    A segment of transcribed text from STT.

    Attributes:
        text: The transcribed text.
        is_partial: Whether this is a partial hypothesis.
        is_final: Whether this is a final result.
        confidence: Confidence score (0.0 to 1.0).
        start_time_ms: Start time in milliseconds.
        end_time_ms: End time in milliseconds.
        word_timestamps: Optional per-word timing info.
    """
    text: str
    is_partial: bool
    is_final: bool
    confidence: float
    start_time_ms: int
    end_time_ms: int
    word_timestamps: Optional[tuple[dict, ...]] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class Entity:
    """
    A medical entity extracted from text.

    Attributes:
        text: The entity text as it appears.
        label: The entity type/label.
        span: Character offsets (start, end).
        normalized: Normalized form of the entity.
        negated: Whether the entity is negated.
    """
    text: str
    label: EntityLabel
    span: tuple[int, int]
    normalized: Optional[str] = None
    negated: bool = False

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text cannot be empty")
        start, end = self.span
        if start < 0 or end < start:
            raise ValueError("invalid span")


@dataclass(frozen=True)
class MedicalTerm:
    """
    A medical term identified from text.

    Attributes:
        text: The term text as it appears.
        normalized: Normalized form for matching.
        category: The term category.
        source: Source lexicon (rxnorm, snomed, etc.).
        span: Character offsets (start, end).
    """
    text: str
    normalized: str
    category: MedicalTermCategory
    source: str
    span: tuple[int, int]

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text cannot be empty")
        if not self.normalized:
            raise ValueError("normalized cannot be empty")


# Result types for metrics

@dataclass(frozen=True)
class TermError:
    """A single term error (substitution, deletion, or insertion)."""
    error_type: str
    category: MedicalTermCategory
    ground_truth_term: Optional[MedicalTerm] = None
    predicted_term: Optional[MedicalTerm] = None
    similarity_score: float = 0.0


@dataclass
class TERResult:
    """Result of Term Error Rate computation."""
    overall_ter: float
    category_ter: dict[str, float]
    total_terms: int
    substitutions: list[TermError]
    deletions: list[TermError]
    insertions: list[TermError]


@dataclass
class EntityMatch:
    """Result of entity matching."""
    ground_truth: Optional[Entity]
    predicted: Optional[Entity]
    match_type: MatchType
    similarity: float


@dataclass
class NERResult:
    """Result of NER accuracy computation."""
    precision: float
    recall: float
    f1_score: float
    entity_distortion_rate: float
    entity_omission_rate: float
    matches: list[EntityMatch]


@dataclass
class SegmentCRSScore:
    """CRS score for a single segment."""
    segment_id: int
    ground_truth_text: str
    predicted_text: str
    semantic_similarity: float
    entities_preserved: int
    entities_lost: int
    negation_flips: int


@dataclass
class CRSResult:
    """Result of Context Retention Score computation."""
    composite_score: float
    semantic_similarity: float
    entity_continuity: float
    negation_consistency: float
    context_drift_rate: float
    segment_scores: list[SegmentCRSScore]
```

## Testing Requirements

- Unit tests required: Yes
- Test file: `tests/unit/core/test_types.py`
- Test cases:
  - [ ] Each type instantiates with valid data
  - [ ] Each type rejects invalid data
  - [ ] Frozen types are actually immutable
  - [ ] Enums have expected values

## Notes

- Use `frozen=True` for immutability where appropriate
- Result types are mutable (contain lists)
- Validation should be strict - fail early
- Healthcare data integrity is critical
