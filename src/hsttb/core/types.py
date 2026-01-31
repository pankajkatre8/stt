"""
Core type definitions for HSTTB.

This module defines all data types used throughout the framework.
All types are designed to be immutable where appropriate and validated.

Types are organized into:
- Enums: Entity labels, term categories, match types
- Audio types: AudioChunk for streaming audio
- Transcript types: TranscriptSegment for STT output
- Entity types: Entity, MedicalTerm for NLP
- Result types: TERResult, NERResult, CRSResult for metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ==============================================================================
# Enums
# ==============================================================================


class EntityLabel(str, Enum):
    """
    Labels for medical entities.

    These labels classify the type of medical information
    represented by an entity extracted from text.
    """

    DRUG = "DRUG"
    DOSAGE = "DOSAGE"
    SYMPTOM = "SYMPTOM"
    DIAGNOSIS = "DIAGNOSIS"
    CONDITION = "CONDITION"  # Diseases/disorders
    ANATOMY = "ANATOMY"
    LAB_VALUE = "LAB_VALUE"
    PROCEDURE = "PROCEDURE"
    FREQUENCY = "FREQUENCY"
    DURATION = "DURATION"
    DATE = "DATE"
    TIME = "TIME"
    OTHER = "OTHER"


class MedicalTermCategory(str, Enum):
    """
    Categories for medical terms.

    Used for categorizing terms from medical lexicons
    and for category-wise TER computation.
    """

    DRUG = "drug"
    DIAGNOSIS = "diagnosis"
    DOSAGE = "dosage"
    ANATOMY = "anatomy"
    PROCEDURE = "procedure"
    SYMPTOM = "symptom"
    LAB = "lab"
    OTHER = "other"


class MatchType(str, Enum):
    """
    Types of entity/term matches in evaluation.

    Used to classify how well a predicted entity matches
    the ground truth entity.
    """

    EXACT = "exact"
    PARTIAL = "partial"
    DISTORTED = "distorted"
    MISSING = "missing"
    HALLUCINATED = "hallucinated"


class ErrorType(str, Enum):
    """
    Types of term errors in TER computation.
    """

    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"


# ==============================================================================
# Audio Types
# ==============================================================================


@dataclass(frozen=True)
class AudioChunk:
    """
    A chunk of audio data for streaming processing.

    This immutable type represents a segment of audio in a stream.
    Used for streaming simulation and STT adapter communication.

    Attributes:
        data: Raw audio bytes (typically PCM format).
        sequence_id: Sequential identifier for ordering chunks.
        timestamp_ms: Start time of this chunk in milliseconds.
        duration_ms: Duration of this chunk in milliseconds.
        is_final: Whether this is the last chunk in the stream.

    Example:
        >>> chunk = AudioChunk(
        ...     data=audio_bytes,
        ...     sequence_id=0,
        ...     timestamp_ms=0,
        ...     duration_ms=1000,
        ...     is_final=False
        ... )
    """

    data: bytes
    sequence_id: int
    timestamp_ms: int
    duration_ms: int
    is_final: bool = False

    def __post_init__(self) -> None:
        """Validate chunk data."""
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")

    @property
    def end_timestamp_ms(self) -> int:
        """End timestamp of this chunk."""
        return self.timestamp_ms + self.duration_ms


# ==============================================================================
# Transcript Types
# ==============================================================================


@dataclass(frozen=True)
class TranscriptSegment:
    """
    A segment of transcribed text from STT.

    Represents output from an STT system, which may be partial
    (in-progress hypothesis) or final (committed result).

    Attributes:
        text: The transcribed text content.
        is_partial: Whether this is a partial (in-progress) hypothesis.
        is_final: Whether this is a final (committed) result.
        confidence: Confidence score from 0.0 to 1.0.
        start_time_ms: Start time in the audio (milliseconds).
        end_time_ms: End time in the audio (milliseconds).
        word_timestamps: Optional per-word timing information.

    Example:
        >>> segment = TranscriptSegment(
        ...     text="Patient reports chest pain",
        ...     is_partial=False,
        ...     is_final=True,
        ...     confidence=0.95,
        ...     start_time_ms=0,
        ...     end_time_ms=3000
        ... )
    """

    text: str
    is_partial: bool
    is_final: bool
    confidence: float
    start_time_ms: int
    end_time_ms: int
    word_timestamps: tuple[dict[str, object], ...] | None = None

    def __post_init__(self) -> None:
        """Validate segment data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.start_time_ms < 0:
            raise ValueError("start_time_ms must be non-negative")
        if self.end_time_ms < self.start_time_ms:
            raise ValueError("end_time_ms must be >= start_time_ms")

    @property
    def duration_ms(self) -> int:
        """Duration of this segment in milliseconds."""
        return self.end_time_ms - self.start_time_ms


# ==============================================================================
# Entity Types
# ==============================================================================


@dataclass(frozen=True)
class Entity:
    """
    A medical entity extracted from text.

    Represents a span of text that has been identified as
    a medical entity (drug, diagnosis, symptom, etc.).

    Attributes:
        text: The entity text as it appears in the source.
        label: The entity type/classification.
        span: Character offsets (start, end) in source text.
        normalized: Normalized/canonical form of the entity.
        negated: Whether the entity is negated in context.
        confidence: Extraction confidence score.

    Example:
        >>> entity = Entity(
        ...     text="metformin",
        ...     label=EntityLabel.DRUG,
        ...     span=(15, 24),
        ...     normalized="metformin",
        ...     negated=False
        ... )
    """

    text: str
    label: EntityLabel
    span: tuple[int, int]
    normalized: str | None = None
    negated: bool = False
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate entity data."""
        if not self.text:
            raise ValueError("text cannot be empty")
        start, end = self.span
        if start < 0:
            raise ValueError("span start must be non-negative")
        if end < start:
            raise ValueError("span end must be >= start")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

    @property
    def span_length(self) -> int:
        """Length of the entity span."""
        return self.span[1] - self.span[0]


@dataclass(frozen=True)
class MedicalTerm:
    """
    A medical term identified from text.

    Represents a term that has been matched against medical
    lexicons (RxNorm, SNOMED CT, ICD-10, etc.).

    Attributes:
        text: The term text as it appears in the source.
        normalized: Normalized form for matching/comparison.
        category: The term category (drug, diagnosis, etc.).
        source: Source lexicon (rxnorm, snomed, icd10, etc.).
        span: Character offsets (start, end) in source text.
        code: Optional code from the source lexicon.

    Example:
        >>> term = MedicalTerm(
        ...     text="Metformin",
        ...     normalized="metformin",
        ...     category=MedicalTermCategory.DRUG,
        ...     source="rxnorm",
        ...     span=(15, 24),
        ...     code="6809"
        ... )
    """

    text: str
    normalized: str
    category: MedicalTermCategory
    source: str
    span: tuple[int, int]
    code: str | None = None

    def __post_init__(self) -> None:
        """Validate term data."""
        if not self.text:
            raise ValueError("text cannot be empty")
        if not self.normalized:
            raise ValueError("normalized cannot be empty")
        if not self.source:
            raise ValueError("source cannot be empty")
        start, end = self.span
        if start < 0:
            raise ValueError("span start must be non-negative")
        if end < start:
            raise ValueError("span end must be >= start")


# ==============================================================================
# Error Types (for TER)
# ==============================================================================


@dataclass(frozen=True)
class TermError:
    """
    A single term error detected during TER computation.

    Represents a substitution, deletion, or insertion of a
    medical term between ground truth and prediction.

    Attributes:
        error_type: Type of error (substitution/deletion/insertion).
        category: Category of the affected term.
        ground_truth_term: The term from ground truth (None for insertions).
        predicted_term: The term from prediction (None for deletions).
        similarity_score: Similarity between terms (for substitutions).
    """

    error_type: ErrorType
    category: MedicalTermCategory
    ground_truth_term: MedicalTerm | None = None
    predicted_term: MedicalTerm | None = None
    similarity_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate error data."""
        if self.error_type == ErrorType.SUBSTITUTION:
            if self.ground_truth_term is None or self.predicted_term is None:
                raise ValueError(
                    "Substitution requires both ground_truth_term and predicted_term"
                )
        elif self.error_type == ErrorType.DELETION:
            if self.ground_truth_term is None:
                raise ValueError("Deletion requires ground_truth_term")
        elif self.error_type == ErrorType.INSERTION and self.predicted_term is None:
            raise ValueError("Insertion requires predicted_term")


# ==============================================================================
# Match Types (for NER)
# ==============================================================================


@dataclass(frozen=True)
class EntityMatch:
    """
    Result of matching a ground truth entity to a predicted entity.

    Attributes:
        ground_truth: The entity from ground truth (None if hallucinated).
        predicted: The entity from prediction (None if missing).
        match_type: Classification of the match quality.
        similarity: Similarity score between entities.
    """

    ground_truth: Entity | None
    predicted: Entity | None
    match_type: MatchType
    similarity: float = 0.0

    def __post_init__(self) -> None:
        """Validate match data."""
        if self.match_type == MatchType.MISSING and self.ground_truth is None:
            raise ValueError("Missing match requires ground_truth")
        if self.match_type == MatchType.HALLUCINATED and self.predicted is None:
            raise ValueError("Hallucinated match requires predicted")


# ==============================================================================
# Result Types
# ==============================================================================


@dataclass
class TERResult:
    """
    Result of Term Error Rate computation.

    Contains overall TER, category-wise breakdown, and detailed
    error information for analysis.

    Attributes:
        overall_ter: Overall term error rate (0.0 to 1.0+).
        category_ter: TER broken down by term category.
        total_terms: Total number of medical terms in ground truth.
        substitutions: List of substitution errors.
        deletions: List of deletion errors.
        insertions: List of insertion errors.
    """

    overall_ter: float
    category_ter: dict[str, float]
    total_terms: int
    substitutions: list[TermError] = field(default_factory=list)
    deletions: list[TermError] = field(default_factory=list)
    insertions: list[TermError] = field(default_factory=list)

    @property
    def total_errors(self) -> int:
        """Total number of errors."""
        return len(self.substitutions) + len(self.deletions) + len(self.insertions)

    @property
    def substitution_count(self) -> int:
        """Number of substitution errors."""
        return len(self.substitutions)

    @property
    def deletion_count(self) -> int:
        """Number of deletion errors."""
        return len(self.deletions)

    @property
    def insertion_count(self) -> int:
        """Number of insertion errors."""
        return len(self.insertions)


@dataclass
class NERResult:
    """
    Result of NER accuracy computation.

    Contains precision, recall, F1 score, and detailed
    match information for analysis.

    Attributes:
        precision: Precision score (0.0 to 1.0).
        recall: Recall score (0.0 to 1.0).
        f1_score: F1 score (0.0 to 1.0).
        entity_distortion_rate: Rate of distorted entities.
        entity_omission_rate: Rate of omitted entities.
        matches: List of entity matches for detailed analysis.
        per_type_metrics: Metrics broken down by entity type.
    """

    precision: float
    recall: float
    f1_score: float
    entity_distortion_rate: float
    entity_omission_rate: float
    matches: list[EntityMatch] = field(default_factory=list)
    per_type_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result data."""
        for name, value in [
            ("precision", self.precision),
            ("recall", self.recall),
            ("f1_score", self.f1_score),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")


@dataclass
class SegmentCRSScore:
    """
    CRS score for a single segment.

    Attributes:
        segment_id: Identifier for this segment.
        ground_truth_text: Ground truth text for this segment.
        predicted_text: Predicted text for this segment.
        semantic_similarity: Semantic similarity score.
        entities_preserved: Number of entities preserved.
        entities_lost: Number of entities lost.
        negation_flips: Number of negation flips detected.
    """

    segment_id: int
    ground_truth_text: str
    predicted_text: str
    semantic_similarity: float
    entities_preserved: int = 0
    entities_lost: int = 0
    negation_flips: int = 0


@dataclass
class CRSResult:
    """
    Result of Context Retention Score computation.

    Contains composite score and component scores for
    semantic similarity, entity continuity, and negation consistency.

    Attributes:
        composite_score: Weighted composite CRS (0.0 to 1.0).
        semantic_similarity: Average semantic similarity score.
        entity_continuity: Entity continuity score.
        negation_consistency: Negation consistency score.
        context_drift_rate: Rate of context drift across segments.
        segment_scores: Per-segment scores for detailed analysis.
    """

    composite_score: float
    semantic_similarity: float
    entity_continuity: float
    negation_consistency: float
    context_drift_rate: float
    segment_scores: list[SegmentCRSScore] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate result data."""
        for name, value in [
            ("composite_score", self.composite_score),
            ("semantic_similarity", self.semantic_similarity),
            ("entity_continuity", self.entity_continuity),
            ("negation_consistency", self.negation_consistency),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")


@dataclass
class SRSResult:
    """
    Result of Streaming Robustness Score computation.

    Compares model performance under different streaming conditions
    to identify whether quality degradation is due to the model
    or the streaming conditions.

    Attributes:
        model_name: Name of the evaluated model.
        ideal_scores: Scores under ideal streaming conditions.
        realtime_scores: Scores under realtime streaming conditions.
        srs: Streaming Robustness Score (realtime/ideal ratio).
        degradation: Per-metric degradation breakdown.
    """

    model_name: str
    ideal_scores: dict[str, float]
    realtime_scores: dict[str, float]
    srs: float
    degradation: dict[str, float] = field(default_factory=dict)


# ==============================================================================
# Benchmark Result Types
# ==============================================================================


@dataclass
class BenchmarkResult:
    """
    Result of benchmarking a single audio file.

    Attributes:
        audio_id: Identifier for the audio file.
        ter: TER computation result.
        ner: NER computation result.
        crs: CRS computation result.
        transcript_ground_truth: Ground truth transcript.
        transcript_predicted: Predicted transcript.
        streaming_profile: Name of streaming profile used.
        adapter_name: Name of STT adapter used.
    """

    audio_id: str
    ter: TERResult
    ner: NERResult
    crs: CRSResult
    transcript_ground_truth: str
    transcript_predicted: str
    streaming_profile: str
    adapter_name: str


@dataclass
class BenchmarkSummary:
    """
    Summary of a complete benchmark run.

    Attributes:
        total_files: Number of files processed.
        avg_ter: Average TER across all files.
        avg_ner_f1: Average NER F1 score.
        avg_crs: Average CRS score.
        results: Individual results per file.
        streaming_profile: Streaming profile used.
        adapter_name: STT adapter used.
    """

    total_files: int
    avg_ter: float
    avg_ner_f1: float
    avg_crs: float
    results: list[BenchmarkResult] = field(default_factory=list)
    streaming_profile: str = ""
    adapter_name: str = ""
