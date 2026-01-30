"""
Tests for hsttb.core.types module.

These tests verify that all core types are properly validated
and behave correctly.
"""
from __future__ import annotations

import pytest

from hsttb.core.types import (
    AudioChunk,
    CRSResult,
    Entity,
    EntityLabel,
    EntityMatch,
    ErrorType,
    MatchType,
    MedicalTerm,
    MedicalTermCategory,
    NERResult,
    TermError,
    TERResult,
    TranscriptSegment,
)

# ==============================================================================
# AudioChunk Tests
# ==============================================================================


class TestAudioChunk:
    """Tests for AudioChunk dataclass."""

    def test_valid_chunk_creation(self) -> None:
        """Should create AudioChunk with valid data."""
        chunk = AudioChunk(
            data=b"\x00\x01\x02",
            sequence_id=0,
            timestamp_ms=0,
            duration_ms=1000,
            is_final=False,
        )
        assert chunk.data == b"\x00\x01\x02"
        assert chunk.sequence_id == 0
        assert chunk.timestamp_ms == 0
        assert chunk.duration_ms == 1000
        assert chunk.is_final is False

    def test_end_timestamp_property(self) -> None:
        """Should calculate end timestamp correctly."""
        chunk = AudioChunk(
            data=b"",
            sequence_id=0,
            timestamp_ms=1000,
            duration_ms=500,
        )
        assert chunk.end_timestamp_ms == 1500

    def test_negative_sequence_id_raises_error(self) -> None:
        """Should raise ValueError for negative sequence_id."""
        with pytest.raises(ValueError, match="sequence_id must be non-negative"):
            AudioChunk(
                data=b"",
                sequence_id=-1,
                timestamp_ms=0,
                duration_ms=1000,
            )

    def test_negative_timestamp_raises_error(self) -> None:
        """Should raise ValueError for negative timestamp."""
        with pytest.raises(ValueError, match="timestamp_ms must be non-negative"):
            AudioChunk(
                data=b"",
                sequence_id=0,
                timestamp_ms=-1,
                duration_ms=1000,
            )

    def test_zero_duration_raises_error(self) -> None:
        """Should raise ValueError for zero duration."""
        with pytest.raises(ValueError, match="duration_ms must be positive"):
            AudioChunk(
                data=b"",
                sequence_id=0,
                timestamp_ms=0,
                duration_ms=0,
            )

    def test_chunk_is_frozen(self) -> None:
        """Should be immutable (frozen)."""
        chunk = AudioChunk(
            data=b"",
            sequence_id=0,
            timestamp_ms=0,
            duration_ms=1000,
        )
        with pytest.raises(AttributeError):
            chunk.sequence_id = 1  # type: ignore


# ==============================================================================
# TranscriptSegment Tests
# ==============================================================================


class TestTranscriptSegment:
    """Tests for TranscriptSegment dataclass."""

    def test_valid_segment_creation(self) -> None:
        """Should create TranscriptSegment with valid data."""
        segment = TranscriptSegment(
            text="Patient reports chest pain",
            is_partial=False,
            is_final=True,
            confidence=0.95,
            start_time_ms=0,
            end_time_ms=3000,
        )
        assert segment.text == "Patient reports chest pain"
        assert segment.is_final is True
        assert segment.confidence == 0.95

    def test_duration_property(self) -> None:
        """Should calculate duration correctly."""
        segment = TranscriptSegment(
            text="test",
            is_partial=False,
            is_final=True,
            confidence=0.9,
            start_time_ms=1000,
            end_time_ms=3000,
        )
        assert segment.duration_ms == 2000

    def test_confidence_below_zero_raises_error(self) -> None:
        """Should raise ValueError for confidence < 0."""
        with pytest.raises(ValueError, match="confidence must be between"):
            TranscriptSegment(
                text="test",
                is_partial=False,
                is_final=True,
                confidence=-0.1,
                start_time_ms=0,
                end_time_ms=1000,
            )

    def test_confidence_above_one_raises_error(self) -> None:
        """Should raise ValueError for confidence > 1."""
        with pytest.raises(ValueError, match="confidence must be between"):
            TranscriptSegment(
                text="test",
                is_partial=False,
                is_final=True,
                confidence=1.1,
                start_time_ms=0,
                end_time_ms=1000,
            )

    def test_end_before_start_raises_error(self) -> None:
        """Should raise ValueError when end_time < start_time."""
        with pytest.raises(ValueError, match="end_time_ms must be >= start_time_ms"):
            TranscriptSegment(
                text="test",
                is_partial=False,
                is_final=True,
                confidence=0.9,
                start_time_ms=2000,
                end_time_ms=1000,
            )


# ==============================================================================
# Entity Tests
# ==============================================================================


class TestEntity:
    """Tests for Entity dataclass."""

    def test_valid_entity_creation(self) -> None:
        """Should create Entity with valid data."""
        entity = Entity(
            text="metformin",
            label=EntityLabel.DRUG,
            span=(15, 24),
            normalized="metformin",
            negated=False,
        )
        assert entity.text == "metformin"
        assert entity.label == EntityLabel.DRUG
        assert entity.span == (15, 24)

    def test_span_length_property(self) -> None:
        """Should calculate span length correctly."""
        entity = Entity(
            text="metformin",
            label=EntityLabel.DRUG,
            span=(15, 24),
        )
        assert entity.span_length == 9

    def test_empty_text_raises_error(self) -> None:
        """Should raise ValueError for empty text."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            Entity(
                text="",
                label=EntityLabel.DRUG,
                span=(0, 0),
            )

    def test_negative_span_start_raises_error(self) -> None:
        """Should raise ValueError for negative span start."""
        with pytest.raises(ValueError, match="span start must be non-negative"):
            Entity(
                text="test",
                label=EntityLabel.DRUG,
                span=(-1, 5),
            )

    def test_invalid_span_order_raises_error(self) -> None:
        """Should raise ValueError when span end < start."""
        with pytest.raises(ValueError, match="span end must be >= start"):
            Entity(
                text="test",
                label=EntityLabel.DRUG,
                span=(10, 5),
            )


# ==============================================================================
# MedicalTerm Tests
# ==============================================================================


class TestMedicalTerm:
    """Tests for MedicalTerm dataclass."""

    def test_valid_term_creation(self) -> None:
        """Should create MedicalTerm with valid data."""
        term = MedicalTerm(
            text="Metformin",
            normalized="metformin",
            category=MedicalTermCategory.DRUG,
            source="rxnorm",
            span=(15, 24),
            code="6809",
        )
        assert term.text == "Metformin"
        assert term.normalized == "metformin"
        assert term.category == MedicalTermCategory.DRUG

    def test_empty_text_raises_error(self) -> None:
        """Should raise ValueError for empty text."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            MedicalTerm(
                text="",
                normalized="test",
                category=MedicalTermCategory.DRUG,
                source="rxnorm",
                span=(0, 0),
            )

    def test_empty_normalized_raises_error(self) -> None:
        """Should raise ValueError for empty normalized."""
        with pytest.raises(ValueError, match="normalized cannot be empty"):
            MedicalTerm(
                text="test",
                normalized="",
                category=MedicalTermCategory.DRUG,
                source="rxnorm",
                span=(0, 4),
            )

    def test_empty_source_raises_error(self) -> None:
        """Should raise ValueError for empty source."""
        with pytest.raises(ValueError, match="source cannot be empty"):
            MedicalTerm(
                text="test",
                normalized="test",
                category=MedicalTermCategory.DRUG,
                source="",
                span=(0, 4),
            )


# ==============================================================================
# TermError Tests
# ==============================================================================


class TestTermError:
    """Tests for TermError dataclass."""

    def test_valid_substitution_error(self) -> None:
        """Should create valid substitution error."""
        gt_term = MedicalTerm(
            text="metformin",
            normalized="metformin",
            category=MedicalTermCategory.DRUG,
            source="rxnorm",
            span=(0, 9),
        )
        pred_term = MedicalTerm(
            text="methotrexate",
            normalized="methotrexate",
            category=MedicalTermCategory.DRUG,
            source="rxnorm",
            span=(0, 12),
        )
        error = TermError(
            error_type=ErrorType.SUBSTITUTION,
            category=MedicalTermCategory.DRUG,
            ground_truth_term=gt_term,
            predicted_term=pred_term,
            similarity_score=0.72,
        )
        assert error.error_type == ErrorType.SUBSTITUTION

    def test_substitution_without_gt_raises_error(self) -> None:
        """Should raise ValueError for substitution without ground_truth."""
        pred_term = MedicalTerm(
            text="test",
            normalized="test",
            category=MedicalTermCategory.DRUG,
            source="rxnorm",
            span=(0, 4),
        )
        with pytest.raises(ValueError, match="Substitution requires both"):
            TermError(
                error_type=ErrorType.SUBSTITUTION,
                category=MedicalTermCategory.DRUG,
                ground_truth_term=None,
                predicted_term=pred_term,
            )

    def test_deletion_without_gt_raises_error(self) -> None:
        """Should raise ValueError for deletion without ground_truth."""
        with pytest.raises(ValueError, match="Deletion requires ground_truth_term"):
            TermError(
                error_type=ErrorType.DELETION,
                category=MedicalTermCategory.DRUG,
                ground_truth_term=None,
            )

    def test_insertion_without_pred_raises_error(self) -> None:
        """Should raise ValueError for insertion without predicted."""
        with pytest.raises(ValueError, match="Insertion requires predicted_term"):
            TermError(
                error_type=ErrorType.INSERTION,
                category=MedicalTermCategory.DRUG,
                predicted_term=None,
            )


# ==============================================================================
# EntityMatch Tests
# ==============================================================================


class TestEntityMatch:
    """Tests for EntityMatch dataclass."""

    def test_valid_exact_match(self) -> None:
        """Should create valid exact match."""
        gt = Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))
        pred = Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))
        match = EntityMatch(
            ground_truth=gt,
            predicted=pred,
            match_type=MatchType.EXACT,
            similarity=1.0,
        )
        assert match.match_type == MatchType.EXACT

    def test_missing_match_without_gt_raises_error(self) -> None:
        """Should raise ValueError for missing match without ground_truth."""
        pred = Entity(text="test", label=EntityLabel.DRUG, span=(0, 4))
        with pytest.raises(ValueError, match="Missing match requires ground_truth"):
            EntityMatch(
                ground_truth=None,
                predicted=pred,
                match_type=MatchType.MISSING,
            )

    def test_hallucinated_match_without_pred_raises_error(self) -> None:
        """Should raise ValueError for hallucinated match without predicted."""
        gt = Entity(text="test", label=EntityLabel.DRUG, span=(0, 4))
        with pytest.raises(ValueError, match="Hallucinated match requires predicted"):
            EntityMatch(
                ground_truth=gt,
                predicted=None,
                match_type=MatchType.HALLUCINATED,
            )


# ==============================================================================
# Result Types Tests
# ==============================================================================


class TestTERResult:
    """Tests for TERResult dataclass."""

    def test_total_errors_property(self) -> None:
        """Should calculate total errors correctly."""
        gt_term = MedicalTerm(
            text="test",
            normalized="test",
            category=MedicalTermCategory.DRUG,
            source="rxnorm",
            span=(0, 4),
        )
        pred_term = MedicalTerm(
            text="test2",
            normalized="test2",
            category=MedicalTermCategory.DRUG,
            source="rxnorm",
            span=(0, 5),
        )
        result = TERResult(
            overall_ter=0.5,
            category_ter={"drug": 0.5},
            total_terms=4,
            substitutions=[
                TermError(
                    error_type=ErrorType.SUBSTITUTION,
                    category=MedicalTermCategory.DRUG,
                    ground_truth_term=gt_term,
                    predicted_term=pred_term,
                )
            ],
            deletions=[
                TermError(
                    error_type=ErrorType.DELETION,
                    category=MedicalTermCategory.DRUG,
                    ground_truth_term=gt_term,
                )
            ],
            insertions=[],
        )
        assert result.total_errors == 2
        assert result.substitution_count == 1
        assert result.deletion_count == 1
        assert result.insertion_count == 0


class TestNERResult:
    """Tests for NERResult dataclass."""

    def test_valid_result_creation(self) -> None:
        """Should create valid NER result."""
        result = NERResult(
            precision=0.9,
            recall=0.85,
            f1_score=0.87,
            entity_distortion_rate=0.05,
            entity_omission_rate=0.1,
        )
        assert result.precision == 0.9
        assert result.f1_score == 0.87

    def test_invalid_precision_raises_error(self) -> None:
        """Should raise ValueError for invalid precision."""
        with pytest.raises(ValueError, match="precision must be between"):
            NERResult(
                precision=1.5,
                recall=0.85,
                f1_score=0.87,
                entity_distortion_rate=0.05,
                entity_omission_rate=0.1,
            )


class TestCRSResult:
    """Tests for CRSResult dataclass."""

    def test_valid_result_creation(self) -> None:
        """Should create valid CRS result."""
        result = CRSResult(
            composite_score=0.88,
            semantic_similarity=0.92,
            entity_continuity=0.85,
            negation_consistency=0.90,
            context_drift_rate=0.05,
        )
        assert result.composite_score == 0.88

    def test_invalid_composite_score_raises_error(self) -> None:
        """Should raise ValueError for invalid composite score."""
        with pytest.raises(ValueError, match="composite_score must be between"):
            CRSResult(
                composite_score=1.5,
                semantic_similarity=0.92,
                entity_continuity=0.85,
                negation_consistency=0.90,
                context_drift_rate=0.05,
            )


# ==============================================================================
# Enum Tests
# ==============================================================================


class TestEnums:
    """Tests for enum types."""

    def test_entity_label_values(self) -> None:
        """Should have expected entity label values."""
        assert EntityLabel.DRUG.value == "DRUG"
        assert EntityLabel.DOSAGE.value == "DOSAGE"
        assert EntityLabel.DIAGNOSIS.value == "DIAGNOSIS"

    def test_medical_term_category_values(self) -> None:
        """Should have expected term category values."""
        assert MedicalTermCategory.DRUG.value == "drug"
        assert MedicalTermCategory.DIAGNOSIS.value == "diagnosis"
        assert MedicalTermCategory.DOSAGE.value == "dosage"

    def test_match_type_values(self) -> None:
        """Should have expected match type values."""
        assert MatchType.EXACT.value == "exact"
        assert MatchType.MISSING.value == "missing"
        assert MatchType.HALLUCINATED.value == "hallucinated"

    def test_error_type_values(self) -> None:
        """Should have expected error type values."""
        assert ErrorType.SUBSTITUTION.value == "substitution"
        assert ErrorType.DELETION.value == "deletion"
        assert ErrorType.INSERTION.value == "insertion"
