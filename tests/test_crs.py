"""
Tests for Context Retention Score (CRS) computation engine.

Tests the CRS engine's ability to compute context retention
across streaming segments.
"""
from __future__ import annotations

import pytest

from hsttb.core.types import Entity, EntityLabel
from hsttb.metrics import (
    CRSConfig,
    CRSEngine,
    EntityContinuityTracker,
    TokenBasedSimilarity,
    compute_crs,
    compute_entity_continuity,
    compute_semantic_similarity,
)
from hsttb.nlp import NegationDetector, check_negation_consistency

# ==============================================================================
# Semantic Similarity Tests
# ==============================================================================


class TestTokenBasedSimilarity:
    """Tests for TokenBasedSimilarity engine."""

    @pytest.fixture
    def engine(self) -> TokenBasedSimilarity:
        """Create similarity engine."""
        return TokenBasedSimilarity()

    def test_identical_texts(self, engine: TokenBasedSimilarity) -> None:
        """Identical texts have similarity 1.0."""
        score = engine.compute_similarity(
            "patient has diabetes",
            "patient has diabetes",
        )
        assert score == 1.0

    def test_empty_texts(self, engine: TokenBasedSimilarity) -> None:
        """Empty texts have similarity 1.0."""
        score = engine.compute_similarity("", "")
        assert score == 1.0

    def test_one_empty(self, engine: TokenBasedSimilarity) -> None:
        """One empty text has similarity 0.0."""
        score = engine.compute_similarity("patient has diabetes", "")
        assert score == 0.0

    def test_completely_different(self, engine: TokenBasedSimilarity) -> None:
        """Completely different texts have low similarity."""
        score = engine.compute_similarity("abc def ghi", "xyz uvw rst")
        assert score < 0.5

    def test_similar_texts(self, engine: TokenBasedSimilarity) -> None:
        """Similar texts have high similarity."""
        score = engine.compute_similarity(
            "patient has diabetes",
            "patient has diabetic condition",
        )
        assert score > 0.3

    def test_segment_similarities(self, engine: TokenBasedSimilarity) -> None:
        """Compute similarities for multiple segments."""
        gt_segments = ["hello world", "foo bar"]
        pred_segments = ["hello world", "foo baz"]

        scores = engine.compute_segment_similarities(gt_segments, pred_segments)

        assert len(scores) == 2
        assert scores[0] == 1.0  # Identical
        assert 0 < scores[1] < 1  # Partially similar

    def test_average_similarity(self, engine: TokenBasedSimilarity) -> None:
        """Compute average similarity."""
        gt = ["hello", "world"]
        pred = ["hello", "world"]

        avg = engine.compute_average_similarity(gt, pred)
        assert avg == 1.0


class TestSemanticSimilarityFunction:
    """Tests for compute_semantic_similarity function."""

    def test_convenience_function(self) -> None:
        """compute_semantic_similarity returns score."""
        score = compute_semantic_similarity("hello world", "hello world")
        assert score == 1.0


# ==============================================================================
# Entity Continuity Tests
# ==============================================================================


class TestEntityContinuityTracker:
    """Tests for EntityContinuityTracker."""

    @pytest.fixture
    def tracker(self) -> EntityContinuityTracker:
        """Create continuity tracker."""
        return EntityContinuityTracker()

    @pytest.fixture
    def sample_entity(self) -> Entity:
        """Create sample entity."""
        return Entity(
            text="metformin",
            label=EntityLabel.DRUG,
            span=(0, 9),
            normalized="metformin",
        )

    def test_perfect_continuity(
        self, tracker: EntityContinuityTracker, sample_entity: Entity
    ) -> None:
        """Perfect entity continuity when all entities match."""
        gt_segments = ["patient takes metformin", "continues metformin"]
        gt_entities: list[list[Entity]] = [[sample_entity], [sample_entity]]
        pred_segments = gt_segments.copy()
        pred_entities: list[list[Entity]] = [[sample_entity], [sample_entity]]

        result = tracker.track(gt_segments, gt_entities, pred_segments, pred_entities)

        assert result.continuity_score == 1.0
        assert result.discontinuity_count == 0

    def test_missing_entity(
        self, tracker: EntityContinuityTracker, sample_entity: Entity
    ) -> None:
        """Detect missing entity in prediction."""
        gt_segments = ["patient takes metformin", "continues metformin"]
        gt_entities: list[list[Entity]] = [[sample_entity], [sample_entity]]
        pred_segments = gt_segments.copy()
        pred_entities: list[list[Entity]] = [[sample_entity], []]  # Missing in seg 2

        result = tracker.track(gt_segments, gt_entities, pred_segments, pred_entities)

        assert result.continuity_score < 1.0
        assert result.has_discontinuities

    def test_empty_segments(self, tracker: EntityContinuityTracker) -> None:
        """Handle empty segments."""
        result = tracker.track([], [], [], [])
        assert result.continuity_score == 1.0

    def test_entity_preservation_rate(
        self, tracker: EntityContinuityTracker, sample_entity: Entity
    ) -> None:
        """Compute entity preservation rate."""
        gt_entities: list[list[Entity]] = [[sample_entity], [sample_entity]]
        pred_entities: list[list[Entity]] = [[sample_entity], []]

        rate = tracker.compute_entity_preservation_rate(gt_entities, pred_entities)
        assert rate == 0.5  # 1 out of 2 preserved


class TestComputeEntityContinuity:
    """Tests for compute_entity_continuity function."""

    def test_convenience_function(self) -> None:
        """compute_entity_continuity returns score."""
        score = compute_entity_continuity(
            ["segment 1"], [[]],
            ["segment 1"], [[]],
        )
        assert score == 1.0


# ==============================================================================
# Negation Detection Tests
# ==============================================================================


class TestNegationDetector:
    """Tests for NegationDetector."""

    @pytest.fixture
    def detector(self) -> NegationDetector:
        """Create negation detector."""
        return NegationDetector()

    def test_detect_denies(self, detector: NegationDetector) -> None:
        """Detect 'denies' negation."""
        negations = detector.detect_negations("patient denies chest pain")
        assert len(negations) >= 1

    def test_detect_no(self, detector: NegationDetector) -> None:
        """Detect 'no' negation."""
        negations = detector.detect_negations("no chest pain")
        assert len(negations) >= 1

    def test_detect_without(self, detector: NegationDetector) -> None:
        """Detect 'without' negation."""
        negations = detector.detect_negations("patient is without fever")
        assert len(negations) >= 1

    def test_no_negation(self, detector: NegationDetector) -> None:
        """No negation detected in affirmative text."""
        # Just ensure no errors when processing affirmative text
        _ = detector.detect_negations("patient has chest pain")

    def test_is_negated_true(self, detector: NegationDetector) -> None:
        """is_negated returns True for negated span."""
        text = "no chest pain"
        # "chest pain" starts after "no "
        is_neg = detector.is_negated(text, (3, 13))
        assert is_neg is True

    def test_negation_consistency_check(self, detector: NegationDetector) -> None:
        """Check negation consistency between GT and pred."""
        result = detector.check_negation_consistency(
            gt_text="patient denies chest pain",
            pred_text="patient has chest pain",
            entity_text="chest pain",
        )
        # GT has negation, pred does not
        assert result["consistent"] is False
        assert result["flip_type"] == "negation_lost"


class TestCheckNegationConsistency:
    """Tests for check_negation_consistency function."""

    def test_all_consistent(self) -> None:
        """All entities have consistent negation."""
        result = check_negation_consistency(
            gt_text="patient has fever",
            pred_text="patient has fever",
            entities=["fever"],
        )
        assert result.consistency_score == 1.0
        assert len(result.flips) == 0

    def test_negation_flip(self) -> None:
        """Detect negation flip."""
        result = check_negation_consistency(
            gt_text="patient denies fever",
            pred_text="patient has fever",
            entities=["fever"],
        )
        assert result.consistency_score == 0.0
        assert len(result.flips) == 1


# ==============================================================================
# CRS Engine Tests
# ==============================================================================


class TestCRSEngine:
    """Tests for CRSEngine."""

    @pytest.fixture
    def engine(self) -> CRSEngine:
        """Create CRS engine."""
        return CRSEngine()

    def test_perfect_score(self, engine: CRSEngine) -> None:
        """Perfect CRS for identical segments."""
        result = engine.compute(
            gt_segments=["patient has diabetes", "takes metformin"],
            pred_segments=["patient has diabetes", "takes metformin"],
        )
        assert result.composite_score == 1.0
        assert result.semantic_similarity == 1.0

    def test_with_entities(self, engine: CRSEngine) -> None:
        """CRS with entity tracking."""
        entity = Entity(
            text="metformin",
            label=EntityLabel.DRUG,
            span=(0, 9),
        )

        result = engine.compute(
            gt_segments=["takes metformin"],
            pred_segments=["takes metformin"],
            gt_entities=[[entity]],
            pred_entities=[[entity]],
        )

        assert result.composite_score > 0.9
        assert result.entity_continuity == 1.0

    def test_missing_segment(self, engine: CRSEngine) -> None:
        """Handle missing segments."""
        result = engine.compute(
            gt_segments=["segment 1", "segment 2"],
            pred_segments=["segment 1"],
        )
        # Should handle different lengths
        assert 0 <= result.composite_score <= 1

    def test_segment_scores(self, engine: CRSEngine) -> None:
        """Per-segment scores are computed."""
        result = engine.compute(
            gt_segments=["hello", "world"],
            pred_segments=["hello", "world"],
        )

        assert len(result.segment_scores) == 2
        assert result.segment_scores[0].segment_id == 0
        assert result.segment_scores[0].semantic_similarity == 1.0

    def test_context_drift_rate(self, engine: CRSEngine) -> None:
        """Context drift rate is computed."""
        result = engine.compute(
            gt_segments=["same text"] * 3,
            pred_segments=["same text"] * 3,
        )
        assert result.context_drift_rate == 0.0

    def test_custom_config(self) -> None:
        """CRS engine with custom config."""
        config = CRSConfig(
            semantic_weight=0.5,
            entity_weight=0.3,
            negation_weight=0.2,
        )
        engine = CRSEngine(config)

        result = engine.compute(
            gt_segments=["hello"],
            pred_segments=["hello"],
        )

        assert result.composite_score == 1.0


class TestCRSConfig:
    """Tests for CRSConfig."""

    def test_default_weights(self) -> None:
        """Default weights sum to 1.0."""
        config = CRSConfig()
        total = config.semantic_weight + config.entity_weight + config.negation_weight
        assert abs(total - 1.0) < 0.001

    def test_weight_normalization(self) -> None:
        """Weights are normalized if they don't sum to 1.0."""
        config = CRSConfig(
            semantic_weight=0.5,
            entity_weight=0.5,
            negation_weight=0.5,
        )
        total = config.semantic_weight + config.entity_weight + config.negation_weight
        assert abs(total - 1.0) < 0.001


class TestComputeCRS:
    """Tests for compute_crs convenience function."""

    def test_basic_usage(self) -> None:
        """Basic usage of compute_crs."""
        score = compute_crs(
            gt_segments=["hello world"],
            pred_segments=["hello world"],
        )
        assert score == 1.0

    def test_different_segments(self) -> None:
        """Different segments have lower score."""
        score = compute_crs(
            gt_segments=["hello world"],
            pred_segments=["goodbye moon"],
        )
        assert score < 1.0


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestCRSIntegration:
    """Integration tests for CRS computation."""

    def test_full_pipeline(self) -> None:
        """Test full CRS computation pipeline."""
        # Create test data
        gt_segments = [
            "patient has diabetes mellitus",
            "takes metformin 500mg twice daily",
            "denies chest pain",
        ]
        pred_segments = [
            "patient has diabetes mellitus",
            "takes metformin 500mg twice daily",
            "denies chest pain",
        ]

        # Create entities
        diabetes = Entity(
            text="diabetes mellitus",
            label=EntityLabel.DIAGNOSIS,
            span=(12, 29),
            normalized="diabetes mellitus",
        )
        metformin = Entity(
            text="metformin",
            label=EntityLabel.DRUG,
            span=(6, 15),
            normalized="metformin",
        )
        chest_pain = Entity(
            text="chest pain",
            label=EntityLabel.SYMPTOM,
            span=(7, 17),
            normalized="chest pain",
            negated=True,
        )

        gt_entities: list[list[Entity]] = [[diabetes], [metformin], [chest_pain]]
        pred_entities: list[list[Entity]] = [[diabetes], [metformin], [chest_pain]]

        # Compute CRS
        engine = CRSEngine()
        result = engine.compute(
            gt_segments=gt_segments,
            pred_segments=pred_segments,
            gt_entities=gt_entities,
            pred_entities=pred_entities,
        )

        # Verify results
        assert result.composite_score == 1.0
        assert result.semantic_similarity == 1.0
        assert result.entity_continuity == 1.0
        assert len(result.segment_scores) == 3

    def test_degraded_transcription(self) -> None:
        """Test CRS with degraded transcription."""
        gt_segments = ["patient has diabetes", "takes metformin"]
        pred_segments = ["patient has diabtes", "taking metformine"]  # Typos

        engine = CRSEngine()
        result = engine.compute(
            gt_segments=gt_segments,
            pred_segments=pred_segments,
        )

        # Should have lower semantic similarity due to typos
        assert result.semantic_similarity < 1.0
        assert result.composite_score < 1.0
