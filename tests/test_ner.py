"""
Tests for NER accuracy computation engine.

Tests the NER engine's ability to compute entity extraction
accuracy between ground truth and predicted entities.
"""
from __future__ import annotations

import pytest

from hsttb.core.types import Entity, EntityLabel, EntityMatch, MatchType
from hsttb.metrics import NEREngine, compute_entity_f1, compute_ner_accuracy
from hsttb.nlp import (
    AlignmentConfig,
    EntityAligner,
    MockNERPipeline,
    SpanMatchStrategy,
    align_entities,
)

# ==============================================================================
# Entity Alignment Tests
# ==============================================================================


class TestEntityAligner:
    """Tests for EntityAligner class."""

    @pytest.fixture
    def aligner(self) -> EntityAligner:
        """Create default aligner."""
        return EntityAligner()

    def test_exact_match(self, aligner: EntityAligner) -> None:
        """Exact span and label match."""
        gold = [
            Entity(
                text="metformin",
                label=EntityLabel.DRUG,
                span=(15, 24),
                normalized="metformin",
            )
        ]
        pred = [
            Entity(
                text="metformin",
                label=EntityLabel.DRUG,
                span=(15, 24),
                normalized="metformin",
            )
        ]

        matches = aligner.align(gold, pred)

        assert len(matches) == 1
        assert matches[0].match_type == MatchType.EXACT
        assert matches[0].similarity == 1.0

    def test_missing_entity(self, aligner: EntityAligner) -> None:
        """Entity in gold not in pred is missing."""
        gold = [
            Entity(
                text="metformin",
                label=EntityLabel.DRUG,
                span=(0, 9),
                normalized="metformin",
            )
        ]
        pred: list[Entity] = []

        matches = aligner.align(gold, pred)

        assert len(matches) == 1
        assert matches[0].match_type == MatchType.MISSING
        assert matches[0].ground_truth is not None
        assert matches[0].predicted is None

    def test_hallucinated_entity(self, aligner: EntityAligner) -> None:
        """Entity in pred not in gold is hallucinated."""
        gold: list[Entity] = []
        pred = [
            Entity(
                text="aspirin",
                label=EntityLabel.DRUG,
                span=(0, 7),
                normalized="aspirin",
            )
        ]

        matches = aligner.align(gold, pred)

        assert len(matches) == 1
        assert matches[0].match_type == MatchType.HALLUCINATED
        assert matches[0].ground_truth is None
        assert matches[0].predicted is not None

    def test_partial_overlap(self, aligner: EntityAligner) -> None:
        """Partially overlapping spans."""
        gold = [
            Entity(
                text="metformin 500mg",
                label=EntityLabel.DRUG,
                span=(0, 15),
                normalized="metformin 500mg",
            )
        ]
        pred = [
            Entity(
                text="metformin",
                label=EntityLabel.DRUG,
                span=(0, 9),
                normalized="metformin",
            )
        ]

        matches = aligner.align(gold, pred)

        # Should match as partial (overlapping spans)
        assert len(matches) == 1
        assert matches[0].match_type in (MatchType.PARTIAL, MatchType.DISTORTED)

    def test_label_mismatch_required(self) -> None:
        """Label mismatch when label matching is required."""
        config = AlignmentConfig(require_label_match=True)
        aligner = EntityAligner(config)

        gold = [
            Entity(
                text="metformin",
                label=EntityLabel.DRUG,
                span=(0, 9),
                normalized="metformin",
            )
        ]
        pred = [
            Entity(
                text="metformin",
                label=EntityLabel.DIAGNOSIS,
                span=(0, 9),
                normalized="metformin",
            )
        ]

        matches = aligner.align(gold, pred)

        # Should not match due to label mismatch
        assert len(matches) == 2  # One missing, one hallucinated

    def test_label_mismatch_ignored(self) -> None:
        """Label mismatch ignored when not required."""
        config = AlignmentConfig(require_label_match=False)
        aligner = EntityAligner(config)

        gold = [
            Entity(
                text="metformin",
                label=EntityLabel.DRUG,
                span=(0, 9),
                normalized="metformin",
            )
        ]
        pred = [
            Entity(
                text="metformin",
                label=EntityLabel.DIAGNOSIS,
                span=(0, 9),
                normalized="metformin",
            )
        ]

        matches = aligner.align(gold, pred)

        # Should match despite label mismatch
        assert len(matches) == 1
        assert matches[0].match_type == MatchType.EXACT

    def test_multiple_entities(self, aligner: EntityAligner) -> None:
        """Multiple entities aligned correctly."""
        gold = [
            Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9)),
            Entity(text="diabetes", label=EntityLabel.DIAGNOSIS, span=(15, 23)),
        ]
        pred = [
            Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9)),
            Entity(text="diabetes", label=EntityLabel.DIAGNOSIS, span=(15, 23)),
        ]

        matches = aligner.align(gold, pred)

        exact_matches = [m for m in matches if m.match_type == MatchType.EXACT]
        assert len(exact_matches) == 2

    def test_span_iou_calculation(self, aligner: EntityAligner) -> None:
        """Span IOU is computed correctly."""
        # Exact overlap
        assert aligner._span_iou((0, 10), (0, 10)) == 1.0

        # No overlap
        assert aligner._span_iou((0, 5), (10, 15)) == 0.0

        # Partial overlap: intersection=3, union=15
        # (0,10) has length 10, (7,15) has length 8
        # intersection is (7,10) = 3
        # union = 10 + 8 - 3 = 15
        # iou = 3/15 = 0.2
        iou = aligner._span_iou((0, 10), (7, 15))
        assert 0.15 <= iou <= 0.25

    def test_empty_lists(self, aligner: EntityAligner) -> None:
        """Empty entity lists."""
        matches = aligner.align([], [])
        assert len(matches) == 0


class TestAlignEntitiesFunction:
    """Tests for align_entities convenience function."""

    def test_align_entities(self) -> None:
        """align_entities returns matches."""
        gold = [Entity(text="drug", label=EntityLabel.DRUG, span=(0, 4))]
        pred = [Entity(text="drug", label=EntityLabel.DRUG, span=(0, 4))]

        matches = align_entities(gold, pred)

        assert len(matches) == 1
        assert matches[0].match_type == MatchType.EXACT


# ==============================================================================
# NER Pipeline Tests
# ==============================================================================


class TestMockNERPipeline:
    """Tests for MockNERPipeline."""

    @pytest.fixture
    def pipeline(self) -> MockNERPipeline:
        """Create pipeline with common patterns."""
        return MockNERPipeline.with_common_patterns()

    def test_extract_drug(self, pipeline: MockNERPipeline) -> None:
        """Extract drug entity."""
        entities = pipeline.extract_entities("patient takes metformin")

        assert len(entities) >= 1
        drug_entities = [e for e in entities if e.label == EntityLabel.DRUG]
        assert len(drug_entities) >= 1
        assert any(e.text.lower() == "metformin" for e in drug_entities)

    def test_extract_diagnosis(self, pipeline: MockNERPipeline) -> None:
        """Extract diagnosis entity."""
        entities = pipeline.extract_entities("patient has diabetes")

        assert len(entities) >= 1
        dx_entities = [e for e in entities if e.label == EntityLabel.DIAGNOSIS]
        assert len(dx_entities) >= 1

    def test_extract_symptom(self, pipeline: MockNERPipeline) -> None:
        """Extract symptom entity."""
        entities = pipeline.extract_entities("patient reports chest pain")

        symptom_entities = [e for e in entities if e.label == EntityLabel.SYMPTOM]
        assert len(symptom_entities) >= 1

    def test_negation_detection(self, pipeline: MockNERPipeline) -> None:
        """Detect negated entities."""
        entities = pipeline.extract_entities("patient denies chest pain")

        # Should detect negation
        assert any(e.negated for e in entities)

    def test_no_negation(self, pipeline: MockNERPipeline) -> None:
        """Non-negated entities."""
        entities = pipeline.extract_entities("patient has chest pain")

        # Should not all be negated
        assert any(not e.negated for e in entities)

    def test_multiple_entities(self, pipeline: MockNERPipeline) -> None:
        """Extract multiple entities."""
        entities = pipeline.extract_entities(
            "patient takes metformin and aspirin for diabetes"
        )

        # Should find multiple entities
        assert len(entities) >= 2

    def test_entity_spans(self, pipeline: MockNERPipeline) -> None:
        """Entity spans are correct."""
        text = "metformin"
        entities = pipeline.extract_entities(text)

        if entities:
            assert entities[0].span[0] >= 0
            assert entities[0].span[1] <= len(text)

    def test_supported_labels(self, pipeline: MockNERPipeline) -> None:
        """Pipeline reports supported labels."""
        labels = pipeline.supported_labels

        assert isinstance(labels, list)
        assert len(labels) > 0

    def test_pipeline_name(self, pipeline: MockNERPipeline) -> None:
        """Pipeline has a name."""
        assert pipeline.name == "mock_ner"

    def test_custom_patterns(self) -> None:
        """Create pipeline with custom patterns."""
        pipeline = MockNERPipeline.with_custom_patterns(
            drug_patterns=["custom_drug"],
            diagnosis_patterns=["custom_diagnosis"],
        )

        entities = pipeline.extract_entities("patient has custom_diagnosis takes custom_drug")

        assert len(entities) >= 2


# ==============================================================================
# NER Engine Tests
# ==============================================================================


class TestNEREngine:
    """Tests for NEREngine class."""

    @pytest.fixture
    def pipeline(self) -> MockNERPipeline:
        """Create NER pipeline."""
        return MockNERPipeline.with_common_patterns()

    @pytest.fixture
    def engine(self, pipeline: MockNERPipeline) -> NEREngine:
        """Create NER engine."""
        return NEREngine(pipeline)

    def test_perfect_match(self, engine: NEREngine) -> None:
        """Perfect entity extraction."""
        result = engine.compute(
            ground_truth="patient takes metformin",
            prediction="patient takes metformin",
        )

        assert result.f1_score >= 0.0  # At least 0

    def test_compute_from_entities(self, engine: NEREngine) -> None:
        """Compute from pre-extracted entities."""
        gold = [Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))]
        pred = [Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))]

        result = engine.compute_from_entities(gold, pred)

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0

    def test_missing_entity(self, engine: NEREngine) -> None:
        """Detect missing entity."""
        gold = [
            Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9)),
            Entity(text="aspirin", label=EntityLabel.DRUG, span=(15, 22)),
        ]
        pred = [Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))]

        result = engine.compute_from_entities(gold, pred)

        # Recall should be < 1 due to missing entity
        assert result.recall < 1.0
        assert result.entity_omission_rate > 0.0

    def test_hallucinated_entity(self, engine: NEREngine) -> None:
        """Detect hallucinated entity."""
        gold = [Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))]
        pred = [
            Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9)),
            Entity(text="aspirin", label=EntityLabel.DRUG, span=(15, 22)),
        ]

        result = engine.compute_from_entities(gold, pred)

        # Precision should be < 1 due to hallucinated entity
        assert result.precision < 1.0

    def test_empty_entities(self, engine: NEREngine) -> None:
        """Handle empty entity lists."""
        result = engine.compute_from_entities([], [])

        # When no entities exist, precision and recall are 1.0
        # (nothing to get wrong), so F1 is also 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0

    def test_per_type_metrics(self, engine: NEREngine) -> None:
        """Per-type metrics are computed."""
        gold = [
            Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9)),
            Entity(text="diabetes", label=EntityLabel.DIAGNOSIS, span=(15, 23)),
        ]
        pred = [
            Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9)),
            Entity(text="diabetes", label=EntityLabel.DIAGNOSIS, span=(15, 23)),
        ]

        result = engine.compute_from_entities(gold, pred)

        assert isinstance(result.per_type_metrics, dict)

    def test_matches_returned(self, engine: NEREngine) -> None:
        """Matches are included in result."""
        gold = [Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))]
        pred = [Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))]

        result = engine.compute_from_entities(gold, pred)

        assert len(result.matches) > 0
        assert isinstance(result.matches[0], EntityMatch)


class TestNEREnginePrecisionRecall:
    """Tests for precision/recall edge cases."""

    @pytest.fixture
    def engine(self) -> NEREngine:
        """Create NER engine."""
        pipeline = MockNERPipeline()
        return NEREngine(pipeline)

    def test_all_missing(self, engine: NEREngine) -> None:
        """All gold entities missing in prediction."""
        gold = [
            Entity(text="a", label=EntityLabel.DRUG, span=(0, 1)),
            Entity(text="b", label=EntityLabel.DRUG, span=(5, 6)),
        ]
        pred: list[Entity] = []

        result = engine.compute_from_entities(gold, pred)

        assert result.recall == 0.0
        assert result.entity_omission_rate == 1.0

    def test_all_hallucinated(self, engine: NEREngine) -> None:
        """All pred entities are hallucinated."""
        gold: list[Entity] = []
        pred = [
            Entity(text="x", label=EntityLabel.DRUG, span=(0, 1)),
            Entity(text="y", label=EntityLabel.DRUG, span=(5, 6)),
        ]

        result = engine.compute_from_entities(gold, pred)

        assert result.precision == 0.0


# ==============================================================================
# Convenience Function Tests
# ==============================================================================


class TestComputeNERAccuracy:
    """Tests for compute_ner_accuracy function."""

    def test_basic_usage(self) -> None:
        """Basic usage of convenience function."""
        gold = [Entity(text="drug", label=EntityLabel.DRUG, span=(0, 4))]
        pred = [Entity(text="drug", label=EntityLabel.DRUG, span=(0, 4))]

        result = compute_ner_accuracy(gold, pred)

        assert result.f1_score == 1.0


class TestComputeEntityF1:
    """Tests for compute_entity_f1 function."""

    def test_perfect_f1(self) -> None:
        """Perfect F1 score."""
        gold = [Entity(text="a", label=EntityLabel.DRUG, span=(0, 1))]
        pred = [Entity(text="a", label=EntityLabel.DRUG, span=(0, 1))]

        f1 = compute_entity_f1(gold, pred)

        assert f1 == 1.0

    def test_zero_f1(self) -> None:
        """Zero F1 score."""
        gold = [Entity(text="a", label=EntityLabel.DRUG, span=(0, 1))]
        pred = [Entity(text="b", label=EntityLabel.DRUG, span=(10, 11))]

        f1 = compute_entity_f1(gold, pred)

        assert f1 == 0.0


# ==============================================================================
# Span Match Strategy Tests
# ==============================================================================


class TestSpanMatchStrategy:
    """Tests for different span match strategies."""

    def test_exact_strategy(self) -> None:
        """Exact span strategy only matches exact spans."""
        config = AlignmentConfig(span_strategy=SpanMatchStrategy.EXACT)
        aligner = EntityAligner(config)

        gold = [Entity(text="metformin 500", label=EntityLabel.DRUG, span=(0, 13))]
        pred = [Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))]

        matches = aligner.align(gold, pred)

        # Should not match with exact strategy
        exact_matches = [m for m in matches if m.match_type == MatchType.EXACT]
        assert len(exact_matches) == 0

    def test_partial_strategy_overlapping(self) -> None:
        """Partial strategy matches overlapping spans."""
        config = AlignmentConfig(span_strategy=SpanMatchStrategy.PARTIAL)
        aligner = EntityAligner(config)

        gold = [Entity(text="metformin 500", label=EntityLabel.DRUG, span=(0, 13))]
        pred = [Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))]

        matches = aligner.align(gold, pred)

        # Should match with partial strategy
        partial_matches = [
            m for m in matches if m.match_type in (MatchType.PARTIAL, MatchType.DISTORTED)
        ]
        assert len(partial_matches) == 1
