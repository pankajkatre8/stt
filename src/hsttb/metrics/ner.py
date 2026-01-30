"""
NER Accuracy computation engine.

This module provides the NER accuracy engine for measuring
entity extraction performance in STT transcriptions.

Metrics computed:
- Precision: Correct predictions / Total predictions
- Recall: Correct predictions / Total ground truth
- F1 Score: Harmonic mean of precision and recall
- Entity Distortion Rate: Distorted entities / Total entities
- Entity Omission Rate: Missing entities / Total entities

Example:
    >>> from hsttb.metrics.ner import NEREngine
    >>> from hsttb.nlp import MockNERPipeline
    >>> pipeline = MockNERPipeline.with_common_patterns()
    >>> engine = NEREngine(pipeline)
    >>> result = engine.compute(
    ...     ground_truth="patient takes metformin for diabetes",
    ...     prediction="patient takes metformin for diabetes",
    ...     gold_entities=gold_entities
    ... )
    >>> print(result.f1_score)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hsttb.core.types import Entity, EntityMatch, MatchType, NERResult
from hsttb.nlp.entity_alignment import AlignmentConfig, EntityAligner

if TYPE_CHECKING:
    from hsttb.nlp.ner_pipeline import NERPipeline


@dataclass
class NEREngineConfig:
    """
    Configuration for NER accuracy engine.

    Attributes:
        alignment_config: Configuration for entity alignment.
        strict_label_matching: Require exact label matches.
        include_negation: Include negation in accuracy computation.
        min_confidence: Minimum confidence for extracted entities.
    """

    alignment_config: AlignmentConfig = field(default_factory=AlignmentConfig)
    strict_label_matching: bool = True
    include_negation: bool = True
    min_confidence: float = 0.0


class NEREngine:
    """
    NER Accuracy computation engine.

    Computes NER accuracy by extracting entities from prediction,
    aligning them with gold entities, and computing metrics.

    Attributes:
        pipeline: NER pipeline for entity extraction.
        config: Engine configuration.

    Example:
        >>> pipeline = MockNERPipeline.with_common_patterns()
        >>> engine = NEREngine(pipeline)
        >>> result = engine.compute(
        ...     ground_truth="patient has diabetes takes metformin",
        ...     prediction="patient has diabetes takes metformin",
        ...     gold_entities=gold_entities
        ... )
        >>> assert result.f1_score == 1.0
    """

    def __init__(
        self,
        pipeline: NERPipeline,
        config: NEREngineConfig | None = None,
    ) -> None:
        """
        Initialize the NER engine.

        Args:
            pipeline: NER pipeline for entity extraction.
            config: Engine configuration (uses defaults if None).
        """
        self.pipeline = pipeline
        self.config = config or NEREngineConfig()
        self.aligner = EntityAligner(self.config.alignment_config)

    def compute(
        self,
        ground_truth: str,
        prediction: str,
        gold_entities: list[Entity] | None = None,
    ) -> NERResult:
        """
        Compute NER accuracy.

        Args:
            ground_truth: Reference transcript (used if gold_entities not provided).
            prediction: Predicted transcript.
            gold_entities: Optional pre-annotated gold entities.
                If not provided, extracts from ground_truth.

        Returns:
            NERResult with detailed accuracy metrics.
        """
        # Get gold entities (either provided or extracted from ground truth)
        if gold_entities is None:
            gold_entities = self.pipeline.extract_entities(ground_truth)

        # Extract entities from prediction
        pred_entities = self.pipeline.extract_entities(prediction)

        # Filter by confidence
        if self.config.min_confidence > 0:
            pred_entities = [
                e for e in pred_entities if e.confidence >= self.config.min_confidence
            ]

        # Align entities
        matches = self.aligner.align(gold_entities, pred_entities)

        # Compute metrics
        return self._compute_metrics(matches, gold_entities, pred_entities)

    def compute_from_entities(
        self,
        gold_entities: list[Entity],
        pred_entities: list[Entity],
    ) -> NERResult:
        """
        Compute NER accuracy from pre-extracted entities.

        Args:
            gold_entities: Ground truth entities.
            pred_entities: Predicted entities.

        Returns:
            NERResult with detailed accuracy metrics.
        """
        # Align entities
        matches = self.aligner.align(gold_entities, pred_entities)

        # Compute metrics
        return self._compute_metrics(matches, gold_entities, pred_entities)

    def _compute_metrics(
        self,
        matches: list[EntityMatch],
        gold_entities: list[Entity],
        pred_entities: list[Entity],  # noqa: ARG002
    ) -> NERResult:
        """
        Compute metrics from alignment results.

        Args:
            matches: Entity alignment matches.
            gold_entities: Ground truth entities.
            pred_entities: Predicted entities.

        Returns:
            NERResult with computed metrics.
        """
        # Count match types
        exact_count = sum(1 for m in matches if m.match_type == MatchType.EXACT)
        partial_count = sum(1 for m in matches if m.match_type == MatchType.PARTIAL)
        distorted_count = sum(1 for m in matches if m.match_type == MatchType.DISTORTED)
        missing_count = sum(1 for m in matches if m.match_type == MatchType.MISSING)
        hallucinated_count = sum(
            1 for m in matches if m.match_type == MatchType.HALLUCINATED
        )

        # True positives: exact + partial matches
        tp = exact_count + partial_count

        # False positives: hallucinated + distorted predictions
        fp = hallucinated_count

        # False negatives: missing entities
        fn = missing_count

        # Compute precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Compute distortion and omission rates
        total_gold = len(gold_entities)
        entity_distortion_rate = (
            distorted_count / total_gold if total_gold > 0 else 0.0
        )
        entity_omission_rate = missing_count / total_gold if total_gold > 0 else 0.0

        # Compute per-type metrics
        per_type_metrics = self._compute_per_type_metrics(matches)

        return NERResult(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            entity_distortion_rate=entity_distortion_rate,
            entity_omission_rate=entity_omission_rate,
            matches=matches,
            per_type_metrics=per_type_metrics,
        )

    def _compute_per_type_metrics(
        self,
        matches: list[EntityMatch],
    ) -> dict[str, dict[str, float]]:
        """
        Compute metrics broken down by entity type.

        Args:
            matches: Entity alignment matches.

        Returns:
            Dictionary mapping entity type to metrics dict.
        """
        # Group matches by entity type
        type_matches: dict[str, list[EntityMatch]] = {}

        for match in matches:
            # Get entity type from ground_truth or predicted
            entity = match.ground_truth or match.predicted
            if entity is None:
                continue

            entity_type = entity.label.value
            if entity_type not in type_matches:
                type_matches[entity_type] = []
            type_matches[entity_type].append(match)

        # Compute metrics per type
        per_type_metrics: dict[str, dict[str, float]] = {}

        for entity_type, type_match_list in type_matches.items():
            exact = sum(1 for m in type_match_list if m.match_type == MatchType.EXACT)
            partial = sum(
                1 for m in type_match_list if m.match_type == MatchType.PARTIAL
            )
            missing = sum(
                1 for m in type_match_list if m.match_type == MatchType.MISSING
            )
            hallucinated = sum(
                1 for m in type_match_list if m.match_type == MatchType.HALLUCINATED
            )

            tp = exact + partial
            fp = hallucinated
            fn = missing

            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_type_metrics[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "exact_matches": float(exact),
                "partial_matches": float(partial),
                "missing": float(missing),
                "hallucinated": float(hallucinated),
            }

        return per_type_metrics


def compute_ner_accuracy(
    gold_entities: list[Entity],
    pred_entities: list[Entity],
    pipeline: NERPipeline | None = None,
) -> NERResult:
    """
    Convenience function to compute NER accuracy.

    Args:
        gold_entities: Ground truth entities.
        pred_entities: Predicted entities.
        pipeline: Optional NER pipeline (uses mock if None).

    Returns:
        NERResult with accuracy metrics.

    Example:
        >>> result = compute_ner_accuracy(gold_list, pred_list)
        >>> print(f"F1: {result.f1_score:.2%}")
    """
    # Import here to avoid circular import
    from hsttb.nlp.ner_pipeline import MockNERPipeline

    if pipeline is None:
        pipeline = MockNERPipeline()

    engine = NEREngine(pipeline)
    return engine.compute_from_entities(gold_entities, pred_entities)


def compute_entity_f1(
    gold_entities: list[Entity],
    pred_entities: list[Entity],
) -> float:
    """
    Quick F1 score computation.

    Args:
        gold_entities: Ground truth entities.
        pred_entities: Predicted entities.

    Returns:
        F1 score (0.0-1.0).
    """
    result = compute_ner_accuracy(gold_entities, pred_entities)
    return result.f1_score
