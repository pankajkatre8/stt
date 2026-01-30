"""
Context Retention Score (CRS) computation engine.

This module provides the CRS computation engine for measuring
how well transcription preserves context across streaming segments.

CRS Components:
- Semantic Similarity: How well meaning is preserved
- Entity Continuity: Whether entities are tracked correctly
- Negation Consistency: Whether negations are preserved

Example:
    >>> from hsttb.metrics.crs import CRSEngine
    >>> engine = CRSEngine()
    >>> result = engine.compute(
    ...     gt_segments=["patient has diabetes", "takes metformin"],
    ...     pred_segments=["patient has diabetes", "takes metformin"],
    ...     gt_entities=[gold_entities_1, gold_entities_2],
    ...     pred_entities=[pred_entities_1, pred_entities_2],
    ... )
    >>> print(result.composite_score)
"""
from __future__ import annotations

from dataclasses import dataclass

from hsttb.core.types import CRSResult, Entity, SegmentCRSScore
from hsttb.metrics.entity_continuity import EntityContinuityTracker
from hsttb.metrics.semantic_similarity import (
    SemanticSimilarityEngine,
    SimilarityConfig,
    TokenBasedSimilarity,
)


@dataclass
class CRSConfig:
    """
    Configuration for CRS computation.

    Attributes:
        semantic_weight: Weight for semantic similarity (0.0-1.0).
        entity_weight: Weight for entity continuity (0.0-1.0).
        negation_weight: Weight for negation consistency (0.0-1.0).
        similarity_config: Config for similarity engine.
        track_negation: Whether to track negation consistency.
    """

    semantic_weight: float = 0.4
    entity_weight: float = 0.4
    negation_weight: float = 0.2
    similarity_config: SimilarityConfig | None = None
    track_negation: bool = True

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0."""
        total = self.semantic_weight + self.entity_weight + self.negation_weight
        if abs(total - 1.0) > 0.001:
            # Normalize weights
            self.semantic_weight /= total
            self.entity_weight /= total
            self.negation_weight /= total


class CRSEngine:
    """
    Context Retention Score computation engine.

    Computes CRS by combining:
    1. Semantic similarity between GT and pred segments
    2. Entity continuity tracking
    3. Negation consistency checking

    Attributes:
        config: Engine configuration.
        similarity_engine: Semantic similarity engine.
        continuity_tracker: Entity continuity tracker.

    Example:
        >>> engine = CRSEngine()
        >>> result = engine.compute(gt_segments, pred_segments, gt_entities, pred_entities)
        >>> print(f"CRS: {result.composite_score:.2%}")
    """

    def __init__(
        self,
        config: CRSConfig | None = None,
        similarity_engine: SemanticSimilarityEngine | None = None,
    ) -> None:
        """
        Initialize the CRS engine.

        Args:
            config: Engine configuration.
            similarity_engine: Custom similarity engine (uses TokenBased if None).
        """
        self.config = config or CRSConfig()
        self.similarity_engine = similarity_engine or TokenBasedSimilarity(
            self.config.similarity_config
        )
        self.continuity_tracker = EntityContinuityTracker(
            track_negation=self.config.track_negation
        )

    def compute(
        self,
        gt_segments: list[str],
        pred_segments: list[str],
        gt_entities: list[list[Entity]] | None = None,
        pred_entities: list[list[Entity]] | None = None,
    ) -> CRSResult:
        """
        Compute Context Retention Score.

        Args:
            gt_segments: Ground truth text segments.
            pred_segments: Predicted text segments.
            gt_entities: Optional entities per GT segment.
            pred_entities: Optional entities per pred segment.

        Returns:
            CRSResult with detailed analysis.
        """
        # Default to empty entity lists if not provided
        if gt_entities is None:
            gt_entities = [[] for _ in gt_segments]
        if pred_entities is None:
            pred_entities = [[] for _ in pred_segments]

        # Ensure same number of segments
        max_segments = max(len(gt_segments), len(pred_segments))
        gt_segments = list(gt_segments) + [""] * (max_segments - len(gt_segments))
        pred_segments = list(pred_segments) + [""] * (max_segments - len(pred_segments))
        gt_entities = list(gt_entities) + [[] for _ in range(max_segments - len(gt_entities))]
        pred_entities = list(pred_entities) + [[] for _ in range(max_segments - len(pred_entities))]

        # Compute semantic similarity
        semantic_scores = self.similarity_engine.compute_segment_similarities(
            gt_segments, pred_segments
        )
        avg_semantic = (
            sum(semantic_scores) / len(semantic_scores) if semantic_scores else 1.0
        )

        # Compute entity continuity
        continuity_result = self.continuity_tracker.track(
            gt_segments, gt_entities, pred_segments, pred_entities
        )
        entity_continuity = continuity_result.continuity_score

        # Compute negation consistency
        negation_score, negation_flips = self._compute_negation_consistency(
            gt_segments, pred_segments, gt_entities
        )

        # Compute composite score
        composite = (
            self.config.semantic_weight * avg_semantic
            + self.config.entity_weight * entity_continuity
            + self.config.negation_weight * negation_score
        )

        # Build per-segment scores
        segment_scores = self._build_segment_scores(
            gt_segments,
            pred_segments,
            semantic_scores,
            gt_entities,
            pred_entities,
        )

        # Compute context drift rate
        context_drift_rate = self._compute_drift_rate(semantic_scores)

        return CRSResult(
            composite_score=composite,
            semantic_similarity=avg_semantic,
            entity_continuity=entity_continuity,
            negation_consistency=negation_score,
            context_drift_rate=context_drift_rate,
            segment_scores=segment_scores,
        )

    def _compute_negation_consistency(
        self,
        gt_segments: list[str],
        pred_segments: list[str],
        gt_entities: list[list[Entity]],
    ) -> tuple[float, int]:
        """
        Compute negation consistency across segments.

        Returns:
            Tuple of (consistency_score, flip_count).
        """
        from hsttb.nlp.negation import NegationDetector

        if not self.config.track_negation:
            return 1.0, 0

        detector = NegationDetector()
        total_entities = 0
        consistent = 0
        flips = 0

        for gt_text, pred_text, entities in zip(gt_segments, pred_segments, gt_entities):
            for entity in entities:
                if not isinstance(entity, Entity):
                    continue

                total_entities += 1
                result = detector.check_negation_consistency(
                    gt_text, pred_text, entity.text
                )

                if result["consistent"]:
                    consistent += 1
                else:
                    flips += 1

        score = consistent / total_entities if total_entities > 0 else 1.0
        return score, flips

    def _build_segment_scores(
        self,
        gt_segments: list[str],
        pred_segments: list[str],
        semantic_scores: list[float],
        gt_entities: list[list[Entity]],
        pred_entities: list[list[Entity]],
    ) -> list[SegmentCRSScore]:
        """Build per-segment score objects."""
        segment_scores: list[SegmentCRSScore] = []

        for i, (gt_seg, pred_seg, sim_score, gt_ents, pred_ents) in enumerate(
            zip(gt_segments, pred_segments, semantic_scores, gt_entities, pred_entities)
        ):
            # Count preserved and lost entities
            gt_keys = {self._entity_key(e) for e in gt_ents if isinstance(e, Entity)}
            pred_keys = {self._entity_key(e) for e in pred_ents if isinstance(e, Entity)}

            preserved = len(gt_keys & pred_keys)
            lost = len(gt_keys - pred_keys)

            # Count negation flips for this segment
            negation_flips = 0
            if self.config.track_negation:
                from hsttb.nlp.negation import NegationDetector

                detector = NegationDetector()
                for entity in gt_ents:
                    if not isinstance(entity, Entity):
                        continue
                    result = detector.check_negation_consistency(
                        gt_seg, pred_seg, entity.text
                    )
                    if not result["consistent"]:
                        negation_flips += 1

            segment_scores.append(
                SegmentCRSScore(
                    segment_id=i,
                    ground_truth_text=gt_seg,
                    predicted_text=pred_seg,
                    semantic_similarity=sim_score,
                    entities_preserved=preserved,
                    entities_lost=lost,
                    negation_flips=negation_flips,
                )
            )

        return segment_scores

    def _entity_key(self, entity: Entity) -> str:
        """Generate normalized key for entity matching."""
        text = (entity.normalized or entity.text).lower()
        return f"{entity.label.value}:{text}"

    def _compute_drift_rate(self, semantic_scores: list[float]) -> float:
        """
        Compute context drift rate.

        Measures how much semantic similarity decreases over segments.

        Args:
            semantic_scores: Similarity scores per segment.

        Returns:
            Drift rate (0.0 = no drift, higher = more drift).
        """
        if len(semantic_scores) < 2:
            return 0.0

        # Compute average decrease in similarity
        decreases = []
        for i in range(1, len(semantic_scores)):
            decrease = max(0, semantic_scores[i - 1] - semantic_scores[i])
            decreases.append(decrease)

        return sum(decreases) / len(decreases) if decreases else 0.0


def compute_crs(
    gt_segments: list[str],
    pred_segments: list[str],
    gt_entities: list[list[Entity]] | None = None,
    pred_entities: list[list[Entity]] | None = None,
) -> float:
    """
    Convenience function to compute CRS.

    Args:
        gt_segments: Ground truth segments.
        pred_segments: Predicted segments.
        gt_entities: Optional entities per GT segment.
        pred_entities: Optional entities per pred segment.

    Returns:
        Composite CRS score (0.0-1.0).

    Example:
        >>> score = compute_crs(
        ...     ["patient has diabetes", "takes metformin"],
        ...     ["patient has diabetes", "takes metformin"]
        ... )
        >>> print(f"CRS: {score:.2%}")
    """
    engine = CRSEngine()
    result = engine.compute(gt_segments, pred_segments, gt_entities, pred_entities)
    return result.composite_score
