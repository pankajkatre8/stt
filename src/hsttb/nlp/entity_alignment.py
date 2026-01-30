"""
Entity alignment for NER accuracy computation.

This module provides algorithms for aligning predicted entities
with ground truth entities to enable accurate NER evaluation.

Supports:
- Exact span matching
- Partial/overlapping span matching
- Normalized text matching
- Label matching

Example:
    >>> from hsttb.nlp.entity_alignment import EntityAligner
    >>> aligner = EntityAligner()
    >>> matches = aligner.align(gold_entities, pred_entities)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hsttb.core.types import Entity, EntityMatch, MatchType


class SpanMatchStrategy(str, Enum):
    """Strategy for matching entity spans."""

    EXACT = "exact"  # Spans must match exactly
    PARTIAL = "partial"  # Any overlap counts as match
    BOUNDARY = "boundary"  # Start or end must match


@dataclass
class AlignmentConfig:
    """
    Configuration for entity alignment.

    Attributes:
        span_strategy: How to match spans (exact, partial, boundary).
        require_label_match: Whether entity labels must match.
        iou_threshold: Minimum IOU for partial matches (0.0-1.0).
        text_similarity_threshold: Minimum text similarity (0.0-1.0).
        prioritize_longer: Prefer longer entities when resolving conflicts.
    """

    span_strategy: SpanMatchStrategy = SpanMatchStrategy.PARTIAL
    require_label_match: bool = True
    iou_threshold: float = 0.5
    text_similarity_threshold: float = 0.8
    prioritize_longer: bool = True


class EntityAligner:
    """
    Align predicted entities with ground truth entities.

    Uses configurable matching strategies to determine which
    predicted entities correspond to which ground truth entities.

    Example:
        >>> aligner = EntityAligner()
        >>> gold = [Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))]
        >>> pred = [Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))]
        >>> matches = aligner.align(gold, pred)
        >>> print(matches[0].match_type)
        MatchType.EXACT
    """

    def __init__(self, config: AlignmentConfig | None = None) -> None:
        """
        Initialize the aligner.

        Args:
            config: Alignment configuration. Uses defaults if None.
        """
        self.config = config or AlignmentConfig()

    def align(
        self,
        gold_entities: list[Entity],
        pred_entities: list[Entity],
    ) -> list[EntityMatch]:
        """
        Align predicted entities with ground truth entities.

        Uses a greedy matching algorithm that prioritizes:
        1. Exact matches
        2. High IOU partial matches
        3. Text similarity matches

        Args:
            gold_entities: Ground truth entities.
            pred_entities: Predicted entities.

        Returns:
            List of EntityMatch objects.
        """
        matches: list[EntityMatch] = []
        used_pred: set[int] = set()
        used_gold: set[int] = set()

        # Sort entities by span for consistent matching
        gold_sorted = sorted(gold_entities, key=lambda e: (e.span[0], -e.span_length))
        pred_sorted = sorted(pred_entities, key=lambda e: (e.span[0], -e.span_length))

        # First pass: exact span matches
        for i, gold in enumerate(gold_sorted):
            if i in used_gold:
                continue

            for j, pred in enumerate(pred_sorted):
                if j in used_pred:
                    continue

                if self._is_exact_match(gold, pred):
                    matches.append(
                        EntityMatch(
                            ground_truth=gold,
                            predicted=pred,
                            match_type=MatchType.EXACT,
                            similarity=1.0,
                        )
                    )
                    used_gold.add(i)
                    used_pred.add(j)
                    break

        # Second pass: partial/overlapping matches
        if self.config.span_strategy != SpanMatchStrategy.EXACT:
            for i, gold in enumerate(gold_sorted):
                if i in used_gold:
                    continue

                best_match: tuple[int, float, MatchType] | None = None

                for j, pred in enumerate(pred_sorted):
                    if j in used_pred:
                        continue

                    match_result = self._compute_match(gold, pred)
                    if match_result is not None:
                        iou, match_type = match_result
                        if best_match is None or iou > best_match[1]:
                            best_match = (j, iou, match_type)

                if best_match is not None:
                    j, iou, match_type = best_match
                    matches.append(
                        EntityMatch(
                            ground_truth=gold,
                            predicted=pred_sorted[j],
                            match_type=match_type,
                            similarity=iou,
                        )
                    )
                    used_gold.add(i)
                    used_pred.add(j)
                else:
                    # No match found - this is a missing entity
                    matches.append(
                        EntityMatch(
                            ground_truth=gold,
                            predicted=None,
                            match_type=MatchType.MISSING,
                            similarity=0.0,
                        )
                    )
                    used_gold.add(i)

        # Third pass: remaining gold entities are missing
        for i, gold in enumerate(gold_sorted):
            if i not in used_gold:
                matches.append(
                    EntityMatch(
                        ground_truth=gold,
                        predicted=None,
                        match_type=MatchType.MISSING,
                        similarity=0.0,
                    )
                )

        # Fourth pass: remaining pred entities are hallucinated
        for j, pred in enumerate(pred_sorted):
            if j not in used_pred:
                matches.append(
                    EntityMatch(
                        ground_truth=None,
                        predicted=pred,
                        match_type=MatchType.HALLUCINATED,
                        similarity=0.0,
                    )
                )

        return matches

    def _is_exact_match(self, gold: Entity, pred: Entity) -> bool:
        """Check if two entities are an exact match."""
        # Check label if required
        if self.config.require_label_match and gold.label != pred.label:
            return False

        # Check span
        return gold.span == pred.span

    def _compute_match(
        self,
        gold: Entity,
        pred: Entity,
    ) -> tuple[float, MatchType] | None:
        """
        Compute match between two entities.

        Returns:
            Tuple of (IOU, match_type) or None if no match.
        """
        # Check label if required
        if self.config.require_label_match and gold.label != pred.label:
            return None

        # Compute span IOU
        iou = self._span_iou(gold.span, pred.span)

        if iou == 0:
            # No overlap - check text similarity
            text_sim = self._text_similarity(gold, pred)
            if text_sim >= self.config.text_similarity_threshold:
                return (text_sim, MatchType.PARTIAL)
            return None

        if iou >= self.config.iou_threshold:
            # Determine match type based on IOU
            if iou >= 0.9:
                return (iou, MatchType.PARTIAL)
            return (iou, MatchType.DISTORTED)

        return None

    def _span_iou(
        self,
        span1: tuple[int, int],
        span2: tuple[int, int],
    ) -> float:
        """
        Compute Intersection over Union for two spans.

        Args:
            span1: First span (start, end).
            span2: Second span (start, end).

        Returns:
            IOU score (0.0-1.0).
        """
        start1, end1 = span1
        start2, end2 = span2

        # Compute intersection
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        intersection = max(0, inter_end - inter_start)

        # Compute union
        union = (end1 - start1) + (end2 - start2) - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _text_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """
        Compute text similarity between entities.

        Uses normalized form if available, otherwise raw text.

        Args:
            entity1: First entity.
            entity2: Second entity.

        Returns:
            Similarity score (0.0-1.0).
        """
        text1 = (entity1.normalized or entity1.text).lower()
        text2 = (entity2.normalized or entity2.text).lower()

        if text1 == text2:
            return 1.0

        # Character-level Jaccard similarity
        set1 = set(text1)
        set2 = set(text2)

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union


def align_entities(
    gold_entities: list[Entity],
    pred_entities: list[Entity],
    config: AlignmentConfig | None = None,
) -> list[EntityMatch]:
    """
    Convenience function to align entities.

    Args:
        gold_entities: Ground truth entities.
        pred_entities: Predicted entities.
        config: Optional alignment configuration.

    Returns:
        List of entity matches.

    Example:
        >>> matches = align_entities(gold_list, pred_list)
        >>> exact = [m for m in matches if m.match_type == MatchType.EXACT]
    """
    aligner = EntityAligner(config)
    return aligner.align(gold_entities, pred_entities)
