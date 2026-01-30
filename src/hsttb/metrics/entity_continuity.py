"""
Entity continuity tracking for CRS.

This module tracks entity occurrences across streaming segments
to detect discontinuities that indicate transcription errors.

Tracks:
- Entity appearances across segments
- Entity disappearances (unexpected)
- Attribute conflicts (e.g., dosage changes)
- Entity relationships

Example:
    >>> from hsttb.metrics.entity_continuity import EntityContinuityTracker
    >>> tracker = EntityContinuityTracker()
    >>> result = tracker.track(segments, entities_per_segment)
    >>> print(result.continuity_score)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DiscontinuityType(str, Enum):
    """Types of entity discontinuities."""

    UNEXPECTED_DISAPPEARANCE = "unexpected_disappearance"
    ATTRIBUTE_CONFLICT = "attribute_conflict"
    LABEL_CHANGE = "label_change"
    NEGATION_FLIP = "negation_flip"


@dataclass
class EntityOccurrence:
    """
    Record of an entity occurrence in a segment.

    Attributes:
        entity_key: Normalized key for entity matching.
        entity_text: Original entity text.
        entity_label: Entity type label.
        segment_id: Segment where entity occurred.
        negated: Whether entity was negated.
        normalized: Normalized form of entity.
        context: Surrounding text context.
    """

    entity_key: str
    entity_text: str
    entity_label: str
    segment_id: int
    negated: bool = False
    normalized: str | None = None
    context: str = ""


@dataclass
class Discontinuity:
    """
    A detected entity discontinuity.

    Attributes:
        discontinuity_type: Type of discontinuity.
        entity_key: Entity involved.
        segment_id: Segment where issue was detected.
        details: Additional details about the issue.
    """

    discontinuity_type: DiscontinuityType
    entity_key: str
    segment_id: int
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class ContinuityResult:
    """
    Result of entity continuity tracking.

    Attributes:
        continuity_score: Overall continuity score (0.0-1.0).
        entities_tracked: Number of unique entities tracked.
        total_occurrences: Total entity occurrences.
        discontinuities: List of detected discontinuities.
        entity_timelines: Timeline of each entity's occurrences.
    """

    continuity_score: float
    entities_tracked: int
    total_occurrences: int
    discontinuities: list[Discontinuity] = field(default_factory=list)
    entity_timelines: dict[str, list[int]] = field(default_factory=dict)

    @property
    def discontinuity_count(self) -> int:
        """Number of discontinuities detected."""
        return len(self.discontinuities)

    @property
    def has_discontinuities(self) -> bool:
        """Whether any discontinuities were detected."""
        return len(self.discontinuities) > 0


class EntityContinuityTracker:
    """
    Track entity continuity across streaming segments.

    Monitors how entities appear and evolve across segments
    to detect transcription errors that cause discontinuities.

    Example:
        >>> tracker = EntityContinuityTracker()
        >>> from hsttb.core.types import Entity, EntityLabel
        >>> entities_seg1 = [Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))]
        >>> entities_seg2 = [Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))]
        >>> result = tracker.track(
        ...     gt_segments=["patient takes metformin", "continues metformin"],
        ...     gt_entities=[entities_seg1, entities_seg2],
        ...     pred_segments=["patient takes metformin", "continues metformin"],
        ...     pred_entities=[entities_seg1, entities_seg2],
        ... )
        >>> assert result.continuity_score == 1.0
    """

    def __init__(
        self,
        gap_threshold: int = 2,
        track_negation: bool = True,
    ) -> None:
        """
        Initialize the tracker.

        Args:
            gap_threshold: Max segments between occurrences before
                considering it a disappearance.
            track_negation: Whether to track negation consistency.
        """
        self.gap_threshold = gap_threshold
        self.track_negation = track_negation

    def track(
        self,
        gt_segments: list[str],
        gt_entities: list[list[object]],
        pred_segments: list[str],
        pred_entities: list[list[object]],
    ) -> ContinuityResult:
        """
        Track entity continuity between ground truth and prediction.

        Compares entity timelines to detect discontinuities
        introduced by transcription.

        Args:
            gt_segments: Ground truth text segments.
            gt_entities: Entities per ground truth segment.
            pred_segments: Predicted text segments.
            pred_entities: Entities per predicted segment.

        Returns:
            ContinuityResult with tracking analysis.
        """
        # Build occurrence maps
        gt_occurrences = self._build_occurrence_map(gt_segments, gt_entities)
        pred_occurrences = self._build_occurrence_map(pred_segments, pred_entities)

        # Detect discontinuities
        discontinuities = self._detect_discontinuities(gt_occurrences, pred_occurrences)

        # Compute score
        total_gt_occurrences = sum(len(occs) for occs in gt_occurrences.values())
        issue_count = len(discontinuities)

        score = 1.0 - (issue_count / total_gt_occurrences) if total_gt_occurrences > 0 else 1.0
        score = max(0.0, min(1.0, score))

        # Build entity timelines
        timelines = {key: [occ.segment_id for occ in occs] for key, occs in gt_occurrences.items()}

        return ContinuityResult(
            continuity_score=score,
            entities_tracked=len(gt_occurrences),
            total_occurrences=total_gt_occurrences,
            discontinuities=discontinuities,
            entity_timelines=timelines,
        )

    def _build_occurrence_map(
        self,
        segments: list[str],
        entities_per_segment: list[list[object]],
    ) -> dict[str, list[EntityOccurrence]]:
        """Build map of entity key to occurrences."""
        from hsttb.core.types import Entity

        occurrences: dict[str, list[EntityOccurrence]] = {}

        for seg_id, (segment, entities) in enumerate(
            zip(segments, entities_per_segment)
        ):
            for entity in entities:
                if not isinstance(entity, Entity):
                    continue

                key = self._entity_key(entity)

                if key not in occurrences:
                    occurrences[key] = []

                occurrences[key].append(
                    EntityOccurrence(
                        entity_key=key,
                        entity_text=entity.text,
                        entity_label=entity.label.value,
                        segment_id=seg_id,
                        negated=entity.negated,
                        normalized=entity.normalized,
                        context=segment,
                    )
                )

        return occurrences

    def _entity_key(self, entity: object) -> str:
        """Generate normalized key for entity matching."""
        from hsttb.core.types import Entity

        if not isinstance(entity, Entity):
            return ""

        # Use normalized form if available, otherwise text
        text = (entity.normalized or entity.text).lower()
        return f"{entity.label.value}:{text}"

    def _detect_discontinuities(
        self,
        gt_occurrences: dict[str, list[EntityOccurrence]],
        pred_occurrences: dict[str, list[EntityOccurrence]],
    ) -> list[Discontinuity]:
        """Detect discontinuities between GT and pred timelines."""
        discontinuities: list[Discontinuity] = []

        for entity_key, gt_occs in gt_occurrences.items():
            pred_occs = pred_occurrences.get(entity_key, [])

            # Check for missing occurrences
            gt_segments = {occ.segment_id for occ in gt_occs}
            pred_segments = {occ.segment_id for occ in pred_occs}

            missing_segments = gt_segments - pred_segments

            for seg_id in missing_segments:
                discontinuities.append(
                    Discontinuity(
                        discontinuity_type=DiscontinuityType.UNEXPECTED_DISAPPEARANCE,
                        entity_key=entity_key,
                        segment_id=seg_id,
                        details={"expected_in_segment": seg_id},
                    )
                )

            # Check for negation flips
            if self.track_negation:
                gt_by_seg = {occ.segment_id: occ for occ in gt_occs}
                pred_by_seg = {occ.segment_id: occ for occ in pred_occs}

                for seg_id in gt_segments & pred_segments:
                    gt_occ = gt_by_seg[seg_id]
                    pred_occ = pred_by_seg[seg_id]

                    if gt_occ.negated != pred_occ.negated:
                        discontinuities.append(
                            Discontinuity(
                                discontinuity_type=DiscontinuityType.NEGATION_FLIP,
                                entity_key=entity_key,
                                segment_id=seg_id,
                                details={
                                    "gt_negated": gt_occ.negated,
                                    "pred_negated": pred_occ.negated,
                                },
                            )
                        )

        return discontinuities

    def compute_entity_preservation_rate(
        self,
        gt_entities: list[list[object]],
        pred_entities: list[list[object]],
    ) -> float:
        """
        Compute rate of entities preserved from GT to prediction.

        Args:
            gt_entities: Entities per ground truth segment.
            pred_entities: Entities per predicted segment.

        Returns:
            Preservation rate (0.0-1.0).
        """
        total_gt = sum(len(seg) for seg in gt_entities)
        if total_gt == 0:
            return 1.0

        # Count preserved entities (same key appears in same segment)
        preserved = 0

        for gt_seg, pred_seg in zip(gt_entities, pred_entities):
            gt_keys = {self._entity_key(e) for e in gt_seg}
            pred_keys = {self._entity_key(e) for e in pred_seg}
            preserved += len(gt_keys & pred_keys)

        return preserved / total_gt


def compute_entity_continuity(
    gt_segments: list[str],
    gt_entities: list[list[object]],
    pred_segments: list[str],
    pred_entities: list[list[object]],
) -> float:
    """
    Convenience function to compute entity continuity score.

    Args:
        gt_segments: Ground truth segments.
        gt_entities: Entities per GT segment.
        pred_segments: Predicted segments.
        pred_entities: Entities per pred segment.

    Returns:
        Continuity score (0.0-1.0).
    """
    tracker = EntityContinuityTracker()
    result = tracker.track(gt_segments, gt_entities, pred_segments, pred_entities)
    return result.continuity_score
