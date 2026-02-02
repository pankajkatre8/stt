"""
Negation detection for medical text.

This module provides negation detection capabilities for identifying
negated medical entities in clinical text.

Supports:
- Rule-based negation detection
- Negation scope detection
- Negation consistency checking

Example:
    >>> from hsttb.nlp.negation import NegationDetector
    >>> detector = NegationDetector()
    >>> negations = detector.detect_negations("patient denies chest pain")
    >>> print(negations[0]["entity"])  # "chest pain"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import NamedTuple


class NegationSpan(NamedTuple):
    """A negation with its span in text."""

    entity_text: str
    start: int
    end: int
    negation_cue: str
    cue_start: int
    cue_end: int


@dataclass
class NegationConfig:
    """
    Configuration for negation detection.

    Attributes:
        max_scope_words: Maximum words between negation cue and entity.
        include_uncertainty: Whether to treat uncertainty as negation.
        custom_cues: Additional negation cues to detect.
    """

    max_scope_words: int = 5
    include_uncertainty: bool = False
    custom_cues: list[str] = field(default_factory=list)


class NegationDetector:
    """
    Rule-based negation detector for medical text.

    Uses pattern matching to identify negated entities based on
    common clinical negation cues.

    Example:
        >>> detector = NegationDetector()
        >>> text = "Patient denies chest pain and shortness of breath"
        >>> negations = detector.detect_negations(text)
        >>> print([n["entity"] for n in negations])
        ['chest pain', 'shortness of breath']
    """

    # Common negation cues (including contractions)
    PRE_NEGATION_CUES = [
        r"\bno\b",
        r"\bnot\b",
        r"\bwithout\b",
        r"\bdenies\b",
        r"\bdenied\b",
        r"\bnegative\s+for\b",
        r"\bno\s+evidence\s+of\b",
        r"\bno\s+signs?\s+of\b",
        r"\brules?\s+out\b",
        r"\bruled\s+out\b",
        r"\bfailed\s+to\b",
        r"\bnever\b",
        r"\bnone\b",
        r"\babsence\s+of\b",
        r"\bfree\s+of\b",
        r"\bunremarkable\b",
        # Contractions
        r"\bdon'?t\b",
        r"\bdoesn'?t\b",
        r"\bdidn'?t\b",
        r"\bwon'?t\b",
        r"\bcan'?t\b",
        r"\bcouldn'?t\b",
        r"\bwouldn'?t\b",
        r"\bhasn'?t\b",
        r"\bhaven'?t\b",
        r"\bisn'?t\b",
        r"\baren'?t\b",
        r"\bwasn'?t\b",
        r"\bweren'?t\b",
    ]

    POST_NEGATION_CUES = [
        r"\bwas\s+ruled\s+out\b",
        r"\bhas\s+been\s+ruled\s+out\b",
        r"\bnot\s+seen\b",
        r"\bnot\s+found\b",
        r"\bnot\s+present\b",
    ]

    UNCERTAINTY_CUES = [
        r"\bpossible\b",
        r"\bprobable\b",
        r"\bsuspect(?:ed)?\b",
        r"\bquestionable\b",
        r"\bmay\s+have\b",
        r"\bcould\s+be\b",
        r"\buncertain\b",
    ]

    TERMINATION_CUES = [
        r"\bbut\b",
        r"\bhowever\b",
        r"\balthough\b",
        r"\bexcept\b",
        r"\b,\b",
        r"\.\s",
    ]

    def __init__(self, config: NegationConfig | None = None) -> None:
        """
        Initialize the negation detector.

        Args:
            config: Configuration options.
        """
        self.config = config or NegationConfig()

        # Compile patterns
        all_pre_cues = self.PRE_NEGATION_CUES + self.config.custom_cues
        self._pre_pattern = re.compile(
            "|".join(f"({cue})" for cue in all_pre_cues), re.IGNORECASE
        )
        self._post_pattern = re.compile(
            "|".join(f"({cue})" for cue in self.POST_NEGATION_CUES), re.IGNORECASE
        )
        self._term_pattern = re.compile(
            "|".join(f"({cue})" for cue in self.TERMINATION_CUES), re.IGNORECASE
        )
        if self.config.include_uncertainty:
            self._uncertainty_pattern = re.compile(
                "|".join(f"({cue})" for cue in self.UNCERTAINTY_CUES), re.IGNORECASE
            )
        else:
            self._uncertainty_pattern = None

    def detect_negations(self, text: str) -> list[dict[str, object]]:
        """
        Detect negated entities in text.

        Args:
            text: Text to analyze.

        Returns:
            List of dictionaries with negation information:
            - entity: The negated text
            - span: (start, end) character offsets
            - negation_cue: The cue that triggered negation
            - cue_span: (start, end) of the negation cue
        """
        negations: list[dict[str, object]] = []
        text_lower = text.lower()

        # Find pre-negation cues
        for match in self._pre_pattern.finditer(text_lower):
            cue_end = match.end()

            # Find scope (text after cue until termination)
            scope_end = self._find_scope_end(text_lower, cue_end)
            scope_text = text[cue_end:scope_end].strip()

            if scope_text:
                negations.append(
                    {
                        "entity": scope_text,
                        "span": (cue_end, scope_end),
                        "negation_cue": match.group(),
                        "cue_span": (match.start(), match.end()),
                    }
                )

        return negations

    def _find_scope_end(self, text: str, start: int) -> int:
        """Find end of negation scope."""
        # Look for termination cue
        search_text = text[start:]
        term_match = self._term_pattern.search(search_text)

        if term_match:
            return start + term_match.start()

        # Limit by word count
        words = search_text.split()
        if len(words) > self.config.max_scope_words:
            # Find position after max_scope_words
            word_count = 0
            for i, char in enumerate(search_text):
                if char.isspace() and i > 0 and not search_text[i - 1].isspace():
                    word_count += 1
                    if word_count >= self.config.max_scope_words:
                        return start + i

        # Return end of text
        return len(text)

    def is_negated(self, text: str, entity_span: tuple[int, int]) -> bool:
        """
        Check if a specific entity span is negated.

        Args:
            text: Full text.
            entity_span: (start, end) of entity to check.

        Returns:
            True if entity is within a negation scope.
        """
        negations = self.detect_negations(text)

        entity_start, entity_end = entity_span

        for neg in negations:
            neg_start, neg_end = neg["span"]  # type: ignore[misc]
            # Check if entity overlaps with negation scope
            if neg_start <= entity_start and entity_end <= neg_end:
                return True

        return False

    def check_negation_consistency(
        self,
        gt_text: str,
        pred_text: str,
        entity_text: str,
    ) -> dict[str, object]:
        """
        Check if negation status is consistent between GT and prediction.

        Args:
            gt_text: Ground truth text.
            pred_text: Predicted text.
            entity_text: Entity to check.

        Returns:
            Dictionary with consistency information.
        """
        gt_negated = self._entity_is_negated(gt_text, entity_text)
        pred_negated = self._entity_is_negated(pred_text, entity_text)

        return {
            "entity": entity_text,
            "gt_negated": gt_negated,
            "pred_negated": pred_negated,
            "consistent": gt_negated == pred_negated,
            "flip_type": self._get_flip_type(gt_negated, pred_negated),
        }

    def _entity_is_negated(self, text: str, entity_text: str) -> bool:
        """Check if entity is negated in text."""
        entity_lower = entity_text.lower()
        text_lower = text.lower()

        # Find entity in text
        idx = text_lower.find(entity_lower)
        if idx == -1:
            return False

        # Check if within negation scope
        return self.is_negated(text, (idx, idx + len(entity_text)))

    def _get_flip_type(
        self,
        gt_negated: bool,
        pred_negated: bool,
    ) -> str | None:
        """Determine type of negation flip."""
        if gt_negated == pred_negated:
            return None
        if gt_negated and not pred_negated:
            return "negation_lost"  # Was negated, now asserted
        return "false_negation"  # Was asserted, now negated


@dataclass
class NegationConsistencyResult:
    """
    Result of negation consistency check.

    Attributes:
        consistency_score: Ratio of consistent negations (0.0-1.0).
        total_checked: Total entities checked.
        flips: List of negation flips detected.
    """

    consistency_score: float
    total_checked: int
    flips: list[dict[str, object]] = field(default_factory=list)


def check_negation_consistency(
    gt_text: str,
    pred_text: str,
    entities: list[str],
) -> NegationConsistencyResult:
    """
    Check negation consistency for a list of entities.

    Args:
        gt_text: Ground truth text.
        pred_text: Predicted text.
        entities: List of entity texts to check.

    Returns:
        NegationConsistencyResult with analysis.
    """
    detector = NegationDetector()
    flips: list[dict[str, object]] = []

    for entity in entities:
        result = detector.check_negation_consistency(gt_text, pred_text, entity)
        if not result["consistent"]:
            flips.append(result)

    total = len(entities)
    consistent = total - len(flips)
    score = consistent / total if total > 0 else 1.0

    return NegationConsistencyResult(
        consistency_score=score,
        total_checked=total,
        flips=flips,
    )
