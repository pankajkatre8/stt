"""
Reference-free transcription quality evaluation.

Combines multiple metrics to assess transcription quality
without requiring ground truth text.

Metrics:
- Perplexity (fluency)
- Grammar (correctness)
- Entity validity (medical terms)
- Medical coherence (drug-condition pairs)

Example:
    >>> from hsttb.metrics.quality import QualityEngine
    >>> engine = QualityEngine()
    >>> result = engine.compute("Patient takes metformin for diabetes")
    >>> print(f"Quality: {result.composite_score:.1%}")
    >>> print(f"Recommendation: {result.recommendation}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    """Configuration for quality scoring."""

    # Component weights (should sum to 1.0)
    perplexity_weight: float = 0.30
    grammar_weight: float = 0.25
    entity_validity_weight: float = 0.25
    coherence_weight: float = 0.20

    # Thresholds for recommendation
    accept_threshold: float = 0.75
    review_threshold: float = 0.50

    # Perplexity normalization
    max_perplexity: float = 500.0

    def __post_init__(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = (
            self.perplexity_weight
            + self.grammar_weight
            + self.entity_validity_weight
            + self.coherence_weight
        )
        if total > 0:
            self.perplexity_weight /= total
            self.grammar_weight /= total
            self.entity_validity_weight /= total
            self.coherence_weight /= total


@dataclass
class QualityResult:
    """Result of quality evaluation."""

    # Composite score (0.0-1.0, higher is better)
    composite_score: float

    # Component scores (0.0-1.0, higher is better)
    perplexity_score: float
    grammar_score: float
    entity_validity_score: float
    coherence_score: float

    # Raw values
    perplexity: float  # Raw perplexity (lower is better)
    log_probability: float

    # Details
    grammar_errors: list[dict] = field(default_factory=list)
    invalid_entities: list[str] = field(default_factory=list)
    entities_found: list[dict] = field(default_factory=list)

    # Metadata
    word_count: int = 0
    medical_entity_count: int = 0
    recommendation: str = "REVIEW"  # "ACCEPT", "REVIEW", "REJECT"

    # Availability flags
    perplexity_available: bool = True
    grammar_available: bool = True


class QualityEngine:
    """
    Reference-free transcription quality scorer.

    Combines perplexity, grammar, entity validity, and
    medical coherence into a composite quality score.

    Attributes:
        config: Quality scoring configuration.

    Example:
        >>> engine = QualityEngine()
        >>> result = engine.compute("Patient has chest pain")
        >>> if result.recommendation == "ACCEPT":
        ...     print("Transcription quality is good")
    """

    def __init__(
        self,
        config: QualityConfig | None = None,
        use_perplexity: bool = True,
        use_grammar: bool = True,
    ) -> None:
        """
        Initialize quality engine.

        Args:
            config: Quality scoring configuration.
            use_perplexity: Whether to use perplexity scoring.
            use_grammar: Whether to use grammar checking.
        """
        self.config = config or QualityConfig()
        self._use_perplexity = use_perplexity
        self._use_grammar = use_grammar

        # Lazy-loaded components
        self._perplexity_scorer: Any = None
        self._grammar_checker: Any = None
        self._coherence_checker: Any = None

        # Availability flags
        self._perplexity_available: bool | None = None
        self._grammar_available: bool | None = None

    def _get_perplexity_scorer(self):
        """Get or create perplexity scorer."""
        if self._perplexity_scorer is None and self._use_perplexity:
            try:
                from hsttb.metrics.perplexity import PerplexityScorer

                self._perplexity_scorer = PerplexityScorer(
                    max_perplexity=self.config.max_perplexity
                )
                self._perplexity_available = True
            except ImportError:
                logger.warning("Perplexity scoring unavailable (transformers not installed)")
                self._perplexity_available = False
        return self._perplexity_scorer

    def _get_grammar_checker(self):
        """Get or create grammar checker."""
        if self._grammar_checker is None and self._use_grammar:
            try:
                from hsttb.metrics.grammar import GrammarChecker

                self._grammar_checker = GrammarChecker()
                self._grammar_available = True
            except ImportError:
                logger.warning("Grammar checking unavailable (language-tool-python not installed)")
                self._grammar_available = False
        return self._grammar_checker

    def _get_coherence_checker(self):
        """Get or create coherence checker."""
        if self._coherence_checker is None:
            from hsttb.metrics.medical_coherence import MedicalCoherenceChecker

            self._coherence_checker = MedicalCoherenceChecker()
        return self._coherence_checker

    def compute(self, text: str) -> QualityResult:
        """
        Compute quality score for transcription.

        Args:
            text: Transcription text to evaluate.

        Returns:
            QualityResult with scores and recommendation.
        """
        if not text or not text.strip():
            return QualityResult(
                composite_score=0.0,
                perplexity_score=0.0,
                grammar_score=0.0,
                entity_validity_score=0.0,
                coherence_score=0.0,
                perplexity=float("inf"),
                log_probability=float("-inf"),
                word_count=0,
                recommendation="REJECT",
            )

        # Compute component scores
        perplexity_score = 0.5  # Default if unavailable
        perplexity = 100.0
        log_probability = 0.0
        perplexity_available = False

        grammar_score = 1.0  # Default if unavailable
        grammar_errors: list[dict] = []
        grammar_available = False

        # Perplexity
        scorer = self._get_perplexity_scorer()
        if scorer is not None:
            try:
                ppl_result = scorer.compute(text)
                perplexity = ppl_result.perplexity
                perplexity_score = ppl_result.normalized_score
                log_probability = ppl_result.log_probability
                perplexity_available = True
            except Exception as e:
                logger.warning(f"Perplexity computation failed: {e}")

        # Grammar
        checker = self._get_grammar_checker()
        if checker is not None:
            try:
                grammar_result = checker.check(text)
                grammar_score = grammar_result.score
                grammar_errors = [
                    {
                        "message": e.message,
                        "text": e.text,
                        "suggestions": e.suggestions[:3],
                    }
                    for e in grammar_result.errors
                ]
                grammar_available = True
            except Exception as e:
                logger.warning(f"Grammar checking failed: {e}")

        # Medical coherence
        coherence_checker = self._get_coherence_checker()
        coherence_result = coherence_checker.check(text)

        entity_validity_score = coherence_result.entity_validity_score
        coherence_score = coherence_result.coherence_score
        invalid_entities = coherence_result.invalid_entities

        entities_found = [
            {
                "text": e.text,
                "type": e.entity_type,
                "valid": e.is_valid,
                "suggestion": e.suggested_correction,
            }
            for e in coherence_result.entities
        ]

        # Compute composite score with adjusted weights
        weights = self._get_adjusted_weights(perplexity_available, grammar_available)

        composite_score = (
            weights["perplexity"] * perplexity_score
            + weights["grammar"] * grammar_score
            + weights["entity_validity"] * entity_validity_score
            + weights["coherence"] * coherence_score
        )

        # Determine recommendation
        if composite_score >= self.config.accept_threshold:
            recommendation = "ACCEPT"
        elif composite_score >= self.config.review_threshold:
            recommendation = "REVIEW"
        else:
            recommendation = "REJECT"

        # Word and entity counts
        word_count = len(text.split())
        medical_entity_count = len([e for e in coherence_result.entities if e.is_valid])

        return QualityResult(
            composite_score=round(composite_score, 4),
            perplexity_score=round(perplexity_score, 4),
            grammar_score=round(grammar_score, 4),
            entity_validity_score=round(entity_validity_score, 4),
            coherence_score=round(coherence_score, 4),
            perplexity=round(perplexity, 2),
            log_probability=round(log_probability, 2),
            grammar_errors=grammar_errors,
            invalid_entities=invalid_entities,
            entities_found=entities_found,
            word_count=word_count,
            medical_entity_count=medical_entity_count,
            recommendation=recommendation,
            perplexity_available=perplexity_available,
            grammar_available=grammar_available,
        )

    def _get_adjusted_weights(
        self, perplexity_available: bool, grammar_available: bool
    ) -> dict[str, float]:
        """Get weights adjusted for available components."""
        weights = {
            "perplexity": self.config.perplexity_weight if perplexity_available else 0.0,
            "grammar": self.config.grammar_weight if grammar_available else 0.0,
            "entity_validity": self.config.entity_validity_weight,
            "coherence": self.config.coherence_weight,
        }

        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def compute_batch(self, texts: list[str]) -> list[QualityResult]:
        """
        Compute quality for multiple texts.

        Args:
            texts: List of texts to evaluate.

        Returns:
            List of QualityResult objects.
        """
        return [self.compute(text) for text in texts]

    @property
    def is_fully_available(self) -> bool:
        """Check if all components are available."""
        # Try to load components
        self._get_perplexity_scorer()
        self._get_grammar_checker()

        return (
            self._perplexity_available is True
            and self._grammar_available is True
        )


# Convenience function
def compute_quality(text: str) -> QualityResult:
    """
    Convenience function to compute quality score.

    Args:
        text: Text to evaluate.

    Returns:
        QualityResult.
    """
    engine = QualityEngine()
    return engine.compute(text)
