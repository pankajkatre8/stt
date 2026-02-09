"""
Reference-free transcription quality evaluation.

Combines multiple metrics to assess transcription quality
without requiring ground truth text.

Metrics:
- Perplexity (fluency)
- Grammar (correctness)
- Entity validity (medical terms)
- Medical coherence (drug-condition pairs)
- Internal contradiction detection
- Speech rate validation
- Embedding drift (semantic stability)
- Confidence variance (next-word probability)

Clinical Risk Scoring (NEW):
- Entity assertion tracking (negated/affirmed/uncertain)
- Soft contradiction detection (conditional conflicts)
- Dosage plausibility validation
- Clinical token weighting by importance
- Risk-adjusted scoring prioritizing clinical safety

Example:
    >>> from hsttb.metrics.quality import QualityEngine
    >>> engine = QualityEngine()
    >>> result = engine.compute("Patient takes metformin for diabetes")
    >>> print(f"Quality: {result.composite_score:.1%}")
    >>> print(f"Clinical Risk Score: {result.clinical_risk_score:.1%}")
    >>> print(f"Recommendation: {result.clinical_recommendation}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from hsttb.nlp.gap_filler import GapFiller

logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    """Configuration for quality scoring."""

    # Component weights (should sum to 1.0)
    perplexity_weight: float = 0.20
    grammar_weight: float = 0.15
    entity_validity_weight: float = 0.15
    coherence_weight: float = 0.15
    contradiction_weight: float = 0.15
    embedding_drift_weight: float = 0.10
    confidence_variance_weight: float = 0.10

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
            + self.contradiction_weight
            + self.embedding_drift_weight
            + self.confidence_variance_weight
        )
        if total > 0:
            self.perplexity_weight /= total
            self.grammar_weight /= total
            self.entity_validity_weight /= total
            self.coherence_weight /= total
            self.contradiction_weight /= total
            self.embedding_drift_weight /= total
            self.confidence_variance_weight /= total


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

    # Raw values (required, no defaults)
    perplexity: float  # Raw perplexity (lower is better)
    log_probability: float

    # New metric scores (0.0-1.0, higher is better)
    contradiction_score: float = 1.0  # Higher = fewer contradictions
    embedding_drift_score: float = 1.0  # Higher = more stable semantics
    confidence_variance_score: float = 1.0  # Higher = more stable confidence

    # Details
    grammar_errors: list[dict] = field(default_factory=list)
    invalid_entities: list[str] = field(default_factory=list)
    entities_found: list[dict] = field(default_factory=list)

    # Contradiction details
    contradictions: list[dict] = field(default_factory=list)

    # Embedding drift details
    drift_points: list[dict] = field(default_factory=list)
    segment_similarities: list[float] = field(default_factory=list)

    # Confidence variance details
    confidence_drop_points: list[dict] = field(default_factory=list)
    token_log_probs: list[float] = field(default_factory=list)

    # Metadata
    word_count: int = 0
    medical_entity_count: int = 0
    recommendation: str = "REVIEW"  # "ACCEPT", "REVIEW", "REJECT"

    # Availability flags
    perplexity_available: bool = True
    grammar_available: bool = True
    embedding_drift_available: bool = True
    confidence_variance_available: bool = True

    # Clinical Risk Scoring (NEW)
    clinical_risk_score: float = 0.0  # Risk-adjusted score (0-1, higher is better)
    clinical_risk_level: str = "low"  # "low", "medium", "high", "critical"
    clinical_recommendation: str = "REVIEW"  # "ACCEPT", "ACCEPT_WITH_REVIEW", "NEEDS_REVIEW", "REJECT"

    # Clinical risk details
    entity_assertion_score: float = 1.0  # Higher = clearer assertion status
    clinical_contradiction_score: float = 1.0  # Higher = fewer clinical contradictions
    dosage_plausibility_score: float = 1.0  # Higher = more plausible dosages
    clinical_token_confidence_score: float = 1.0  # Higher = more confident clinical terms

    # Clinical concerns and flags
    clinical_concerns: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    assertion_details: list[dict] = field(default_factory=list)  # Entity assertion info
    dosage_issues: list[str] = field(default_factory=list)  # Dosage problems
    clinical_contradictions: list[dict] = field(default_factory=list)  # Soft/hard contradictions

    # Transcription Error Detection (NEW - for reference-free evaluation)
    transcription_error_score: float = 1.0  # Higher = fewer detected errors
    potential_transcription_errors: list[dict] = field(default_factory=list)  # Detected misspellings
    spelling_inconsistencies: list[dict] = field(default_factory=list)  # Same word spelled differently
    known_terms_found: list[dict] = field(default_factory=list)  # Recognized medical terms

    missing_context_alerts: list[dict] = field(default_factory=list)

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
        use_embedding_drift: bool = True,
        use_confidence_variance: bool = True,
    ) -> None:
        """
        Initialize quality engine.

        Args:
            config: Quality scoring configuration.
            use_perplexity: Whether to use perplexity scoring.
            use_grammar: Whether to use grammar checking.
            use_embedding_drift: Whether to use embedding drift detection.
            use_confidence_variance: Whether to use confidence variance analysis.
        """
        self.config = config or QualityConfig()
        self._use_perplexity = use_perplexity
        self._use_grammar = use_grammar
        self._use_embedding_drift = use_embedding_drift
        self._use_confidence_variance = use_confidence_variance
        self._gap_filler: Any = None
        # Lazy-loaded components
        self._perplexity_scorer: Any = None
        self._grammar_checker: Any = None
        self._coherence_checker: Any = None
        self._contradiction_detector: Any = None
        self._embedding_drift_detector: Any = None
        self._confidence_analyzer: Any = None
        self._clinical_risk_scorer: Any = None
        self._term_matcher: Any = None

        # Availability flags
        self._perplexity_available: bool | None = None
        self._grammar_available: bool | None = None
        self._embedding_drift_available: bool | None = None
        self._confidence_variance_available: bool | None = None

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

    def _get_contradiction_detector(self):
        """Get or create contradiction detector."""
        if self._contradiction_detector is None:
            from hsttb.metrics.contradiction import ContradictionDetector

            self._contradiction_detector = ContradictionDetector()
        return self._contradiction_detector

    def _get_embedding_drift_detector(self):
        """Get or create embedding drift detector."""
        if self._embedding_drift_detector is None and self._use_embedding_drift:
            try:
                from hsttb.metrics.embedding_drift import EmbeddingDriftDetector

                self._embedding_drift_detector = EmbeddingDriftDetector()
                self._embedding_drift_available = True
            except ImportError:
                logger.warning("Embedding drift unavailable (sentence-transformers not installed)")
                self._embedding_drift_available = False
        return self._embedding_drift_detector

    def _get_confidence_analyzer(self):
        """Get or create confidence variance analyzer."""
        if self._confidence_analyzer is None and self._use_confidence_variance:
            try:
                from hsttb.metrics.confidence_variance import ConfidenceAnalyzer

                self._confidence_analyzer = ConfidenceAnalyzer()
                self._confidence_variance_available = True
            except ImportError:
                logger.warning("Confidence variance unavailable (transformers not installed)")
                self._confidence_variance_available = False
        return self._confidence_analyzer

    def _get_clinical_risk_scorer(self):
        """Get or create clinical risk scorer."""
        if self._clinical_risk_scorer is None:
            from hsttb.metrics.clinical_risk import ClinicalRiskScorer

            self._clinical_risk_scorer = ClinicalRiskScorer()
        return self._clinical_risk_scorer

    def _get_term_matcher(self):
        """Get or create medical term matcher for error detection."""
        if self._term_matcher is None:
            try:
                from hsttb.metrics.term_matcher import MedicalTermMatcher

                self._term_matcher = MedicalTermMatcher()
            except Exception as e:
                logger.warning(f"Term matcher unavailable: {e}")
        return self._term_matcher
    
    def _get_gap_filler(self):
        """Lazy load the GapFiller."""
        if self._gap_filler is None:
            try:
                self._gap_filler = GapFiller()
            except Exception as e:
                logger.warning(f"Gap filler unavailable: {e}")
        return self._gap_filler

    def compute(self, text: str, audio_duration_seconds: float | None = None) -> QualityResult:
        """
        Compute quality score for transcription.

        Args:
            text: Transcription text to evaluate.
            audio_duration_seconds: Optional audio duration for speech rate validation.

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
                contradiction_score=1.0,
                embedding_drift_score=1.0,
                confidence_variance_score=1.0,
                perplexity=float("inf"),
                log_probability=float("-inf"),
                word_count=0,
                recommendation="REJECT",
                clinical_risk_score=0.0,
                clinical_risk_level="critical",
                clinical_recommendation="REJECT",
            )

        # Compute component scores
        perplexity_score = 0.5  # Default if unavailable
        perplexity = 100.0
        log_probability = 0.0
        perplexity_available = False
        
        missing_context_alerts: list[dict] = []

        grammar_score = 1.0  # Default if unavailable
        grammar_errors: list[dict] = []
        grammar_available = False

        # New metric defaults
        contradiction_score = 1.0
        contradictions: list[dict] = []
        embedding_drift_score = 1.0
        embedding_drift_available = True
        drift_points: list[dict] = []
        segment_similarities: list[float] = []
        confidence_variance_score = 1.0
        confidence_variance_available = True
        confidence_drop_points: list[dict] = []
        token_log_probs: list[float] = []

        # Perplexity
        scorer = self._get_perplexity_scorer()
        if scorer is not None:
            try:
                ppl_result = scorer.compute(text)
                perplexity = ppl_result.perplexity
                perplexity_score = ppl_result.normalized_score
                log_probability = ppl_result.log_probability
                perplexity_available = True
                
                # --- NEW CODE: Missing Word Detection Strategy ---
                # 1. Check for spikes found by PerplexityScorer
                if hasattr(ppl_result, 'spikes') and ppl_result.spikes:
                    gap_filler = self._get_gap_filler()
                    term_matcher = self._get_term_matcher()
                    
                    for spike in ppl_result.spikes:
                        # 2. Filter: Only check spikes near medical terms or key verbs
                        prev_word = spike['prev_token']
                        next_word = spike['token']
                        
                        is_med_context = False
                        if term_matcher:
                            # Loose check: is the area near a medical term OR a critical verb?
                            is_med_context = (
                                term_matcher.is_medical_term(prev_word) or 
                                term_matcher.is_medical_term(next_word) or
                                prev_word.lower() in ["take", "prescribe", "have", "diagnosed"]
                            )
                        
                        # 3. Predict: Ask PubMedBERT what is missing
                        if is_med_context and gap_filler:
                            prediction = gap_filler.predict_gap(text, prev_word, next_word)
                            
                            if prediction.get("detected") and prediction.get("confidence") > 0.1:
                                top_guess = prediction['top_predictions'][0][0]
                                
                                alert = {
                                    "type": "MISSING_WORD",
                                    "severity": "HIGH",
                                    "location": f"{prev_word} ... {next_word}",
                                    "prediction": top_guess,
                                    "reason": f"High perplexity spike ({spike['surprisal']}). AI predicts '{top_guess}' is missing."
                                }
                                missing_context_alerts.append(alert)
                # -------------------------------------------------
                
                
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

        # Contradiction detection
        contradiction_detector = self._get_contradiction_detector()
        try:
            contradiction_result = contradiction_detector.detect(text)
            contradiction_score = contradiction_result.consistency_score
            contradictions = [
                {
                    "statement1": c.statement1,
                    "statement2": c.statement2,
                    "entity": c.entity,
                    "type": c.contradiction_type,
                    "severity": c.severity,
                    "explanation": c.explanation,
                }
                for c in contradiction_result.contradictions
            ]
        except Exception as e:
            logger.warning(f"Contradiction detection failed: {e}")

        # Embedding drift detection
        drift_detector = self._get_embedding_drift_detector()
        if drift_detector is not None:
            try:
                drift_result = drift_detector.analyze(text)
                embedding_drift_score = drift_result.stability_score
                embedding_drift_available = True
                segment_similarities = drift_result.segment_similarities
                drift_points = [
                    {
                        "segment_index": dp.segment_index,
                        "from_segment": dp.from_segment,
                        "to_segment": dp.to_segment,
                        "similarity": round(dp.similarity, 3),
                        "drop_magnitude": round(dp.drop_magnitude, 3),
                        "is_anomaly": dp.is_anomaly,
                    }
                    for dp in drift_result.drift_points
                ]
            except Exception as e:
                logger.warning(f"Embedding drift detection failed: {e}")
                embedding_drift_available = False
        else:
            embedding_drift_available = self._embedding_drift_available or False

        # Confidence variance analysis
        confidence_analyzer = self._get_confidence_analyzer()
        if confidence_analyzer is not None:
            try:
                conf_result = confidence_analyzer.analyze(text)
                confidence_variance_score = conf_result.stability_score
                confidence_variance_available = conf_result.model_available
                token_log_probs = conf_result.token_log_probs
                confidence_drop_points = [
                    {
                        "position": dp.position,
                        "token": dp.token,
                        "context": dp.context,
                        "log_prob": round(dp.log_prob, 3),
                        "drop_magnitude": round(dp.drop_magnitude, 3),
                        "is_anomaly": dp.is_anomaly,
                    }
                    for dp in conf_result.drop_points
                ]
            except Exception as e:
                logger.warning(f"Confidence variance analysis failed: {e}")
                confidence_variance_available = False
        else:
            confidence_variance_available = self._confidence_variance_available or False

        # Clinical Risk Scoring (NEW) - Combines assertion, contradictions, dosages
        clinical_risk_score = 0.5
        clinical_risk_level = "medium"
        clinical_recommendation = "REVIEW"
        entity_assertion_score = 1.0
        clinical_contradiction_score = 1.0
        dosage_plausibility_score = 1.0
        clinical_token_confidence_score = 1.0
        clinical_concerns: list[str] = []
        risk_factors: list[str] = []
        assertion_details: list[dict] = []
        dosage_issues: list[str] = []
        clinical_contradictions: list[dict] = []

        try:
            clinical_risk_scorer = self._get_clinical_risk_scorer()

            # Build quality result dict for context
            quality_context = {
                "embedding_drift_score": embedding_drift_score,
                "perplexity_score": perplexity_score,
                "grammar_score": grammar_score,
                "coherence_score": coherence_score,
                "confidence_drops": confidence_drop_points,
            }

            risk_result = clinical_risk_scorer.score(text, quality_context)

            clinical_risk_score = risk_result.final_score
            if missing_context_alerts:
             clinical_risk_score = max(0.0, clinical_risk_score - (0.15 * len(missing_context_alerts)))
             if clinical_risk_level != "critical":
                 clinical_risk_level = "high"
             clinical_recommendation = "NEEDS_REVIEW"
            clinical_risk_level = risk_result.risk_level.value
            clinical_recommendation = risk_result.recommendation.value
            clinical_concerns = risk_result.clinical_concerns
            risk_factors = risk_result.risk_factors

            # Extract individual signal scores
            for signal in risk_result.signals:
                if signal.name == "entity_assertion":
                    entity_assertion_score = signal.value
                    assertion_details = [{"detail": d} for d in signal.details]
                elif signal.name == "contradiction":
                    clinical_contradiction_score = signal.value
                    clinical_contradictions = [{"detail": d} for d in signal.details]
                elif signal.name == "dosage_plausibility":
                    dosage_plausibility_score = signal.value
                    dosage_issues = signal.details
                elif signal.name == "clinical_token_confidence":
                    clinical_token_confidence_score = signal.value

        except Exception as e:
            logger.warning(f"Clinical risk scoring failed: {e}")

        # Transcription Error Detection (NEW) - Detects potential misspellings
        transcription_error_score = 1.0
        potential_transcription_errors: list[dict] = []
        spelling_inconsistencies: list[dict] = []
        known_terms_found: list[dict] = []

        try:
            term_matcher = self._get_term_matcher()
            if term_matcher is not None:
                # Analyze text for potential transcription errors
                match_result = term_matcher.analyze(text)

                # Collect known terms found
                known_terms_found = [
                    {
                        "term": kt["term"],
                        "position": kt["position"],
                        "category": kt["info"].get("category", "unknown"),
                        "source": kt["info"].get("source", "unknown"),
                    }
                    for kt in match_result.known_terms_found
                ]

                # Collect potential transcription errors (misspellings)
                potential_transcription_errors = [
                    {
                        "found_term": pe.found_term,
                        "suggested_term": pe.suggested_term,
                        "confidence": round(pe.confidence, 3),
                        "similarity": round(pe.similarity, 3),
                        "match_type": pe.match_type,
                        "position": pe.position,
                        "context": pe.context[:100] if pe.context else "",
                        "source": pe.source,
                    }
                    for pe in match_result.potential_errors
                ]

                # Find spelling inconsistencies (same word spelled differently)
                spelling_inconsistencies = term_matcher.find_inconsistencies(text)

                # Calculate transcription error score
                # Start with the error_score from match_result
                transcription_error_score = match_result.error_score

                # Penalize for inconsistencies
                if spelling_inconsistencies:
                    inconsistency_penalty = min(0.3, len(spelling_inconsistencies) * 0.1)
                    transcription_error_score = max(0.0, transcription_error_score - inconsistency_penalty)

                logger.debug(
                    f"Transcription error detection: {len(potential_transcription_errors)} errors, "
                    f"{len(spelling_inconsistencies)} inconsistencies, score={transcription_error_score:.2f}"
                )

        except Exception as e:
            logger.warning(f"Transcription error detection failed: {e}")

        # Compute composite score with adjusted weights
        weights = self._get_adjusted_weights(
            perplexity_available,
            grammar_available,
            embedding_drift_available,
            confidence_variance_available,
        )

        composite_score = (
            weights["perplexity"] * perplexity_score
            + weights["grammar"] * grammar_score
            + weights["entity_validity"] * entity_validity_score
            + weights["coherence"] * coherence_score
            + weights["contradiction"] * contradiction_score
            + weights["embedding_drift"] * embedding_drift_score
            + weights["confidence_variance"] * confidence_variance_score
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
            contradiction_score=round(contradiction_score, 4),
            embedding_drift_score=round(embedding_drift_score, 4),
            confidence_variance_score=round(confidence_variance_score, 4),
            perplexity=round(perplexity, 2),
            log_probability=round(log_probability, 2),
            grammar_errors=grammar_errors,
            invalid_entities=invalid_entities,
            entities_found=entities_found,
            contradictions=contradictions,
            drift_points=drift_points,
            segment_similarities=segment_similarities,
            confidence_drop_points=confidence_drop_points,
            token_log_probs=token_log_probs,
            word_count=word_count,
            medical_entity_count=medical_entity_count,
            recommendation=recommendation,
            perplexity_available=perplexity_available,
            grammar_available=grammar_available,
            embedding_drift_available=embedding_drift_available,
            confidence_variance_available=confidence_variance_available,
            
            missing_context_alerts=missing_context_alerts,
            # Clinical Risk Scoring
            clinical_risk_score=round(clinical_risk_score, 4),
            clinical_risk_level=clinical_risk_level,
            clinical_recommendation=clinical_recommendation,
            entity_assertion_score=round(entity_assertion_score, 4),
            clinical_contradiction_score=round(clinical_contradiction_score, 4),
            dosage_plausibility_score=round(dosage_plausibility_score, 4),
            clinical_token_confidence_score=round(clinical_token_confidence_score, 4),
            clinical_concerns=clinical_concerns,
            risk_factors=risk_factors,
            assertion_details=assertion_details,
            dosage_issues=dosage_issues,
            clinical_contradictions=clinical_contradictions,
            # Transcription Error Detection
            transcription_error_score=round(transcription_error_score, 4),
            potential_transcription_errors=potential_transcription_errors,
            spelling_inconsistencies=spelling_inconsistencies,
            known_terms_found=known_terms_found,
        )

    def _get_adjusted_weights(
        self,
        perplexity_available: bool,
        grammar_available: bool,
        embedding_drift_available: bool = True,
        confidence_variance_available: bool = True,
    ) -> dict[str, float]:
        """Get weights adjusted for available components."""
        weights = {
            "perplexity": self.config.perplexity_weight if perplexity_available else 0.0,
            "grammar": self.config.grammar_weight if grammar_available else 0.0,
            "entity_validity": self.config.entity_validity_weight,
            "coherence": self.config.coherence_weight,
            "contradiction": self.config.contradiction_weight,
            "embedding_drift": self.config.embedding_drift_weight if embedding_drift_available else 0.0,
            "confidence_variance": self.config.confidence_variance_weight if confidence_variance_available else 0.0,
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
