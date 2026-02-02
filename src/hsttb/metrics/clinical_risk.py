"""
Clinical Risk Scoring for Transcription Quality.

Combines multiple quality signals with clinical weighting to produce
a risk-adjusted quality score. Prioritizes clinical safety over
surface-level fluency.

Example:
    >>> from hsttb.metrics.clinical_risk import ClinicalRiskScorer
    >>> scorer = ClinicalRiskScorer()
    >>> result = scorer.score(text)
    >>> print(f"Risk-Adjusted Score: {result.final_score:.1%}")
    >>> print(f"Recommendation: {result.recommendation}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Clinical risk level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Recommendation(Enum):
    """Recommendation for transcript."""
    ACCEPT = "ACCEPT"  # Safe for clinical use
    ACCEPT_WITH_REVIEW = "ACCEPT_WITH_REVIEW"  # Minor issues, review recommended
    NEEDS_REVIEW = "NEEDS_REVIEW"  # Significant issues, human review required
    REJECT = "REJECT"  # Too many issues, not safe for clinical use


@dataclass
class RiskSignal:
    """A risk signal with its weight and value."""
    name: str
    value: float  # 0-1, where 1 is best
    weight: float  # How important this signal is
    clinical_impact: str  # Description of clinical impact
    details: list[str] = field(default_factory=list)


@dataclass
class ClinicalRiskResult:
    """Result of clinical risk scoring."""
    text: str
    final_score: float  # Risk-adjusted quality score (0-1)
    risk_level: RiskLevel
    recommendation: Recommendation
    signals: list[RiskSignal] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    clinical_concerns: list[str] = field(default_factory=list)
    raw_quality_score: float = 0.0  # Original quality score before adjustment


class ClinicalRiskScorer:
    """
    Score transcription quality with clinical risk weighting.

    Unlike surface-level quality metrics, this scorer:
    1. Penalizes clinically important errors heavily
    2. Tracks entity assertion status (negated vs affirmed)
    3. Detects soft contradictions
    4. Validates dosage plausibility
    5. Weights low-confidence tokens by clinical importance
    """

    # Signal weights (must sum to ~1.0 for interpretability)
    SIGNAL_WEIGHTS = {
        "entity_assertion": 0.20,  # Negated entities counted as present
        "contradiction": 0.20,  # Internal contradictions
        "dosage_plausibility": 0.15,  # Medication dosages
        "clinical_token_confidence": 0.15,  # Confidence on important tokens
        "semantic_stability": 0.10,  # Meaning consistency
        "fluency": 0.10,  # Basic perplexity
        "grammar": 0.05,  # Grammar errors
        "coherence": 0.05,  # Drug-condition pairs
    }

    # Clinical tokens that matter more
    CLINICAL_TOKEN_CATEGORIES = {
        "critical": [  # Highest weight
            "mg", "milligram", "mcg", "microgram", "units",
            "daily", "twice", "three", "times", "bid", "tid",
            "metformin", "insulin", "warfarin", "methotrexate",
            "allergic", "allergy", "anaphylaxis",
        ],
        "high": [  # High weight
            "diabetes", "hypertension", "heart", "stroke",
            "cancer", "renal", "hepatic", "failure",
            "pain", "chest", "breathing", "pressure",
            "blood", "sugar", "glucose", "a1c",
        ],
        "medium": [  # Medium weight
            "aspirin", "lisinopril", "amlodipine", "atorvastatin",
            "fatigue", "dizziness", "nausea", "swelling",
            "history", "family", "previous", "diagnosed",
        ],
        "low": [  # Low weight (can be wrong without danger)
            "okay", "alright", "doctor", "patient",
            "yes", "no", "good", "morning",
        ],
    }

    def __init__(self) -> None:
        """Initialize scorer."""
        pass

    def score(
        self,
        text: str,
        quality_result: dict | None = None,
    ) -> ClinicalRiskResult:
        """
        Score transcript with clinical risk weighting.

        Args:
            text: Transcript text.
            quality_result: Optional existing quality result dict.

        Returns:
            ClinicalRiskResult with risk-adjusted score.
        """
        signals: list[RiskSignal] = []
        risk_factors: list[str] = []
        clinical_concerns: list[str] = []

        # 1. Entity Assertion Analysis
        assertion_signal = self._analyze_entity_assertion(text)
        signals.append(assertion_signal)
        if assertion_signal.value < 0.9:
            risk_factors.append("Entity assertion issues detected")
            clinical_concerns.extend(assertion_signal.details)

        # 2. Contradiction Analysis
        contradiction_signal = self._analyze_contradictions(text)
        signals.append(contradiction_signal)
        if contradiction_signal.value < 0.9:
            risk_factors.append("Contradictions detected")
            clinical_concerns.extend(contradiction_signal.details)

        # 3. Dosage Plausibility
        dosage_signal = self._analyze_dosages(text)
        signals.append(dosage_signal)
        if dosage_signal.value < 0.9:
            risk_factors.append("Dosage issues detected")
            clinical_concerns.extend(dosage_signal.details)

        # 4. Clinical Token Confidence
        token_signal = self._analyze_clinical_token_confidence(text, quality_result)
        signals.append(token_signal)
        if token_signal.value < 0.8:
            risk_factors.append("Low confidence on clinical terms")
            clinical_concerns.extend(token_signal.details)

        # 5. Semantic Stability (from quality result if available)
        stability_signal = self._get_stability_signal(quality_result)
        signals.append(stability_signal)

        # 6. Fluency (from quality result if available)
        fluency_signal = self._get_fluency_signal(quality_result)
        signals.append(fluency_signal)

        # 7. Grammar (from quality result if available)
        grammar_signal = self._get_grammar_signal(quality_result)
        signals.append(grammar_signal)

        # 8. Coherence (from quality result if available)
        coherence_signal = self._get_coherence_signal(quality_result)
        signals.append(coherence_signal)

        # Calculate weighted score
        total_weight = sum(s.weight for s in signals)
        weighted_sum = sum(s.value * s.weight for s in signals)
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Apply additional penalties for specific risk patterns
        final_score = self._apply_risk_penalties(
            final_score, signals, risk_factors, clinical_concerns
        )

        # Determine risk level and recommendation
        risk_level = self._determine_risk_level(final_score, clinical_concerns)
        recommendation = self._determine_recommendation(final_score, risk_level)

        # Get raw quality score for comparison
        raw_quality = quality_result.get("composite_score", 0.0) if quality_result else 0.0

        return ClinicalRiskResult(
            text=text,
            final_score=final_score,
            risk_level=risk_level,
            recommendation=recommendation,
            signals=signals,
            risk_factors=risk_factors,
            clinical_concerns=clinical_concerns,
            raw_quality_score=raw_quality,
        )

    def _analyze_entity_assertion(self, text: str) -> RiskSignal:
        """Analyze entity assertion status."""
        try:
            from hsttb.metrics.entity_assertion import analyze_entity_assertions

            result = analyze_entity_assertions(text)

            details = []
            for flag in result.clinical_risk_flags:
                details.append(flag)

            # Count problematic entities
            unsafe_entities = [e for e in result.entities if not e.is_clinically_safe]
            for entity in unsafe_entities[:3]:  # Limit to 3
                details.append(
                    f"'{entity.text}' is {entity.assertion.value} "
                    f"({entity.certainty.value})"
                )

            return RiskSignal(
                name="entity_assertion",
                value=result.assertion_score,
                weight=self.SIGNAL_WEIGHTS["entity_assertion"],
                clinical_impact="Negated or uncertain entities may be miscounted as present",
                details=details,
            )

        except Exception as e:
            logger.warning(f"Entity assertion analysis failed: {e}")
            return RiskSignal(
                name="entity_assertion",
                value=0.8,  # Assume some risk
                weight=self.SIGNAL_WEIGHTS["entity_assertion"],
                clinical_impact="Analysis unavailable",
                details=[str(e)],
            )

    def _analyze_contradictions(self, text: str) -> RiskSignal:
        """Analyze for clinical contradictions."""
        try:
            from hsttb.metrics.clinical_contradiction import detect_clinical_contradictions

            result = detect_clinical_contradictions(text)

            details = []
            for contradiction in result.contradictions[:3]:
                details.append(
                    f"{contradiction.severity.value}: {contradiction.explanation}"
                )

            return RiskSignal(
                name="contradiction",
                value=result.score,
                weight=self.SIGNAL_WEIGHTS["contradiction"],
                clinical_impact="Contradictions can lead to clinical misinterpretation",
                details=details,
            )

        except Exception as e:
            logger.warning(f"Contradiction analysis failed: {e}")
            return RiskSignal(
                name="contradiction",
                value=0.9,
                weight=self.SIGNAL_WEIGHTS["contradiction"],
                clinical_impact="Analysis unavailable",
                details=[str(e)],
            )

    def _analyze_dosages(self, text: str) -> RiskSignal:
        """Analyze dosage plausibility."""
        try:
            from hsttb.metrics.dosage_plausibility import check_dosage_plausibility

            result = check_dosage_plausibility(text)

            details = result.issues + result.warnings

            return RiskSignal(
                name="dosage_plausibility",
                value=result.score,
                weight=self.SIGNAL_WEIGHTS["dosage_plausibility"],
                clinical_impact="Incorrect dosages can cause patient harm",
                details=details[:5],
            )

        except Exception as e:
            logger.warning(f"Dosage analysis failed: {e}")
            return RiskSignal(
                name="dosage_plausibility",
                value=0.9,
                weight=self.SIGNAL_WEIGHTS["dosage_plausibility"],
                clinical_impact="Analysis unavailable",
                details=[str(e)],
            )

    def _analyze_clinical_token_confidence(
        self,
        text: str,
        quality_result: dict | None,
    ) -> RiskSignal:
        """Analyze confidence on clinically important tokens."""
        details = []

        if not quality_result or "confidence_drops" not in quality_result:
            return RiskSignal(
                name="clinical_token_confidence",
                value=0.85,
                weight=self.SIGNAL_WEIGHTS["clinical_token_confidence"],
                clinical_impact="Low-confidence clinical terms may be errors",
                details=["Confidence data unavailable"],
            )

        # Get low-confidence tokens
        low_conf_tokens = quality_result.get("confidence_drops", [])

        # Weight by clinical importance
        critical_low_conf = 0
        high_low_conf = 0
        medium_low_conf = 0

        for token_info in low_conf_tokens:
            token = token_info.get("token", "").lower().strip()

            if any(t in token for t in self.CLINICAL_TOKEN_CATEGORIES["critical"]):
                critical_low_conf += 1
                details.append(f"CRITICAL: Low confidence on '{token}'")
            elif any(t in token for t in self.CLINICAL_TOKEN_CATEGORIES["high"]):
                high_low_conf += 1
                details.append(f"HIGH: Low confidence on '{token}'")
            elif any(t in token for t in self.CLINICAL_TOKEN_CATEGORIES["medium"]):
                medium_low_conf += 1

        # Calculate score with weighted penalties
        penalty = (critical_low_conf * 0.15) + (high_low_conf * 0.08) + (medium_low_conf * 0.03)
        score = max(0.0, 1.0 - penalty)

        return RiskSignal(
            name="clinical_token_confidence",
            value=score,
            weight=self.SIGNAL_WEIGHTS["clinical_token_confidence"],
            clinical_impact="Low-confidence clinical terms may be errors",
            details=details[:5],
        )

    def _get_stability_signal(self, quality_result: dict | None) -> RiskSignal:
        """Get semantic stability signal from quality result."""
        value = 0.9
        if quality_result and "embedding_drift_score" in quality_result:
            value = quality_result["embedding_drift_score"]

        return RiskSignal(
            name="semantic_stability",
            value=value,
            weight=self.SIGNAL_WEIGHTS["semantic_stability"],
            clinical_impact="Semantic drift may indicate transcription errors",
            details=[],
        )

    def _get_fluency_signal(self, quality_result: dict | None) -> RiskSignal:
        """Get fluency signal from quality result."""
        value = 0.9
        if quality_result and "perplexity_score" in quality_result:
            value = quality_result["perplexity_score"]

        return RiskSignal(
            name="fluency",
            value=value,
            weight=self.SIGNAL_WEIGHTS["fluency"],
            clinical_impact="Poor fluency may indicate garbled transcription",
            details=[],
        )

    def _get_grammar_signal(self, quality_result: dict | None) -> RiskSignal:
        """Get grammar signal from quality result."""
        value = 0.95
        if quality_result and "grammar_score" in quality_result:
            value = quality_result["grammar_score"]

        return RiskSignal(
            name="grammar",
            value=value,
            weight=self.SIGNAL_WEIGHTS["grammar"],
            clinical_impact="Grammar errors are usually less clinically significant",
            details=[],
        )

    def _get_coherence_signal(self, quality_result: dict | None) -> RiskSignal:
        """Get coherence signal from quality result."""
        value = 0.9
        if quality_result and "coherence_score" in quality_result:
            value = quality_result["coherence_score"]

        return RiskSignal(
            name="coherence",
            value=value,
            weight=self.SIGNAL_WEIGHTS["coherence"],
            clinical_impact="Drug-condition mismatches may indicate errors",
            details=[],
        )

    def _apply_risk_penalties(
        self,
        score: float,
        signals: list[RiskSignal],
        risk_factors: list[str],
        clinical_concerns: list[str],
    ) -> float:
        """Apply additional penalties for specific risk patterns."""
        # Severe penalty if multiple clinical signals are bad
        bad_clinical_signals = sum(
            1 for s in signals
            if s.name in ["entity_assertion", "contradiction", "dosage_plausibility"]
            and s.value < 0.8
        )

        if bad_clinical_signals >= 2:
            score *= 0.85  # 15% penalty for multiple clinical issues

        # Penalty for critical concerns
        critical_count = sum(1 for c in clinical_concerns if "CRITICAL" in c)
        if critical_count > 0:
            score *= (1 - critical_count * 0.1)

        return max(0.0, min(1.0, score))

    def _determine_risk_level(
        self,
        score: float,
        clinical_concerns: list[str],
    ) -> RiskLevel:
        """Determine overall risk level."""
        critical_count = sum(1 for c in clinical_concerns if "CRITICAL" in c)

        if critical_count > 0 or score < 0.5:
            return RiskLevel.CRITICAL
        elif score < 0.65:
            return RiskLevel.HIGH
        elif score < 0.80:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _determine_recommendation(
        self,
        score: float,
        risk_level: RiskLevel,
    ) -> Recommendation:
        """Determine recommendation based on score and risk."""
        if risk_level == RiskLevel.CRITICAL:
            return Recommendation.REJECT
        elif risk_level == RiskLevel.HIGH:
            return Recommendation.NEEDS_REVIEW
        elif risk_level == RiskLevel.MEDIUM:
            return Recommendation.ACCEPT_WITH_REVIEW
        else:
            return Recommendation.ACCEPT


# Singleton
_scorer: ClinicalRiskScorer | None = None


def get_clinical_risk_scorer() -> ClinicalRiskScorer:
    """Get singleton scorer."""
    global _scorer
    if _scorer is None:
        _scorer = ClinicalRiskScorer()
    return _scorer


def score_clinical_risk(
    text: str,
    quality_result: dict | None = None,
) -> ClinicalRiskResult:
    """Convenience function to score clinical risk."""
    return get_clinical_risk_scorer().score(text, quality_result)
