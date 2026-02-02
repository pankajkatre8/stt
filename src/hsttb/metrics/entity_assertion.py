"""
Entity Assertion and Certainty Detection.

Tracks assertion status (affirmed/negated/uncertain) for medical entities.
Critical for clinical accuracy - "borderline sugar but not diabetes" should
NOT be scored as "diabetes present".

Example:
    >>> from hsttb.metrics.entity_assertion import EntityAssertionAnalyzer
    >>> analyzer = EntityAssertionAnalyzer()
    >>> result = analyzer.analyze("Patient denies diabetes but has borderline sugar")
    >>> for entity in result.entities:
    ...     print(f"{entity.text}: {entity.assertion} ({entity.certainty})")
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AssertionStatus(Enum):
    """Entity assertion status."""
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"
    HYPOTHETICAL = "hypothetical"  # "if you have...", "rule out..."
    HISTORICAL = "historical"  # "history of...", "previous..."
    FAMILY = "family"  # "family history of..."


class CertaintyLevel(Enum):
    """Certainty level of assertion."""
    DEFINITE = "definite"  # "has diabetes"
    PROBABLE = "probable"  # "likely has diabetes"
    POSSIBLE = "possible"  # "may have diabetes"
    BORDERLINE = "borderline"  # "borderline diabetes"
    RULED_OUT = "ruled_out"  # "diabetes ruled out"


@dataclass
class AssertedEntity:
    """Medical entity with assertion status."""
    text: str
    entity_type: str  # drug, diagnosis, symptom, etc.
    assertion: AssertionStatus
    certainty: CertaintyLevel
    span: tuple[int, int]
    context: str  # surrounding text
    is_clinically_safe: bool = True  # False if ambiguous/risky


@dataclass
class AssertionAnalysisResult:
    """Result of entity assertion analysis."""
    text: str
    entities: list[AssertedEntity] = field(default_factory=list)
    assertion_score: float = 1.0  # Penalized for uncertain/ambiguous
    negated_count: int = 0
    uncertain_count: int = 0
    affirmed_count: int = 0
    clinical_risk_flags: list[str] = field(default_factory=list)


class EntityAssertionAnalyzer:
    """
    Analyze assertion status of medical entities.

    Detects:
    - Negations: "no diabetes", "denies pain", "without symptoms"
    - Uncertainty: "possible diabetes", "borderline sugar", "may have"
    - Hypotheticals: "if you have", "rule out", "suspect"
    - Historical: "history of", "previous", "past"
    - Family: "family history of", "father had"
    """

    # Negation patterns (entity follows)
    NEGATION_PRE = [
        r"\bno\s+",
        r"\bnot\s+",
        r"\bdenies?\s+",
        r"\bdeny\s+",
        r"\bdenied\s+",
        r"\bwithout\s+",
        r"\babsence\s+of\s+",
        r"\bnegative\s+for\s+",
        r"\bfree\s+of\s+",
        r"\brules?\s+out\s+",
        r"\bruled\s+out\s+",
        r"\bnever\s+had\s+",
        r"\bdoes\s+not\s+have\s+",
        r"\bdo\s+not\s+have\s+",
        r"\bdid\s+not\s+have\s+",
        r"\bhasn'?t\s+",
        r"\bhaven'?t\s+",
        r"\bdidn'?t\s+have\s+",
    ]

    # Negation patterns (entity precedes)
    NEGATION_POST = [
        r"\s+not\s+present",
        r"\s+not\s+found",
        r"\s+ruled\s+out",
        r"\s+negative",
        r"\s+absent",
        r"\s+denied",
    ]

    # Uncertainty patterns
    UNCERTAINTY_PRE = [
        r"\bpossible\s+",
        r"\bprobable\s+",
        r"\blikely\s+",
        r"\bsuspected?\s+",
        r"\bquestionable\s+",
        r"\bpotential\s+",
        r"\bmay\s+have\s+",
        r"\bmight\s+have\s+",
        r"\bcould\s+have\s+",
        r"\bappears?\s+to\s+have\s+",
        r"\bseems?\s+like\s+",
        r"\bworried\s+about\s+",
        r"\bconcerned\s+about\s+",
        r"\bthink\s+",
        r"\bthinks?\s+",
        r"\bbelieve\s+",
    ]

    # Borderline/subclinical patterns
    BORDERLINE_PATTERNS = [
        r"\bborderline\s+",
        r"\bpre[-\s]?",  # pre-diabetes, prediabetes
        r"\bsub[-\s]?clinical\s+",
        r"\bmild\s+",
        r"\bearly\s+",
        r"\bslightly\s+elevated\s+",
    ]

    # Hypothetical patterns
    HYPOTHETICAL_PATTERNS = [
        r"\bif\s+you\s+have\s+",
        r"\bif\s+there\s+is\s+",
        r"\brule\s+out\s+",
        r"\bto\s+exclude\s+",
        r"\bwatch\s+for\s+",
        r"\bmonitor\s+for\s+",
    ]

    # Historical patterns
    HISTORICAL_PATTERNS = [
        r"\bhistory\s+of\s+",
        r"\bprevious\s+",
        r"\bpast\s+",
        r"\bformer\s+",
        r"\bused\s+to\s+have\s+",
        r"\bhad\s+",  # context dependent
        r"\bprior\s+",
    ]

    # Family history patterns
    FAMILY_PATTERNS = [
        r"\bfamily\s+history\s+of\s+",
        r"\bfather\s+had\s+",
        r"\bmother\s+had\s+",
        r"\bparent\s+had\s+",
        r"\bsibling\s+had\s+",
        r"\bbrother\s+had\s+",
        r"\bsister\s+had\s+",
        r"\brelative\s+had\s+",
        r"\bfamilial\s+",
    ]

    # Affirmation patterns (explicit positive)
    AFFIRMATION_PATTERNS = [
        r"\bhas\s+",
        r"\bhave\s+",
        r"\bdiagnosed\s+with\s+",
        r"\bsuffers?\s+from\s+",
        r"\bexperiencing\s+",
        r"\breports?\s+",
        r"\bcomplains?\s+of\s+",
        r"\bpositive\s+for\s+",
        r"\bconfirmed\s+",
    ]

    # Medical entities loaded dynamically from lexicon
    # Fallback minimal set only used if lexicon unavailable
    _FALLBACK_SYMPTOMS = {
        "chest pain", "chest pressure", "shortness of breath",
        "breathless", "breathlessness", "fatigue", "tired",
        "dizziness", "dizzy", "nausea", "sweating", "swelling",
        "palpitations", "weakness", "pain", "headache", "fever",
    }

    def _get_medical_entities(self) -> dict[str, list[str]]:
        """Get medical entities dynamically from lexicon."""
        drugs = []
        conditions = []
        symptoms = list(self._FALLBACK_SYMPTOMS)

        try:
            from hsttb.metrics.medical_terms import get_medical_terms
            provider = get_medical_terms()

            drugs = list(provider.get_drugs())
            loaded_conditions = list(provider.get_conditions())
            loaded_symptoms = list(provider.get_symptoms())

            if loaded_conditions:
                conditions = loaded_conditions
            if loaded_symptoms:
                symptoms = loaded_symptoms
        except Exception as e:
            logger.warning(f"Could not load medical entities from provider: {e}")

        return {
            "diagnosis": conditions,
            "drug": drugs,
            "symptom": symptoms,
        }

    def __init__(self) -> None:
        """Initialize analyzer."""
        # Compile patterns for efficiency
        self._negation_pre_re = [re.compile(p, re.IGNORECASE) for p in self.NEGATION_PRE]
        self._negation_post_re = [re.compile(p, re.IGNORECASE) for p in self.NEGATION_POST]
        self._uncertainty_re = [re.compile(p, re.IGNORECASE) for p in self.UNCERTAINTY_PRE]
        self._borderline_re = [re.compile(p, re.IGNORECASE) for p in self.BORDERLINE_PATTERNS]
        self._hypothetical_re = [re.compile(p, re.IGNORECASE) for p in self.HYPOTHETICAL_PATTERNS]
        self._historical_re = [re.compile(p, re.IGNORECASE) for p in self.HISTORICAL_PATTERNS]
        self._family_re = [re.compile(p, re.IGNORECASE) for p in self.FAMILY_PATTERNS]
        self._affirmation_re = [re.compile(p, re.IGNORECASE) for p in self.AFFIRMATION_PATTERNS]

    def analyze(self, text: str) -> AssertionAnalysisResult:
        """
        Analyze entity assertions in text.

        Args:
            text: Clinical text to analyze.

        Returns:
            AssertionAnalysisResult with entities and scores.
        """
        entities: list[AssertedEntity] = []
        risk_flags: list[str] = []
        text_lower = text.lower()

        # Find all medical entities and their assertion status
        medical_entities = self._get_medical_entities()
        for entity_type, entity_list in medical_entities.items():
            for entity_text in entity_list:
                # Find all occurrences
                for match in re.finditer(
                    rf"\b{re.escape(entity_text)}\b",
                    text_lower
                ):
                    start, end = match.span()

                    # Get surrounding context (100 chars before/after)
                    ctx_start = max(0, start - 100)
                    ctx_end = min(len(text), end + 100)
                    context = text[ctx_start:ctx_end]

                    # Determine assertion status
                    assertion, certainty = self._determine_assertion(
                        text, start, end, context
                    )

                    # Check if clinically safe
                    is_safe = self._is_clinically_safe(assertion, certainty)

                    entities.append(AssertedEntity(
                        text=entity_text,
                        entity_type=entity_type,
                        assertion=assertion,
                        certainty=certainty,
                        span=(start, end),
                        context=context,
                        is_clinically_safe=is_safe,
                    ))

        # Count assertion types
        negated = sum(1 for e in entities if e.assertion == AssertionStatus.NEGATED)
        uncertain = sum(1 for e in entities
                       if e.assertion == AssertionStatus.UNCERTAIN
                       or e.certainty in [CertaintyLevel.POSSIBLE, CertaintyLevel.BORDERLINE])
        affirmed = sum(1 for e in entities if e.assertion == AssertionStatus.AFFIRMED)

        # Calculate assertion score
        # Penalize for:
        # - Uncertain entities that should be clear
        # - Negated entities counted as present
        # - Borderline conditions without resolution
        total = len(entities) or 1
        unsafe_count = sum(1 for e in entities if not e.is_clinically_safe)
        assertion_score = max(0.0, 1.0 - (unsafe_count * 0.1) - (uncertain * 0.05))

        # Add risk flags
        if uncertain > 0:
            risk_flags.append(f"{uncertain} uncertain/ambiguous entities")

        # Check for specific risky patterns
        if "borderline" in text_lower and "diabetes" in text_lower:
            if "not diabetes" in text_lower or "but not" in text_lower:
                risk_flags.append("Borderline condition with negated diagnosis - needs clarity")

        return AssertionAnalysisResult(
            text=text,
            entities=entities,
            assertion_score=assertion_score,
            negated_count=negated,
            uncertain_count=uncertain,
            affirmed_count=affirmed,
            clinical_risk_flags=risk_flags,
        )

    def _determine_assertion(
        self,
        text: str,
        start: int,
        end: int,
        context: str,
    ) -> tuple[AssertionStatus, CertaintyLevel]:
        """Determine assertion status and certainty for an entity."""
        # Get text before entity (for prefix patterns)
        prefix_start = max(0, start - 50)
        prefix = text[prefix_start:start].lower()

        # Get text after entity (for suffix patterns)
        suffix_end = min(len(text), end + 50)
        suffix = text[end:suffix_end].lower()

        context_lower = context.lower()

        # Check family history first (specific case)
        for pattern in self._family_re:
            if pattern.search(prefix):
                return AssertionStatus.FAMILY, CertaintyLevel.DEFINITE

        # Check historical
        for pattern in self._historical_re:
            if pattern.search(prefix):
                return AssertionStatus.HISTORICAL, CertaintyLevel.DEFINITE

        # Check hypothetical
        for pattern in self._hypothetical_re:
            if pattern.search(prefix):
                return AssertionStatus.HYPOTHETICAL, CertaintyLevel.POSSIBLE

        # Check negation (prefix)
        for pattern in self._negation_pre_re:
            if pattern.search(prefix):
                return AssertionStatus.NEGATED, CertaintyLevel.DEFINITE

        # Check negation (suffix)
        for pattern in self._negation_post_re:
            if pattern.search(suffix):
                return AssertionStatus.NEGATED, CertaintyLevel.DEFINITE

        # Check borderline
        for pattern in self._borderline_re:
            if pattern.search(prefix):
                return AssertionStatus.UNCERTAIN, CertaintyLevel.BORDERLINE

        # Check uncertainty
        for pattern in self._uncertainty_re:
            if pattern.search(prefix):
                return AssertionStatus.UNCERTAIN, CertaintyLevel.POSSIBLE

        # Check explicit affirmation
        for pattern in self._affirmation_re:
            if pattern.search(prefix):
                return AssertionStatus.AFFIRMED, CertaintyLevel.DEFINITE

        # Default: affirmed with probable certainty
        return AssertionStatus.AFFIRMED, CertaintyLevel.PROBABLE

    def _is_clinically_safe(
        self,
        assertion: AssertionStatus,
        certainty: CertaintyLevel,
    ) -> bool:
        """Determine if assertion status is clinically safe/clear."""
        # Unsafe combinations
        if assertion == AssertionStatus.UNCERTAIN:
            return False
        if certainty in [CertaintyLevel.POSSIBLE, CertaintyLevel.BORDERLINE]:
            return False
        if assertion == AssertionStatus.HYPOTHETICAL:
            return False
        return True


# Singleton accessor
_analyzer: EntityAssertionAnalyzer | None = None


def get_entity_assertion_analyzer() -> EntityAssertionAnalyzer:
    """Get singleton analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = EntityAssertionAnalyzer()
    return _analyzer


def analyze_entity_assertions(text: str) -> AssertionAnalysisResult:
    """Convenience function to analyze entity assertions."""
    return get_entity_assertion_analyzer().analyze(text)
