"""
Clinical Contradiction Detection (Soft & Hard).

Detects both hard contradictions ("has diabetes" vs "no diabetes")
and soft/conditional contradictions ("no SOB at rest" vs "breathless on exertion").

Example:
    >>> from hsttb.metrics.clinical_contradiction import ClinicalContradictionDetector
    >>> detector = ClinicalContradictionDetector()
    >>> result = detector.analyze(text)
    >>> print(f"Contradiction score: {result.score:.1%}")
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ContradictionSeverity(Enum):
    """Severity of contradiction."""
    HARD = "hard"  # Direct contradiction: "has X" vs "no X"
    SOFT = "soft"  # Conditional contradiction: "no X usually" vs "sometimes X"
    AMBIGUOUS = "ambiguous"  # Unclear but potentially conflicting
    TEMPORAL = "temporal"  # Different timeframes: "had X" vs "no X now"


@dataclass
class Contradiction:
    """A detected contradiction."""
    entity: str
    statement1: str
    statement2: str
    severity: ContradictionSeverity
    explanation: str
    clinical_impact: str  # low, medium, high, critical
    span1: tuple[int, int] | None = None
    span2: tuple[int, int] | None = None


@dataclass
class ContradictionResult:
    """Result of contradiction analysis."""
    text: str
    contradictions: list[Contradiction] = field(default_factory=list)
    score: float = 1.0  # 1.0 = no contradictions, 0.0 = severe
    hard_count: int = 0
    soft_count: int = 0
    ambiguous_count: int = 0
    clinical_risk_level: str = "low"  # low, medium, high, critical


class ClinicalContradictionDetector:
    """
    Detect clinical contradictions in medical transcripts.

    Handles:
    - Hard contradictions: "has diabetes" vs "no diabetes"
    - Soft contradictions: "no SOB at rest" vs "SOB on exertion"
    - Temporal contradictions: "had heart attack" vs "no heart attack"
    - Medication contradictions: "takes aspirin" vs "stopped aspirin"
    """

    # Symptom pairs that are commonly conditionally related
    CONDITIONAL_SYMPTOM_PAIRS = {
        # (symptom, modifier_context) -> can coexist with negated form
        ("shortness of breath", "at rest"): ["on exertion", "when walking", "climbing stairs"],
        ("breathless", "at rest"): ["on exertion", "when walking", "climbing stairs"],
        ("chest pain", "at rest"): ["on exertion", "with activity"],
        ("pain", "at rest"): ["on movement", "with activity"],
        ("dizziness", "usually"): ["sometimes", "occasionally"],
        ("nausea", "usually"): ["sometimes", "occasionally"],
    }

    # Hedging words that indicate soft assertions
    HEDGING_WORDS = [
        "usually", "sometimes", "occasionally", "rarely",
        "often", "mostly", "typically", "generally",
        "not always", "not every time",
    ]

    # Negation patterns
    NEGATION_PATTERNS = [
        r"\bno\b", r"\bnot\b", r"\bdenies?\b", r"\bwithout\b",
        r"\babsent\b", r"\bnegative\b", r"\bdoesn'?t\b", r"\bdon'?t\b",
    ]

    # Affirmation patterns
    AFFIRMATION_PATTERNS = [
        r"\bhas\b", r"\bhave\b", r"\bfeel\b", r"\bfeels?\b",
        r"\breports?\b", r"\bexperiencing\b", r"\bnoticed?\b",
    ]

    def __init__(self) -> None:
        """Initialize detector."""
        self._negation_re = re.compile(
            "|".join(self.NEGATION_PATTERNS), re.IGNORECASE
        )
        self._affirmation_re = re.compile(
            "|".join(self.AFFIRMATION_PATTERNS), re.IGNORECASE
        )

    def analyze(self, text: str) -> ContradictionResult:
        """
        Analyze text for clinical contradictions.

        Args:
            text: Clinical transcript text.

        Returns:
            ContradictionResult with detected contradictions.
        """
        contradictions: list[Contradiction] = []

        # Split into sentences
        sentences = self._split_sentences(text)

        # Track entity states across sentences
        entity_states: dict[str, list[tuple[str, str, int]]] = {}  # entity -> [(state, context, sent_idx)]

        # Analyze each sentence
        for idx, sentence in enumerate(sentences):
            self._extract_entity_states(sentence, idx, entity_states)

        # Find contradictions
        for entity, states in entity_states.items():
            if len(states) < 2:
                continue

            # Compare all pairs of states
            for i, (state1, ctx1, idx1) in enumerate(states):
                for state2, ctx2, idx2 in states[i + 1:]:
                    contradiction = self._check_contradiction(
                        entity, state1, ctx1, idx1,
                        state2, ctx2, idx2, sentences
                    )
                    if contradiction:
                        contradictions.append(contradiction)

        # Also check for medication contradictions
        med_contradictions = self._check_medication_contradictions(text, sentences)
        contradictions.extend(med_contradictions)

        # Calculate score
        hard_count = sum(1 for c in contradictions if c.severity == ContradictionSeverity.HARD)
        soft_count = sum(1 for c in contradictions if c.severity == ContradictionSeverity.SOFT)
        ambiguous_count = sum(1 for c in contradictions if c.severity == ContradictionSeverity.AMBIGUOUS)

        # Score calculation:
        # - Hard contradiction: -0.20 each
        # - Soft contradiction: -0.10 each
        # - Ambiguous: -0.05 each
        penalty = (hard_count * 0.20) + (soft_count * 0.10) + (ambiguous_count * 0.05)
        score = max(0.0, 1.0 - penalty)

        # Determine clinical risk level
        if hard_count > 0:
            risk_level = "high" if hard_count > 1 else "medium"
        elif soft_count > 1:
            risk_level = "medium"
        elif soft_count > 0 or ambiguous_count > 0:
            risk_level = "low"
        else:
            risk_level = "none"

        return ContradictionResult(
            text=text,
            contradictions=contradictions,
            score=score,
            hard_count=hard_count,
            soft_count=soft_count,
            ambiguous_count=ambiguous_count,
            clinical_risk_level=risk_level,
        )

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Handle common transcript patterns
        # Split on newlines, periods, or speaker changes
        sentences = re.split(r'[\n.?!]+|(?=\b(?:Doctor|Patient|Dr|Pt):\s*)', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_trackable_entities(self) -> list[str]:
        """Get medical entities to track for contradictions."""
        try:
            from hsttb.metrics.medical_terms import get_medical_terms
            provider = get_medical_terms()

            # Combine symptoms and conditions
            entities = list(provider.get_symptoms() | provider.get_conditions())
            if entities:
                return entities
        except Exception as e:
            logger.warning(f"Could not load medical terms: {e}")

        # Minimal fallback
        return [
            "shortness of breath", "breathless", "chest pain",
            "fatigue", "dizziness", "nausea", "swelling",
            "diabetes", "hypertension", "heart disease", "heart failure",
        ]

    def _extract_entity_states(
        self,
        sentence: str,
        sent_idx: int,
        entity_states: dict[str, list[tuple[str, str, int]]],
    ) -> None:
        """Extract entity states from a sentence."""
        sentence_lower = sentence.lower()

        # Get entities dynamically
        entities = self._get_trackable_entities()

        for entity in entities:
            if entity in sentence_lower:
                # Determine state
                state = self._determine_state(sentence_lower, entity)
                # Get context (modifiers around entity)
                context = self._get_context(sentence_lower, entity)

                if entity not in entity_states:
                    entity_states[entity] = []
                entity_states[entity].append((state, context, sent_idx))

    def _determine_state(self, sentence: str, entity: str) -> str:
        """Determine if entity is affirmed, negated, or conditional."""
        # Find entity position
        entity_pos = sentence.find(entity)
        if entity_pos == -1:
            return "unknown"

        # Check prefix for negation/affirmation
        prefix = sentence[max(0, entity_pos - 50):entity_pos]

        # Check for hedging
        for hedge in self.HEDGING_WORDS:
            if hedge in prefix or hedge in sentence[entity_pos:entity_pos + 50]:
                if self._negation_re.search(prefix):
                    return "negated_conditional"
                return "affirmed_conditional"

        # Check for negation - but handle conversational patterns
        negation_match = self._negation_re.search(prefix)
        if negation_match:
            # Get text between negation and entity
            text_after_negation = prefix[negation_match.end():]

            # Handle conversational "No, ..." patterns where "No" answers a question
            # e.g., "No, just the headache" - here "No" answers "any other symptoms?"
            # and "just the headache" AFFIRMS headache, not negates it
            if negation_match.group().lower() == "no":
                # Check for "No, just" or "No, only" patterns - these AFFIRM the entity
                if re.search(r"^[,\s]+(?:just|only|mainly)\b", text_after_negation):
                    return "affirmed"

                # Check for sentence-start "No," followed by punctuation
                # This usually answers a question, not negates what follows
                if re.match(r"^[,\s]*$", text_after_negation):
                    # "No" is immediately followed by entity with only comma/space
                    # Check if "No" is at start of sentence/clause
                    prefix_before_no = prefix[:negation_match.start()].strip()
                    if not prefix_before_no or prefix_before_no.endswith((".","?","!",":")):
                        # "No" at start - likely answering a question
                        return "mentioned"

            return "negated"

        # Check for affirmation
        if self._affirmation_re.search(prefix):
            return "affirmed"

        return "mentioned"

    def _get_context(self, sentence: str, entity: str) -> str:
        """Get context modifiers around entity."""
        entity_pos = sentence.find(entity)
        if entity_pos == -1:
            return ""

        # Get surrounding text
        start = max(0, entity_pos - 30)
        end = min(len(sentence), entity_pos + len(entity) + 30)
        context = sentence[start:end]

        # Extract key modifiers
        modifiers = []
        if "at rest" in context:
            modifiers.append("at_rest")
        if "on exertion" in context or "when walking" in context or "climbing" in context:
            modifiers.append("on_exertion")
        if "sometimes" in context or "occasionally" in context:
            modifiers.append("sometimes")
        if "usually" in context or "mostly" in context:
            modifiers.append("usually")

        return "_".join(modifiers) if modifiers else "general"

    def _check_contradiction(
        self,
        entity: str,
        state1: str,
        ctx1: str,
        idx1: int,
        state2: str,
        ctx2: str,
        idx2: int,
        sentences: list[str],
    ) -> Contradiction | None:
        """Check if two states are contradictory."""
        # Get actual sentences
        sent1 = sentences[idx1] if idx1 < len(sentences) else ""
        sent2 = sentences[idx2] if idx2 < len(sentences) else ""

        # Hard contradiction: negated vs affirmed (no conditionals)
        if state1 == "negated" and state2 == "affirmed":
            return Contradiction(
                entity=entity,
                statement1=sent1[:100],
                statement2=sent2[:100],
                severity=ContradictionSeverity.HARD,
                explanation=f"'{entity}' is both denied and affirmed",
                clinical_impact="high",
            )

        if state1 == "affirmed" and state2 == "negated":
            return Contradiction(
                entity=entity,
                statement1=sent1[:100],
                statement2=sent2[:100],
                severity=ContradictionSeverity.HARD,
                explanation=f"'{entity}' is both affirmed and denied",
                clinical_impact="high",
            )

        # Soft contradiction: negated_conditional vs affirmed_conditional
        # with incompatible contexts
        if "conditional" in state1 or "conditional" in state2:
            # Check if contexts explain the difference
            if ctx1 == ctx2:
                # Same context but different states = soft contradiction
                return Contradiction(
                    entity=entity,
                    statement1=sent1[:100],
                    statement2=sent2[:100],
                    severity=ContradictionSeverity.SOFT,
                    explanation=f"'{entity}' has conflicting conditional states",
                    clinical_impact="medium",
                )

            # Different contexts might be valid (e.g., "no SOB at rest" + "SOB on exertion")
            if self._contexts_complementary(ctx1, ctx2):
                # This is actually OK - not a contradiction
                return None

            # Ambiguous - unclear if truly contradictory
            return Contradiction(
                entity=entity,
                statement1=sent1[:100],
                statement2=sent2[:100],
                severity=ContradictionSeverity.AMBIGUOUS,
                explanation=f"'{entity}' appears in potentially conflicting contexts",
                clinical_impact="low",
            )

        return None

    def _contexts_complementary(self, ctx1: str, ctx2: str) -> bool:
        """Check if two contexts are complementary (not contradictory)."""
        # "at_rest" and "on_exertion" are complementary
        complementary_pairs = [
            ("at_rest", "on_exertion"),
            ("usually", "sometimes"),
        ]

        for c1, c2 in complementary_pairs:
            if (c1 in ctx1 and c2 in ctx2) or (c2 in ctx1 and c1 in ctx2):
                return True

        return False

    def _get_medication_list(self) -> list[str]:
        """Get medication names dynamically."""
        try:
            from hsttb.metrics.medical_terms import get_medical_terms
            provider = get_medical_terms()
            drugs = list(provider.get_drugs())
            if drugs:
                return drugs
        except Exception as e:
            logger.warning(f"Could not load drug list: {e}")

        # Minimal fallback
        return []

    def _check_medication_contradictions(
        self,
        text: str,
        sentences: list[str],
    ) -> list[Contradiction]:
        """Check for medication-related contradictions."""
        contradictions = []
        text_lower = text.lower()

        # Get medications dynamically
        medications = self._get_medication_list()

        for med in medications:
            if med not in text_lower:
                continue

            # Find all mentions
            mentions = list(re.finditer(rf"\b{med}\b", text_lower))
            if len(mentions) < 2:
                continue

            # Check for conflicting usage patterns
            takes_pattern = re.search(
                rf"(?:takes?|taking|on)\s+{med}", text_lower
            )
            stopped_pattern = re.search(
                rf"(?:stopped?|avoid|stop|not\s+taking|discontinued?)\s+{med}",
                text_lower
            )
            sometimes_pattern = re.search(
                rf"(?:sometimes|occasionally|when|as\s+needed)\s+.*{med}",
                text_lower
            )

            if takes_pattern and stopped_pattern:
                contradictions.append(Contradiction(
                    entity=med,
                    statement1=takes_pattern.group()[:50],
                    statement2=stopped_pattern.group()[:50],
                    severity=ContradictionSeverity.SOFT,
                    explanation=f"Conflicting statements about {med} usage",
                    clinical_impact="medium",
                ))

            # Check for inconsistent dosing
            doses = re.findall(
                rf"{med}\s+(\d+)\s*(?:mg|milligram)",
                text_lower
            )
            if len(set(doses)) > 1:
                contradictions.append(Contradiction(
                    entity=med,
                    statement1=f"{med} {doses[0]}mg",
                    statement2=f"{med} {doses[1]}mg",
                    severity=ContradictionSeverity.AMBIGUOUS,
                    explanation=f"Multiple doses mentioned for {med}",
                    clinical_impact="medium",
                ))

        return contradictions


# Singleton
_detector: ClinicalContradictionDetector | None = None


def get_contradiction_detector() -> ClinicalContradictionDetector:
    """Get singleton detector."""
    global _detector
    if _detector is None:
        _detector = ClinicalContradictionDetector()
    return _detector


def detect_clinical_contradictions(text: str) -> ContradictionResult:
    """Convenience function to detect contradictions."""
    return get_contradiction_detector().analyze(text)
