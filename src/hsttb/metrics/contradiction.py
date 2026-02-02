"""
Internal contradiction detection for transcription quality.

Detects contradictions within a single transcript without requiring
ground truth. Catches issues like:
- "I don't have diabetes" followed by "diabetes medication"
- Conflicting statements about the same entity

Example:
    >>> from hsttb.metrics.contradiction import ContradictionDetector
    >>> detector = ContradictionDetector()
    >>> result = detector.detect("I don't have diabetes. My diabetes is controlled.")
    >>> print(f"Contradictions: {len(result.contradictions)}")
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)


class ContradictionPair(NamedTuple):
    """A detected contradiction."""

    statement1: str
    statement2: str
    entity: str
    contradiction_type: str  # "negation_conflict", "value_conflict", "temporal_conflict"
    severity: str  # "critical", "high", "medium"
    explanation: str


@dataclass
class ContradictionResult:
    """Result of contradiction detection."""

    # Score (0.0-1.0, higher = more consistent, fewer contradictions)
    consistency_score: float

    # Detected contradictions
    contradictions: list[ContradictionPair] = field(default_factory=list)

    # Statistics
    statements_analyzed: int = 0
    entities_tracked: int = 0

    @property
    def has_contradictions(self) -> bool:
        return len(self.contradictions) > 0

    @property
    def critical_count(self) -> int:
        return sum(1 for c in self.contradictions if c.severity == "critical")


class ContradictionDetector:
    """
    Detect internal contradictions in transcripts.

    Uses pattern matching and entity tracking to find:
    - Negation conflicts (affirm then deny, or vice versa)
    - Value conflicts (different values for same attribute)
    - Temporal conflicts (impossible timelines)
    """

    # Negation patterns
    NEGATION_PATTERNS = [
        r"\b(no|not|don'?t|doesn'?t|didn'?t|never|none|without|denies?|denied)\b",
        r"\b(isn'?t|aren'?t|wasn'?t|weren'?t|haven'?t|hasn'?t|won'?t|can'?t|couldn'?t)\b",
    ]

    # Medical entities to track
    MEDICAL_ENTITIES = [
        # Conditions
        r"\b(diabetes|diabetic)\b",
        r"\b(hypertension|high blood pressure|htn)\b",
        r"\b(chest pain|angina)\b",
        r"\b(shortness of breath|dyspnea|sob)\b",
        r"\b(nausea|vomiting)\b",
        r"\b(dizziness|dizzy|vertigo)\b",
        r"\b(headache|migraine)\b",
        r"\b(fever|febrile)\b",
        r"\b(cough|coughing)\b",
        r"\b(pain|painful)\b",
        r"\b(swelling|edema)\b",
        r"\b(bleeding|hemorrhage)\b",
        r"\b(infection|infected)\b",
        r"\b(allergy|allergic|allergies)\b",
        r"\b(pregnant|pregnancy)\b",
        # Medications
        r"\b(medication|medicine|drug|prescription)\b",
        r"\b(insulin)\b",
        r"\b(metformin)\b",
        r"\b(aspirin)\b",
    ]

    # Value patterns (for detecting value conflicts)
    VALUE_PATTERNS = [
        # Blood pressure
        (r"blood pressure[^\d]*(\d+)/(\d+)", "blood_pressure"),
        # Heart rate
        (r"(heart rate|pulse|hr)[^\d]*(\d+)", "heart_rate"),
        # Temperature
        (r"(temperature|temp)[^\d]*([\d.]+)", "temperature"),
        # Dosage
        (r"(\d+)\s*(mg|mcg|ml|units?)", "dosage"),
        # Age
        (r"(\d+)\s*(year|yr)s?\s*old", "age"),
    ]

    def __init__(self) -> None:
        """Initialize the contradiction detector."""
        self._negation_pattern = re.compile(
            "|".join(self.NEGATION_PATTERNS), re.IGNORECASE
        )
        self._entity_patterns = [
            (re.compile(p, re.IGNORECASE), p) for p in self.MEDICAL_ENTITIES
        ]
        self._value_patterns = [
            (re.compile(p, re.IGNORECASE), name) for p, name in self.VALUE_PATTERNS
        ]

    def detect(self, text: str) -> ContradictionResult:
        """
        Detect contradictions in text.

        Args:
            text: Transcript text to analyze.

        Returns:
            ContradictionResult with detected contradictions.
        """
        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) < 2:
            return ContradictionResult(
                consistency_score=1.0,
                statements_analyzed=len(sentences),
            )

        contradictions: list[ContradictionPair] = []
        entity_states: dict[str, list[tuple[str, bool, int]]] = {}  # entity -> [(sentence, is_negated, index)]

        # Track entity states across sentences
        for idx, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()

            # Skip questions - they don't make assertions about entities
            # Questions typically ask IF something exists, not assert it does
            is_question = self._is_question(sentence)
            if is_question:
                continue

            is_negated = bool(self._negation_pattern.search(sentence_lower))

            # Check each entity pattern
            for pattern, pattern_str in self._entity_patterns:
                if pattern.search(sentence_lower):
                    entity_name = self._extract_entity_name(pattern_str)

                    if entity_name not in entity_states:
                        entity_states[entity_name] = []

                    # Check for contradiction with previous states
                    for prev_sentence, prev_negated, prev_idx in entity_states[entity_name]:
                        if is_negated != prev_negated:
                            # Found negation conflict!
                            contradiction = ContradictionPair(
                                statement1=prev_sentence,
                                statement2=sentence,
                                entity=entity_name,
                                contradiction_type="negation_conflict",
                                severity=self._get_severity(entity_name),
                                explanation=f"Conflicting statements about '{entity_name}': "
                                           f"one affirms, one denies",
                            )
                            contradictions.append(contradiction)

                    entity_states[entity_name].append((sentence, is_negated, idx))

        # Check for value conflicts
        value_contradictions = self._detect_value_conflicts(sentences)
        contradictions.extend(value_contradictions)

        # Calculate consistency score
        if len(sentences) > 0:
            # Penalize based on number and severity of contradictions
            penalty = 0.0
            for c in contradictions:
                if c.severity == "critical":
                    penalty += 0.3
                elif c.severity == "high":
                    penalty += 0.2
                else:
                    penalty += 0.1

            consistency_score = max(0.0, 1.0 - penalty)
        else:
            consistency_score = 1.0

        return ContradictionResult(
            consistency_score=consistency_score,
            contradictions=contradictions,
            statements_analyzed=len(sentences),
            entities_tracked=len(entity_states),
        )

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_question(self, sentence: str) -> bool:
        """Detect if a sentence is a question.

        Questions don't make assertions - they ask about things.
        "Do you have diabetes?" is not asserting diabetes exists.
        """
        sentence_lower = sentence.lower().strip()

        # Check for question words at the start
        question_starters = [
            'do you', 'did you', 'have you', 'are you', 'were you',
            'is there', 'was there', 'can you', 'could you', 'would you',
            'what', 'when', 'where', 'why', 'how', 'which', 'who',
            'does', 'has', 'had', 'will', 'shall', 'should', 'may', 'might',
            'any history', 'any symptoms', 'noticed any', 'experienced any',
        ]

        for starter in question_starters:
            if sentence_lower.startswith(starter):
                return True

        # Check for question patterns in the middle
        question_patterns = [
            r'\bdo you have\b',
            r'\bhave you (had|noticed|experienced|been)\b',
            r'\bare you (taking|experiencing|having)\b',
            r'\bany (history|symptoms|pain|issues)\b.*\?',
        ]

        for pattern in question_patterns:
            if re.search(pattern, sentence_lower):
                return True

        return False

    def _extract_entity_name(self, pattern: str) -> str:
        """Extract readable entity name from pattern."""
        # Remove regex syntax
        clean = re.sub(r'\\b|\(|\)|\||\?', '', pattern)
        # Take first option if there are alternatives
        parts = clean.split('|') if '|' in clean else [clean]
        return parts[0].strip()

    def _get_severity(self, entity: str) -> str:
        """Determine severity based on entity type."""
        critical_entities = {'diabetes', 'diabetic', 'allergy', 'allergic', 'pregnant', 'pregnancy'}
        high_entities = {'chest pain', 'shortness of breath', 'bleeding', 'medication', 'insulin'}

        entity_lower = entity.lower()
        if entity_lower in critical_entities:
            return "critical"
        elif entity_lower in high_entities:
            return "high"
        return "medium"

    def _detect_value_conflicts(self, sentences: list[str]) -> list[ContradictionPair]:
        """Detect conflicting values for the same attribute."""
        contradictions: list[ContradictionPair] = []
        value_occurrences: dict[str, list[tuple[str, str, str]]] = {}  # name -> [(value, unit, sentence)]

        for sentence in sentences:
            for pattern, name in self._value_patterns:
                match = pattern.search(sentence)
                if match:
                    # Extract value
                    groups = match.groups()
                    value = groups[0] if groups else match.group()
                    unit = groups[1] if len(groups) > 1 else ""

                    if name not in value_occurrences:
                        value_occurrences[name] = []

                    # Check for conflicts with significant differences
                    for prev_value, prev_unit, prev_sentence in value_occurrences[name]:
                        if self._values_conflict(value, prev_value, name):
                            contradiction = ContradictionPair(
                                statement1=prev_sentence,
                                statement2=sentence,
                                entity=name,
                                contradiction_type="value_conflict",
                                severity="high",
                                explanation=f"Conflicting values for '{name}': "
                                           f"{prev_value} vs {value}",
                            )
                            contradictions.append(contradiction)

                    value_occurrences[name].append((value, unit, sentence))

        return contradictions

    def _values_conflict(self, val1: str, val2: str, value_type: str) -> bool:
        """Check if two values conflict significantly."""
        try:
            v1 = float(val1.replace('/', '.'))
            v2 = float(val2.replace('/', '.'))

            # Significant difference thresholds by type
            thresholds = {
                "blood_pressure": 20,
                "heart_rate": 20,
                "temperature": 1.0,
                "dosage": 0,  # Any difference in dosage is significant
                "age": 0,  # Age shouldn't change
            }

            threshold = thresholds.get(value_type, 10)
            return abs(v1 - v2) > threshold
        except (ValueError, TypeError):
            return val1 != val2


def detect_contradictions(text: str) -> ContradictionResult:
    """
    Convenience function to detect contradictions.

    Args:
        text: Text to analyze.

    Returns:
        ContradictionResult.
    """
    detector = ContradictionDetector()
    return detector.detect(text)
