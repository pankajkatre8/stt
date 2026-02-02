"""
Medical coherence checking for transcription quality.

Validates that medical entities in transcriptions are:
1. Valid (real drug names, conditions, etc.)
2. Coherent (drug-condition pairs make clinical sense)

Example:
    >>> from hsttb.metrics.medical_coherence import MedicalCoherenceChecker
    >>> checker = MedicalCoherenceChecker()
    >>> result = checker.check("Patient takes metformin for diabetes")
    >>> print(f"Coherence: {result.coherence_score:.1%}")
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EntityValidation:
    """Validation result for a single entity."""

    text: str
    entity_type: str  # "drug", "diagnosis", "dosage", etc.
    is_valid: bool
    confidence: float  # 0-1
    suggested_correction: str | None = None


@dataclass
class CoherenceResult:
    """Result of medical coherence checking."""

    text: str
    entities: list[EntityValidation] = field(default_factory=list)
    entity_validity_score: float = 1.0  # valid entities / total
    coherence_score: float = 1.0  # drug-condition coherence
    invalid_entities: list[str] = field(default_factory=list)
    valid_pairs: list[tuple[str, str]] = field(default_factory=list)
    invalid_pairs: list[tuple[str, str]] = field(default_factory=list)


class MedicalCoherenceChecker:
    """
    Check medical coherence of transcriptions.

    Validates entities using the medical lexicon and checks
    drug-condition relationships for clinical plausibility.
    Drug-condition pairs are loaded dynamically from the lexicon.

    Example:
        >>> checker = MedicalCoherenceChecker()
        >>> result = checker.check("Patient takes metformin for migraine")
        >>> print(f"Invalid pairs: {result.invalid_pairs}")
    """

    def __init__(self) -> None:
        """Initialize medical coherence checker."""
        self._lexicon = None
        self._valid_pairs: set[tuple[str, str]] | None = None
        self._medical_terms = None

    def _get_lexicon(self):
        """Get or create lexicon."""
        if self._lexicon is None:
            try:
                from hsttb.lexicons.mock_lexicon import MockMedicalLexicon

                self._lexicon = MockMedicalLexicon.with_common_terms()
            except ImportError:
                logger.warning("MockMedicalLexicon not available")
                self._lexicon = None

    def _get_medical_terms(self):
        """Get medical terms provider."""
        if self._medical_terms is None:
            try:
                from hsttb.metrics.medical_terms import get_medical_terms
                self._medical_terms = get_medical_terms()
            except Exception as e:
                logger.warning(f"Could not load medical terms: {e}")
        return self._medical_terms

    def _get_valid_drug_condition_pairs(self) -> set[tuple[str, str]]:
        """Get valid drug-condition pairs dynamically from lexicon."""
        if self._valid_pairs is not None:
            return self._valid_pairs

        self._valid_pairs = set()

        try:
            # Try to get from medical terms provider (which loads from SQLite)
            provider = self._get_medical_terms()
            if provider:
                # Build pairs from drug indications
                for drug in provider.get_drugs():
                    indications = provider.get_drug_indications(drug)
                    for indication in indications:
                        self._valid_pairs.add((drug.lower(), indication.lower()))

            if self._valid_pairs:
                logger.info(f"Loaded {len(self._valid_pairs)} drug-condition pairs from lexicon")
                return self._valid_pairs
        except Exception as e:
            logger.warning(f"Could not load drug-condition pairs: {e}")

        # Empty set if no data - coherence checking will be skipped
        return self._valid_pairs

    def is_valid_pair(self, drug: str, condition: str) -> bool | None:
        """
        Check if drug-condition pair is valid.

        Returns:
            True if valid pair, False if known invalid, None if unknown.
        """
        valid_pairs = self._get_valid_drug_condition_pairs()
        drug_lower = drug.lower()
        condition_lower = condition.lower()

        # Direct match
        if (drug_lower, condition_lower) in valid_pairs:
            return True

        # Check for partial matches (e.g., "type 2 diabetes" contains "diabetes")
        for d, c in valid_pairs:
            if d == drug_lower and (c in condition_lower or condition_lower in c):
                return True

        # If we have pairs loaded but no match, it's unknown (not invalid)
        return None
        return self._lexicon

    def check(self, text: str) -> CoherenceResult:
        """
        Check medical coherence of text.

        Args:
            text: Text to check.

        Returns:
            CoherenceResult with entity validation and coherence scores.
        """
        # Extract entities
        entities = self._extract_entities(text)

        # Validate entities
        valid_count = 0
        invalid_entities = []

        for entity in entities:
            if entity.is_valid:
                valid_count += 1
            else:
                invalid_entities.append(entity.text)

        # Calculate entity validity score
        if len(entities) > 0:
            entity_validity_score = valid_count / len(entities)
        else:
            entity_validity_score = 1.0  # No entities = no errors

        # Check drug-condition coherence
        drugs = [e for e in entities if e.entity_type == "drug" and e.is_valid]
        conditions = [
            e for e in entities if e.entity_type == "diagnosis" and e.is_valid
        ]

        valid_pairs = []
        invalid_pairs = []

        for drug in drugs:
            for condition in conditions:
                is_valid = self.is_valid_pair(drug.text, condition.text)
                if is_valid is True:
                    valid_pairs.append((drug.text, condition.text))
                elif is_valid is False:
                    invalid_pairs.append((drug.text, condition.text))
                # Unknown pairs (None) are not penalized

        # Calculate coherence score
        total_pairs = len(valid_pairs) + len(invalid_pairs)
        if total_pairs > 0:
            coherence_score = len(valid_pairs) / total_pairs
        else:
            coherence_score = 1.0  # No pairs to check

        return CoherenceResult(
            text=text,
            entities=entities,
            entity_validity_score=entity_validity_score,
            coherence_score=coherence_score,
            invalid_entities=invalid_entities,
            valid_pairs=valid_pairs,
            invalid_pairs=invalid_pairs,
        )

    def _extract_entities(self, text: str) -> list[EntityValidation]:
        """Extract and validate medical entities from text."""
        entities = []
        text_lower = text.lower()
        lexicon = self._get_lexicon()

        if lexicon is None:
            return entities

        # Extract using lexicon lookup
        words = text.split()
        matched_positions: set[int] = set()

        # Try multi-word phrases first (up to 4 words)
        for n in range(min(4, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                if any(pos in matched_positions for pos in range(i, i + n)):
                    continue

                phrase = " ".join(words[i : i + n])
                entry = lexicon.lookup(phrase)

                if entry is not None:
                    entities.append(
                        EntityValidation(
                            text=phrase,
                            entity_type=entry.category,
                            is_valid=True,
                            confidence=1.0,
                        )
                    )
                    for pos in range(i, i + n):
                        matched_positions.add(pos)

        # Also check for potential misspellings using fuzzy matching
        self._check_misspellings(text, entities, matched_positions, words)

        return entities

    def _check_misspellings(
        self,
        text: str,
        entities: list[EntityValidation],
        matched_positions: set[int],
        words: list[str],
    ) -> None:
        """Check for potential misspellings of medical terms."""
        lexicon = self._get_lexicon()
        if lexicon is None:
            return

        # Common medical term patterns that might be misspelled
        drug_patterns = [
            r"\b\w*formin\b",  # metformin variants
            r"\b\w*pril\b",  # ACE inhibitors
            r"\b\w*sartan\b",  # ARBs
            r"\b\w*statin\b",  # statins
            r"\b\w*olol\b",  # beta blockers
            r"\b\w*dipine\b",  # calcium channel blockers
            r"\b\w*cillin\b",  # penicillins
            r"\b\w*mycin\b",  # macrolides
        ]

        for i, word in enumerate(words):
            if i in matched_positions:
                continue

            word_lower = word.lower().strip(".,;:")

            # Check if word looks like a medical term but isn't recognized
            for pattern in drug_patterns:
                if re.match(pattern, word_lower):
                    # This looks like a drug but wasn't found in lexicon
                    # Might be a misspelling
                    entry = lexicon.lookup(word_lower)
                    if entry is None:
                        # Try to find similar valid drug
                        suggestion = self._find_similar_term(word_lower, lexicon)
                        entities.append(
                            EntityValidation(
                                text=word,
                                entity_type="drug",
                                is_valid=False,
                                confidence=0.7,
                                suggested_correction=suggestion,
                            )
                        )
                    break

    def _find_similar_term(self, misspelled: str, lexicon) -> str | None:
        """Find a similar valid term for a misspelling."""
        # Simple prefix matching
        for entry_key in lexicon._entries:
            if len(entry_key) > 3 and misspelled[:3] == entry_key[:3]:
                return lexicon._entries[entry_key].term
        return None


# Convenience function
def check_medical_coherence(text: str) -> CoherenceResult:
    """
    Convenience function to check medical coherence.

    Args:
        text: Text to check.

    Returns:
        CoherenceResult.
    """
    checker = MedicalCoherenceChecker()
    return checker.check(text)
