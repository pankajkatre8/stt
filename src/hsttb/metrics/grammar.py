"""
Grammar checking for transcription quality evaluation.

Uses language-tool-python for rule-based grammar checking with
medical term awareness to avoid false positives.

Example:
    >>> from hsttb.metrics.grammar import GrammarChecker
    >>> checker = GrammarChecker()
    >>> result = checker.check("Patient have chest pain")
    >>> print(f"Errors: {len(result.errors)}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class GrammarError:
    """A single grammar error."""

    message: str
    offset: int
    length: int
    text: str  # The erroneous text
    suggestions: list[str]
    rule_id: str
    category: str


@dataclass
class GrammarResult:
    """Result of grammar checking."""

    text: str
    errors: list[GrammarError] = field(default_factory=list)
    score: float = 1.0  # 0-1 score (higher is better)
    word_count: int = 0
    error_rate: float = 0.0  # errors per word


class GrammarChecker:
    """
    Grammar checker with medical term awareness.

    Uses language-tool-python for rule-based checking.
    Filters out false positives for common medical terms.

    Attributes:
        language: Language code (default: "en-US").
        medical_terms: Set of medical terms to ignore.

    Example:
        >>> checker = GrammarChecker()
        >>> result = checker.check("Patient takes metformin daily")
        >>> print(f"Score: {result.score:.1%}")
    """

    # Common medical terms that might be flagged incorrectly
    MEDICAL_TERMS = {
        # Drugs
        "metformin", "lisinopril", "atorvastatin", "omeprazole", "amlodipine",
        "metoprolol", "losartan", "gabapentin", "hydrochlorothiazide", "warfarin",
        "prednisone", "amoxicillin", "azithromycin", "ciprofloxacin", "sertraline",
        "fluoxetine", "escitalopram", "alprazolam", "lorazepam", "methotrexate",
        "hydroxychloroquine", "sulfasalazine", "levothyroxine", "pantoprazole",
        # Conditions
        "hypertension", "hyperlipidemia", "dyslipidemia", "hypothyroidism",
        "hyperthyroidism", "tachycardia", "bradycardia", "arrhythmia",
        "myocardial", "infarction", "cerebrovascular", "atherosclerosis",
        "dyspnea", "orthopnea", "tachypnea", "bradypnea", "hypoxia", "hypoxemia",
        "hyperglycemia", "hypoglycemia", "hyperkalemia", "hypokalemia",
        "hypernatremia", "hyponatremia", "hypercalcemia", "hypocalcemia",
        # Anatomy
        "subcutaneous", "intramuscular", "intravenous", "sublingual",
        "epigastric", "periumbilical", "suprapubic", "retrosternal",
        # Procedures
        "colonoscopy", "endoscopy", "bronchoscopy", "laparoscopy",
        "echocardiogram", "electrocardiogram",
        # Abbreviations
        "mg", "mcg", "ml", "kg", "bid", "tid", "qid", "prn", "qhs", "qam",
        "po", "iv", "im", "sq", "sl", "pr", "ou", "od", "os",
        "ekg", "ecg", "ct", "mri", "cbc", "bmp", "cmp", "hba1c", "a1c",
        "bp", "hr", "rr", "spo2", "o2", "bmi", "gfr", "egfr",
    }

    # Singleton instance
    _instance: GrammarChecker | None = None

    def __new__(cls, language: str = "en-US") -> GrammarChecker:
        """Return singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        language: str = "en-US",
        additional_medical_terms: set[str] | None = None,
    ) -> None:
        """
        Initialize grammar checker.

        Args:
            language: Language code for grammar rules.
            additional_medical_terms: Extra medical terms to ignore.
        """
        if hasattr(self, "_initialized"):
            return

        self.language = language
        self.medical_terms = self.MEDICAL_TERMS.copy()
        if additional_medical_terms:
            self.medical_terms.update(t.lower() for t in additional_medical_terms)

        self._tool: Any = None
        self._initialized = True

    def _ensure_loaded(self) -> None:
        """Load language tool if not already loaded."""
        if self._tool is not None:
            return

        try:
            import language_tool_python

            logger.info(f"Loading LanguageTool for {self.language}")
            self._tool = language_tool_python.LanguageTool(self.language)
            logger.info("LanguageTool loaded")

        except ImportError as e:
            raise ImportError(
                "language-tool-python required for grammar checking. "
                "Install with: pip install language-tool-python"
            ) from e

    def check(self, text: str) -> GrammarResult:
        """
        Check grammar in text.

        Args:
            text: Text to check.

        Returns:
            GrammarResult with errors and score.
        """
        self._ensure_loaded()

        # Get all matches
        matches = self._tool.check(text)

        # Filter out medical term false positives
        filtered_errors = []
        for match in matches:
            error_text = text[match.offset : match.offset + match.errorLength]

            # Skip if it's a medical term
            if error_text.lower() in self.medical_terms:
                continue

            # Skip if error is within a medical term
            if self._is_within_medical_term(text, match.offset, match.errorLength):
                continue

            filtered_errors.append(
                GrammarError(
                    message=match.message,
                    offset=match.offset,
                    length=match.errorLength,
                    text=error_text,
                    suggestions=match.replacements[:5],  # Limit suggestions
                    rule_id=match.ruleId,
                    category=match.category,
                )
            )

        # Calculate score
        words = text.split()
        word_count = len(words)
        error_count = len(filtered_errors)

        if word_count == 0:
            score = 1.0
            error_rate = 0.0
        else:
            # Score decreases with more errors
            error_rate = error_count / word_count
            score = max(0.0, 1.0 - error_rate)

        return GrammarResult(
            text=text,
            errors=filtered_errors,
            score=score,
            word_count=word_count,
            error_rate=error_rate,
        )

    def _is_within_medical_term(self, text: str, offset: int, length: int) -> bool:
        """Check if error position is within a medical term."""
        text_lower = text.lower()
        for term in self.medical_terms:
            idx = text_lower.find(term)
            while idx != -1:
                term_end = idx + len(term)
                # Check if error overlaps with this term
                if offset < term_end and offset + length > idx:
                    return True
                idx = text_lower.find(term, idx + 1)
        return False

    def close(self) -> None:
        """Close the language tool server."""
        if self._tool is not None:
            try:
                self._tool.close()
            except Exception:
                pass
            self._tool = None

    @property
    def is_available(self) -> bool:
        """Check if grammar checking is available."""
        try:
            import language_tool_python  # noqa: F401

            return True
        except ImportError:
            return False


# Convenience function
def check_grammar(text: str) -> GrammarResult:
    """
    Convenience function to check grammar.

    Args:
        text: Text to check.

    Returns:
        GrammarResult.
    """
    checker = GrammarChecker()
    return checker.check(text)
