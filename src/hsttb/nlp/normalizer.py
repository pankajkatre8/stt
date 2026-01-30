"""
Medical text normalizer for TER computation.

This module provides text normalization utilities specifically
designed for medical text, including abbreviation expansion,
case normalization, and dosage standardization.

Example:
    >>> normalizer = MedicalTextNormalizer()
    >>> normalizer.normalize("Patient takes 500MG of metformin BID")
    'patient takes 500 mg of metformin twice daily'
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class NormalizerConfig:
    """
    Configuration for text normalization.

    Attributes:
        lowercase: Convert to lowercase.
        expand_abbreviations: Expand medical abbreviations.
        normalize_whitespace: Collapse multiple spaces.
        normalize_numbers: Standardize number formats.
        normalize_dosages: Standardize dosage formats.
        preserve_case_for_acronyms: Keep acronyms in original case.
    """

    lowercase: bool = True
    expand_abbreviations: bool = True
    normalize_whitespace: bool = True
    normalize_numbers: bool = True
    normalize_dosages: bool = True
    preserve_case_for_acronyms: bool = False


@dataclass
class MedicalTextNormalizer:
    """
    Normalizer for medical text.

    Provides various normalization functions for medical text
    to enable accurate term matching and comparison.

    Attributes:
        config: Normalization configuration.

    Example:
        >>> normalizer = MedicalTextNormalizer()
        >>> normalizer.normalize("BP 120/80 mmHg")
        'blood pressure 120/80 mmhg'
    """

    config: NormalizerConfig = field(default_factory=NormalizerConfig)

    # Common medical abbreviations
    ABBREVIATIONS: dict[str, str] = field(
        default_factory=lambda: {
            # Frequency
            "prn": "as needed",
            "bid": "twice daily",
            "tid": "three times daily",
            "qid": "four times daily",
            "qd": "once daily",
            "qhs": "at bedtime",
            "qam": "every morning",
            "qpm": "every evening",
            "q4h": "every 4 hours",
            "q6h": "every 6 hours",
            "q8h": "every 8 hours",
            "q12h": "every 12 hours",
            # Routes
            "po": "by mouth",
            "iv": "intravenous",
            "im": "intramuscular",
            "subq": "subcutaneous",
            "sl": "sublingual",
            "pr": "per rectum",
            "top": "topical",
            # Units
            "mg": "milligram",
            "mcg": "microgram",
            "ml": "milliliter",
            "cc": "cubic centimeter",
            "g": "gram",
            "kg": "kilogram",
            "l": "liter",
            # Vital signs
            "bp": "blood pressure",
            "hr": "heart rate",
            "rr": "respiratory rate",
            "temp": "temperature",
            "spo2": "oxygen saturation",
            "bpm": "beats per minute",
            # Medical history
            "hx": "history",
            "dx": "diagnosis",
            "rx": "prescription",
            "tx": "treatment",
            "sx": "symptoms",
            "cx": "culture",
            # Common conditions
            "htn": "hypertension",
            "dm": "diabetes mellitus",
            "cad": "coronary artery disease",
            "chf": "congestive heart failure",
            "copd": "chronic obstructive pulmonary disease",
            "ckd": "chronic kidney disease",
            "uti": "urinary tract infection",
            "gerd": "gastroesophageal reflux disease",
            "afib": "atrial fibrillation",
            "mi": "myocardial infarction",
            "cva": "cerebrovascular accident",
            "dvt": "deep vein thrombosis",
            "pe": "pulmonary embolism",
            # Labs
            "cbc": "complete blood count",
            "bmp": "basic metabolic panel",
            "cmp": "comprehensive metabolic panel",
            "lfts": "liver function tests",
            "ua": "urinalysis",
            "ekg": "electrocardiogram",
            "ecg": "electrocardiogram",
            # Common terms
            "pt": "patient",
            "yo": "year old",
            "y/o": "year old",
            "h/o": "history of",
            "c/o": "complains of",
            "s/p": "status post",
            "w/o": "without",
            "w/": "with",
            "nka": "no known allergies",
            "nkda": "no known drug allergies",
        }
    )

    # Number words to digits
    NUMBER_WORDS: dict[str, str] = field(
        default_factory=lambda: {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "twenty": "20",
            "thirty": "30",
            "forty": "40",
            "fifty": "50",
            "hundred": "100",
        }
    )

    def normalize(self, text: str) -> str:
        """
        Apply all normalizations.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text.
        """
        if not text:
            return text

        result = text

        if self.config.normalize_whitespace:
            result = self.normalize_whitespace(result)

        if self.config.normalize_dosages:
            result = self.normalize_dosages(result)

        if self.config.normalize_numbers:
            result = self.normalize_numbers(result)

        if self.config.expand_abbreviations:
            result = self.expand_abbreviations(result)

        if self.config.lowercase:
            result = self.normalize_case(result)

        if self.config.normalize_whitespace:
            result = self.normalize_whitespace(result)

        return result

    def normalize_case(self, text: str) -> str:
        """
        Convert text to lowercase.

        Args:
            text: Text to normalize.

        Returns:
            Lowercase text.
        """
        return text.lower()

    def normalize_whitespace(self, text: str) -> str:
        """
        Collapse multiple whitespace to single space.

        Args:
            text: Text to normalize.

        Returns:
            Text with normalized whitespace.
        """
        return " ".join(text.split())

    def expand_abbreviations(self, text: str) -> str:
        """
        Expand medical abbreviations.

        Args:
            text: Text with abbreviations.

        Returns:
            Text with expanded abbreviations.
        """
        result = text
        for abbrev, expansion in self.ABBREVIATIONS.items():
            # Match word boundaries (case-insensitive)
            pattern = rf"\b{re.escape(abbrev)}\b"
            result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
        return result

    def normalize_numbers(self, text: str) -> str:
        """
        Normalize number representations.

        Converts number words to digits.

        Args:
            text: Text with numbers.

        Returns:
            Text with normalized numbers.
        """
        result = text
        for word, digit in self.NUMBER_WORDS.items():
            pattern = rf"\b{word}\b"
            result = re.sub(pattern, digit, result, flags=re.IGNORECASE)
        return result

    def normalize_dosages(self, text: str) -> str:
        """
        Standardize dosage formats.

        Adds space between number and unit, standardizes units.

        Args:
            text: Text with dosages.

        Returns:
            Text with normalized dosages.

        Example:
            >>> normalizer.normalize_dosages("500mg")
            '500 mg'
            >>> normalizer.normalize_dosages("0.5mcg")
            '0.5 mcg'
        """
        # Add space between number and unit
        # Matches: 500mg, 0.5ml, 100mcg, etc.
        pattern = r"(\d+(?:\.\d+)?)\s*(mg|mcg|ml|cc|g|kg|l|iu|meq)\b"
        result = re.sub(pattern, r"\1 \2", text, flags=re.IGNORECASE)

        # Standardize unit variations
        result = re.sub(r"\bmg\b", "mg", result, flags=re.IGNORECASE)
        result = re.sub(r"\bmcg\b", "mcg", result, flags=re.IGNORECASE)
        result = re.sub(r"\bml\b", "ml", result, flags=re.IGNORECASE)

        return result

    def normalize_for_comparison(
        self,
        text1: str,
        text2: str,
    ) -> tuple[str, str]:
        """
        Normalize two texts for comparison.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Tuple of normalized texts.
        """
        return self.normalize(text1), self.normalize(text2)

    def are_equivalent(
        self,
        text1: str,
        text2: str,
    ) -> bool:
        """
        Check if two texts are equivalent after normalization.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            True if equivalent after normalization.
        """
        norm1, norm2 = self.normalize_for_comparison(text1, text2)
        return norm1 == norm2

    def add_abbreviation(self, abbrev: str, expansion: str) -> None:
        """
        Add custom abbreviation.

        Args:
            abbrev: Abbreviation to add.
            expansion: Expansion for the abbreviation.
        """
        self.ABBREVIATIONS[abbrev.lower()] = expansion

    def remove_abbreviation(self, abbrev: str) -> bool:
        """
        Remove an abbreviation.

        Args:
            abbrev: Abbreviation to remove.

        Returns:
            True if removed, False if not found.
        """
        if abbrev.lower() in self.ABBREVIATIONS:
            del self.ABBREVIATIONS[abbrev.lower()]
            return True
        return False


def create_normalizer(
    lowercase: bool = True,
    expand_abbreviations: bool = True,
    normalize_dosages: bool = True,
) -> MedicalTextNormalizer:
    """
    Create a normalizer with custom configuration.

    Args:
        lowercase: Whether to lowercase text.
        expand_abbreviations: Whether to expand abbreviations.
        normalize_dosages: Whether to normalize dosages.

    Returns:
        Configured MedicalTextNormalizer.
    """
    config = NormalizerConfig(
        lowercase=lowercase,
        expand_abbreviations=expand_abbreviations,
        normalize_dosages=normalize_dosages,
    )
    return MedicalTextNormalizer(config=config)


def normalize_for_ter(text: str) -> str:
    """
    Normalize text for TER computation.

    Applies standard normalization suitable for comparing
    ground truth and predicted transcriptions.

    Args:
        text: Text to normalize.

    Returns:
        Normalized text.
    """
    normalizer = MedicalTextNormalizer()
    return normalizer.normalize(text)
