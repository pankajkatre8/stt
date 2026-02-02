"""
Medication Dosage Plausibility Checker.

Validates that medication dosages mentioned in transcripts are
clinically reasonable without needing ground truth.

Example:
    >>> from hsttb.metrics.dosage_plausibility import DosagePlausibilityChecker
    >>> checker = DosagePlausibilityChecker()
    >>> result = checker.check("Patient takes amlodipine 5mg twice daily")
    >>> print(f"Plausibility: {result.score:.1%}")
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DosageInfo:
    """Information about a detected dosage."""
    drug: str
    dose_value: float
    dose_unit: str
    frequency: str | None
    raw_text: str
    is_plausible: bool
    issue: str | None = None
    typical_range: str | None = None


@dataclass
class DosagePlausibilityResult:
    """Result of dosage plausibility check."""
    text: str
    dosages: list[DosageInfo] = field(default_factory=list)
    score: float = 1.0
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DosagePlausibilityChecker:
    """
    Check if medication dosages are clinically plausible.

    Uses known typical dosage ranges for common medications.
    Flags unusual dosages without needing ground truth.
    Dosage ranges are loaded dynamically from the medical terms provider.
    """

    # Frequency patterns for detecting dosing frequency
    FREQUENCY_PATTERNS = {
        "once daily": [r"once\s+daily", r"once\s+a\s+day", r"qd", r"daily"],
        "twice daily": [r"twice\s+daily", r"twice\s+a\s+day", r"bid", r"two\s+times", r"2x"],
        "three times": [r"three\s+times", r"tid", r"3x", r"three\s+times\s+daily"],
        "four times": [r"four\s+times", r"qid", r"4x"],
        "at night": [r"at\s+night", r"bedtime", r"hs", r"at\s+bedtime"],
        "as needed": [r"as\s+needed", r"prn", r"when\s+needed"],
        "morning": [r"in\s+the\s+morning", r"morning", r"am"],
        "evening": [r"in\s+the\s+evening", r"evening", r"pm"],
    }

    def __init__(self) -> None:
        """Initialize checker."""
        self._dosage_ranges: dict | None = None
        # Compile frequency patterns
        self._freq_patterns = {}
        for freq_name, patterns in self.FREQUENCY_PATTERNS.items():
            self._freq_patterns[freq_name] = re.compile(
                "|".join(patterns), re.IGNORECASE
            )

    def _get_dosage_ranges(self) -> dict:
        """Get dosage ranges dynamically from provider."""
        if self._dosage_ranges is not None:
            return self._dosage_ranges

        ranges = {}

        try:
            from hsttb.metrics.medical_terms import get_medical_terms
            provider = get_medical_terms()

            for drug, dosage in provider.get_all_dosage_ranges().items():
                ranges[drug] = {
                    "min": dosage.min_dose,
                    "max": dosage.max_dose,
                    "common": dosage.common_doses,
                    "unit": dosage.unit,
                    "typical_freq": dosage.typical_frequencies,
                    "unusual_freq": dosage.unusual_frequencies,
                }
        except Exception as e:
            logger.warning(f"Could not load dosage ranges from provider: {e}")

        # Always store result (even if empty)
        self._dosage_ranges = ranges
        return self._dosage_ranges

    def check(self, text: str) -> DosagePlausibilityResult:
        """
        Check dosage plausibility in text.

        Args:
            text: Clinical text to analyze.

        Returns:
            DosagePlausibilityResult with detected dosages and issues.
        """
        dosages: list[DosageInfo] = []
        issues: list[str] = []
        warnings: list[str] = []

        text_lower = text.lower()

        # Get dosage ranges dynamically
        dosage_ranges = self._get_dosage_ranges()

        # Find all drug mentions with dosages
        for drug, ranges in dosage_ranges.items():
            if drug not in text_lower:
                continue

            # Find dosage patterns near drug name
            drug_pattern = rf"\b{drug}\b\s*(\d+(?:\.\d+)?)\s*(mg|mcg|milligram|microgram|units?)?"
            for match in re.finditer(drug_pattern, text_lower):
                dose_value = float(match.group(1))
                dose_unit = match.group(2) or ranges["unit"]

                # Normalize unit
                if "milligram" in dose_unit:
                    dose_unit = "mg"
                elif "microgram" in dose_unit:
                    dose_unit = "mcg"

                # Get surrounding context for frequency
                context_start = max(0, match.start() - 20)
                context_end = min(len(text_lower), match.end() + 50)
                context = text_lower[context_start:context_end]
                frequency = self._extract_frequency(context)

                # Check plausibility
                is_plausible, issue = self._check_dose_plausible(
                    drug, dose_value, dose_unit, frequency, ranges
                )

                dosages.append(DosageInfo(
                    drug=drug,
                    dose_value=dose_value,
                    dose_unit=dose_unit,
                    frequency=frequency,
                    raw_text=match.group(),
                    is_plausible=is_plausible,
                    issue=issue,
                    typical_range=f"{ranges['min']}-{ranges['max']} {ranges['unit']}",
                ))

                if issue:
                    issues.append(f"{drug}: {issue}")

        # Also check for general dosage anomalies
        general_issues = self._check_general_anomalies(text_lower)
        warnings.extend(general_issues)

        # Calculate score
        total = len(dosages) or 1
        plausible_count = sum(1 for d in dosages if d.is_plausible)
        score = plausible_count / total if dosages else 1.0

        # Apply penalty for warnings
        score = max(0.0, score - len(warnings) * 0.05)

        return DosagePlausibilityResult(
            text=text,
            dosages=dosages,
            score=score,
            issues=issues,
            warnings=warnings,
        )

    def _extract_frequency(self, context: str) -> str | None:
        """Extract dosing frequency from context."""
        for freq_name, pattern in self._freq_patterns.items():
            if pattern.search(context):
                return freq_name
        return None

    def _check_dose_plausible(
        self,
        drug: str,
        dose_value: float,
        dose_unit: str,
        frequency: str | None,
        ranges: dict,
    ) -> tuple[bool, str | None]:
        """Check if a specific dose is plausible."""
        expected_unit = ranges["unit"]

        # Unit mismatch
        if dose_unit != expected_unit:
            # Check if it's a conversion issue
            if dose_unit == "mg" and expected_unit == "mcg":
                dose_value *= 1000  # Convert mg to mcg
            elif dose_unit == "mcg" and expected_unit == "mg":
                dose_value /= 1000  # Convert mcg to mg

        # Check range
        min_dose = ranges["min"]
        max_dose = ranges["max"]

        if dose_value < min_dose * 0.5:  # More than 50% below min
            return False, f"Dose {dose_value}{dose_unit} is unusually low (typical: {min_dose}-{max_dose}{expected_unit})"

        if dose_value > max_dose * 1.5:  # More than 50% above max
            return False, f"Dose {dose_value}{dose_unit} exceeds typical maximum (typical: {min_dose}-{max_dose}{expected_unit})"

        # Check for unlikely round numbers (potential transcription error)
        if dose_value not in ranges["common"] and dose_value > max_dose:
            return False, f"Dose {dose_value}{dose_unit} is unusual (common doses: {ranges['common']})"

        # Check frequency plausibility
        if frequency and "unusual_freq" in ranges:
            if frequency in ranges["unusual_freq"]:
                return False, f"Frequency '{frequency}' is unusual for {drug}"

        return True, None

    def _check_general_anomalies(self, text: str) -> list[str]:
        """Check for general dosage anomalies."""
        warnings = []

        # Check for very high numbers that might be errors
        high_numbers = re.findall(r"\b(\d{4,})\s*(?:mg|mcg|milligram)", text)
        for num in high_numbers:
            if int(num) > 5000:
                warnings.append(f"Unusually high dose value: {num}mg")

        # Check for decimal point issues (e.g., "5 00" instead of "500")
        space_numbers = re.findall(r"(\d+)\s+(\d+)\s*mg", text)
        for n1, n2 in space_numbers:
            if n2 == "00" or n2 == "0":
                warnings.append(f"Possible transcription error: '{n1} {n2}mg'")

        return warnings


# Singleton
_checker: DosagePlausibilityChecker | None = None


def get_dosage_checker() -> DosagePlausibilityChecker:
    """Get singleton checker."""
    global _checker
    if _checker is None:
        _checker = DosagePlausibilityChecker()
    return _checker


def check_dosage_plausibility(text: str) -> DosagePlausibilityResult:
    """Convenience function to check dosage plausibility."""
    return get_dosage_checker().check(text)
