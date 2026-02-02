"""
Term Error Rate (TER) computation engine.

This module provides the TER computation engine for measuring
medical term accuracy in STT transcriptions.

TER measures:
- Substitutions: Wrong term predicted
- Deletions: Ground truth term missing
- Insertions: Extra term predicted

Example:
    >>> from hsttb.metrics.ter import TEREngine
    >>> from hsttb.lexicons import MockMedicalLexicon
    >>> lexicon = MockMedicalLexicon.with_common_terms()
    >>> engine = TEREngine(lexicon)
    >>> result = engine.compute(
    ...     ground_truth="patient takes metformin for diabetes",
    ...     prediction="patient takes metformin for diabetes"
    ... )
    >>> print(result.overall_ter)  # 0.0 (perfect match)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hsttb.core.types import ErrorType, MedicalTerm, MedicalTermCategory, TermError
from hsttb.nlp.normalizer import MedicalTextNormalizer

if TYPE_CHECKING:
    from hsttb.lexicons.base import MedicalLexicon


@dataclass
class TERResult:
    """
    Result of TER computation.

    Attributes:
        overall_ter: Overall Term Error Rate (0.0-1.0+).
        category_ter: TER broken down by category.
        total_gt_terms: Total ground truth terms.
        total_pred_terms: Total predicted terms.
        correct_matches: Number of correct matches.
        substitutions: List of substitution errors.
        deletions: List of deletion errors.
        insertions: List of insertion errors.

    Example:
        >>> result.overall_ter  # 0.15 = 15% error rate
        >>> result.category_ter["drug"]  # 0.10 for drugs
    """

    overall_ter: float
    category_ter: dict[str, float]
    total_gt_terms: int
    total_pred_terms: int
    correct_matches: int
    substitutions: list[TermError] = field(default_factory=list)
    deletions: list[TermError] = field(default_factory=list)
    insertions: list[TermError] = field(default_factory=list)

    @property
    def total_errors(self) -> int:
        """Total number of errors."""
        return len(self.substitutions) + len(self.deletions) + len(self.insertions)

    @property
    def precision(self) -> float:
        """Precision: correct / (correct + insertions)."""
        if self.total_pred_terms == 0:
            return 1.0
        return self.correct_matches / self.total_pred_terms

    @property
    def recall(self) -> float:
        """Recall: correct / (correct + deletions)."""
        if self.total_gt_terms == 0:
            return 1.0
        return self.correct_matches / self.total_gt_terms

    @property
    def f1_score(self) -> float:
        """F1 score: harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


@dataclass
class TermMatch:
    """
    A match between ground truth and predicted terms.

    Attributes:
        gt_term: Ground truth term (None for insertions).
        pred_term: Predicted term (None for deletions).
        match_type: Type of match (correct, substitution, deletion, insertion).
        similarity: Similarity score (0.0-1.0).
    """

    gt_term: MedicalTerm | None
    pred_term: MedicalTerm | None
    match_type: str  # "correct", "substitution", "deletion", "insertion"
    similarity: float = 1.0


class TEREngine:
    """
    Term Error Rate computation engine.

    Computes TER by extracting medical terms from both ground truth
    and prediction, aligning them, and measuring errors.

    Attributes:
        lexicon: Medical lexicon for term identification.
        normalizer: Text normalizer for comparison.
        fuzzy_threshold: Minimum similarity for fuzzy matching.

    Example:
        >>> lexicon = MockMedicalLexicon.with_common_terms()
        >>> engine = TEREngine(lexicon)
        >>> result = engine.compute(
        ...     "patient has diabetes takes metformin",
        ...     "patient has diabetes takes metformin"
        ... )
        >>> assert result.overall_ter == 0.0
    """

    def __init__(
        self,
        lexicon: MedicalLexicon,
        normalizer: MedicalTextNormalizer | None = None,
        fuzzy_threshold: float = 0.6,  # Lower threshold to catch drug name confusions
    ) -> None:
        """
        Initialize the TER engine.

        Args:
            lexicon: Medical lexicon for term extraction.
            normalizer: Text normalizer (creates default if None).
            fuzzy_threshold: Threshold for fuzzy matching (0.0-1.0).
        """
        self.lexicon = lexicon
        self.normalizer = normalizer or MedicalTextNormalizer()
        self.fuzzy_threshold = fuzzy_threshold

    def compute(
        self,
        ground_truth: str,
        prediction: str,
    ) -> TERResult:
        """
        Compute Term Error Rate.

        Args:
            ground_truth: Reference transcript.
            prediction: Predicted transcript.

        Returns:
            TERResult with detailed error analysis.
        """
        # Normalize texts
        gt_normalized = self.normalizer.normalize(ground_truth)
        pred_normalized = self.normalizer.normalize(prediction)

        # Extract terms
        gt_terms = self._extract_terms(gt_normalized)
        pred_terms = self._extract_terms(pred_normalized)

        # Align terms and detect errors
        matches = self._align_terms(gt_terms, pred_terms)

        # Categorize matches
        substitutions: list[TermError] = []
        deletions: list[TermError] = []
        insertions: list[TermError] = []
        correct = 0

        for match in matches:
            if match.match_type == "correct":
                correct += 1
            elif match.match_type == "substitution" and match.gt_term and match.pred_term:
                substitutions.append(
                    TermError(
                        error_type=ErrorType.SUBSTITUTION,
                        category=match.gt_term.category,
                        ground_truth_term=match.gt_term,
                        predicted_term=match.pred_term,
                    )
                )
            elif match.match_type == "deletion" and match.gt_term:
                deletions.append(
                    TermError(
                        error_type=ErrorType.DELETION,
                        category=match.gt_term.category,
                        ground_truth_term=match.gt_term,
                        predicted_term=None,
                    )
                )
            elif match.match_type == "insertion" and match.pred_term:
                insertions.append(
                    TermError(
                        error_type=ErrorType.INSERTION,
                        category=match.pred_term.category,
                        ground_truth_term=None,
                        predicted_term=match.pred_term,
                    )
                )

        # Compute TER
        total_errors = len(substitutions) + len(deletions) + len(insertions)
        overall_ter = total_errors / len(gt_terms) if gt_terms else 0.0

        # Category-wise TER
        category_ter = self._compute_category_ter(
            gt_terms, substitutions, deletions
        )

        return TERResult(
            overall_ter=overall_ter,
            category_ter=category_ter,
            total_gt_terms=len(gt_terms),
            total_pred_terms=len(pred_terms),
            correct_matches=correct,
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
        )

    def _extract_terms(self, text: str) -> list[MedicalTerm]:
        """
        Extract medical terms from text.

        Uses lexicon's extract_terms if available (NER-based),
        otherwise falls back to n-gram lookup.

        Args:
            text: Text to extract terms from.

        Returns:
            List of identified medical terms.
        """
        # Check if lexicon has NER-based extraction (more efficient)
        if hasattr(self.lexicon, "extract_terms"):
            entries = self.lexicon.extract_terms(text)
            terms = []
            for entry in entries:
                # Find span in text
                start = text.lower().find(entry.term.lower())
                end = start + len(entry.term) if start >= 0 else 0
                terms.append(
                    MedicalTerm(
                        text=entry.term,
                        normalized=entry.normalized,
                        category=self._map_category(entry.category),
                        source=entry.source.value,
                        span=(start, end),
                    )
                )
            terms.sort(key=lambda t: t.span[0])
            return terms

        # Fallback: n-gram lookup
        terms: list[MedicalTerm] = []
        words = text.split()

        # Track which positions have been matched
        matched_positions: set[int] = set()

        # Try n-grams from largest to smallest (4 down to 1)
        for n in range(min(4, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                # Skip if any position already matched
                if any(pos in matched_positions for pos in range(i, i + n)):
                    continue

                phrase = " ".join(words[i : i + n])
                entry = self.lexicon.lookup(phrase)

                if entry is not None:
                    # Find span in original text
                    start = text.find(phrase)
                    if start == -1:
                        # Try case-insensitive
                        start = text.lower().find(phrase.lower())
                    end = start + len(phrase) if start >= 0 else 0

                    terms.append(
                        MedicalTerm(
                            text=phrase,
                            normalized=entry.normalized,
                            category=self._map_category(entry.category),
                            source=entry.source.value,
                            span=(start, end),
                        )
                    )

                    # Mark positions as matched
                    for pos in range(i, i + n):
                        matched_positions.add(pos)

        # Sort by span start
        terms.sort(key=lambda t: t.span[0])
        return terms

    def _align_terms(
        self,
        gt_terms: list[MedicalTerm],
        pred_terms: list[MedicalTerm],
    ) -> list[TermMatch]:
        """
        Align ground truth and predicted terms.

        Uses greedy matching based on normalized form.

        Args:
            gt_terms: Ground truth terms.
            pred_terms: Predicted terms.

        Returns:
            List of term matches.
        """
        matches: list[TermMatch] = []
        used_pred: set[int] = set()

        # First pass: exact matches
        for gt_term in gt_terms:
            for j, pred_term in enumerate(pred_terms):
                if j in used_pred:
                    continue

                if self._terms_match(gt_term, pred_term):
                    matches.append(
                        TermMatch(
                            gt_term=gt_term,
                            pred_term=pred_term,
                            match_type="correct",
                            similarity=1.0,
                        )
                    )
                    used_pred.add(j)
                    break
            else:
                # No exact match found - check for fuzzy match
                best_match_idx = -1
                best_similarity = 0.0

                for j, pred_term in enumerate(pred_terms):
                    if j in used_pred:
                        continue

                    sim = self._similarity(gt_term, pred_term)
                    if sim >= self.fuzzy_threshold and sim > best_similarity:
                        best_similarity = sim
                        best_match_idx = j

                if best_match_idx >= 0:
                    # Substitution
                    matches.append(
                        TermMatch(
                            gt_term=gt_term,
                            pred_term=pred_terms[best_match_idx],
                            match_type="substitution",
                            similarity=best_similarity,
                        )
                    )
                    used_pred.add(best_match_idx)
                else:
                    # Deletion
                    matches.append(
                        TermMatch(
                            gt_term=gt_term,
                            pred_term=None,
                            match_type="deletion",
                            similarity=0.0,
                        )
                    )

        # Remaining predicted terms are insertions
        for j, pred_term in enumerate(pred_terms):
            if j not in used_pred:
                matches.append(
                    TermMatch(
                        gt_term=None,
                        pred_term=pred_term,
                        match_type="insertion",
                        similarity=0.0,
                    )
                )

        return matches

    def _terms_match(self, term1: MedicalTerm, term2: MedicalTerm) -> bool:
        """Check if two terms match (exact or normalized)."""
        return (
            term1.normalized == term2.normalized
            or term1.text.lower() == term2.text.lower()
        )

    def _similarity(self, term1: MedicalTerm, term2: MedicalTerm) -> float:
        """
        Compute similarity between two terms.

        Uses multiple similarity measures for robust matching.

        Args:
            term1: First term.
            term2: Second term.

        Returns:
            Similarity score (0.0-1.0).
        """
        s1 = term1.normalized.lower()
        s2 = term2.normalized.lower()

        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # 1. Character overlap (Jaccard)
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = intersection / union if union > 0 else 0.0

        # 2. Prefix match (important for catching metformin/methotrexate)
        common_prefix = 0
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                common_prefix += 1
            else:
                break
        prefix_ratio = common_prefix / max(len1, len2)

        # 3. Same category boost - terms in same category are more likely substitutions
        category_boost = 0.2 if term1.category == term2.category else 0.0

        # 4. Length similarity - very different lengths are less likely substitutions
        len_ratio = min(len1, len2) / max(len1, len2)

        # Weighted combination - prioritize prefix match and category
        base_sim = 0.3 * jaccard + 0.4 * prefix_ratio + 0.3 * len_ratio

        # Apply category boost (if same category, boost similarity)
        return min(1.0, base_sim + category_boost)

    def _compute_category_ter(
        self,
        gt_terms: list[MedicalTerm],
        substitutions: list[TermError],
        deletions: list[TermError],
    ) -> dict[str, float]:
        """
        Compute TER per category.

        Args:
            gt_terms: Ground truth terms.
            substitutions: Substitution errors.
            deletions: Deletion errors.

        Returns:
            Dictionary mapping category to TER.
        """
        # Count terms per category
        category_counts: dict[str, int] = {}
        for term in gt_terms:
            cat = term.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Count errors per category
        category_errors: dict[str, int] = {}
        for error in substitutions:
            if error.ground_truth_term:
                cat = error.ground_truth_term.category.value
                category_errors[cat] = category_errors.get(cat, 0) + 1

        for error in deletions:
            if error.ground_truth_term:
                cat = error.ground_truth_term.category.value
                category_errors[cat] = category_errors.get(cat, 0) + 1

        # Compute TER per category
        category_ter: dict[str, float] = {}
        for cat, count in category_counts.items():
            errors = category_errors.get(cat, 0)
            category_ter[cat] = errors / count if count > 0 else 0.0

        return category_ter

    def _map_category(self, category_str: str) -> MedicalTermCategory:
        """Map string category to MedicalTermCategory enum."""
        mapping = {
            "drug": MedicalTermCategory.DRUG,
            "diagnosis": MedicalTermCategory.DIAGNOSIS,
            "dosage": MedicalTermCategory.DOSAGE,
            "anatomy": MedicalTermCategory.ANATOMY,
            "procedure": MedicalTermCategory.PROCEDURE,
        }
        return mapping.get(category_str.lower(), MedicalTermCategory.DRUG)


def compute_ter(
    ground_truth: str,
    prediction: str,
    lexicon: MedicalLexicon,
) -> float:
    """
    Convenience function to compute TER.

    Args:
        ground_truth: Reference transcript.
        prediction: Predicted transcript.
        lexicon: Medical lexicon.

    Returns:
        Overall TER value.
    """
    engine = TEREngine(lexicon)
    result = engine.compute(ground_truth, prediction)
    return result.overall_ter
