"""
Unified medical lexicon for multi-source lookup.

This module provides a unified interface for looking up terms
across multiple medical lexicons (RxNorm, SNOMED, ICD-10, etc.).

Example:
    >>> unified = UnifiedMedicalLexicon()
    >>> unified.add_lexicon("rxnorm", rxnorm_lexicon)
    >>> unified.add_lexicon("snomed", snomed_lexicon)
    >>> entry = unified.lookup("metformin")  # Searches all lexicons
"""
from __future__ import annotations

from dataclasses import dataclass

from hsttb.core.types import MedicalTerm, MedicalTermCategory
from hsttb.lexicons.base import (
    LexiconEntry,
    LexiconSource,
    LexiconStats,
    MedicalLexicon,
)


@dataclass
class UnifiedLexiconStats:
    """Statistics for the unified lexicon."""

    total_entries: int
    lexicon_count: int
    lexicon_stats: dict[str, LexiconStats]


class UnifiedMedicalLexicon:
    """
    Combines multiple lexicons for comprehensive lookup.

    The unified lexicon searches across all registered lexicons
    and returns the first match found. Priority is determined
    by registration order.

    Attributes:
        lexicons: Dictionary of lexicon name to MedicalLexicon.

    Example:
        >>> unified = UnifiedMedicalLexicon()
        >>> unified.add_lexicon("drugs", drug_lexicon)
        >>> unified.add_lexicon("diagnoses", diagnosis_lexicon)
        >>> entry = unified.lookup("diabetes")
    """

    def __init__(self) -> None:
        """Initialize the unified lexicon."""
        self._lexicons: dict[str, MedicalLexicon] = {}
        self._priority_order: list[str] = []

    @property
    def lexicons(self) -> dict[str, MedicalLexicon]:
        """Get registered lexicons."""
        return self._lexicons.copy()

    def add_lexicon(
        self,
        name: str,
        lexicon: MedicalLexicon,
        priority: int | None = None,
    ) -> None:
        """
        Add a lexicon to the unified lookup.

        Args:
            name: Unique name for the lexicon.
            lexicon: The lexicon instance.
            priority: Optional priority index (lower = higher priority).
        """
        self._lexicons[name] = lexicon

        if priority is not None and priority < len(self._priority_order):
            self._priority_order.insert(priority, name)
        else:
            self._priority_order.append(name)

    def remove_lexicon(self, name: str) -> bool:
        """
        Remove a lexicon from the unified lookup.

        Args:
            name: Name of the lexicon to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self._lexicons:
            del self._lexicons[name]
            self._priority_order.remove(name)
            return True
        return False

    def lookup(self, term: str) -> LexiconEntry | None:
        """
        Look up a term across all lexicons.

        Searches lexicons in priority order and returns first match.

        Args:
            term: Term to look up.

        Returns:
            LexiconEntry if found, None otherwise.
        """
        for name in self._priority_order:
            lexicon = self._lexicons[name]
            entry = lexicon.lookup(term)
            if entry is not None:
                return entry
        return None

    def lookup_all(self, term: str) -> list[LexiconEntry]:
        """
        Look up a term in all lexicons.

        Unlike lookup(), returns all matches from all lexicons.

        Args:
            term: Term to look up.

        Returns:
            List of all matching entries.
        """
        results = []
        for name in self._priority_order:
            lexicon = self._lexicons[name]
            entry = lexicon.lookup(term)
            if entry is not None:
                results.append(entry)
        return results

    def contains(self, term: str) -> bool:
        """
        Check if term exists in any lexicon.

        Args:
            term: Term to check.

        Returns:
            True if term exists in any lexicon.
        """
        return any(
            self._lexicons[name].contains(term) for name in self._priority_order
        )

    def get_category(self, term: str) -> str | None:
        """
        Get the category of a term.

        Args:
            term: Term to categorize.

        Returns:
            Category string if found.
        """
        entry = self.lookup(term)
        return entry.category if entry else None

    def get_source(self, term: str) -> LexiconSource | None:
        """
        Get the source of a term.

        Args:
            term: Term to look up.

        Returns:
            LexiconSource if found.
        """
        entry = self.lookup(term)
        return entry.source if entry else None

    def identify_terms(self, text: str) -> list[MedicalTerm]:
        """
        Identify medical terms in text using lexicon matching.

        This is a simple word-based matcher. For more sophisticated
        term extraction, use the MedicalTermExtractor with NLP.

        Args:
            text: Text to analyze.

        Returns:
            List of identified MedicalTerm objects.
        """
        terms: list[MedicalTerm] = []
        words = text.split()

        # Simple n-gram matching (1-4 words)
        for n in range(4, 0, -1):
            i = 0
            while i <= len(words) - n:
                phrase = " ".join(words[i : i + n])
                entry = self.lookup(phrase)

                if entry:
                    # Find span in original text
                    start = text.lower().find(phrase.lower())
                    if start >= 0:
                        end = start + len(phrase)

                        # Map category
                        category = self._map_category(entry.category)

                        term = MedicalTerm(
                            text=phrase,
                            normalized=entry.normalized,
                            category=category,
                            source=entry.source.value,
                            span=(start, end),
                        )
                        terms.append(term)

                        # Skip matched words
                        i += n
                        continue

                i += 1

        return terms

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

    def get_stats(self) -> UnifiedLexiconStats:
        """
        Get statistics about all lexicons.

        Returns:
            UnifiedLexiconStats with combined statistics.
        """
        lexicon_stats: dict[str, LexiconStats] = {}
        total_entries = 0

        for name, lexicon in self._lexicons.items():
            stats = lexicon.get_stats()
            if stats:
                lexicon_stats[name] = stats
                total_entries += stats.entry_count

        return UnifiedLexiconStats(
            total_entries=total_entries,
            lexicon_count=len(self._lexicons),
            lexicon_stats=lexicon_stats,
        )

    def fuzzy_lookup(
        self,
        term: str,
        threshold: float = 0.8,
        max_results: int = 10,
    ) -> list[tuple[LexiconEntry, float]]:
        """
        Fuzzy lookup across all lexicons.

        Args:
            term: Term to look up.
            threshold: Minimum similarity score.
            max_results: Maximum results to return.

        Returns:
            List of (entry, score) tuples sorted by score.
        """
        all_results: list[tuple[LexiconEntry, float]] = []

        for name in self._priority_order:
            lexicon = self._lexicons[name]
            results = lexicon.fuzzy_lookup(term, threshold)
            all_results.extend(results)

        # Sort by score descending and limit
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:max_results]
