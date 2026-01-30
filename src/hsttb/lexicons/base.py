"""
Base classes for medical lexicons.

This module defines the abstract interface for medical lexicons
used in TER computation and medical term identification.

Example:
    >>> class MyLexicon(MedicalLexicon):
    ...     def lookup(self, term: str) -> LexiconEntry | None:
    ...         # Implementation
    ...         pass
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class LexiconSource(str, Enum):
    """Source identifier for medical lexicons."""

    RXNORM = "rxnorm"
    SNOMED = "snomed"
    ICD10 = "icd10"
    CUSTOM = "custom"
    MOCK = "mock"


@dataclass(frozen=True)
class LexiconEntry:
    """
    A single entry in a medical lexicon.

    Attributes:
        term: The original term text.
        normalized: Normalized form for matching.
        code: Standard code (RxCUI, SNOMED ID, etc.).
        category: Medical category (drug, diagnosis, etc.).
        source: Source lexicon identifier.
        synonyms: Alternative names for this term.

    Example:
        >>> entry = LexiconEntry(
        ...     term="Metformin",
        ...     normalized="metformin",
        ...     code="6809",
        ...     category="drug",
        ...     source=LexiconSource.RXNORM,
        ...     synonyms=["Glucophage", "metformin hydrochloride"]
        ... )
    """

    term: str
    normalized: str
    code: str
    category: str
    source: LexiconSource
    synonyms: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate entry fields."""
        if not self.term:
            raise ValueError("term cannot be empty")
        if not self.normalized:
            raise ValueError("normalized cannot be empty")
        if not self.code:
            raise ValueError("code cannot be empty")


@dataclass
class LexiconStats:
    """Statistics about a loaded lexicon."""

    entry_count: int
    source: LexiconSource
    categories: dict[str, int]
    load_time_ms: float


class MedicalLexicon(ABC):
    """
    Abstract base class for medical lexicons.

    All medical lexicon implementations must implement this interface
    to be used with the TER engine.

    Example:
        >>> lexicon = RxNormLexicon()
        >>> lexicon.load("path/to/rxnorm")
        >>> entry = lexicon.lookup("metformin")
        >>> if entry:
        ...     print(f"Found: {entry.code}")
    """

    @property
    @abstractmethod
    def source(self) -> LexiconSource:
        """Return the lexicon source identifier."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the lexicon is loaded."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load lexicon from file or directory.

        Args:
            path: Path to lexicon data.

        Raises:
            LexiconLoadError: If loading fails.
        """
        ...

    @abstractmethod
    def lookup(self, term: str) -> LexiconEntry | None:
        """
        Look up a term in the lexicon.

        Args:
            term: Term to look up (case-insensitive).

        Returns:
            LexiconEntry if found, None otherwise.
        """
        ...

    @abstractmethod
    def contains(self, term: str) -> bool:
        """
        Check if term exists in lexicon.

        Args:
            term: Term to check.

        Returns:
            True if term exists, False otherwise.
        """
        ...

    @abstractmethod
    def get_category(self, term: str) -> str | None:
        """
        Get the category of a term.

        Args:
            term: Term to categorize.

        Returns:
            Category string if found, None otherwise.
        """
        ...

    def get_stats(self) -> LexiconStats | None:
        """
        Get statistics about the loaded lexicon.

        Returns:
            LexiconStats if loaded, None otherwise.
        """
        return None

    def normalize_term(self, term: str) -> str:
        """
        Normalize a term for lookup.

        Default implementation: lowercase and strip whitespace.

        Args:
            term: Term to normalize.

        Returns:
            Normalized term.
        """
        return term.lower().strip()

    def fuzzy_lookup(
        self,
        term: str,  # noqa: ARG002
        threshold: float = 0.8,  # noqa: ARG002
    ) -> list[tuple[LexiconEntry, float]]:
        """
        Fuzzy lookup with similarity scores.

        Default implementation returns empty list.
        Subclasses should override for fuzzy matching support.

        Args:
            term: Term to look up.
            threshold: Minimum similarity score (0.0-1.0).

        Returns:
            List of (entry, score) tuples sorted by score descending.
        """
        return []
