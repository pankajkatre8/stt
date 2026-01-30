"""
Medical lexicons for term identification and normalization.

This module provides interfaces and implementations for medical
lexicons used in TER (Term Error Rate) computation.

Example:
    >>> from hsttb.lexicons import MockMedicalLexicon, UnifiedMedicalLexicon
    >>> # Create mock lexicons for testing
    >>> drugs = MockMedicalLexicon.with_common_drugs()
    >>> diagnoses = MockMedicalLexicon.with_common_diagnoses()
    >>> # Combine into unified lexicon
    >>> unified = UnifiedMedicalLexicon()
    >>> unified.add_lexicon("drugs", drugs)
    >>> unified.add_lexicon("diagnoses", diagnoses)
    >>> # Look up terms
    >>> entry = unified.lookup("metformin")
"""
from __future__ import annotations

from hsttb.lexicons.base import (
    LexiconEntry,
    LexiconSource,
    LexiconStats,
    MedicalLexicon,
)
from hsttb.lexicons.mock_lexicon import MockMedicalLexicon
from hsttb.lexicons.unified import UnifiedLexiconStats, UnifiedMedicalLexicon

__all__ = [
    # Base classes
    "MedicalLexicon",
    "LexiconEntry",
    "LexiconSource",
    "LexiconStats",
    # Implementations
    "MockMedicalLexicon",
    "UnifiedMedicalLexicon",
    "UnifiedLexiconStats",
]
