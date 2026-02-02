"""
Medical lexicons for term identification and normalization.

This module provides interfaces and implementations for medical
lexicons used in TER (Term Error Rate) computation.

Available Backends:
    - MockMedicalLexicon: For testing (hardcoded terms)
    - SQLiteMedicalLexicon: SQLite-backed, fetches from RxNorm/ICD-10 APIs
    - DynamicMedicalLexicon: API-based with caching
    - MedCATLexicon: UMLS/SNOMED-CT linking via MedCAT
    - SciSpacyLexicon: Biomedical NER via scispaCy
    - SciSpacyWithLinker: scispaCy + UMLS entity linking

Example:
    >>> from hsttb.lexicons import SQLiteMedicalLexicon
    >>> # Production: Use SQLite lexicon (auto-fetches from APIs)
    >>> lexicon = SQLiteMedicalLexicon()
    >>> lexicon.load()  # Fetches from RxNorm, OpenFDA, ICD-10 if needed
    >>> entry = lexicon.lookup("metformin")
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

# Lazy imports for optional dependencies
def __getattr__(name: str):
    """Lazy import for optional lexicon backends."""
    if name == "MedCATLexicon":
        from hsttb.lexicons.medcat_lexicon import MedCATLexicon
        return MedCATLexicon
    elif name == "SciSpacyLexicon":
        from hsttb.lexicons.scispacy_lexicon import SciSpacyLexicon
        return SciSpacyLexicon
    elif name == "SciSpacyWithLinker":
        from hsttb.lexicons.scispacy_lexicon import SciSpacyWithLinker
        return SciSpacyWithLinker
    elif name == "SQLiteMedicalLexicon":
        from hsttb.lexicons.sqlite_lexicon import SQLiteMedicalLexicon
        return SQLiteMedicalLexicon
    elif name == "DynamicMedicalLexicon":
        from hsttb.lexicons.dynamic_lexicon import DynamicMedicalLexicon
        return DynamicMedicalLexicon
    elif name == "get_sqlite_lexicon":
        from hsttb.lexicons.sqlite_lexicon import get_sqlite_lexicon
        return get_sqlite_lexicon
    elif name == "MedicalTermFetcher":
        from hsttb.lexicons.api_fetcher import MedicalTermFetcher
        return MedicalTermFetcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Base classes
    "MedicalLexicon",
    "LexiconEntry",
    "LexiconSource",
    "LexiconStats",
    # Mock (testing)
    "MockMedicalLexicon",
    # API-based lexicons
    "SQLiteMedicalLexicon",
    "DynamicMedicalLexicon",
    "MedicalTermFetcher",
    "get_sqlite_lexicon",
    # Production backends
    "MedCATLexicon",
    "SciSpacyLexicon",
    "SciSpacyWithLinker",
    # Unified
    "UnifiedMedicalLexicon",
    "UnifiedLexiconStats",
]
