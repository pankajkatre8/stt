"""
Medical lexicons for term identification and normalization.

This module provides interfaces and implementations for medical
lexicons used in TER (Term Error Rate) computation.

Available Backends:
    - MockMedicalLexicon: For testing (hardcoded terms)
    - MedCATLexicon: UMLS/SNOMED-CT linking via MedCAT
    - SciSpacyLexicon: Biomedical NER via scispaCy
    - SciSpacyWithLinker: scispaCy + UMLS entity linking

Example:
    >>> from hsttb.lexicons import MedCATLexicon, SciSpacyLexicon
    >>> # Production: Use MedCAT for SNOMED linking
    >>> medcat = MedCATLexicon()
    >>> medcat.load("path/to/medcat_model.zip")
    >>> # Or scispaCy for biomedical NER
    >>> scispacy = SciSpacyLexicon()
    >>> scispacy.load("en_ner_bc5cdr_md")
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Base classes
    "MedicalLexicon",
    "LexiconEntry",
    "LexiconSource",
    "LexiconStats",
    # Mock (testing)
    "MockMedicalLexicon",
    # Production backends
    "MedCATLexicon",
    "SciSpacyLexicon",
    "SciSpacyWithLinker",
    # Unified
    "UnifiedMedicalLexicon",
    "UnifiedLexiconStats",
]
