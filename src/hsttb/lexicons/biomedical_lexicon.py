"""
Production biomedical lexicon using HuggingFace NER.

This lexicon uses the d4data/biomedical-ner-all model for
medical term extraction, eliminating the need for manual
term lists or UMLS licenses.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from hsttb.lexicons.base import (
    LexiconEntry,
    LexiconSource,
    LexiconStats,
    MedicalLexicon,
)

if TYPE_CHECKING:
    from hsttb.nlp.biomedical_ner import BiomedicalNERPipeline


class BiomedicalLexicon(MedicalLexicon):
    """
    Production medical lexicon using HuggingFace biomedical NER.

    Instead of a static term list, this lexicon uses a neural
    NER model to identify medical terms dynamically. Any medical
    entity recognized by the model becomes a valid lexicon entry.

    Categories:
        - drug: Medications
        - diagnosis: Diseases, disorders, symptoms
        - procedure: Therapeutic and diagnostic procedures
        - anatomy: Biological structures
        - dosage: Dosage information
        - lab: Lab values

    Example:
        >>> lexicon = BiomedicalLexicon()
        >>> entry = lexicon.lookup("metformin")
        >>> print(entry.category)  # "drug"
    """

    def __init__(self) -> None:
        """Initialize biomedical lexicon."""
        self._pipeline: BiomedicalNERPipeline | None = None
        self._is_loaded = False
        self._load_time_ms = 0.0
        self._cache: dict[str, LexiconEntry | None] = {}

    @property
    def source(self) -> LexiconSource:
        """Return the lexicon source identifier."""
        return LexiconSource.CUSTOM

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    def load(self, path: str = "") -> None:
        """
        Load the biomedical NER model.

        Args:
            path: Ignored (model loaded from HuggingFace).
        """
        start = time.perf_counter()

        from hsttb.nlp.biomedical_ner import BiomedicalNERPipeline
        self._pipeline = BiomedicalNERPipeline()
        self._pipeline._ensure_loaded()

        self._load_time_ms = (time.perf_counter() - start) * 1000
        self._is_loaded = True

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._is_loaded:
            self.load()

    def lookup(self, term: str) -> LexiconEntry | None:
        """
        Look up a term using the NER model.

        Args:
            term: Term to look up.

        Returns:
            LexiconEntry if recognized as medical term.
        """
        self._ensure_loaded()

        # Check cache
        cache_key = self.normalize_term(term)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Run NER on the term
        entities = self._pipeline.extract(term)

        if not entities:
            self._cache[cache_key] = None
            return None

        # Use first (best) entity
        entity = entities[0]
        category = self._map_label_to_category(entity.label)

        entry = LexiconEntry(
            term=term,
            normalized=entity.text.lower(),
            code=f"BIO:{entity.label}",
            category=category,
            source=LexiconSource.CUSTOM,
            synonyms=(),
        )

        self._cache[cache_key] = entry
        return entry

    def contains(self, term: str) -> bool:
        """Check if term is recognized."""
        return self.lookup(term) is not None

    def get_category(self, term: str) -> str | None:
        """Get the category of a term."""
        entry = self.lookup(term)
        return entry.category if entry else None

    def get_stats(self) -> LexiconStats | None:
        """Get statistics about the lexicon."""
        if not self._is_loaded:
            return None

        return LexiconStats(
            entry_count=len(self._cache),
            source=LexiconSource.CUSTOM,
            categories={
                "drug": 0,
                "diagnosis": 0,
                "procedure": 0,
                "anatomy": 0,
                "dosage": 0,
                "lab": 0,
            },
            load_time_ms=self._load_time_ms,
        )

    def extract_terms(self, text: str) -> list[LexiconEntry]:
        """
        Extract all medical terms from text.

        This is more efficient than multiple lookup() calls
        as it runs NER once on the full text.

        Args:
            text: Clinical text to process.

        Returns:
            List of LexiconEntry for all recognized terms.
        """
        self._ensure_loaded()

        entities = self._pipeline.extract(text)
        entries = []

        for entity in entities:
            category = self._map_label_to_category(entity.label)
            entry = LexiconEntry(
                term=entity.text,
                normalized=entity.text.lower(),
                code=f"BIO:{entity.label}",
                category=category,
                source=LexiconSource.CUSTOM,
                synonyms=(),
            )
            entries.append(entry)

            # Also cache
            cache_key = self.normalize_term(entity.text)
            self._cache[cache_key] = entry

        return entries

    def _map_label_to_category(self, label: str) -> str:
        """Map NER label to category string."""
        mapping = {
            "Medication": "drug",
            "Disease_disorder": "diagnosis",
            "Sign_symptom": "diagnosis",
            "Therapeutic_procedure": "procedure",
            "Diagnostic_procedure": "procedure",
            "Lab_value": "lab",
            "Biological_structure": "anatomy",
            "Dosage": "dosage",
            "Duration": "dosage",
            "Frequency": "dosage",
        }
        return mapping.get(label, "other")


# Singleton instance for convenience
_default_lexicon: BiomedicalLexicon | None = None


def get_biomedical_lexicon() -> BiomedicalLexicon:
    """Get the default biomedical lexicon instance."""
    global _default_lexicon
    if _default_lexicon is None:
        _default_lexicon = BiomedicalLexicon()
    return _default_lexicon
