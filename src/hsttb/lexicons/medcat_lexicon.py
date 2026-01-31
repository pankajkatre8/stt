"""
MedCAT-based medical lexicon for clinical term extraction.

MedCAT (Medical Concept Annotation Tool) links text to UMLS/SNOMED-CT
concepts for accurate medical term identification.

Example:
    >>> from hsttb.lexicons.medcat_lexicon import MedCATLexicon
    >>> lexicon = MedCATLexicon()
    >>> lexicon.load("path/to/medcat/model")
    >>> entry = lexicon.lookup("diabetes mellitus")
    >>> print(entry.code)  # SNOMED code
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
    pass


class MedCATLexicon(MedicalLexicon):
    """
    Medical lexicon using MedCAT for UMLS/SNOMED-CT linking.

    MedCAT provides:
    - Concept recognition and linking
    - UMLS CUI and SNOMED-CT codes
    - Context-aware disambiguation

    Requires: pip install medcat

    Example:
        >>> lexicon = MedCATLexicon()
        >>> lexicon.load("path/to/medcat_model.zip")
        >>> entities = lexicon.extract_entities("Patient has diabetes")
    """

    def __init__(self) -> None:
        """Initialize MedCAT lexicon."""
        self._cat: Any = None
        self._is_loaded = False
        self._load_time_ms = 0.0
        self._entity_cache: dict[str, LexiconEntry | None] = {}

    @property
    def source(self) -> LexiconSource:
        """Return the lexicon source identifier."""
        return LexiconSource.SNOMED

    @property
    def is_loaded(self) -> bool:
        """Check if the MedCAT model is loaded."""
        return self._is_loaded

    def load(self, path: str) -> None:
        """
        Load MedCAT model from path.

        Args:
            path: Path to MedCAT model pack (.zip file).

        Raises:
            ImportError: If medcat is not installed.
            FileNotFoundError: If model path doesn't exist.
        """
        try:
            from medcat.cat import CAT
        except ImportError as e:
            raise ImportError(
                "MedCAT not installed. Install with: pip install medcat"
            ) from e

        start = time.perf_counter()
        self._cat = CAT.load_model_pack(path)
        self._load_time_ms = (time.perf_counter() - start) * 1000
        self._is_loaded = True

    def lookup(self, term: str) -> LexiconEntry | None:
        """
        Look up a term using MedCAT.

        Args:
            term: Term to look up.

        Returns:
            LexiconEntry if concept found, None otherwise.
        """
        if not self._is_loaded or self._cat is None:
            return None

        # Check cache
        cache_key = self.normalize_term(term)
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        # Get entities from MedCAT
        result = self._cat.get_entities(term)
        entities = result.get("entities", {})

        if not entities:
            self._entity_cache[cache_key] = None
            return None

        # Get first (best) match
        entity = list(entities.values())[0]
        cui = entity.get("cui", "")
        pretty_name = entity.get("pretty_name", term)
        type_ids = entity.get("type_ids", [])

        # Map UMLS type to category
        category = self._map_type_to_category(type_ids)

        entry = LexiconEntry(
            term=term,
            normalized=pretty_name.lower(),
            code=cui,
            category=category,
            source=LexiconSource.SNOMED,
            synonyms=(),
        )

        self._entity_cache[cache_key] = entry
        return entry

    def contains(self, term: str) -> bool:
        """Check if term exists in MedCAT's vocabulary."""
        return self.lookup(term) is not None

    def get_category(self, term: str) -> str | None:
        """Get the category of a term."""
        entry = self.lookup(term)
        return entry.category if entry else None

    def get_stats(self) -> LexiconStats | None:
        """Get statistics about the loaded model."""
        if not self._is_loaded or self._cat is None:
            return None

        # Get concept count from MedCAT's CDB
        cdb = self._cat.cdb
        concept_count = len(cdb.cui2names) if hasattr(cdb, "cui2names") else 0

        return LexiconStats(
            entry_count=concept_count,
            source=LexiconSource.SNOMED,
            categories={"all": concept_count},
            load_time_ms=self._load_time_ms,
        )

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """
        Extract all medical entities from text.

        Args:
            text: Clinical text to process.

        Returns:
            List of entity dictionaries with cui, name, span, category.
        """
        if not self._is_loaded or self._cat is None:
            return []

        result = self._cat.get_entities(text)
        entities = result.get("entities", {})

        extracted = []
        for entity in entities.values():
            extracted.append({
                "cui": entity.get("cui", ""),
                "name": entity.get("pretty_name", ""),
                "text": entity.get("source_value", ""),
                "span": (entity.get("start", 0), entity.get("end", 0)),
                "category": self._map_type_to_category(entity.get("type_ids", [])),
                "confidence": entity.get("context_similarity", 1.0),
            })

        return extracted

    def _map_type_to_category(self, type_ids: list[str]) -> str:
        """
        Map UMLS semantic type IDs to categories.

        Args:
            type_ids: List of UMLS type IDs (e.g., T121 for drug).

        Returns:
            Category string.
        """
        # UMLS Semantic Type mappings
        drug_types = {"T121", "T200", "T195", "T123", "T122", "T109"}
        diagnosis_types = {"T047", "T048", "T046", "T191", "T190", "T033"}
        procedure_types = {"T060", "T061", "T058", "T059"}
        anatomy_types = {"T023", "T024", "T025", "T029", "T030"}

        type_set = set(type_ids)

        if type_set & drug_types:
            return "drug"
        elif type_set & diagnosis_types:
            return "diagnosis"
        elif type_set & procedure_types:
            return "procedure"
        elif type_set & anatomy_types:
            return "anatomy"
        else:
            return "other"
