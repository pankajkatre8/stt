"""
scispaCy-based medical lexicon for biomedical NER.

scispaCy provides spaCy models trained on biomedical and clinical text
for named entity recognition.

Example:
    >>> from hsttb.lexicons.scispacy_lexicon import SciSpacyLexicon
    >>> lexicon = SciSpacyLexicon()
    >>> lexicon.load("en_core_sci_md")
    >>> entities = lexicon.extract_entities("Patient has diabetes")
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


class SciSpacyLexicon(MedicalLexicon):
    """
    Medical lexicon using scispaCy for biomedical NER.

    Available models:
    - en_core_sci_sm: Small model (100k vocab)
    - en_core_sci_md: Medium model (360k vocab)
    - en_core_sci_lg: Large model (785k vocab)
    - en_ner_bc5cdr_md: BC5CDR (diseases/chemicals)
    - en_ner_bionlp13cg_md: BioNLP13CG (cancer genetics)

    Requires: pip install scispacy
              pip install <model_url>

    Example:
        >>> lexicon = SciSpacyLexicon()
        >>> lexicon.load("en_ner_bc5cdr_md")  # diseases & chemicals
        >>> entities = lexicon.extract_entities("metformin for diabetes")
    """

    # Model to entity label mappings
    MODEL_LABELS: dict[str, dict[str, str]] = {
        "en_ner_bc5cdr_md": {
            "DISEASE": "diagnosis",
            "CHEMICAL": "drug",
        },
        "en_ner_bionlp13cg_md": {
            "CANCER": "diagnosis",
            "ORGAN": "anatomy",
            "SIMPLE_CHEMICAL": "drug",
            "GENE_OR_GENE_PRODUCT": "other",
        },
        "default": {
            "ENTITY": "other",
        },
    }

    def __init__(self) -> None:
        """Initialize scispaCy lexicon."""
        self._nlp: Any = None
        self._model_name: str = ""
        self._is_loaded = False
        self._load_time_ms = 0.0
        self._entity_cache: dict[str, LexiconEntry | None] = {}
        self._label_mapping: dict[str, str] = {}

    @property
    def source(self) -> LexiconSource:
        """Return the lexicon source identifier."""
        return LexiconSource.CUSTOM

    @property
    def is_loaded(self) -> bool:
        """Check if the scispaCy model is loaded."""
        return self._is_loaded

    @property
    def model_name(self) -> str:
        """Return the loaded model name."""
        return self._model_name

    def load(self, model_name: str = "en_ner_bc5cdr_md") -> None:
        """
        Load scispaCy model.

        Args:
            model_name: Name of the scispaCy model to load.

        Raises:
            ImportError: If spacy/scispacy is not installed.
            OSError: If model is not found.
        """
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "spaCy not installed. Install with: pip install spacy scispacy"
            ) from e

        start = time.perf_counter()

        try:
            self._nlp = spacy.load(model_name)
        except OSError as e:
            raise OSError(
                f"Model '{model_name}' not found. Install with:\n"
                f"pip install {self._get_model_url(model_name)}"
            ) from e

        self._model_name = model_name
        self._label_mapping = self.MODEL_LABELS.get(
            model_name, self.MODEL_LABELS["default"]
        )
        self._load_time_ms = (time.perf_counter() - start) * 1000
        self._is_loaded = True

    def extract_terms(self, text: str) -> list:
        """
        Extract all medical terms from text using NER.

        More efficient than multiple lookup() calls as it
        runs NER once on the full text.

        Args:
            text: Clinical text to process.

        Returns:
            List of LexiconEntry for all recognized terms.
        """
        if not self._is_loaded or self._nlp is None:
            self.load()

        doc = self._nlp(text)
        entries = []

        for ent in doc.ents:
            category = self._label_mapping.get(ent.label_, "other")
            entry = LexiconEntry(
                term=ent.text,
                normalized=ent.text.lower(),
                code=f"{self._model_name}:{ent.label_}",
                category=category,
                source=LexiconSource.CUSTOM,
                synonyms=(),
            )
            entries.append(entry)

            # Also cache
            cache_key = self.normalize_term(ent.text)
            self._entity_cache[cache_key] = entry

        return entries

    def lookup(self, term: str) -> LexiconEntry | None:
        """
        Look up a term using scispaCy NER.

        Args:
            term: Term to look up.

        Returns:
            LexiconEntry if entity found, None otherwise.
        """
        if not self._is_loaded or self._nlp is None:
            return None

        # Check cache
        cache_key = self.normalize_term(term)
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        # Process with scispaCy
        doc = self._nlp(term)

        if not doc.ents:
            self._entity_cache[cache_key] = None
            return None

        # Get first entity
        ent = doc.ents[0]
        category = self._label_mapping.get(ent.label_, "other")

        entry = LexiconEntry(
            term=term,
            normalized=ent.text.lower(),
            code=f"{self._model_name}:{ent.label_}",
            category=category,
            source=LexiconSource.CUSTOM,
            synonyms=(),
        )

        self._entity_cache[cache_key] = entry
        return entry

    def contains(self, term: str) -> bool:
        """Check if term is recognized as an entity."""
        return self.lookup(term) is not None

    def get_category(self, term: str) -> str | None:
        """Get the category of a term."""
        entry = self.lookup(term)
        return entry.category if entry else None

    def get_stats(self) -> LexiconStats | None:
        """Get statistics about the loaded model."""
        if not self._is_loaded or self._nlp is None:
            return None

        return LexiconStats(
            entry_count=len(self._nlp.vocab),
            source=LexiconSource.CUSTOM,
            categories=dict(self._label_mapping),
            load_time_ms=self._load_time_ms,
        )

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """
        Extract all medical entities from text.

        Args:
            text: Clinical text to process.

        Returns:
            List of entity dictionaries with label, text, span, category.
        """
        if not self._is_loaded or self._nlp is None:
            return []

        doc = self._nlp(text)

        extracted = []
        for ent in doc.ents:
            extracted.append({
                "label": ent.label_,
                "text": ent.text,
                "span": (ent.start_char, ent.end_char),
                "category": self._label_mapping.get(ent.label_, "other"),
            })

        return extracted

    def _get_model_url(self, model_name: str) -> str:
        """Get download URL for a scispaCy model."""
        base_url = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4"
        return f"{base_url}/{model_name}-0.5.4.tar.gz"


class SciSpacyWithLinker(SciSpacyLexicon):
    """
    scispaCy with UMLS entity linking for code resolution.

    Adds UMLS CUI codes to extracted entities using scispaCy's
    EntityLinker component.

    Requires: pip install scispacy
              pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

    Example:
        >>> lexicon = SciSpacyWithLinker()
        >>> lexicon.load("en_core_sci_md", linker="umls")
        >>> entry = lexicon.lookup("diabetes mellitus")
        >>> print(entry.code)  # UMLS CUI
    """

    def __init__(self) -> None:
        """Initialize scispaCy with linker."""
        super().__init__()
        self._linker: Any = None

    def load(
        self,
        model_name: str = "en_core_sci_md",
        linker: str = "umls",
    ) -> None:
        """
        Load scispaCy model with entity linker.

        Args:
            model_name: Name of the scispaCy model.
            linker: Linker to use ('umls', 'mesh', 'rxnorm', 'go', 'hpo').

        Raises:
            ImportError: If scispacy is not installed.
        """
        try:
            import spacy
            from scispacy.linking import EntityLinker  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "scispaCy not installed. Install with:\n"
                "pip install spacy scispacy\n"
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz"
            ) from e

        start = time.perf_counter()

        self._nlp = spacy.load(model_name)
        self._nlp.add_pipe(
            "scispacy_linker",
            config={"resolve_abbreviations": True, "linker_name": linker},
        )
        self._linker = self._nlp.get_pipe("scispacy_linker")

        self._model_name = f"{model_name}+{linker}"
        self._load_time_ms = (time.perf_counter() - start) * 1000
        self._is_loaded = True

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """
        Extract entities with UMLS linking.

        Args:
            text: Clinical text to process.

        Returns:
            List of entity dictionaries with UMLS CUI codes.
        """
        if not self._is_loaded or self._nlp is None:
            return []

        doc = self._nlp(text)

        extracted = []
        for ent in doc.ents:
            # Get linked UMLS concepts
            cui = ""
            umls_name = ""
            if hasattr(ent, "_") and hasattr(ent._, "kb_ents") and ent._.kb_ents:
                top_match = ent._.kb_ents[0]
                cui = top_match[0]
                # Get concept name from linker KB
                if self._linker and hasattr(self._linker, "kb"):
                    concept = self._linker.kb.cui_to_entity.get(cui)
                    if concept:
                        umls_name = concept.canonical_name

            extracted.append({
                "label": ent.label_,
                "text": ent.text,
                "span": (ent.start_char, ent.end_char),
                "category": self._map_cui_to_category(cui),
                "cui": cui,
                "umls_name": umls_name,
            })

        return extracted

    def _map_cui_to_category(self, cui: str) -> str:
        """Map UMLS CUI to category based on semantic type."""
        # This would need the UMLS semantic type lookup
        # For now, return generic category
        if cui:
            return "medical"
        return "other"
