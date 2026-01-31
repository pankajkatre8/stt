"""
Production NER using scispaCy BC5CDR model.

Extracts:
- CHEMICAL: Drugs and chemicals
- DISEASE: Diseases and conditions
"""
from __future__ import annotations

from typing import Any

from hsttb.core.types import Entity, EntityLabel
from hsttb.nlp.ner_pipeline import NERPipeline


class SciSpacyNERPipeline(NERPipeline):
    """
    Production NER pipeline using scispaCy BC5CDR model.

    Trained on BC5CDR corpus for detecting:
    - Chemicals/Drugs (CHEMICAL)
    - Diseases (DISEASE)

    Example:
        >>> pipeline = SciSpacyNERPipeline()
        >>> entities = pipeline.extract_entities("metformin for diabetes")
        >>> for e in entities:
        ...     print(f"{e.label}: {e.text}")
        DRUG: metformin
        DIAGNOSIS: diabetes
    """

    _instance: SciSpacyNERPipeline | None = None
    _nlp: Any = None

    # Map scispaCy labels to our EntityLabel
    LABEL_MAP = {
        "CHEMICAL": EntityLabel.DRUG,
        "DISEASE": EntityLabel.DIAGNOSIS,
    }

    def __new__(cls) -> SciSpacyNERPipeline:
        """Singleton for model reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize (model loaded on first use)."""
        pass

    @property
    def name(self) -> str:
        """Return pipeline name."""
        return "scispacy-bc5cdr"

    @property
    def supported_labels(self) -> list[EntityLabel]:
        """Return supported entity labels."""
        return [EntityLabel.DRUG, EntityLabel.DIAGNOSIS]

    def _ensure_loaded(self) -> None:
        """Load model if not already loaded."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_ner_bc5cdr_md")

    def extract_entities(self, text: str) -> list[Entity]:
        """
        Extract medical entities from text.

        Args:
            text: Text to analyze.

        Returns:
            List of Entity objects.
        """
        self._ensure_loaded()

        doc = self._nlp(text)
        entities = []

        for ent in doc.ents:
            label = self.LABEL_MAP.get(ent.label_, EntityLabel.OTHER)
            entities.append(Entity(
                text=ent.text,
                label=label,
                span=(ent.start_char, ent.end_char),
                confidence=1.0,  # spaCy doesn't provide confidence
            ))

        return entities


# Singleton instance
_default_pipeline: SciSpacyNERPipeline | None = None


def get_scispacy_pipeline() -> SciSpacyNERPipeline:
    """Get the default scispaCy NER pipeline."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = SciSpacyNERPipeline()
    return _default_pipeline
