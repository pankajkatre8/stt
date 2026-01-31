"""
Production biomedical NER using HuggingFace transformers.

Uses d4data/biomedical-ner-all model for entity extraction:
- Medication
- Disease_disorder
- Sign_symptom
- Therapeutic_procedure
- Diagnostic_procedure
- Lab_value
- etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hsttb.core.types import Entity, EntityLabel
from hsttb.nlp.ner_pipeline import NERPipeline


@dataclass
class BiomedicalEntity:
    """A biomedical entity extracted from text."""
    text: str
    label: str
    start: int
    end: int
    score: float

    def to_entity(self) -> Entity:
        """Convert to core Entity type."""
        label_map = {
            "Medication": EntityLabel.DRUG,
            "Disease_disorder": EntityLabel.CONDITION,
            "Sign_symptom": EntityLabel.CONDITION,
            "Therapeutic_procedure": EntityLabel.PROCEDURE,
            "Diagnostic_procedure": EntityLabel.PROCEDURE,
            "Lab_value": EntityLabel.LAB_VALUE,
            "Severity": EntityLabel.OTHER,
            "Biological_structure": EntityLabel.ANATOMY,
            "Age": EntityLabel.OTHER,
            "Sex": EntityLabel.OTHER,
            "Clinical_event": EntityLabel.OTHER,
            "Dosage": EntityLabel.DOSAGE,
            "Duration": EntityLabel.OTHER,
            "Frequency": EntityLabel.OTHER,
            "Date": EntityLabel.DATE,
            "Time": EntityLabel.TIME,
        }
        return Entity(
            text=self.text,
            label=label_map.get(self.label, EntityLabel.OTHER),
            span=(self.start, self.end),
            confidence=self.score,
        )


class BiomedicalNERPipeline(NERPipeline):
    """
    Production NER pipeline using HuggingFace biomedical model.

    Uses d4data/biomedical-ner-all which covers:
    - Medications/drugs
    - Diseases and disorders
    - Signs and symptoms
    - Procedures (therapeutic and diagnostic)
    - Lab values
    - Anatomy
    - Dosages, frequencies, durations

    Example:
        >>> pipeline = BiomedicalNERPipeline()
        >>> entities = pipeline.extract_entities("Patient takes metformin for diabetes")
        >>> for e in entities:
        ...     print(f"{e.label}: {e.text}")
        DRUG: metformin
        CONDITION: diabetes
    """

    _instance: BiomedicalNERPipeline | None = None
    _hf_pipeline: Any = None

    def __new__(cls) -> BiomedicalNERPipeline:
        """Singleton pattern for efficient model reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the NER pipeline (lazy loading)."""
        pass  # Model loaded on first use

    @property
    def name(self) -> str:
        """Return pipeline name."""
        return "biomedical-ner-all"

    @property
    def supported_labels(self) -> list[EntityLabel]:
        """Return list of supported entity labels."""
        return [
            EntityLabel.DRUG,
            EntityLabel.CONDITION,
            EntityLabel.PROCEDURE,
            EntityLabel.ANATOMY,
            EntityLabel.DOSAGE,
            EntityLabel.LAB_VALUE,
            EntityLabel.DATE,
            EntityLabel.TIME,
            EntityLabel.OTHER,
        ]

    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded."""
        if self._hf_pipeline is None:
            from transformers import pipeline
            self._hf_pipeline = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple",
            )

    def extract(self, text: str) -> list[BiomedicalEntity]:
        """
        Extract biomedical entities from text.

        Args:
            text: Clinical text to process.

        Returns:
            List of BiomedicalEntity objects.
        """
        self._ensure_loaded()

        results = self._hf_pipeline(text)

        entities = []
        for r in results:
            # Use original text span instead of tokenized word
            start = r["start"]
            end = r["end"]
            actual_text = text[start:end]

            # Skip very short or whitespace-only extractions
            if len(actual_text.strip()) < 2:
                continue

            entities.append(BiomedicalEntity(
                text=actual_text,
                label=r["entity_group"],
                start=start,
                end=end,
                score=r["score"],
            ))

        # Merge adjacent entities of same type
        return self._merge_adjacent(entities, text)

    def extract_entities(self, text: str) -> list[Entity]:
        """
        Extract medical entities from text (NERPipeline interface).

        Args:
            text: Text to analyze.

        Returns:
            List of Entity objects.
        """
        entities = [e.to_entity() for e in self.extract(text)]
        return self._post_process_entities(entities, text)

    def _post_process_entities(
        self,
        entities: list[Entity],
        text: str,
    ) -> list[Entity]:
        """
        Post-process entities to fix tokenization issues.

        The HuggingFace model sometimes has word boundary issues where
        entities don't align with word boundaries. This method fixes:
        - Entities starting or ending mid-word
        - Entities with leading/trailing spaces
        - Partial word matches that should be full words

        Args:
            entities: Raw extracted entities.
            text: Original text for boundary checking.

        Returns:
            Corrected list of Entity objects.
        """
        import re

        corrected = []

        for entity in entities:
            start, end = entity.span
            entity_text = entity.text

            # Skip empty or whitespace-only entities
            if not entity_text.strip():
                continue

            # Strip leading/trailing whitespace and adjust spans
            stripped = entity_text.strip()
            leading_spaces = len(entity_text) - len(entity_text.lstrip())
            start += leading_spaces
            end = start + len(stripped)
            entity_text = stripped

            # Expand to word boundaries if we're mid-word
            # Check character before start
            if start > 0 and text[start - 1].isalnum():
                # Find the start of the word
                while start > 0 and text[start - 1].isalnum():
                    start -= 1

            # Check character after end
            if end < len(text) and text[end].isalnum():
                # Find the end of the word
                while end < len(text) and text[end].isalnum():
                    end += 1

            # Get the corrected text
            corrected_text = text[start:end].strip()

            # Skip if too short after correction
            if len(corrected_text) < 2:
                continue

            # Skip if it's now just punctuation or numbers
            if not re.search(r'[a-zA-Z]', corrected_text):
                continue

            corrected.append(Entity(
                text=corrected_text,
                label=entity.label,
                span=(start, end),
                normalized=corrected_text.lower(),
                negated=entity.negated,
                confidence=entity.confidence,
            ))

        # Remove duplicates that may have been created by boundary expansion
        return self._deduplicate_entities(corrected)

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """
        Remove duplicate entities with overlapping spans.

        Keeps the entity with higher confidence or longer span.

        Args:
            entities: List of entities that may have duplicates.

        Returns:
            Deduplicated list of entities.
        """
        if not entities:
            return []

        # Sort by span start, then by span length (descending)
        sorted_entities = sorted(
            entities,
            key=lambda e: (e.span[0], -(e.span[1] - e.span[0])),
        )

        result: list[Entity] = []
        last_end = -1

        for entity in sorted_entities:
            # Skip if this entity overlaps with the previous one
            if entity.span[0] < last_end:
                continue

            result.append(entity)
            last_end = entity.span[1]

        return result

    def _merge_adjacent(
        self,
        entities: list[BiomedicalEntity],
        text: str,
    ) -> list[BiomedicalEntity]:
        """Merge adjacent entities of the same type."""
        if not entities:
            return []

        merged = []
        current = entities[0]

        for next_ent in entities[1:]:
            # Check if adjacent (within 1 char) and same type
            gap = next_ent.start - current.end
            if gap <= 1 and next_ent.label == current.label:
                # Merge: extend current entity
                actual_text = text[current.start:next_ent.end]
                current = BiomedicalEntity(
                    text=actual_text,
                    label=current.label,
                    start=current.start,
                    end=next_ent.end,
                    score=min(current.score, next_ent.score),
                )
            else:
                merged.append(current)
                current = next_ent

        merged.append(current)
        return merged

    def get_medications(self, text: str) -> list[str]:
        """Extract just medication names."""
        return [
            e.text for e in self.extract(text)
            if e.label == "Medication"
        ]

    def get_conditions(self, text: str) -> list[str]:
        """Extract diseases and symptoms."""
        return [
            e.text for e in self.extract(text)
            if e.label in ("Disease_disorder", "Sign_symptom")
        ]

    def get_procedures(self, text: str) -> list[str]:
        """Extract procedures."""
        return [
            e.text for e in self.extract(text)
            if e.label in ("Therapeutic_procedure", "Diagnostic_procedure")
        ]
