"""
Medical NER pipeline for entity extraction.

This module provides the NER pipeline interface and implementations
for extracting medical entities from text.

Example:
    >>> from hsttb.nlp.ner_pipeline import MockNERPipeline
    >>> pipeline = MockNERPipeline.with_common_patterns()
    >>> entities = pipeline.extract_entities("patient takes metformin 500mg")
    >>> print([e.text for e in entities])
    ['metformin', '500mg']
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from hsttb.core.types import Entity, EntityLabel


@dataclass
class NERPipelineConfig:
    """
    Configuration for NER pipeline.

    Attributes:
        detect_negation: Whether to detect negated entities.
        normalize_entities: Whether to normalize entity text.
        min_entity_length: Minimum entity length to extract.
    """

    detect_negation: bool = True
    normalize_entities: bool = True
    min_entity_length: int = 2


class NERPipeline(ABC):
    """
    Abstract base class for NER pipelines.

    All NER implementations must implement this interface
    to be used with the NER accuracy engine.

    Example:
        >>> class MyNER(NERPipeline):
        ...     def extract_entities(self, text: str) -> list[Entity]:
        ...         # Implementation
        ...         pass
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return pipeline name."""
        ...

    @property
    @abstractmethod
    def supported_labels(self) -> list[EntityLabel]:
        """Return list of supported entity labels."""
        ...

    @abstractmethod
    def extract_entities(self, text: str) -> list[Entity]:
        """
        Extract medical entities from text.

        Args:
            text: Text to analyze.

        Returns:
            List of Entity objects.
        """
        ...

    def extract_entities_with_context(
        self,
        text: str,
        context: str | None = None,  # noqa: ARG002
    ) -> list[Entity]:
        """
        Extract entities with additional context.

        Default implementation ignores context.

        Args:
            text: Text to analyze.
            context: Optional surrounding context.

        Returns:
            List of Entity objects.
        """
        return self.extract_entities(text)


@dataclass
class MockNERPipeline(NERPipeline):
    """
    Mock NER pipeline for testing.

    Uses pattern matching to identify entities.
    Useful for testing the NER accuracy engine without
    requiring external NLP models.

    Attributes:
        patterns: Dictionary mapping patterns to entity labels.
        config: Pipeline configuration.

    Example:
        >>> pipeline = MockNERPipeline.with_common_patterns()
        >>> entities = pipeline.extract_entities("patient has diabetes")
        >>> print(entities[0].label)
        EntityLabel.DIAGNOSIS
    """

    patterns: dict[str, EntityLabel] = field(default_factory=dict)
    config: NERPipelineConfig = field(default_factory=NERPipelineConfig)
    _name: str = "mock_ner"

    @property
    def name(self) -> str:
        """Return pipeline name."""
        return self._name

    @property
    def supported_labels(self) -> list[EntityLabel]:
        """Return supported entity labels."""
        return list(set(self.patterns.values()))

    def add_pattern(self, pattern: str, label: EntityLabel) -> None:
        """
        Add a pattern to recognize.

        Args:
            pattern: Regex pattern or exact string.
            label: Entity label for matches.
        """
        self.patterns[pattern.lower()] = label

    def extract_entities(self, text: str) -> list[Entity]:
        """
        Extract entities using pattern matching.

        Args:
            text: Text to analyze.

        Returns:
            List of Entity objects.
        """
        entities: list[Entity] = []
        text_lower = text.lower()

        # Check for negation patterns
        negation_patterns = [
            r"\bno\s+",
            r"\bnot\s+",
            r"\bwithout\s+",
            r"\bdenies\s+",
            r"\bnegative\s+for\s+",
            r"\bno\s+evidence\s+of\s+",
        ]

        for pattern, label in self.patterns.items():
            # Find all matches
            for match in re.finditer(
                rf"\b{re.escape(pattern)}\b", text_lower, re.IGNORECASE
            ):
                start, end = match.start(), match.end()
                entity_text = text[start:end]

                if len(entity_text) < self.config.min_entity_length:
                    continue

                # Check for negation
                negated = False
                if self.config.detect_negation:
                    # Look for negation patterns before the entity
                    prefix = text_lower[max(0, start - 30) : start]
                    for neg_pattern in negation_patterns:
                        if re.search(neg_pattern, prefix):
                            negated = True
                            break

                # Normalize if configured
                normalized = entity_text.lower().strip() if self.config.normalize_entities else None

                entities.append(
                    Entity(
                        text=entity_text,
                        label=label,
                        span=(start, end),
                        normalized=normalized,
                        negated=negated,
                    )
                )

        # Sort by span start and remove overlapping
        entities = self._remove_overlapping(entities)
        return entities

    def _remove_overlapping(self, entities: list[Entity]) -> list[Entity]:
        """Remove overlapping entities, keeping longer ones."""
        if not entities:
            return entities

        # Sort by span start
        entities.sort(key=lambda e: (e.span[0], -(e.span[1] - e.span[0])))

        result: list[Entity] = []
        last_end = -1

        for entity in entities:
            if entity.span[0] >= last_end:
                result.append(entity)
                last_end = entity.span[1]

        return result

    @classmethod
    def with_common_patterns(cls) -> MockNERPipeline:
        """
        Create pipeline with common medical patterns.

        Returns:
            MockNERPipeline with drugs, diagnoses, dosages, etc.
        """
        pipeline = cls()

        # Drug patterns - comprehensive list including commonly confused drugs
        drugs = [
            # Diabetes medications
            "metformin", "glipizide", "glyburide", "insulin", "ozempic",
            "jardiance", "farxiga", "trulicity", "victoza",
            # Cardiovascular medications
            "aspirin", "lisinopril", "atorvastatin", "amlodipine", "metoprolol",
            "losartan", "hydrochlorothiazide", "warfarin", "clopidogrel",
            "simvastatin", "pravastatin", "rosuvastatin", "carvedilol",
            "furosemide", "spironolactone", "digoxin", "diltiazem", "verapamil",
            # Pain/Anti-inflammatory
            "ibuprofen", "acetaminophen", "naproxen", "celecoxib", "tramadol",
            "oxycodone", "hydrocodone", "morphine", "fentanyl", "gabapentin",
            "pregabalin", "meloxicam", "prednisone", "methylprednisolone",
            # Antibiotics
            "amoxicillin", "azithromycin", "ciprofloxacin", "levofloxacin",
            "doxycycline", "metronidazole", "clindamycin", "cephalexin",
            "sulfamethoxazole", "trimethoprim", "nitrofurantoin", "penicillin",
            # GI medications
            "omeprazole", "pantoprazole", "esomeprazole", "famotidine",
            "ranitidine", "ondansetron", "metoclopramide", "dicyclomine",
            # Psych medications
            "sertraline", "fluoxetine", "escitalopram", "citalopram",
            "duloxetine", "venlafaxine", "bupropion", "trazodone",
            "alprazolam", "lorazepam", "diazepam", "clonazepam",
            "quetiapine", "risperidone", "olanzapine", "aripiprazole",
            # Thyroid
            "levothyroxine", "synthroid", "armour thyroid",
            # Commonly confused/misheard drugs (critical for STT evaluation)
            "methotrexate", "hydroxychloroquine", "sulfasalazine",  # RA drugs
            "lisinopril", "losartan",  # Sound-alike
            "clonidine", "klonopin",   # Sound-alike
            "lamotrigine", "lamictal",
            "labetalol", "levetiracetam",
            # Urinary/Antibiotics from the example
            "ciprofloxacin", "phenazopyridine", "nitrofurantoin",
        ]
        for drug in drugs:
            pipeline.add_pattern(drug, EntityLabel.DRUG)

        # Diagnosis patterns
        diagnoses = [
            "diabetes", "diabetes mellitus", "type 2 diabetes",
            "hypertension", "high blood pressure",
            "hyperlipidemia", "high cholesterol",
            "coronary artery disease", "heart disease",
            "heart failure", "congestive heart failure",
            "atrial fibrillation", "afib",
            "copd", "chronic obstructive pulmonary disease",
            "asthma", "pneumonia",
            "depression", "anxiety",
            "arthritis", "osteoarthritis",
            "chronic kidney disease", "renal failure",
            "stroke", "myocardial infarction", "heart attack",
        ]
        for dx in diagnoses:
            pipeline.add_pattern(dx, EntityLabel.DIAGNOSIS)

        # Dosage patterns (using regex-friendly patterns)
        pipeline.patterns[r"\d+\s*mg"] = EntityLabel.DOSAGE
        pipeline.patterns[r"\d+\s*mcg"] = EntityLabel.DOSAGE
        pipeline.patterns[r"\d+\s*ml"] = EntityLabel.DOSAGE
        pipeline.patterns[r"\d+\s*units"] = EntityLabel.DOSAGE

        # Anatomy patterns
        anatomy = [
            "heart", "lung", "liver", "kidney", "brain",
            "chest", "abdomen", "head", "neck", "back",
            "arm", "leg", "hand", "foot",
        ]
        for part in anatomy:
            pipeline.add_pattern(part, EntityLabel.ANATOMY)

        # Symptom patterns
        symptoms = [
            "pain", "chest pain", "headache", "fatigue",
            "nausea", "vomiting", "dizziness", "shortness of breath",
            "cough", "fever", "swelling", "weakness",
        ]
        for symptom in symptoms:
            pipeline.add_pattern(symptom, EntityLabel.SYMPTOM)

        # Procedure patterns
        procedures = [
            "surgery", "biopsy", "mri", "ct scan", "x-ray",
            "ultrasound", "endoscopy", "colonoscopy",
            "echocardiogram", "ekg", "blood test",
        ]
        for proc in procedures:
            pipeline.add_pattern(proc, EntityLabel.PROCEDURE)

        # Lab value patterns
        labs = [
            "hemoglobin", "hematocrit", "white blood cell",
            "platelet", "creatinine", "bun", "glucose",
            "potassium", "sodium", "calcium",
            "a1c", "hemoglobin a1c",
        ]
        for lab in labs:
            pipeline.add_pattern(lab, EntityLabel.LAB_VALUE)

        return pipeline

    @classmethod
    def with_custom_patterns(
        cls,
        drug_patterns: list[str] | None = None,
        diagnosis_patterns: list[str] | None = None,
    ) -> MockNERPipeline:
        """
        Create pipeline with custom patterns.

        Args:
            drug_patterns: Custom drug patterns.
            diagnosis_patterns: Custom diagnosis patterns.

        Returns:
            Configured MockNERPipeline.
        """
        pipeline = cls()

        for pattern in drug_patterns or []:
            pipeline.add_pattern(pattern, EntityLabel.DRUG)

        for pattern in diagnosis_patterns or []:
            pipeline.add_pattern(pattern, EntityLabel.DIAGNOSIS)

        return pipeline
