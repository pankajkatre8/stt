"""
MedSpaCy NER pipeline for clinical NLP.

Uses MedSpaCy which builds on spaCy with additional clinical NLP
components including:
- Clinical NER
- Section detection
- Context analysis (negation, uncertainty, family history)
- Target matching

Example:
    >>> pipeline = MedSpacyNERPipeline()
    >>> entities = pipeline.extract_entities("Patient denies chest pain")
    >>> for e in entities:
    ...     print(f"{e.label}: {e.text} (negated={e.negated})")
"""
from __future__ import annotations

import logging
from typing import Any

from hsttb.core.types import Entity, EntityLabel
from hsttb.nlp.ner_pipeline import NERPipeline

logger = logging.getLogger(__name__)


class MedSpacyNERPipeline(NERPipeline):
    """
    Clinical NER pipeline using MedSpaCy.

    MedSpaCy provides enhanced clinical NLP capabilities including
    context detection for negation, uncertainty, and family history.
    This is particularly important for healthcare applications where
    "no chest pain" means something very different from "chest pain".

    Entity label mapping:
        - PROBLEM -> CONDITION (diseases, symptoms)
        - TREATMENT -> DRUG (medications, therapies)
        - TEST -> PROCEDURE (diagnostic tests)
        - ANATOMY -> ANATOMY (body parts)
        - And others from QuickUMLS or custom rules

    Attributes:
        detect_negation: Whether to detect negated entities.
        detect_uncertainty: Whether to detect uncertain entities.
        enable_context: Whether to use ConText algorithm.

    Example:
        >>> pipeline = MedSpacyNERPipeline()
        >>> entities = pipeline.extract_entities(
        ...     "Patient has diabetes but denies chest pain"
        ... )
        >>> for e in entities:
        ...     status = "negated" if e.negated else "affirmed"
        ...     print(f"{e.text}: {status}")
        diabetes: affirmed
        chest pain: negated
    """

    _instance: MedSpacyNERPipeline | None = None
    _nlp: Any = None

    # Map MedSpaCy labels to our EntityLabel
    LABEL_MAP: dict[str, EntityLabel] = {
        # Clinical concepts
        "PROBLEM": EntityLabel.CONDITION,
        "TREATMENT": EntityLabel.DRUG,
        "TEST": EntityLabel.PROCEDURE,
        "MEDICATION": EntityLabel.DRUG,
        "DRUG": EntityLabel.DRUG,
        "DISEASE": EntityLabel.CONDITION,
        "SYMPTOM": EntityLabel.SYMPTOM,
        "ANATOMY": EntityLabel.ANATOMY,
        "PROCEDURE": EntityLabel.PROCEDURE,
        "LAB": EntityLabel.LAB_VALUE,
        "LAB_VALUE": EntityLabel.LAB_VALUE,
        # If using QuickUMLS semantic types
        "T047": EntityLabel.CONDITION,  # Disease or Syndrome
        "T184": EntityLabel.SYMPTOM,  # Sign or Symptom
        "T121": EntityLabel.DRUG,  # Pharmacologic Substance
        "T200": EntityLabel.DRUG,  # Clinical Drug
        "T023": EntityLabel.ANATOMY,  # Body Part
        "T060": EntityLabel.PROCEDURE,  # Diagnostic Procedure
        "T061": EntityLabel.PROCEDURE,  # Therapeutic Procedure
        "T034": EntityLabel.LAB_VALUE,  # Laboratory Finding
    }

    def __new__(cls) -> MedSpacyNERPipeline:
        """Singleton pattern for model reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        detect_negation: bool = True,
        detect_uncertainty: bool = True,
        enable_context: bool = True,
    ) -> None:
        """
        Initialize the MedSpaCy pipeline.

        Args:
            detect_negation: Enable negation detection.
            detect_uncertainty: Enable uncertainty detection.
            enable_context: Enable ConText algorithm.
        """
        self._detect_negation = detect_negation
        self._detect_uncertainty = detect_uncertainty
        self._enable_context = enable_context

    @property
    def name(self) -> str:
        """Return pipeline name."""
        return "medspacy"

    @property
    def supported_labels(self) -> list[EntityLabel]:
        """Return supported entity labels."""
        return [
            EntityLabel.DRUG,
            EntityLabel.CONDITION,
            EntityLabel.SYMPTOM,
            EntityLabel.PROCEDURE,
            EntityLabel.ANATOMY,
            EntityLabel.LAB_VALUE,
        ]

    def _ensure_loaded(self) -> None:
        """Load MedSpaCy model if not already loaded."""
        if self._nlp is not None:
            return

        try:
            import medspacy
        except ImportError as e:
            raise ImportError(
                "MedSpaCy is not installed. Install with: "
                "pip install medspacy"
            ) from e

        try:
            # Load MedSpaCy with clinical components
            self._nlp = medspacy.load(
                enable=["medspacy_pyrush", "medspacy_context"]
                if self._enable_context
                else ["medspacy_pyrush"]
            )

            # Add target rules for common clinical concepts if not using QuickUMLS
            self._add_default_target_rules()

            logger.info("Loaded MedSpaCy pipeline with clinical components")

        except Exception as e:
            logger.warning(f"Failed to load full MedSpaCy, falling back to basic: {e}")
            # Fallback to basic MedSpaCy
            try:
                import medspacy
                self._nlp = medspacy.load()
                self._add_default_target_rules()
            except Exception as e2:
                raise ImportError(
                    f"Failed to load MedSpaCy pipeline: {e2}"
                ) from e2

    def _add_default_target_rules(self) -> None:
        """Add default target matching rules for clinical concepts."""
        try:
            from medspacy.ner import TargetRule
            from medspacy.target_matcher import TargetMatcher
        except ImportError:
            logger.warning("Could not import TargetMatcher, skipping default rules")
            return

        # Check if target_matcher is in the pipeline
        if "medspacy_target_matcher" not in self._nlp.pipe_names:
            target_matcher = TargetMatcher(self._nlp)
            self._nlp.add_pipe(target_matcher)

        # Get the target matcher
        target_matcher = self._nlp.get_pipe("medspacy_target_matcher")

        # Common medications
        medication_patterns = [
            "metformin", "aspirin", "lisinopril", "atorvastatin",
            "omeprazole", "amlodipine", "metoprolol", "losartan",
            "gabapentin", "hydrochlorothiazide", "warfarin", "prednisone",
            "amoxicillin", "ibuprofen", "acetaminophen", "insulin",
            "levothyroxine", "simvastatin", "pantoprazole", "clopidogrel",
            "methotrexate", "morphine", "oxycodone", "fentanyl",
        ]

        # Common conditions/problems
        condition_patterns = [
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
            "cancer", "tumor", "malignancy",
        ]

        # Symptoms
        symptom_patterns = [
            "chest pain", "shortness of breath", "dyspnea",
            "headache", "fatigue", "nausea", "vomiting",
            "dizziness", "syncope", "pain", "weakness",
            "fever", "cough", "swelling", "edema",
            "palpitations", "abdominal pain",
        ]

        # Procedures
        procedure_patterns = [
            "surgery", "biopsy", "mri", "ct scan", "x-ray",
            "ultrasound", "endoscopy", "colonoscopy",
            "echocardiogram", "ekg", "electrocardiogram",
            "blood test", "cbc", "bmp", "cmp",
            "catheterization", "angiography",
        ]

        # Add rules
        rules = []

        for med in medication_patterns:
            rules.append(TargetRule(med, "TREATMENT"))

        for condition in condition_patterns:
            rules.append(TargetRule(condition, "PROBLEM"))

        for symptom in symptom_patterns:
            rules.append(TargetRule(symptom, "PROBLEM"))

        for proc in procedure_patterns:
            rules.append(TargetRule(proc, "TEST"))

        try:
            target_matcher.add(rules)
        except Exception as e:
            logger.warning(f"Could not add target rules: {e}")

    def extract_entities(self, text: str) -> list[Entity]:
        """
        Extract medical entities from text with context.

        Args:
            text: Clinical text to analyze.

        Returns:
            List of Entity objects with negation status.
        """
        self._ensure_loaded()

        doc = self._nlp(text)
        entities = []

        for ent in doc.ents:
            # Map label to our EntityLabel
            label = self.LABEL_MAP.get(ent.label_, EntityLabel.OTHER)

            # Check for negation using ConText
            negated = False
            if self._detect_negation and hasattr(ent._, "is_negated"):
                negated = bool(ent._.is_negated)

            # Check for uncertainty (if available)
            uncertain = False
            if self._detect_uncertainty and hasattr(ent._, "is_uncertain"):
                uncertain = bool(ent._.is_uncertain)

            # Skip if it's just whitespace
            if not ent.text.strip():
                continue

            entities.append(Entity(
                text=ent.text,
                label=label,
                span=(ent.start_char, ent.end_char),
                normalized=ent.text.lower().strip(),
                negated=negated,
                confidence=0.9 if not uncertain else 0.5,
            ))

        return entities

    def extract_entities_with_context(
        self,
        text: str,
        context: str | None = None,
    ) -> list[Entity]:
        """
        Extract entities with additional context.

        MedSpaCy benefits from having full document context for
        better section and modifier detection.

        Args:
            text: Text to analyze.
            context: Optional surrounding context.

        Returns:
            List of Entity objects.
        """
        if context:
            # Combine context with text for better analysis
            full_text = f"{context} {text}"
            offset = len(context) + 1

            entities = self.extract_entities(full_text)

            # Adjust spans to be relative to original text
            adjusted_entities = []
            for ent in entities:
                start, end = ent.span
                if start >= offset:
                    adjusted_entities.append(Entity(
                        text=ent.text,
                        label=ent.label,
                        span=(start - offset, end - offset),
                        normalized=ent.normalized,
                        negated=ent.negated,
                        confidence=ent.confidence,
                    ))

            return adjusted_entities

        return self.extract_entities(text)

    def get_sections(self, text: str) -> list[dict[str, Any]]:
        """
        Detect clinical document sections.

        Args:
            text: Clinical document text.

        Returns:
            List of section dictionaries with title and text.
        """
        self._ensure_loaded()

        doc = self._nlp(text)
        sections = []

        if hasattr(doc, "_.sections"):
            for section in doc._.sections:
                sections.append({
                    "title": section.title_span.text if section.title_span else "",
                    "category": section.category or "Unknown",
                    "text": section.text,
                    "start": section.start,
                    "end": section.end,
                })

        return sections


# Singleton accessor
_default_pipeline: MedSpacyNERPipeline | None = None


def get_medspacy_pipeline() -> MedSpacyNERPipeline:
    """Get the default MedSpaCy NER pipeline."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = MedSpacyNERPipeline()
    return _default_pipeline
