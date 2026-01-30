"""
NLP pipelines for medical text processing.

This module provides medical NER, text normalization,
negation detection, entity alignment, and other NLP utilities.

Example:
    >>> from hsttb.nlp import MedicalTextNormalizer, normalize_for_ter
    >>> normalizer = MedicalTextNormalizer()
    >>> normalizer.normalize("Patient has HTN, takes 500mg metformin BID")
    'patient has hypertension, takes 500 mg metformin twice daily'

    >>> from hsttb.nlp import MockNERPipeline
    >>> pipeline = MockNERPipeline.with_common_patterns()
    >>> entities = pipeline.extract_entities("patient takes metformin")
    >>> print([e.text for e in entities])
    ['metformin']

    >>> from hsttb.nlp import EntityAligner, align_entities
    >>> matches = align_entities(gold_entities, pred_entities)
"""
from __future__ import annotations

from hsttb.nlp.entity_alignment import (
    AlignmentConfig,
    EntityAligner,
    SpanMatchStrategy,
    align_entities,
)
from hsttb.nlp.ner_pipeline import (
    MockNERPipeline,
    NERPipeline,
    NERPipelineConfig,
)
from hsttb.nlp.normalizer import (
    MedicalTextNormalizer,
    NormalizerConfig,
    create_normalizer,
    normalize_for_ter,
)

__all__ = [
    "AlignmentConfig",
    "EntityAligner",
    "MedicalTextNormalizer",
    "MockNERPipeline",
    "NERPipeline",
    "NERPipelineConfig",
    "NormalizerConfig",
    "SpanMatchStrategy",
    "align_entities",
    "create_normalizer",
    "normalize_for_ter",
]
