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

    >>> from hsttb.nlp import get_nlp_pipeline, list_nlp_pipelines
    >>> print(list_nlp_pipelines())
    ['biomedical', 'medspacy', 'mock', 'scispacy']
    >>> pipeline = get_nlp_pipeline("scispacy")

    >>> from hsttb.nlp import EntityAligner, align_entities
    >>> matches = align_entities(gold_entities, pred_entities)

    >>> from hsttb.nlp import NegationDetector
    >>> detector = NegationDetector()
    >>> negations = detector.detect_negations("patient denies chest pain")
"""
from __future__ import annotations

from hsttb.nlp.entity_alignment import (
    AlignmentConfig,
    EntityAligner,
    SpanMatchStrategy,
    align_entities,
)
from hsttb.nlp.negation import (
    NegationConfig,
    NegationConsistencyResult,
    NegationDetector,
    NegationSpan,
    check_negation_consistency,
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
from hsttb.nlp.registry import (
    clear_nlp_registry,
    get_nlp_pipeline,
    get_pipeline_info,
    is_nlp_pipeline_registered,
    list_nlp_pipelines,
    register_nlp_pipeline,
    register_nlp_pipeline_factory,
    unregister_nlp_pipeline,
)


# Lazy loading for heavy NLP pipelines
def __getattr__(name: str) -> type:
    """Lazy import NLP pipelines."""
    if name == "SciSpacyNERPipeline":
        from hsttb.nlp.scispacy_ner import SciSpacyNERPipeline
        return SciSpacyNERPipeline

    if name == "BiomedicalNERPipeline":
        from hsttb.nlp.biomedical_ner import BiomedicalNERPipeline
        return BiomedicalNERPipeline

    if name == "MedSpacyNERPipeline":
        from hsttb.nlp.medspacy_ner import MedSpacyNERPipeline
        return MedSpacyNERPipeline

    if name == "TransformerSimilarityEngine":
        from hsttb.nlp.semantic_similarity import TransformerSimilarityEngine
        return TransformerSimilarityEngine

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Entity alignment
    "AlignmentConfig",
    "EntityAligner",
    "SpanMatchStrategy",
    "align_entities",
    # Negation detection
    "NegationConfig",
    "NegationConsistencyResult",
    "NegationDetector",
    "NegationSpan",
    "check_negation_consistency",
    # NER pipelines
    "MockNERPipeline",
    "NERPipeline",
    "NERPipelineConfig",
    # Lazy-loaded pipelines
    "SciSpacyNERPipeline",
    "BiomedicalNERPipeline",
    "MedSpacyNERPipeline",
    "TransformerSimilarityEngine",
    # Text normalization
    "MedicalTextNormalizer",
    "NormalizerConfig",
    "create_normalizer",
    "normalize_for_ter",
    # Pipeline registry
    "clear_nlp_registry",
    "get_nlp_pipeline",
    "get_pipeline_info",
    "is_nlp_pipeline_registered",
    "list_nlp_pipelines",
    "register_nlp_pipeline",
    "register_nlp_pipeline_factory",
    "unregister_nlp_pipeline",
]
