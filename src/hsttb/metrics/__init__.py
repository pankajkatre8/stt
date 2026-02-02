"""
Metric computation engines.

This module provides the core metric engines:
- TER (Term Error Rate)
- NER Accuracy
- CRS (Context Retention Score)
- SRS (Streaming Robustness Score)

Example:
    >>> from hsttb.metrics import TEREngine, compute_ter
    >>> from hsttb.lexicons import MockMedicalLexicon
    >>> lexicon = MockMedicalLexicon.with_common_terms()
    >>> ter = compute_ter("patient has diabetes", "patient has diabetes", lexicon)

    >>> from hsttb.metrics import NEREngine, compute_ner_accuracy
    >>> from hsttb.nlp import MockNERPipeline
    >>> pipeline = MockNERPipeline.with_common_patterns()
    >>> engine = NEREngine(pipeline)

    >>> from hsttb.metrics import CRSEngine, compute_crs
    >>> engine = CRSEngine()
    >>> crs = compute_crs(["segment 1"], ["segment 1"])

    >>> from hsttb.metrics import SRSEngine
    >>> engine = SRSEngine()
"""
from __future__ import annotations

from hsttb.metrics.crs import CRSConfig, CRSEngine, compute_crs
from hsttb.metrics.entity_continuity import (
    ContinuityResult,
    Discontinuity,
    DiscontinuityType,
    EntityContinuityTracker,
    EntityOccurrence,
    compute_entity_continuity,
)
from hsttb.metrics.ner import (
    NEREngine,
    NEREngineConfig,
    compute_entity_f1,
    compute_ner_accuracy,
)
from hsttb.metrics.semantic_similarity import (
    EmbeddingBasedSimilarity,
    SemanticSimilarityEngine,
    SimilarityConfig,
    TokenBasedSimilarity,
    compute_semantic_similarity,
    create_similarity_engine,
)
from hsttb.metrics.srs import SRSConfig, SRSEngine, compute_srs
from hsttb.metrics.ter import TEREngine, TERResult, compute_ter

# Quality metrics (optional - require transformers/language-tool-python)
try:
    from hsttb.metrics.quality import QualityConfig, QualityEngine, QualityResult, compute_quality
    _QUALITY_AVAILABLE = True
except ImportError:
    _QUALITY_AVAILABLE = False

# New reference-free metrics
from hsttb.metrics.contradiction import (
    ContradictionDetector,
    ContradictionPair,
    ContradictionResult,
    detect_contradictions,
)
from hsttb.metrics.speech_rate import (
    SpeechRateCategory,
    SpeechRateResult,
    SpeechRateValidator,
    validate_speech_rate,
)
from hsttb.metrics.embedding_drift import (
    DriftPoint,
    EmbeddingDriftDetector,
    EmbeddingDriftResult,
    analyze_embedding_drift,
)
from hsttb.metrics.confidence_variance import (
    ConfidenceAnalyzer,
    ConfidenceDropPoint,
    ConfidenceVarianceResult,
    analyze_confidence_variance,
)

__all__ = [
    "CRSConfig",
    "CRSEngine",
    "ContinuityResult",
    "Discontinuity",
    "DiscontinuityType",
    "EmbeddingBasedSimilarity",
    "EntityContinuityTracker",
    "EntityOccurrence",
    "NEREngine",
    "NEREngineConfig",
    "SRSConfig",
    "SRSEngine",
    "SemanticSimilarityEngine",
    "SimilarityConfig",
    "TEREngine",
    "TERResult",
    "TokenBasedSimilarity",
    "compute_crs",
    "compute_entity_continuity",
    "compute_entity_f1",
    "compute_ner_accuracy",
    "compute_semantic_similarity",
    "compute_srs",
    "compute_ter",
    "create_similarity_engine",
    # New reference-free metrics
    "ContradictionDetector",
    "ContradictionPair",
    "ContradictionResult",
    "detect_contradictions",
    "SpeechRateCategory",
    "SpeechRateResult",
    "SpeechRateValidator",
    "validate_speech_rate",
    "DriftPoint",
    "EmbeddingDriftDetector",
    "EmbeddingDriftResult",
    "analyze_embedding_drift",
    "ConfidenceAnalyzer",
    "ConfidenceDropPoint",
    "ConfidenceVarianceResult",
    "analyze_confidence_variance",
]

# Add quality exports if available
if _QUALITY_AVAILABLE:
    __all__.extend([
        "QualityConfig",
        "QualityEngine",
        "QualityResult",
        "compute_quality",
    ])
