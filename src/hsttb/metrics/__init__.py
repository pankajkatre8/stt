"""
Metric computation engines.

This module provides the core metric engines:
- TER (Term Error Rate)
- NER Accuracy
- CRS (Context Retention Score) (to be implemented)
- SRS (Streaming Robustness Score) (to be implemented)

Example:
    >>> from hsttb.metrics import TEREngine, compute_ter
    >>> from hsttb.lexicons import MockMedicalLexicon
    >>> lexicon = MockMedicalLexicon.with_common_terms()
    >>> ter = compute_ter("patient has diabetes", "patient has diabetes", lexicon)

    >>> from hsttb.metrics import NEREngine, compute_ner_accuracy
    >>> from hsttb.nlp import MockNERPipeline
    >>> pipeline = MockNERPipeline.with_common_patterns()
    >>> engine = NEREngine(pipeline)
"""
from __future__ import annotations

from hsttb.metrics.ner import (
    NEREngine,
    NEREngineConfig,
    compute_entity_f1,
    compute_ner_accuracy,
)
from hsttb.metrics.ter import TEREngine, TERResult, compute_ter

__all__ = [
    "NEREngine",
    "NEREngineConfig",
    "TEREngine",
    "TERResult",
    "compute_entity_f1",
    "compute_ner_accuracy",
    "compute_ter",
]
