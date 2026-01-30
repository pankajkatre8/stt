"""
Metric computation engines.

This module provides the core metric engines:
- TER (Term Error Rate)
- NER Accuracy (to be implemented)
- CRS (Context Retention Score) (to be implemented)
- SRS (Streaming Robustness Score) (to be implemented)

Example:
    >>> from hsttb.metrics import TEREngine, compute_ter
    >>> from hsttb.lexicons import MockMedicalLexicon
    >>> lexicon = MockMedicalLexicon.with_common_terms()
    >>> ter = compute_ter("patient has diabetes", "patient has diabetes", lexicon)
"""
from __future__ import annotations

from hsttb.metrics.ter import TEREngine, TERResult, compute_ter

__all__ = [
    "TEREngine",
    "TERResult",
    "compute_ter",
]
