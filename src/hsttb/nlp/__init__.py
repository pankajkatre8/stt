"""
NLP pipelines for medical text processing.

This module provides medical NER, text normalization,
negation detection, and other NLP utilities.

Example:
    >>> from hsttb.nlp import MedicalTextNormalizer, normalize_for_ter
    >>> normalizer = MedicalTextNormalizer()
    >>> normalizer.normalize("Patient has HTN, takes 500mg metformin BID")
    'patient has hypertension, takes 500 mg metformin twice daily'
"""
from __future__ import annotations

from hsttb.nlp.normalizer import (
    MedicalTextNormalizer,
    NormalizerConfig,
    create_normalizer,
    normalize_for_ter,
)

__all__ = [
    "MedicalTextNormalizer",
    "NormalizerConfig",
    "create_normalizer",
    "normalize_for_ter",
]
