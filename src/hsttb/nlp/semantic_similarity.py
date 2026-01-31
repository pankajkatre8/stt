"""
Production semantic similarity using sentence-transformers.

Uses clinical/biomedical embedding models for accurate
semantic similarity in healthcare text.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from hsttb.metrics.semantic_similarity import (
    SemanticSimilarityEngine as BaseEngine,
)


class TransformerSimilarityEngine(BaseEngine):
    """
    Production semantic similarity using sentence-transformers.

    Uses PubMedBERT-based model fine-tuned for clinical NLI tasks,
    providing accurate semantic similarity for medical text.

    Example:
        >>> engine = SemanticSimilarityEngine()
        >>> score = engine.similarity(
        ...     "Patient has diabetes mellitus",
        ...     "Patient diagnosed with type 2 diabetes"
        ... )
        >>> print(f"Similarity: {score:.2f}")  # ~0.85
    """

    # Available models in order of preference
    MODELS = [
        "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "sentence-transformers/all-MiniLM-L6-v2",  # Fallback
    ]

    _instance: TransformerSimilarityEngine | None = None
    _model: Any = None
    _model_name: str = ""

    def __new__(cls) -> TransformerSimilarityEngine:
        """Singleton pattern for efficient model reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize (model loaded on first use)."""
        pass

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        # Try models in order of preference
        for model_name in self.MODELS:
            try:
                self._model = SentenceTransformer(model_name)
                self._model_name = model_name
                break
            except Exception:
                continue

        if self._model is None:
            raise RuntimeError(
                "Failed to load any sentence-transformer model. "
                "Install with: pip install sentence-transformers"
            )

    @property
    def model_name(self) -> str:
        """Return the loaded model name."""
        self._ensure_loaded()
        return self._model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode.

        Returns:
            Numpy array of embeddings.
        """
        self._ensure_loaded()
        return self._model.encode(texts, convert_to_numpy=True)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score (0.0-1.0).
        """
        if not text1 or not text2:
            return 0.0 if (text1 or text2) else 1.0

        embeddings = self.encode([text1, text2])
        return float(self._cosine_similarity(embeddings[0], embeddings[1]))

    def similarity(self, text1: str, text2: str) -> float:
        """Alias for compute_similarity."""
        return self.compute_similarity(text1, text2)

    def batch_similarity(
        self,
        texts1: list[str],
        texts2: list[str],
    ) -> list[float]:
        """
        Compute similarities for pairs of texts.

        Args:
            texts1: First list of texts.
            texts2: Second list of texts (same length).

        Returns:
            List of similarity scores.
        """
        if len(texts1) != len(texts2):
            raise ValueError("Lists must have same length")

        if not texts1:
            return []

        # Encode all texts at once for efficiency
        all_texts = texts1 + texts2
        embeddings = self.encode(all_texts)

        n = len(texts1)
        emb1 = embeddings[:n]
        emb2 = embeddings[n:]

        scores = []
        for i in range(n):
            if not texts1[i] or not texts2[i]:
                scores.append(0.0 if (texts1[i] or texts2[i]) else 1.0)
            else:
                scores.append(float(self._cosine_similarity(emb1[i], emb2[i])))

        return scores

    def average_similarity(
        self,
        texts1: list[str],
        texts2: list[str],
    ) -> float:
        """
        Compute average similarity across text pairs.

        Args:
            texts1: First list of texts.
            texts2: Second list of texts.

        Returns:
            Average similarity score.
        """
        scores = self.batch_similarity(texts1, texts2)
        return sum(scores) / len(scores) if scores else 0.0

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between vectors."""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)


# Singleton instance
_default_engine: TransformerSimilarityEngine | None = None


def get_transformer_engine() -> TransformerSimilarityEngine:
    """Get the default transformer similarity engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = TransformerSimilarityEngine()
    return _default_engine


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Convenience function for semantic similarity."""
    return get_transformer_engine().compute_similarity(text1, text2)
