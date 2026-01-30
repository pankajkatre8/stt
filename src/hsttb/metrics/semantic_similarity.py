"""
Semantic similarity computation for CRS.

This module provides semantic similarity computation between
text segments using embedding-based or token-based methods.

Supports:
- Embedding-based similarity (requires sentence-transformers)
- Token-based similarity (fallback, no dependencies)
- Segment-wise similarity computation

Example:
    >>> from hsttb.metrics.semantic_similarity import SemanticSimilarityEngine
    >>> engine = SemanticSimilarityEngine()
    >>> score = engine.compute_similarity("patient has diabetes", "patient has diabetes")
    >>> print(score)  # 1.0
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SimilarityConfig:
    """
    Configuration for semantic similarity computation.

    Attributes:
        model_name: Embedding model name (for embedding-based).
        use_embeddings: Whether to use embeddings (False = token-based).
        normalize_text: Whether to normalize text before comparison.
    """

    model_name: str = "all-MiniLM-L6-v2"
    use_embeddings: bool = False  # Default to token-based (no deps)
    normalize_text: bool = True


class SemanticSimilarityEngine(ABC):
    """
    Abstract base class for semantic similarity engines.

    Defines the interface for computing similarity between texts.
    """

    @abstractmethod
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score (0.0-1.0).
        """
        ...

    def compute_segment_similarities(
        self,
        gt_segments: list[str],
        pred_segments: list[str],
    ) -> list[float]:
        """
        Compute similarity for each segment pair.

        Args:
            gt_segments: Ground truth segments.
            pred_segments: Predicted segments.

        Returns:
            List of similarity scores.
        """
        if len(gt_segments) != len(pred_segments):
            # Pad shorter list with empty strings
            max_len = max(len(gt_segments), len(pred_segments))
            gt_segments = list(gt_segments) + [""] * (max_len - len(gt_segments))
            pred_segments = list(pred_segments) + [""] * (max_len - len(pred_segments))

        similarities = []
        for gt, pred in zip(gt_segments, pred_segments):
            similarities.append(self.compute_similarity(gt, pred))
        return similarities

    def compute_average_similarity(
        self,
        gt_segments: list[str],
        pred_segments: list[str],
    ) -> float:
        """
        Compute average similarity across all segments.

        Args:
            gt_segments: Ground truth segments.
            pred_segments: Predicted segments.

        Returns:
            Average similarity score.
        """
        similarities = self.compute_segment_similarities(gt_segments, pred_segments)
        if not similarities:
            return 1.0
        return sum(similarities) / len(similarities)


class TokenBasedSimilarity(SemanticSimilarityEngine):
    """
    Token-based semantic similarity using Jaccard/overlap measures.

    Does not require any external dependencies. Uses word overlap
    and n-gram similarity for comparison.

    Example:
        >>> engine = TokenBasedSimilarity()
        >>> engine.compute_similarity("cat sat mat", "cat on mat")
        0.5  # 2 common words out of 4 unique
    """

    def __init__(self, config: SimilarityConfig | None = None) -> None:
        """
        Initialize the engine.

        Args:
            config: Configuration options.
        """
        self.config = config or SimilarityConfig()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity using token overlap.

        Uses a combination of:
        - Word-level Jaccard similarity
        - Character n-gram similarity
        - Sequence matching

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score (0.0-1.0).
        """
        if self.config.normalize_text:
            text1 = text1.lower().strip()
            text2 = text2.lower().strip()

        # Handle empty strings
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        # Exact match
        if text1 == text2:
            return 1.0

        # Word-level Jaccard
        words1 = set(text1.split())
        words2 = set(text2.split())
        word_jaccard = self._jaccard(words1, words2)

        # Character 3-gram similarity
        ngrams1 = self._get_ngrams(text1, 3)
        ngrams2 = self._get_ngrams(text2, 3)
        ngram_jaccard = self._jaccard(ngrams1, ngrams2)

        # Sequence similarity (longest common subsequence ratio)
        lcs_ratio = self._lcs_ratio(text1, text2)

        # Weighted combination
        return 0.4 * word_jaccard + 0.3 * ngram_jaccard + 0.3 * lcs_ratio

    def _jaccard(self, set1: set[str], set2: set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _get_ngrams(self, text: str, n: int) -> set[str]:
        """Extract character n-grams from text."""
        if len(text) < n:
            return {text}
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    def _lcs_ratio(self, text1: str, text2: str) -> float:
        """
        Compute LCS length ratio.

        Returns ratio of LCS length to average text length.
        """
        lcs_len = self._lcs_length(text1, text2)
        avg_len = (len(text1) + len(text2)) / 2
        return lcs_len / avg_len if avg_len > 0 else 0.0

    def _lcs_length(self, text1: str, text2: str) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(text1), len(text2)
        if m == 0 or n == 0:
            return 0

        # Use space-optimized DP
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, prev

        return prev[n]


class EmbeddingBasedSimilarity(SemanticSimilarityEngine):
    """
    Embedding-based semantic similarity using sentence transformers.

    Requires sentence-transformers package to be installed.

    Example:
        >>> engine = EmbeddingBasedSimilarity()  # Requires sentence-transformers
        >>> engine.compute_similarity("patient has diabetes", "patient is diabetic")
        0.95  # High semantic similarity
    """

    def __init__(self, config: SimilarityConfig | None = None) -> None:
        """
        Initialize the engine.

        Args:
            config: Configuration options.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        self.config = config or SimilarityConfig()
        self._model = None

    def _load_model(self) -> None:
        """Load the embedding model lazily."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.config.model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for embedding-based similarity. "
                    "Install it with: pip install sentence-transformers"
                ) from e

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between text embeddings.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Cosine similarity score (0.0-1.0).
        """
        import numpy as np

        if self.config.normalize_text:
            text1 = text1.lower().strip()
            text2 = text2.lower().strip()

        # Handle empty strings
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        # Exact match
        if text1 == text2:
            return 1.0

        self._load_model()

        # Compute embeddings
        embeddings = self._model.encode([text1, text2])  # type: ignore[union-attr]

        # Cosine similarity
        similarity = float(
            np.dot(embeddings[0], embeddings[1])
            / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        )

        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))


def create_similarity_engine(
    config: SimilarityConfig | None = None,
) -> SemanticSimilarityEngine:
    """
    Factory function to create appropriate similarity engine.

    Args:
        config: Configuration options.

    Returns:
        Appropriate similarity engine based on config.

    Example:
        >>> engine = create_similarity_engine()
        >>> engine.compute_similarity("hello", "hi")
    """
    config = config or SimilarityConfig()

    if config.use_embeddings:
        return EmbeddingBasedSimilarity(config)
    return TokenBasedSimilarity(config)


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Convenience function to compute semantic similarity.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Similarity score (0.0-1.0).
    """
    engine = TokenBasedSimilarity()
    return engine.compute_similarity(text1, text2)
