"""
Embedding drift detection for transcription quality.

Measures semantic stability across transcript segments using
sentence embeddings. Sudden drops in similarity indicate
potential transcription failures or context loss.

Example:
    >>> from hsttb.metrics.embedding_drift import EmbeddingDriftDetector
    >>> detector = EmbeddingDriftDetector()
    >>> result = detector.analyze("First sentence. Second sentence. Third sentence.")
    >>> print(f"Stability: {result.stability_score:.1%}")
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DriftPoint:
    """A detected semantic drift point."""

    segment_index: int
    from_segment: str
    to_segment: str
    similarity: float
    drop_magnitude: float  # How much similarity dropped
    is_anomaly: bool


@dataclass
class EmbeddingDriftResult:
    """Result of embedding drift analysis."""

    # Score (0.0-1.0, higher = more stable)
    stability_score: float

    # Segment similarities
    segment_similarities: list[float] = field(default_factory=list)

    # Drift points
    drift_points: list[DriftPoint] = field(default_factory=list)

    # Statistics
    mean_similarity: float = 0.0
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    similarity_variance: float = 0.0
    segment_count: int = 0

    # Flags
    has_anomalies: bool = False
    anomaly_count: int = 0

    @property
    def is_stable(self) -> bool:
        return self.stability_score >= 0.7 and not self.has_anomalies


class EmbeddingDriftDetector:
    """
    Detect semantic drift across transcript segments.

    Uses sentence embeddings to measure semantic similarity
    between consecutive segments. Large drops indicate
    potential transcription issues.
    """

    # Thresholds - adjusted for dialogue where topic changes are normal
    MIN_EXPECTED_SIMILARITY = 0.15  # Below this = anomaly (lowered for dialogue)
    DRIFT_THRESHOLD = 0.35  # Drop of this magnitude = drift point (raised for dialogue)

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        min_similarity: float = MIN_EXPECTED_SIMILARITY,
        drift_threshold: float = DRIFT_THRESHOLD,
    ) -> None:
        """
        Initialize drift detector.

        Args:
            model_name: Sentence transformer model name.
            min_similarity: Minimum expected similarity.
            drift_threshold: Similarity drop threshold for drift detection.
        """
        self.model_name = model_name
        self.min_similarity = min_similarity
        self.drift_threshold = drift_threshold
        self._model: Any = None

    def _ensure_model_loaded(self) -> bool:
        """Load embedding model if not already loaded."""
        if self._model is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            return True
        except ImportError:
            logger.warning("sentence-transformers not available for embedding drift")
            return False

    def analyze(self, text: str, segment_method: str = "sentence") -> EmbeddingDriftResult:
        """
        Analyze semantic drift across text segments.

        Args:
            text: Full transcript text.
            segment_method: How to segment text ("sentence" or "chunk").

        Returns:
            EmbeddingDriftResult with stability metrics.
        """
        # Split into segments
        segments = self._split_segments(text, segment_method)

        if len(segments) < 2:
            return EmbeddingDriftResult(
                stability_score=1.0,
                segment_count=len(segments),
                mean_similarity=1.0,
                min_similarity=1.0,
                max_similarity=1.0,
            )

        # Check if model is available
        if not self._ensure_model_loaded():
            # Fallback to simple word overlap similarity
            return self._analyze_fallback(segments)

        # Get embeddings
        try:
            embeddings = self._model.encode(segments)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return self._analyze_fallback(segments)

        # Calculate consecutive similarities
        similarities: list[float] = []
        drift_points: list[DriftPoint] = []

        for i in range(len(segments) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

            # Check for drift
            is_anomaly = sim < self.min_similarity
            drop_magnitude = 0.0

            if i > 0:
                drop_magnitude = similarities[i - 1] - sim

            if drop_magnitude > self.drift_threshold or is_anomaly:
                drift_points.append(DriftPoint(
                    segment_index=i + 1,
                    from_segment=segments[i][:50] + "..." if len(segments[i]) > 50 else segments[i],
                    to_segment=segments[i + 1][:50] + "..." if len(segments[i + 1]) > 50 else segments[i + 1],
                    similarity=sim,
                    drop_magnitude=drop_magnitude,
                    is_anomaly=is_anomaly,
                ))

        # Calculate statistics
        mean_sim = sum(similarities) / len(similarities) if similarities else 1.0
        min_sim = min(similarities) if similarities else 1.0
        max_sim = max(similarities) if similarities else 1.0

        variance = 0.0
        if len(similarities) > 1:
            variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)

        # Calculate stability score
        # Based on mean similarity and absence of anomalies
        # For dialogue, we expect some topic changes, so we're more lenient
        anomaly_count = sum(1 for dp in drift_points if dp.is_anomaly)

        # Scale mean similarity to 0-1 (typical similarities range 0.2-0.8)
        # A mean of 0.5 should give a score around 0.8
        base_score = min(1.0, mean_sim * 1.5 + 0.25)

        # Penalize anomalies less harshly (5% per anomaly, max 30% penalty)
        anomaly_penalty = min(0.3, 0.05 * anomaly_count)

        # Variance penalty (high variance is expected in dialogue)
        variance_penalty = min(0.1, variance * 0.5)

        stability_score = max(0.0, base_score - anomaly_penalty - variance_penalty)

        return EmbeddingDriftResult(
            stability_score=stability_score,
            segment_similarities=similarities,
            drift_points=drift_points,
            mean_similarity=mean_sim,
            min_similarity=min_sim,
            max_similarity=max_sim,
            similarity_variance=variance,
            segment_count=len(segments),
            has_anomalies=anomaly_count > 0,
            anomaly_count=anomaly_count,
        )

    def _split_segments(self, text: str, method: str) -> list[str]:
        """Split text into segments."""
        if method == "sentence":
            # Split on sentence boundaries
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        else:
            # Split into chunks of ~50 words
            words = text.split()
            chunk_size = 50
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk)
            return chunks

    def _cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _analyze_fallback(self, segments: list[str]) -> EmbeddingDriftResult:
        """Fallback analysis using word overlap."""
        if len(segments) < 2:
            return EmbeddingDriftResult(
                stability_score=1.0,
                segment_count=len(segments),
            )

        similarities: list[float] = []
        drift_points: list[DriftPoint] = []

        for i in range(len(segments) - 1):
            # Simple Jaccard similarity
            words1 = set(segments[i].lower().split())
            words2 = set(segments[i + 1].lower().split())

            if not words1 or not words2:
                sim = 0.0
            else:
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                sim = intersection / union if union > 0 else 0.0

            similarities.append(sim)

            # Check for anomalies (lower threshold for word overlap)
            is_anomaly = sim < 0.1

            if is_anomaly:
                drift_points.append(DriftPoint(
                    segment_index=i + 1,
                    from_segment=segments[i][:50] + "...",
                    to_segment=segments[i + 1][:50] + "...",
                    similarity=sim,
                    drop_magnitude=0.0,
                    is_anomaly=is_anomaly,
                ))

        mean_sim = sum(similarities) / len(similarities) if similarities else 1.0
        min_sim = min(similarities) if similarities else 1.0
        max_sim = max(similarities) if similarities else 1.0

        return EmbeddingDriftResult(
            stability_score=max(0.0, mean_sim * 2),  # Scale up since Jaccard is lower
            segment_similarities=similarities,
            drift_points=drift_points,
            mean_similarity=mean_sim,
            min_similarity=min_sim,
            max_similarity=max_sim,
            segment_count=len(segments),
            has_anomalies=len(drift_points) > 0,
            anomaly_count=len(drift_points),
        )


def analyze_embedding_drift(text: str) -> EmbeddingDriftResult:
    """
    Convenience function to analyze embedding drift.

    Args:
        text: Text to analyze.

    Returns:
        EmbeddingDriftResult.
    """
    detector = EmbeddingDriftDetector()
    return detector.analyze(text)
