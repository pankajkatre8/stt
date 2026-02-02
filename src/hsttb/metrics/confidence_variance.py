"""
Next-word probability variance for transcription quality.

Measures the stability of language model confidence across
the transcript. High variance or sudden drops indicate
potential transcription issues.

Example:
    >>> from hsttb.metrics.confidence_variance import ConfidenceAnalyzer
    >>> analyzer = ConfidenceAnalyzer()
    >>> result = analyzer.analyze("Patient takes metformin daily")
    >>> print(f"Confidence stability: {result.stability_score:.1%}")
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceDropPoint:
    """A detected confidence drop."""

    position: int
    token: str
    context: str
    log_prob: float
    drop_magnitude: float
    is_anomaly: bool


@dataclass
class ConfidenceVarianceResult:
    """Result of confidence variance analysis."""

    # Score (0.0-1.0, higher = more stable confidence)
    stability_score: float

    # Statistics
    mean_log_prob: float = 0.0
    variance: float = 0.0
    std_dev: float = 0.0
    min_log_prob: float = 0.0
    max_log_prob: float = 0.0

    # Token-level data
    token_log_probs: list[float] = field(default_factory=list)

    # Anomalies
    drop_points: list[ConfidenceDropPoint] = field(default_factory=list)
    anomaly_count: int = 0

    # Metadata
    token_count: int = 0
    model_available: bool = True

    @property
    def has_confidence_issues(self) -> bool:
        return self.anomaly_count > 0 or self.variance > 2.0


class ConfidenceAnalyzer:
    """
    Analyze next-word probability stability.

    Uses a language model to get token-level log probabilities
    and identifies confidence drops that may indicate
    transcription errors.
    """

    # Thresholds - adjusted to reduce false positives in dialogue
    MIN_LOG_PROB = -18.0  # Below this = very unlikely token (lowered for dialogue)
    DROP_THRESHOLD = 8.0  # Log prob drop of this magnitude = anomaly (raised for dialogue)

    def __init__(
        self,
        model_name: str = "gpt2",
        min_log_prob: float = MIN_LOG_PROB,
        drop_threshold: float = DROP_THRESHOLD,
    ) -> None:
        """
        Initialize analyzer.

        Args:
            model_name: Language model name.
            min_log_prob: Minimum expected log probability.
            drop_threshold: Log prob drop threshold for anomaly detection.
        """
        self.model_name = model_name
        self.min_log_prob = min_log_prob
        self.drop_threshold = drop_threshold
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_model_loaded(self) -> bool:
        """Load model if not already loaded."""
        if self._model is not None:
            return True

        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast

            logger.info(f"Loading confidence model: {self.model_name}")
            self._tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
            self._model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self._model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()

            return True
        except ImportError:
            logger.warning("transformers not available for confidence analysis")
            return False

    def analyze(self, text: str) -> ConfidenceVarianceResult:
        """
        Analyze confidence variance across text.

        Args:
            text: Text to analyze.

        Returns:
            ConfidenceVarianceResult with stability metrics.
        """
        if not text.strip():
            return ConfidenceVarianceResult(
                stability_score=1.0,
                model_available=True,
            )

        if not self._ensure_model_loaded():
            return self._analyze_fallback(text)

        try:
            return self._analyze_with_model(text)
        except Exception as e:
            logger.warning(f"Confidence analysis failed: {e}")
            return self._analyze_fallback(text)

    def _analyze_with_model(self, text: str) -> ConfidenceVarianceResult:
        """Analyze using language model."""
        import torch
        import torch.nn.functional as F

        # Tokenize
        encodings = self._tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids

        if self._model.device.type == "cuda":
            input_ids = input_ids.cuda()

        # Get log probabilities
        with torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits

        # Calculate token-level log probabilities
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Get log probs for actual tokens
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = []

        for i in range(shift_labels.shape[1]):
            token_id = shift_labels[0, i].item()
            log_prob = log_probs[0, i, token_id].item()
            token_log_probs.append(log_prob)

        # Get tokens for context
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        # Analyze log probabilities
        return self._analyze_log_probs(token_log_probs, tokens[1:], text)

    def _analyze_log_probs(
        self,
        log_probs: list[float],
        tokens: list[str],
        text: str,
    ) -> ConfidenceVarianceResult:
        """Analyze log probability sequence."""
        if not log_probs:
            return ConfidenceVarianceResult(
                stability_score=1.0,
                token_count=0,
            )

        # Basic statistics
        mean_lp = sum(log_probs) / len(log_probs)
        variance = sum((lp - mean_lp) ** 2 for lp in log_probs) / len(log_probs)
        std_dev = math.sqrt(variance)
        min_lp = min(log_probs)
        max_lp = max(log_probs)

        # Detect drop points
        drop_points: list[ConfidenceDropPoint] = []

        for i, lp in enumerate(log_probs):
            # Check for anomaly
            is_anomaly = lp < self.min_log_prob

            # Check for significant drop from previous
            drop_magnitude = 0.0
            if i > 0:
                drop_magnitude = log_probs[i - 1] - lp

            if is_anomaly or drop_magnitude > self.drop_threshold:
                # Get context
                start = max(0, i - 2)
                end = min(len(tokens), i + 3)
                context_tokens = tokens[start:end]
                context = "".join(context_tokens).replace("Ġ", " ").strip()

                token = tokens[i] if i < len(tokens) else "?"
                token = token.replace("Ġ", " ").strip()

                drop_points.append(ConfidenceDropPoint(
                    position=i,
                    token=token,
                    context=context,
                    log_prob=lp,
                    drop_magnitude=drop_magnitude,
                    is_anomaly=is_anomaly,
                ))

        # Calculate stability score
        # Higher mean log prob and lower variance = more stable
        # Normalize mean log prob (typical range: -12 to -2 for dialogue)
        # A mean of -5 should give about 0.7, -3 should give about 0.9
        normalized_mean = (mean_lp + 12) / 10  # Map [-12, -2] to [0, 1]
        normalized_mean = max(0.0, min(1.0, normalized_mean))

        # Penalize high variance less (dialogue has natural variance)
        variance_penalty = min(0.2, variance / 15)

        # Only penalize actual anomalies (tokens with is_anomaly=True)
        actual_anomalies = len([dp for dp in drop_points if dp.is_anomaly])
        anomaly_penalty = min(0.3, 0.1 * actual_anomalies)

        stability_score = max(0.0, normalized_mean - variance_penalty - anomaly_penalty)

        return ConfidenceVarianceResult(
            stability_score=stability_score,
            mean_log_prob=mean_lp,
            variance=variance,
            std_dev=std_dev,
            min_log_prob=min_lp,
            max_log_prob=max_lp,
            token_log_probs=log_probs,
            drop_points=drop_points,
            anomaly_count=len([dp for dp in drop_points if dp.is_anomaly]),
            token_count=len(log_probs),
            model_available=True,
        )

    def _analyze_fallback(self, text: str) -> ConfidenceVarianceResult:
        """Fallback analysis without model."""
        # Simple heuristic based on text characteristics
        words = text.split()
        word_count = len(words)

        if word_count == 0:
            return ConfidenceVarianceResult(
                stability_score=1.0,
                model_available=False,
            )

        # Check for potential issues
        issues = 0

        # Repeated words
        for i in range(1, len(words)):
            if words[i].lower() == words[i - 1].lower():
                issues += 1

        # Very short words in sequence (potential garbage)
        short_word_streak = 0
        max_streak = 0
        for word in words:
            if len(word) <= 2:
                short_word_streak += 1
                max_streak = max(max_streak, short_word_streak)
            else:
                short_word_streak = 0

        if max_streak >= 3:
            issues += max_streak - 2

        # Calculate stability score
        issue_rate = issues / word_count if word_count > 0 else 0
        stability_score = max(0.0, 1.0 - issue_rate * 2)

        return ConfidenceVarianceResult(
            stability_score=stability_score,
            token_count=word_count,
            model_available=False,
        )


def analyze_confidence_variance(text: str) -> ConfidenceVarianceResult:
    """
    Convenience function to analyze confidence variance.

    Args:
        text: Text to analyze.

    Returns:
        ConfidenceVarianceResult.
    """
    analyzer = ConfidenceAnalyzer()
    return analyzer.analyze(text)
