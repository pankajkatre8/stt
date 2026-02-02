"""
Perplexity scoring for transcription fluency evaluation.

Uses language models (GPT-2 by default) to measure how "natural"
a transcription sounds. Lower perplexity indicates more fluent text.

Example:
    >>> from hsttb.metrics.perplexity import PerplexityScorer
    >>> scorer = PerplexityScorer()
    >>> result = scorer.compute("Patient takes metformin daily")
    >>> print(f"Perplexity: {result.perplexity:.2f}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class PerplexityResult:
    """Result of perplexity computation."""

    text: str
    perplexity: float  # Raw perplexity (lower is better)
    log_probability: float  # Log probability of sentence
    normalized_score: float  # 0-1 score (higher is better)
    token_count: int
    model_name: str


class PerplexityScorer:
    """
    Compute perplexity using language models.

    Supports GPT-2 (default) and other causal LMs from Hugging Face.
    Lazy-loads the model on first use to avoid startup overhead.

    Attributes:
        model_name: Name of the Hugging Face model to use.
        max_perplexity: Maximum perplexity for normalization.

    Example:
        >>> scorer = PerplexityScorer("gpt2")
        >>> result = scorer.compute("Patient has diabetes")
        >>> print(f"Score: {result.normalized_score:.1%}")
    """

    # Singleton instances per model
    _instances: dict[str, PerplexityScorer] = {}

    def __new__(cls, model_name: str = "gpt2") -> PerplexityScorer:
        """Return singleton instance per model."""
        if model_name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[model_name] = instance
        return cls._instances[model_name]

    def __init__(
        self,
        model_name: str = "gpt2",
        max_perplexity: float = 500.0,
        device: str | None = None,
    ) -> None:
        """
        Initialize perplexity scorer.

        Args:
            model_name: Hugging Face model name (e.g., "gpt2", "gpt2-medium").
            max_perplexity: Maximum perplexity for score normalization.
            device: Device to use ("cuda", "cpu", or None for auto).
        """
        if hasattr(self, "_initialized"):
            return

        self.model_name = model_name
        self.max_perplexity = max_perplexity
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._initialized = True

    def _ensure_loaded(self) -> None:
        """Load model and tokenizer if not already loaded."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast

            logger.info(f"Loading perplexity model: {self.model_name}")

            self._tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
            self._model = GPT2LMHeadModel.from_pretrained(self.model_name)

            # Determine device
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model.to(self._device)
            self._model.eval()

            logger.info(f"Perplexity model loaded on {self._device}")

        except ImportError as e:
            raise ImportError(
                "transformers and torch required for perplexity scoring. "
                "Install with: pip install transformers torch"
            ) from e

    def compute(self, text: str) -> PerplexityResult:
        """
        Compute perplexity for a single text.

        Args:
            text: Text to evaluate.

        Returns:
            PerplexityResult with perplexity and normalized score.
        """
        self._ensure_loaded()

        import torch

        # Tokenize
        encodings = self._tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self._device)

        # Handle empty or very short text
        if input_ids.shape[1] == 0:
            return PerplexityResult(
                text=text,
                perplexity=float("inf"),
                log_probability=float("-inf"),
                normalized_score=0.0,
                token_count=0,
                model_name=self.model_name,
            )

        # Compute loss (negative log likelihood)
        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            loss = outputs.loss

        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()

        # Log probability
        log_prob = -loss.item() * input_ids.shape[1]

        # Normalize to 0-1 score (higher is better)
        # Use sigmoid-like transformation
        normalized = 1.0 / (1.0 + perplexity / self.max_perplexity)

        return PerplexityResult(
            text=text,
            perplexity=perplexity,
            log_probability=log_prob,
            normalized_score=min(1.0, max(0.0, normalized)),
            token_count=input_ids.shape[1],
            model_name=self.model_name,
        )

    def compute_batch(self, texts: list[str]) -> list[PerplexityResult]:
        """
        Compute perplexity for multiple texts.

        Args:
            texts: List of texts to evaluate.

        Returns:
            List of PerplexityResult objects.
        """
        return [self.compute(text) for text in texts]

    @property
    def is_available(self) -> bool:
        """Check if perplexity scoring is available."""
        try:
            import torch  # noqa: F401
            from transformers import GPT2LMHeadModel  # noqa: F401

            return True
        except ImportError:
            return False


# Convenience function
def compute_perplexity(text: str, model: str = "gpt2") -> PerplexityResult:
    """
    Convenience function to compute perplexity.

    Args:
        text: Text to evaluate.
        model: Model name.

    Returns:
        PerplexityResult.
    """
    scorer = PerplexityScorer(model)
    return scorer.compute(text)
