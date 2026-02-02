"""
Speech rate validation for transcription quality.

Validates that the transcript word count is plausible given
the audio duration. Catches hallucinations and missing content.

Normal speech rates:
- Slow: 100-120 words/minute
- Normal: 120-150 words/minute
- Fast: 150-180 words/minute
- Very fast: 180+ words/minute

Example:
    >>> from hsttb.metrics.speech_rate import SpeechRateValidator
    >>> validator = SpeechRateValidator()
    >>> result = validator.validate(text="...", audio_duration_seconds=60)
    >>> print(f"Speech rate: {result.words_per_minute:.0f} WPM")
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SpeechRateCategory(Enum):
    """Speech rate classification."""

    IMPLAUSIBLY_LOW = "implausibly_low"  # < 50 WPM - missing content
    SLOW = "slow"                         # 50-100 WPM
    NORMAL = "normal"                     # 100-180 WPM
    FAST = "fast"                         # 180-220 WPM
    IMPLAUSIBLY_HIGH = "implausibly_high" # > 220 WPM - likely hallucination


@dataclass
class SpeechRateResult:
    """Result of speech rate validation."""

    # Score (0.0-1.0, higher = more plausible rate)
    plausibility_score: float

    # Rate metrics
    words_per_minute: float
    word_count: int
    audio_duration_seconds: float

    # Classification
    category: SpeechRateCategory

    # Flags
    is_plausible: bool
    warning: str | None = None

    @property
    def is_too_fast(self) -> bool:
        return self.category == SpeechRateCategory.IMPLAUSIBLY_HIGH

    @property
    def is_too_slow(self) -> bool:
        return self.category == SpeechRateCategory.IMPLAUSIBLY_LOW


class SpeechRateValidator:
    """
    Validate transcript length against audio duration.

    Uses expected speech rates to detect:
    - Hallucinated content (too many words for audio duration)
    - Missing content (too few words for audio duration)
    """

    # Speech rate thresholds (words per minute)
    MIN_PLAUSIBLE_WPM = 50   # Below this = likely missing content
    SLOW_WPM = 100
    NORMAL_MIN_WPM = 100
    NORMAL_MAX_WPM = 180
    FAST_WPM = 220
    MAX_PLAUSIBLE_WPM = 250  # Above this = likely hallucination

    # Optimal range for scoring
    OPTIMAL_MIN_WPM = 110
    OPTIMAL_MAX_WPM = 160

    def __init__(
        self,
        min_plausible_wpm: float = MIN_PLAUSIBLE_WPM,
        max_plausible_wpm: float = MAX_PLAUSIBLE_WPM,
    ) -> None:
        """
        Initialize validator.

        Args:
            min_plausible_wpm: Minimum plausible speech rate.
            max_plausible_wpm: Maximum plausible speech rate.
        """
        self.min_plausible_wpm = min_plausible_wpm
        self.max_plausible_wpm = max_plausible_wpm

    def validate(
        self,
        text: str,
        audio_duration_seconds: float,
    ) -> SpeechRateResult:
        """
        Validate speech rate.

        Args:
            text: Transcript text.
            audio_duration_seconds: Audio duration in seconds.

        Returns:
            SpeechRateResult with plausibility assessment.
        """
        # Count words
        word_count = self._count_words(text)

        # Handle edge cases
        if audio_duration_seconds <= 0:
            return SpeechRateResult(
                plausibility_score=0.0,
                words_per_minute=0.0,
                word_count=word_count,
                audio_duration_seconds=audio_duration_seconds,
                category=SpeechRateCategory.IMPLAUSIBLY_LOW,
                is_plausible=False,
                warning="Invalid audio duration",
            )

        if word_count == 0:
            return SpeechRateResult(
                plausibility_score=0.5,  # Might be silence
                words_per_minute=0.0,
                word_count=0,
                audio_duration_seconds=audio_duration_seconds,
                category=SpeechRateCategory.IMPLAUSIBLY_LOW,
                is_plausible=audio_duration_seconds < 5,  # OK for very short audio
                warning="No words in transcript" if audio_duration_seconds >= 5 else None,
            )

        # Calculate WPM
        duration_minutes = audio_duration_seconds / 60.0
        wpm = word_count / duration_minutes

        # Classify
        category = self._classify_rate(wpm)

        # Calculate plausibility score
        plausibility_score = self._calculate_score(wpm)

        # Determine if plausible
        is_plausible = self.min_plausible_wpm <= wpm <= self.max_plausible_wpm

        # Generate warning
        warning = None
        if wpm < self.min_plausible_wpm:
            expected_min = int(self.min_plausible_wpm * duration_minutes)
            warning = f"Too few words ({word_count}) for {audio_duration_seconds:.0f}s audio. Expected at least {expected_min} words."
        elif wpm > self.max_plausible_wpm:
            expected_max = int(self.max_plausible_wpm * duration_minutes)
            warning = f"Too many words ({word_count}) for {audio_duration_seconds:.0f}s audio. Expected at most {expected_max} words. Possible hallucination."

        return SpeechRateResult(
            plausibility_score=plausibility_score,
            words_per_minute=wpm,
            word_count=word_count,
            audio_duration_seconds=audio_duration_seconds,
            category=category,
            is_plausible=is_plausible,
            warning=warning,
        )

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        # Split on whitespace and filter empty
        words = text.split()
        return len(words)

    def _classify_rate(self, wpm: float) -> SpeechRateCategory:
        """Classify speech rate."""
        if wpm < self.min_plausible_wpm:
            return SpeechRateCategory.IMPLAUSIBLY_LOW
        elif wpm < self.SLOW_WPM:
            return SpeechRateCategory.SLOW
        elif wpm <= self.NORMAL_MAX_WPM:
            return SpeechRateCategory.NORMAL
        elif wpm <= self.max_plausible_wpm:
            return SpeechRateCategory.FAST
        else:
            return SpeechRateCategory.IMPLAUSIBLY_HIGH

    def _calculate_score(self, wpm: float) -> float:
        """
        Calculate plausibility score (0-1).

        Score is highest in optimal range and decreases as
        rate moves away from optimal.
        """
        if wpm <= 0:
            return 0.0

        # Perfect score in optimal range
        if self.OPTIMAL_MIN_WPM <= wpm <= self.OPTIMAL_MAX_WPM:
            return 1.0

        # Below optimal
        if wpm < self.OPTIMAL_MIN_WPM:
            if wpm < self.min_plausible_wpm:
                # Scale from 0 to 0.5 as wpm goes from 0 to min_plausible
                return 0.5 * (wpm / self.min_plausible_wpm)
            else:
                # Scale from 0.5 to 1.0 as wpm goes from min_plausible to optimal_min
                range_size = self.OPTIMAL_MIN_WPM - self.min_plausible_wpm
                position = wpm - self.min_plausible_wpm
                return 0.5 + 0.5 * (position / range_size)

        # Above optimal
        if wpm > self.OPTIMAL_MAX_WPM:
            if wpm > self.max_plausible_wpm:
                # Scale from 0.5 down to 0 as wpm goes beyond max_plausible
                excess = wpm - self.max_plausible_wpm
                return max(0.0, 0.5 - 0.5 * (excess / 100))
            else:
                # Scale from 1.0 to 0.5 as wpm goes from optimal_max to max_plausible
                range_size = self.max_plausible_wpm - self.OPTIMAL_MAX_WPM
                position = wpm - self.OPTIMAL_MAX_WPM
                return 1.0 - 0.5 * (position / range_size)

        return 0.5


def validate_speech_rate(
    text: str,
    audio_duration_seconds: float,
) -> SpeechRateResult:
    """
    Convenience function to validate speech rate.

    Args:
        text: Transcript text.
        audio_duration_seconds: Audio duration in seconds.

    Returns:
        SpeechRateResult.
    """
    validator = SpeechRateValidator()
    return validator.validate(text, audio_duration_seconds)
