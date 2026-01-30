"""
Streaming Robustness Score (SRS) computation engine.

This module provides the SRS computation engine for measuring
how well an STT model handles realistic streaming conditions
compared to ideal conditions.

SRS = Performance(Realtime) / Performance(Ideal)

A score close to 1.0 indicates the model handles streaming well.
A lower score indicates streaming-specific degradation.

Example:
    >>> from hsttb.metrics.srs import SRSEngine
    >>> engine = SRSEngine()
    >>> result = await engine.compute(adapter, audio_dir, gt_dir)
    >>> print(f"SRS: {result.srs:.2%}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from hsttb.core.types import SRSResult

if TYPE_CHECKING:
    from hsttb.adapters.base import STTAdapter


@dataclass
class SRSConfig:
    """
    Configuration for SRS computation.

    Attributes:
        ideal_profile: Profile for ideal conditions.
        realtime_profile: Profile for realistic conditions.
        metric_weights: Weights for combining metrics.
    """

    ideal_profile: str = "ideal"
    realtime_profile: str = "realtime_mobile"
    metric_weights: dict[str, float] = field(
        default_factory=lambda: {"ter": 0.4, "ner_f1": 0.3, "crs": 0.3}
    )


class SRSEngine:
    """
    Streaming Robustness Score computation engine.

    Computes SRS by running the same model under ideal and
    realistic streaming conditions and comparing the results.

    Attributes:
        config: Engine configuration.

    Example:
        >>> engine = SRSEngine()
        >>> result = await engine.compute(adapter, audio_dir, gt_dir)
        >>> print(f"SRS: {result.srs:.2%}")
        >>> print(f"TER degradation: {result.degradation['ter']:.2%}")
    """

    def __init__(self, config: SRSConfig | None = None) -> None:
        """
        Initialize the SRS engine.

        Args:
            config: Engine configuration.
        """
        self.config = config or SRSConfig()

    async def compute(
        self,
        adapter: STTAdapter,
        audio_dir: Path | str,
        ground_truth_dir: Path | str,
    ) -> SRSResult:
        """
        Compute Streaming Robustness Score.

        Runs evaluation under both ideal and realtime conditions.

        Args:
            adapter: STT adapter to evaluate.
            audio_dir: Directory containing audio files.
            ground_truth_dir: Directory containing ground truth.

        Returns:
            SRSResult with comparison data.
        """
        from hsttb.evaluation.runner import BenchmarkConfig, BenchmarkRunner

        audio_dir = Path(audio_dir)
        ground_truth_dir = Path(ground_truth_dir)

        # Run with ideal profile
        ideal_config = BenchmarkConfig(streaming_profile=self.config.ideal_profile)
        ideal_runner = BenchmarkRunner(adapter, ideal_config)
        ideal_summary = await ideal_runner.evaluate(audio_dir, ground_truth_dir)

        # Run with realtime profile
        realtime_config = BenchmarkConfig(streaming_profile=self.config.realtime_profile)
        realtime_runner = BenchmarkRunner(adapter, realtime_config)
        realtime_summary = await realtime_runner.evaluate(audio_dir, ground_truth_dir)

        # Extract scores
        ideal_scores = {
            "ter": ideal_summary.avg_ter,
            "ner_f1": ideal_summary.avg_ner_f1,
            "crs": ideal_summary.avg_crs,
        }

        realtime_scores = {
            "ter": realtime_summary.avg_ter,
            "ner_f1": realtime_summary.avg_ner_f1,
            "crs": realtime_summary.avg_crs,
        }

        # Compute degradation and SRS
        degradation = self._compute_degradation(ideal_scores, realtime_scores)
        srs = self._compute_srs(ideal_scores, realtime_scores)

        return SRSResult(
            model_name=adapter.name,
            ideal_scores=ideal_scores,
            realtime_scores=realtime_scores,
            srs=srs,
            degradation=degradation,
        )

    def compute_from_summaries(
        self,
        model_name: str,
        ideal_scores: dict[str, float],
        realtime_scores: dict[str, float],
    ) -> SRSResult:
        """
        Compute SRS from pre-computed scores.

        Args:
            model_name: Name of the model.
            ideal_scores: Scores under ideal conditions.
            realtime_scores: Scores under realtime conditions.

        Returns:
            SRSResult.
        """
        degradation = self._compute_degradation(ideal_scores, realtime_scores)
        srs = self._compute_srs(ideal_scores, realtime_scores)

        return SRSResult(
            model_name=model_name,
            ideal_scores=ideal_scores,
            realtime_scores=realtime_scores,
            srs=srs,
            degradation=degradation,
        )

    def _compute_degradation(
        self,
        ideal_scores: dict[str, float],
        realtime_scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute per-metric degradation.

        For error metrics (TER), higher is worse, so degradation is positive
        when realtime is higher than ideal.

        For accuracy metrics (NER F1, CRS), higher is better, so degradation
        is positive when realtime is lower than ideal.

        Args:
            ideal_scores: Scores under ideal conditions.
            realtime_scores: Scores under realtime conditions.

        Returns:
            Dictionary of degradation values.
        """
        degradation: dict[str, float] = {}

        for metric in ideal_scores:
            ideal = ideal_scores[metric]
            realtime = realtime_scores.get(metric, ideal)

            if metric == "ter":
                # TER: lower is better, so degradation = realtime - ideal
                degradation[metric] = realtime - ideal
            else:
                # NER F1, CRS: higher is better, so degradation = ideal - realtime
                degradation[metric] = ideal - realtime

        return degradation

    def _compute_srs(
        self,
        ideal_scores: dict[str, float],
        realtime_scores: dict[str, float],
    ) -> float:
        """
        Compute composite SRS.

        SRS measures how well performance is preserved under streaming.
        A score of 1.0 means no degradation.

        For error metrics (TER), we compute 1 - (realtime - ideal) / max(ideal, epsilon).
        For accuracy metrics, we compute realtime / max(ideal, epsilon).

        Args:
            ideal_scores: Scores under ideal conditions.
            realtime_scores: Scores under realtime conditions.

        Returns:
            Composite SRS (0.0-1.0, higher is better).
        """
        epsilon = 0.001  # Avoid division by zero
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in self.config.metric_weights.items():
            ideal = ideal_scores.get(metric, 0.0)
            realtime = realtime_scores.get(metric, ideal)

            if metric == "ter":
                # For TER: lower is better
                # If ideal=0 and realtime=0, ratio = 1.0 (perfect)
                # If ideal=0 and realtime>0, ratio = 0.0 (degraded)
                # Otherwise, ratio = max(0, 1 - (realtime - ideal))
                if ideal <= epsilon and realtime <= epsilon:
                    ratio = 1.0
                elif ideal <= epsilon:
                    ratio = 0.0
                else:
                    ratio = max(0.0, min(1.0, 1.0 - (realtime - ideal) / ideal))
            else:
                # For accuracy metrics: higher is better
                if ideal <= epsilon:
                    ratio = 1.0 if realtime <= epsilon else 0.0
                else:
                    ratio = min(1.0, realtime / ideal)

            weighted_sum += weight * ratio
            total_weight += weight

        if total_weight == 0:
            return 1.0

        return weighted_sum / total_weight


async def compute_srs(
    adapter: STTAdapter,
    audio_dir: Path | str,
    ground_truth_dir: Path | str,
    ideal_profile: str = "ideal",
    realtime_profile: str = "realtime_mobile",
) -> float:
    """
    Convenience function to compute SRS.

    Args:
        adapter: STT adapter to evaluate.
        audio_dir: Directory containing audio files.
        ground_truth_dir: Directory containing ground truth.
        ideal_profile: Profile for ideal conditions.
        realtime_profile: Profile for realistic conditions.

    Returns:
        SRS score (0.0-1.0).
    """
    config = SRSConfig(
        ideal_profile=ideal_profile,
        realtime_profile=realtime_profile,
    )
    engine = SRSEngine(config)
    result = await engine.compute(adapter, audio_dir, ground_truth_dir)
    return result.srs
