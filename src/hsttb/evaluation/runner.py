"""
Benchmark runner for STT evaluation.

This module provides the main evaluation orchestrator that coordinates
audio loading, STT transcription, and metric computation.

Example:
    >>> from hsttb.evaluation.runner import BenchmarkRunner
    >>> from hsttb.adapters import MockSTTAdapter
    >>> adapter = MockSTTAdapter(transcripts={"test": "hello world"})
    >>> runner = BenchmarkRunner(adapter)
    >>> result = await runner.evaluate_file(audio_path, ground_truth)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from hsttb.core.config import get_builtin_profile
from hsttb.core.types import BenchmarkResult, BenchmarkSummary

if TYPE_CHECKING:
    from hsttb.adapters.base import STTAdapter
    from hsttb.lexicons.base import MedicalLexicon
    from hsttb.nlp.ner_pipeline import NERPipeline


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark runner.

    Attributes:
        streaming_profile: Name of streaming profile to use.
        compute_ter: Whether to compute TER.
        compute_ner: Whether to compute NER accuracy.
        compute_crs: Whether to compute CRS.
        parallel_files: Number of files to process in parallel.
        continue_on_error: Whether to continue if a file fails.
    """

    streaming_profile: str = "ideal"
    compute_ter: bool = True
    compute_ner: bool = True
    compute_crs: bool = True
    parallel_files: int = 1
    continue_on_error: bool = True


@dataclass
class EvaluationResult:
    """
    Result of evaluating a single audio file.

    Attributes:
        audio_id: Identifier for the audio file.
        ground_truth: Ground truth transcript.
        prediction: Predicted transcript.
        segments: List of predicted segments.
        ter_score: TER score (if computed).
        ner_f1: NER F1 score (if computed).
        crs_score: CRS score (if computed).
        metadata: Additional metadata.
    """

    audio_id: str
    ground_truth: str
    prediction: str
    segments: list[str] = field(default_factory=list)
    ter_score: float | None = None
    ner_f1: float | None = None
    crs_score: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Whether evaluation completed successfully."""
        return self.prediction is not None


class BenchmarkRunner:
    """
    Main benchmark orchestrator.

    Coordinates the full evaluation pipeline:
    1. Load audio files
    2. Stream audio to STT adapter
    3. Collect transcription results
    4. Compute metrics (TER, NER, CRS)
    5. Generate summary statistics

    Attributes:
        adapter: STT adapter for transcription.
        config: Benchmark configuration.
        profile: Streaming profile for simulation.

    Example:
        >>> adapter = MockSTTAdapter()
        >>> runner = BenchmarkRunner(adapter)
        >>> summary = await runner.evaluate(audio_dir, gt_dir)
        >>> print(f"Average TER: {summary.avg_ter:.2%}")
    """

    def __init__(
        self,
        adapter: STTAdapter,
        config: BenchmarkConfig | None = None,
        lexicon: MedicalLexicon | None = None,
        ner_pipeline: NERPipeline | None = None,
    ) -> None:
        """
        Initialize the benchmark runner.

        Args:
            adapter: STT adapter for transcription.
            config: Benchmark configuration.
            lexicon: Medical lexicon for TER (uses mock if None).
            ner_pipeline: NER pipeline (uses mock if None).
        """
        self.adapter = adapter
        self.config = config or BenchmarkConfig()

        # Load streaming profile
        self.profile = get_builtin_profile(self.config.streaming_profile)

        # Initialize metric engines lazily
        self._ter_engine = None
        self._ner_engine = None
        self._crs_engine = None
        self._lexicon = lexicon
        self._ner_pipeline = ner_pipeline

    @property
    def ter_engine(self) -> object:
        """Get or create TER engine."""
        if self._ter_engine is None:
            from hsttb.lexicons import MockMedicalLexicon
            from hsttb.metrics.ter import TEREngine

            lexicon = self._lexicon or MockMedicalLexicon.with_common_terms()
            self._ter_engine = TEREngine(lexicon)
        return self._ter_engine

    @property
    def ner_engine(self) -> object:
        """Get or create NER engine."""
        if self._ner_engine is None:
            from hsttb.metrics.ner import NEREngine
            from hsttb.nlp import MockNERPipeline

            pipeline = self._ner_pipeline or MockNERPipeline.with_common_patterns()
            self._ner_engine = NEREngine(pipeline)
        return self._ner_engine

    @property
    def crs_engine(self) -> object:
        """Get or create CRS engine."""
        if self._crs_engine is None:
            from hsttb.metrics.crs import CRSEngine

            self._crs_engine = CRSEngine()
        return self._crs_engine

    async def evaluate(
        self,
        audio_dir: Path | str,
        ground_truth_dir: Path | str,
        file_extension: str = ".wav",
    ) -> BenchmarkSummary:
        """
        Run benchmark on all audio files in directory.

        Args:
            audio_dir: Directory containing audio files.
            ground_truth_dir: Directory containing ground truth files.
            file_extension: Audio file extension to look for.

        Returns:
            BenchmarkSummary with aggregate results.
        """
        audio_dir = Path(audio_dir)
        ground_truth_dir = Path(ground_truth_dir)

        # Find audio files
        audio_files = list(audio_dir.glob(f"*{file_extension}"))

        if not audio_files:
            return BenchmarkSummary(
                total_files=0,
                avg_ter=0.0,
                avg_ner_f1=0.0,
                avg_crs=0.0,
                results=[],
                streaming_profile=self.config.streaming_profile,
                adapter_name=self.adapter.name,
            )

        # Initialize adapter
        await self.adapter.initialize()

        try:
            # Evaluate files
            results: list[BenchmarkResult] = []

            for audio_file in audio_files:
                try:
                    gt_file = ground_truth_dir / f"{audio_file.stem}.txt"
                    if not gt_file.exists():
                        continue

                    ground_truth = gt_file.read_text().strip()
                    result = await self.evaluate_file(audio_file, ground_truth)

                    # Convert EvaluationResult to BenchmarkResult
                    benchmark_result = self._to_benchmark_result(result)
                    results.append(benchmark_result)

                except Exception as e:
                    if not self.config.continue_on_error:
                        raise
                    # Log error but continue
                    print(f"Error processing {audio_file}: {e}")

            return self._create_summary(results)

        finally:
            await self.adapter.cleanup()

    async def evaluate_file(
        self,
        audio_path: Path | str,
        ground_truth: str,
    ) -> EvaluationResult:
        """
        Evaluate a single audio file.

        Args:
            audio_path: Path to audio file.
            ground_truth: Ground truth transcript.

        Returns:
            EvaluationResult with metrics.
        """
        audio_path = Path(audio_path)
        audio_id = audio_path.stem

        # Load and stream audio
        prediction, segments = await self._transcribe_file(audio_path)

        # Compute metrics
        ter_score = None
        ner_f1 = None
        crs_score = None

        if self.config.compute_ter:
            ter_result = self.ter_engine.compute(ground_truth, prediction)  # type: ignore[union-attr]
            ter_score = ter_result.overall_ter

        if self.config.compute_ner:
            ner_result = self.ner_engine.compute(ground_truth, prediction)  # type: ignore[union-attr]
            ner_f1 = ner_result.f1_score

        if self.config.compute_crs:
            # Segment ground truth for CRS
            gt_segments = self._segment_text(ground_truth, len(segments))
            crs_result = self.crs_engine.compute(gt_segments, segments)  # type: ignore[union-attr]
            crs_score = crs_result.composite_score

        return EvaluationResult(
            audio_id=audio_id,
            ground_truth=ground_truth,
            prediction=prediction,
            segments=segments,
            ter_score=ter_score,
            ner_f1=ner_f1,
            crs_score=crs_score,
            metadata={
                "streaming_profile": self.config.streaming_profile,
                "adapter_name": self.adapter.name,
            },
        )

    async def _transcribe_file(
        self,
        audio_path: Path,
    ) -> tuple[str, list[str]]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            Tuple of (full_transcript, segments).
        """
        from hsttb.audio.chunker import StreamingChunker
        from hsttb.audio.loader import AudioLoader

        # Load audio
        loader = AudioLoader()
        audio_data, sample_rate = loader.load(audio_path)

        # Create chunker
        chunker = StreamingChunker(self.profile)

        # Stream and transcribe
        segments: list[str] = []
        audio_stream = chunker.stream_audio(audio_data, sample_rate)

        async for segment in self.adapter.transcribe_stream(audio_stream):
            if segment.is_final and segment.text.strip():
                segments.append(segment.text.strip())

        prediction = " ".join(segments)
        return prediction, segments

    def _segment_text(self, text: str, num_segments: int) -> list[str]:
        """
        Split text into segments for CRS comparison.

        Args:
            text: Text to segment.
            num_segments: Target number of segments.

        Returns:
            List of text segments.
        """
        if num_segments <= 0:
            return [text]

        if num_segments == 1:
            return [text]

        # Split by sentences first
        sentences = self._split_sentences(text)

        if len(sentences) <= num_segments:
            # Pad with empty strings if needed
            return sentences + [""] * (num_segments - len(sentences))

        # Combine sentences into num_segments groups
        segments: list[str] = []
        sentences_per_segment = len(sentences) // num_segments

        for i in range(num_segments):
            start = i * sentences_per_segment
            if i == num_segments - 1:
                # Last segment gets remaining sentences
                segment_sentences = sentences[start:]
            else:
                segment_sentences = sentences[start : start + sentences_per_segment]
            segments.append(" ".join(segment_sentences))

        return segments

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        # Simple sentence splitting
        sentences = re.split(r"[.!?]+\s*", text)
        return [s.strip() for s in sentences if s.strip()]

    def _to_benchmark_result(self, eval_result: EvaluationResult) -> BenchmarkResult:
        """Convert EvaluationResult to BenchmarkResult."""
        from hsttb.core.types import CRSResult, NERResult, TERResult

        # Create placeholder results
        ter_result = TERResult(
            overall_ter=eval_result.ter_score or 0.0,
            category_ter={},
            total_terms=0,
        )

        ner_result = NERResult(
            precision=0.0,
            recall=0.0,
            f1_score=eval_result.ner_f1 or 0.0,
            entity_distortion_rate=0.0,
            entity_omission_rate=0.0,
        )

        crs_result = CRSResult(
            composite_score=eval_result.crs_score or 0.0,
            semantic_similarity=0.0,
            entity_continuity=0.0,
            negation_consistency=0.0,
            context_drift_rate=0.0,
        )

        return BenchmarkResult(
            audio_id=eval_result.audio_id,
            ter=ter_result,
            ner=ner_result,
            crs=crs_result,
            transcript_ground_truth=eval_result.ground_truth,
            transcript_predicted=eval_result.prediction,
            streaming_profile=self.config.streaming_profile,
            adapter_name=self.adapter.name,
        )

    def _create_summary(self, results: list[BenchmarkResult]) -> BenchmarkSummary:
        """Create summary from results."""
        if not results:
            return BenchmarkSummary(
                total_files=0,
                avg_ter=0.0,
                avg_ner_f1=0.0,
                avg_crs=0.0,
                results=[],
                streaming_profile=self.config.streaming_profile,
                adapter_name=self.adapter.name,
            )

        avg_ter = sum(r.ter.overall_ter for r in results) / len(results)
        avg_ner_f1 = sum(r.ner.f1_score for r in results) / len(results)
        avg_crs = sum(r.crs.composite_score for r in results) / len(results)

        return BenchmarkSummary(
            total_files=len(results),
            avg_ter=avg_ter,
            avg_ner_f1=avg_ner_f1,
            avg_crs=avg_crs,
            results=results,
            streaming_profile=self.config.streaming_profile,
            adapter_name=self.adapter.name,
        )


def create_benchmark_runner(
    adapter: STTAdapter,
    profile: str = "ideal",
    **kwargs: object,
) -> BenchmarkRunner:
    """
    Factory function to create benchmark runner.

    Args:
        adapter: STT adapter.
        profile: Streaming profile name.
        **kwargs: Additional config options.

    Returns:
        Configured BenchmarkRunner.
    """
    config = BenchmarkConfig(streaming_profile=profile, **kwargs)  # type: ignore[arg-type]
    return BenchmarkRunner(adapter, config)
