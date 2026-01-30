"""
Tests for evaluation orchestration module.

Tests the benchmark runner and SRS computation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hsttb.adapters import MockSTTAdapter
from hsttb.evaluation import BenchmarkConfig, BenchmarkRunner, EvaluationResult
from hsttb.metrics import SRSConfig, SRSEngine

# ==============================================================================
# Benchmark Config Tests
# ==============================================================================


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_config(self) -> None:
        """Default config has reasonable values."""
        config = BenchmarkConfig()

        assert config.streaming_profile == "ideal"
        assert config.compute_ter is True
        assert config.compute_ner is True
        assert config.compute_crs is True
        assert config.continue_on_error is True

    def test_custom_config(self) -> None:
        """Config accepts custom values."""
        config = BenchmarkConfig(
            streaming_profile="realtime_mobile",
            compute_ter=False,
            parallel_files=4,
        )

        assert config.streaming_profile == "realtime_mobile"
        assert config.compute_ter is False
        assert config.parallel_files == 4


# ==============================================================================
# Evaluation Result Tests
# ==============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult."""

    def test_successful_result(self) -> None:
        """Result with prediction is successful."""
        result = EvaluationResult(
            audio_id="test",
            ground_truth="hello world",
            prediction="hello world",
            ter_score=0.0,
        )

        assert result.is_successful is True

    def test_result_with_scores(self) -> None:
        """Result stores scores correctly."""
        result = EvaluationResult(
            audio_id="test",
            ground_truth="hello",
            prediction="hello",
            ter_score=0.1,
            ner_f1=0.9,
            crs_score=0.95,
        )

        assert result.ter_score == 0.1
        assert result.ner_f1 == 0.9
        assert result.crs_score == 0.95


# ==============================================================================
# Benchmark Runner Tests
# ==============================================================================


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    @pytest.fixture
    def mock_adapter(self) -> MockSTTAdapter:
        """Create mock adapter."""
        return MockSTTAdapter(
            responses=["hello world", "transcribed text"],
        )

    @pytest.fixture
    def runner(self, mock_adapter: MockSTTAdapter) -> BenchmarkRunner:
        """Create benchmark runner."""
        config = BenchmarkConfig()
        return BenchmarkRunner(mock_adapter, config)

    def test_runner_creation(self, runner: BenchmarkRunner) -> None:
        """Runner is created with correct config."""
        assert runner.adapter is not None
        assert runner.config.streaming_profile == "ideal"

    def test_segment_text_single(self, runner: BenchmarkRunner) -> None:
        """Segment text into single segment."""
        segments = runner._segment_text("hello world", 1)
        assert len(segments) == 1
        assert segments[0] == "hello world"

    def test_segment_text_multiple(self, runner: BenchmarkRunner) -> None:
        """Segment text into multiple segments."""
        text = "First sentence. Second sentence. Third sentence."
        segments = runner._segment_text(text, 3)
        assert len(segments) == 3

    def test_split_sentences(self, runner: BenchmarkRunner) -> None:
        """Split text into sentences."""
        text = "Hello world. How are you? I am fine!"
        sentences = runner._split_sentences(text)
        assert len(sentences) == 3

    def test_ter_engine_lazy_init(self, runner: BenchmarkRunner) -> None:
        """TER engine is lazily initialized."""
        assert runner._ter_engine is None
        _ = runner.ter_engine
        assert runner._ter_engine is not None

    def test_ner_engine_lazy_init(self, runner: BenchmarkRunner) -> None:
        """NER engine is lazily initialized."""
        assert runner._ner_engine is None
        _ = runner.ner_engine
        assert runner._ner_engine is not None

    def test_crs_engine_lazy_init(self, runner: BenchmarkRunner) -> None:
        """CRS engine is lazily initialized."""
        assert runner._crs_engine is None
        _ = runner.crs_engine
        assert runner._crs_engine is not None


class TestBenchmarkRunnerEvaluate:
    """Tests for benchmark runner evaluation."""

    @pytest.fixture
    def mock_adapter(self) -> MockSTTAdapter:
        """Create mock adapter with fixed transcripts."""
        return MockSTTAdapter(responses=["patient takes metformin for diabetes"])

    @pytest.fixture
    def runner(self, mock_adapter: MockSTTAdapter) -> BenchmarkRunner:
        """Create runner with mock adapter."""
        return BenchmarkRunner(mock_adapter)

    @pytest.mark.asyncio
    async def test_evaluate_empty_dir(
        self, runner: BenchmarkRunner, tmp_path: Path
    ) -> None:
        """Evaluate returns empty summary for empty directory."""
        audio_dir = tmp_path / "audio"
        gt_dir = tmp_path / "ground_truth"
        audio_dir.mkdir()
        gt_dir.mkdir()

        summary = await runner.evaluate(audio_dir, gt_dir)

        assert summary.total_files == 0
        assert summary.avg_ter == 0.0

    @pytest.mark.asyncio
    async def test_create_summary_empty(self, runner: BenchmarkRunner) -> None:
        """Create summary from empty results."""
        summary = runner._create_summary([])

        assert summary.total_files == 0
        assert summary.avg_ter == 0.0
        assert summary.avg_ner_f1 == 0.0
        assert summary.avg_crs == 0.0


# ==============================================================================
# SRS Engine Tests
# ==============================================================================


class TestSRSConfig:
    """Tests for SRSConfig."""

    def test_default_config(self) -> None:
        """Default config has reasonable values."""
        config = SRSConfig()

        assert config.ideal_profile == "ideal"
        assert config.realtime_profile == "realtime_mobile"
        assert "ter" in config.metric_weights

    def test_custom_config(self) -> None:
        """Config accepts custom values."""
        config = SRSConfig(
            ideal_profile="custom_ideal",
            realtime_profile="custom_realtime",
        )

        assert config.ideal_profile == "custom_ideal"
        assert config.realtime_profile == "custom_realtime"


class TestSRSEngine:
    """Tests for SRSEngine."""

    @pytest.fixture
    def engine(self) -> SRSEngine:
        """Create SRS engine."""
        return SRSEngine()

    def test_compute_from_summaries_perfect(self, engine: SRSEngine) -> None:
        """Perfect scores under both conditions gives SRS = 1.0."""
        result = engine.compute_from_summaries(
            model_name="test_model",
            ideal_scores={"ter": 0.0, "ner_f1": 1.0, "crs": 1.0},
            realtime_scores={"ter": 0.0, "ner_f1": 1.0, "crs": 1.0},
        )

        assert result.srs == 1.0
        assert result.degradation["ter"] == 0.0
        assert result.degradation["ner_f1"] == 0.0

    def test_compute_from_summaries_degraded(self, engine: SRSEngine) -> None:
        """Degraded realtime scores give SRS < 1.0."""
        result = engine.compute_from_summaries(
            model_name="test_model",
            ideal_scores={"ter": 0.1, "ner_f1": 0.9, "crs": 0.9},
            realtime_scores={"ter": 0.2, "ner_f1": 0.8, "crs": 0.8},
        )

        assert result.srs < 1.0
        assert result.degradation["ter"] > 0  # TER increased (worse)
        assert result.degradation["ner_f1"] > 0  # NER decreased (worse)

    def test_degradation_computation(self, engine: SRSEngine) -> None:
        """Degradation is computed correctly."""
        degradation = engine._compute_degradation(
            ideal_scores={"ter": 0.1, "ner_f1": 0.9},
            realtime_scores={"ter": 0.15, "ner_f1": 0.85},
        )

        # TER: higher is worse, so positive degradation
        assert degradation["ter"] == pytest.approx(0.05)
        # NER F1: higher is better, so positive degradation when lower
        assert degradation["ner_f1"] == pytest.approx(0.05)

    def test_srs_computation_no_ideal_errors(self, engine: SRSEngine) -> None:
        """SRS handles zero ideal TER."""
        result = engine.compute_from_summaries(
            model_name="test",
            ideal_scores={"ter": 0.0, "ner_f1": 1.0, "crs": 1.0},
            realtime_scores={"ter": 0.1, "ner_f1": 0.9, "crs": 0.9},
        )

        # Should handle gracefully
        assert 0 <= result.srs <= 1


class TestSRSIntegration:
    """Integration tests for SRS."""

    def test_srs_result_structure(self) -> None:
        """SRS result has correct structure."""
        engine = SRSEngine()
        result = engine.compute_from_summaries(
            model_name="test",
            ideal_scores={"ter": 0.1, "ner_f1": 0.9, "crs": 0.9},
            realtime_scores={"ter": 0.1, "ner_f1": 0.9, "crs": 0.9},
        )

        assert hasattr(result, "model_name")
        assert hasattr(result, "ideal_scores")
        assert hasattr(result, "realtime_scores")
        assert hasattr(result, "srs")
        assert hasattr(result, "degradation")


# ==============================================================================
# Factory Function Tests
# ==============================================================================


class TestCreateBenchmarkRunner:
    """Tests for create_benchmark_runner factory."""

    def test_create_with_defaults(self) -> None:
        """Create runner with default settings."""
        from hsttb.evaluation.runner import create_benchmark_runner

        adapter = MockSTTAdapter()
        runner = create_benchmark_runner(adapter)

        assert runner.config.streaming_profile == "ideal"

    def test_create_with_custom_profile(self) -> None:
        """Create runner with custom profile."""
        from hsttb.evaluation.runner import create_benchmark_runner

        adapter = MockSTTAdapter()
        runner = create_benchmark_runner(adapter, profile="realtime_mobile")

        assert runner.config.streaming_profile == "realtime_mobile"
