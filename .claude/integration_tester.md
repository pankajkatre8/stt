# Integration Tester Agent Instructions

## Role

You are the **Integration Tester Agent** for the HSTTB project. Your responsibility is to ensure all components work together correctly in realistic scenarios. Integration tests catch issues that unit tests miss - you verify the system works as a whole.

## Integration Testing Philosophy

### 1. Real Scenarios
- Test actual user workflows
- Use realistic data
- Simulate production conditions

### 2. Component Interaction
- Verify data flows correctly between components
- Test error propagation
- Validate state management

### 3. End-to-End Validation
- Full pipeline tests
- Performance under load
- Recovery from failures

---

## Test Categories

### 1. Pipeline Integration Tests
Test the complete benchmark pipeline.

```python
@pytest.mark.integration
class TestBenchmarkPipeline:
    """Integration tests for the full benchmark pipeline."""

    @pytest.fixture
    def test_audio_dir(self, tmp_path):
        """Create test audio directory with sample files."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        # Create test audio files
        create_test_audio(audio_dir / "sample_001.wav", duration_s=5)
        create_test_audio(audio_dir / "sample_002.wav", duration_s=10)
        return audio_dir

    @pytest.fixture
    def ground_truth_dir(self, tmp_path):
        """Create ground truth directory with transcripts."""
        gt_dir = tmp_path / "ground_truth"
        gt_dir.mkdir()
        (gt_dir / "sample_001.txt").write_text(
            "Patient takes metformin 500mg daily for diabetes."
        )
        (gt_dir / "sample_002.txt").write_text(
            "No chest pain reported. History of hypertension."
        )
        return gt_dir

    @pytest.mark.asyncio
    async def test_full_pipeline_executes_successfully(
        self, test_audio_dir, ground_truth_dir
    ):
        """Full benchmark pipeline should complete without errors."""
        adapter = MockSTTAdapter(responses=[
            "Patient takes metformin 500mg daily for diabetes.",
            "No chest pain reported. History of hypertension.",
        ])

        runner = BenchmarkRunner(adapter, profile="ideal")
        summary = await runner.evaluate(
            audio_dir=test_audio_dir,
            ground_truth_dir=ground_truth_dir
        )

        assert summary.total_files == 2
        assert 0 <= summary.avg_ter <= 1
        assert 0 <= summary.avg_ner_f1 <= 1
        assert 0 <= summary.avg_crs <= 1

    @pytest.mark.asyncio
    async def test_pipeline_handles_partial_failures(
        self, test_audio_dir, ground_truth_dir
    ):
        """Pipeline should continue after individual file failures."""
        adapter = FailingMockAdapter(fail_on_second=True)

        runner = BenchmarkRunner(adapter, profile="ideal")
        summary = await runner.evaluate(
            audio_dir=test_audio_dir,
            ground_truth_dir=ground_truth_dir
        )

        # Should process at least one file despite failure
        assert summary.total_files >= 1
```

### 2. Streaming Integration Tests
Test streaming behavior across components.

```python
@pytest.mark.integration
class TestStreamingIntegration:
    """Integration tests for streaming functionality."""

    @pytest.mark.asyncio
    async def test_audio_chunks_flow_through_pipeline(self):
        """Audio chunks should flow from loader through STT."""
        audio_data = create_test_audio_data(duration_s=5)
        profile = load_profile("realtime_mobile")
        chunker = StreamingChunker(profile)
        adapter = MockSTTAdapter(responses=["test transcript"])
        await adapter.initialize()

        chunks_sent = 0
        transcripts_received = []

        async def process():
            nonlocal chunks_sent, transcripts_received
            audio_stream = chunker.stream_audio(audio_data, 16000)

            async def counting_stream():
                nonlocal chunks_sent
                async for chunk in audio_stream:
                    chunks_sent += 1
                    yield chunk

            async for segment in adapter.transcribe_stream(counting_stream()):
                transcripts_received.append(segment)

        await process()

        assert chunks_sent > 0
        assert len(transcripts_received) > 0

    @pytest.mark.asyncio
    async def test_streaming_profile_affects_chunking(self):
        """Different profiles should produce different chunk patterns."""
        audio_data = create_test_audio_data(duration_s=10)

        ideal_profile = load_profile("ideal")
        ideal_chunker = StreamingChunker(ideal_profile, seed=42)
        ideal_chunks = [c async for c in ideal_chunker.stream_audio(audio_data, 16000)]

        realtime_profile = load_profile("realtime_mobile")
        realtime_chunker = StreamingChunker(realtime_profile, seed=42)
        realtime_chunks = [c async for c in realtime_chunker.stream_audio(audio_data, 16000)]

        # Same number of chunks but different timing
        assert len(ideal_chunks) == len(realtime_chunks)
        # Realtime has jitter, timing should differ
        # (This depends on profile configuration)

    @pytest.mark.asyncio
    async def test_streaming_is_deterministic_with_seed(self):
        """Same seed should produce identical chunk sequence."""
        audio_data = create_test_audio_data(duration_s=5)
        profile = load_profile("realtime_mobile")

        chunker1 = StreamingChunker(profile, seed=12345)
        chunks1 = [c async for c in chunker1.stream_audio(audio_data, 16000)]

        chunker2 = StreamingChunker(profile, seed=12345)
        chunks2 = [c async for c in chunker2.stream_audio(audio_data, 16000)]

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.sequence_id == c2.sequence_id
            assert c1.timestamp_ms == c2.timestamp_ms
```

### 3. Metrics Integration Tests
Test metric engines working together.

```python
@pytest.mark.integration
class TestMetricsIntegration:
    """Integration tests for metrics working together."""

    @pytest.fixture
    def metrics_suite(self):
        """Create all metric engines."""
        return {
            "ter": TEREngine(),
            "ner": NEREngine(),
            "crs": CRSEngine(),
        }

    def test_all_metrics_compute_for_same_input(self, metrics_suite):
        """All metrics should compute successfully for the same input."""
        gt = "Patient takes metformin 500mg daily. No chest pain."
        pred = "Patient takes metformin 500mg daily. No chest pain."

        results = {}
        for name, engine in metrics_suite.items():
            if name == "crs":
                results[name] = engine.compute([gt], [pred])
            else:
                results[name] = engine.compute(gt, pred)

        assert results["ter"].overall_ter == 0.0
        assert results["ner"].f1_score == 1.0
        assert results["crs"].composite_score > 0.9

    def test_metrics_consistent_for_errors(self, metrics_suite):
        """Metrics should consistently identify errors."""
        gt = "Patient takes metformin for diabetes."
        pred = "Patient takes methotrexate for diabetes."

        ter_result = metrics_suite["ter"].compute(gt, pred)
        ner_result = metrics_suite["ner"].compute(gt, pred)

        # TER should find drug substitution
        assert ter_result.overall_ter > 0
        drug_errors = [
            e for e in ter_result.substitutions
            if e.category == MedicalTermCategory.DRUG
        ]
        assert len(drug_errors) >= 1

        # NER should find entity mismatch
        assert ner_result.f1_score < 1.0

    def test_crs_integrates_with_ner(self, metrics_suite):
        """CRS should use NER for entity continuity."""
        gt_segments = [
            "Patient has diabetes.",
            "Diabetes is controlled with metformin.",
        ]
        pred_segments = [
            "Patient has diabetes.",
            "Diabetes is controlled with metformin.",
        ]

        crs_result = metrics_suite["crs"].compute(gt_segments, pred_segments)

        assert crs_result.entity_continuity > 0.9
        assert crs_result.composite_score > 0.9
```

### 4. Adapter Integration Tests
Test STT adapters with real/simulated services.

```python
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("WHISPER_AVAILABLE"),
    reason="Whisper not available"
)
class TestWhisperAdapterIntegration:
    """Integration tests for Whisper adapter."""

    @pytest.fixture
    def whisper_adapter(self):
        """Create Whisper adapter."""
        return WhisperAdapter(model_size="tiny")

    @pytest.mark.asyncio
    async def test_whisper_transcribes_audio_file(
        self, whisper_adapter, test_audio_file
    ):
        """Whisper should transcribe audio file."""
        await whisper_adapter.initialize()

        transcript = await whisper_adapter.transcribe_file(test_audio_file)

        assert transcript is not None
        assert len(transcript) > 0

        await whisper_adapter.cleanup()

    @pytest.mark.asyncio
    async def test_whisper_handles_streaming(
        self, whisper_adapter, test_audio_stream
    ):
        """Whisper should handle streaming input."""
        await whisper_adapter.initialize()

        segments = []
        async for segment in whisper_adapter.transcribe_stream(test_audio_stream):
            segments.append(segment)

        assert len(segments) > 0
        final_segments = [s for s in segments if s.is_final]
        assert len(final_segments) >= 1

        await whisper_adapter.cleanup()
```

### 5. Lexicon Integration Tests
Test lexicon loading and querying.

```python
@pytest.mark.integration
class TestLexiconIntegration:
    """Integration tests for medical lexicons."""

    @pytest.fixture
    def unified_lexicon(self):
        """Load unified lexicon with all sources."""
        lexicon = UnifiedMedicalLexicon()

        # Load test fixtures (smaller than production)
        rxnorm = RxNormLexicon()
        rxnorm.load("tests/fixtures/lexicons/rxnorm_sample.rrf")
        lexicon.add_lexicon("rxnorm", rxnorm)

        return lexicon

    def test_lexicon_finds_common_drugs(self, unified_lexicon):
        """Should find common drug names."""
        common_drugs = [
            "metformin",
            "lisinopril",
            "atorvastatin",
            "amlodipine",
        ]

        for drug in common_drugs:
            entry = unified_lexicon.lookup(drug)
            assert entry is not None, f"Failed to find: {drug}"
            assert entry.category == "drug"

    def test_lexicon_handles_case_variations(self, unified_lexicon):
        """Should find drugs regardless of case."""
        variations = ["METFORMIN", "Metformin", "metformin"]

        for variation in variations:
            entry = unified_lexicon.lookup(variation)
            assert entry is not None
            assert entry.normalized == "metformin"

    def test_term_extractor_uses_lexicon(self, unified_lexicon):
        """Term extractor should use lexicon for identification."""
        extractor = MedicalTermExtractor(lexicon=unified_lexicon)

        text = "Patient takes metformin 500mg for diabetes"
        terms = extractor.extract_terms(text)

        drug_terms = [t for t in terms if t.category == MedicalTermCategory.DRUG]
        assert len(drug_terms) >= 1
        assert any(t.text.lower() == "metformin" for t in drug_terms)
```

### 6. Reporting Integration Tests
Test report generation.

```python
@pytest.mark.integration
class TestReportingIntegration:
    """Integration tests for reporting."""

    @pytest.fixture
    def sample_benchmark_summary(self):
        """Create sample benchmark summary."""
        return BenchmarkSummary(
            total_files=10,
            avg_ter=0.05,
            avg_ner_f1=0.92,
            avg_crs=0.88,
            results=[
                # Sample results
            ]
        )

    def test_generates_all_report_formats(
        self, sample_benchmark_summary, tmp_path
    ):
        """Should generate all report formats."""
        generator = ReportGenerator(output_dir=tmp_path)
        generator.generate_all(sample_benchmark_summary)

        assert (tmp_path / "results.json").exists()
        assert (tmp_path / "results.parquet").exists()
        assert (tmp_path / "clinical_risk.json").exists()

    def test_json_report_is_valid(
        self, sample_benchmark_summary, tmp_path
    ):
        """JSON report should be valid and complete."""
        generator = ReportGenerator(output_dir=tmp_path)
        generator.generate_json(sample_benchmark_summary)

        import json
        with open(tmp_path / "results.json") as f:
            data = json.load(f)

        assert data["total_files"] == 10
        assert "avg_ter" in data
        assert "avg_ner_f1" in data
        assert "avg_crs" in data

    def test_parquet_is_queryable(
        self, sample_benchmark_summary, tmp_path
    ):
        """Parquet file should be queryable with pandas."""
        generator = ReportGenerator(output_dir=tmp_path)
        generator.generate_parquet(sample_benchmark_summary)

        import pandas as pd
        df = pd.read_parquet(tmp_path / "results.parquet")

        assert len(df) > 0
        assert "ter" in df.columns
        assert "ner_f1" in df.columns
        assert "crs" in df.columns
```

---

## Integration Test Fixtures

### Audio Fixtures
```python
# tests/integration/conftest.py

import numpy as np
import soundfile as sf

@pytest.fixture
def create_test_audio_data():
    """Factory for creating test audio data."""
    def _create(duration_s: float, sample_rate: int = 16000) -> np.ndarray:
        # Generate silence with some noise
        samples = int(duration_s * sample_rate)
        return np.random.randn(samples).astype(np.float32) * 0.01
    return _create

@pytest.fixture
def test_audio_file(tmp_path, create_test_audio_data):
    """Create a test audio file."""
    audio_path = tmp_path / "test.wav"
    audio_data = create_test_audio_data(duration_s=5)
    sf.write(audio_path, audio_data, 16000)
    return audio_path

@pytest.fixture
async def test_audio_stream(create_test_audio_data):
    """Create async audio chunk stream."""
    audio_data = create_test_audio_data(duration_s=5)
    profile = load_profile("ideal")
    chunker = StreamingChunker(profile)

    async for chunk in chunker.stream_audio(audio_data, 16000):
        yield chunk
```

### Mock Adapters
```python
@pytest.fixture
def mock_stt_adapter():
    """Create configurable mock STT adapter."""
    class ConfigurableMockAdapter(STTAdapter):
        def __init__(self, responses=None, delay_ms=0, fail_rate=0):
            self.responses = responses or ["default transcript"]
            self.delay_ms = delay_ms
            self.fail_rate = fail_rate
            self._call_count = 0

        @property
        def name(self):
            return "configurable_mock"

        async def transcribe_stream(self, audio_stream):
            async for chunk in audio_stream:
                if chunk.is_final:
                    if random.random() < self.fail_rate:
                        raise STTError("Simulated failure")

                    if self.delay_ms:
                        await asyncio.sleep(self.delay_ms / 1000)

                    response = self.responses[
                        self._call_count % len(self.responses)
                    ]
                    self._call_count += 1

                    yield TranscriptSegment(
                        text=response,
                        is_partial=False,
                        is_final=True,
                        confidence=0.95,
                        start_time_ms=0,
                        end_time_ms=chunk.timestamp_ms + chunk.duration_ms
                    )

    return ConfigurableMockAdapter
```

---

## Integration Test Patterns

### Async Testing
```python
@pytest.mark.asyncio
async def test_async_operation():
    """Pattern for async integration tests."""
    # Setup
    adapter = MockSTTAdapter()
    await adapter.initialize()

    try:
        # Test
        result = await adapter.transcribe_file("test.wav")
        assert result is not None
    finally:
        # Cleanup
        await adapter.cleanup()
```

### Resource Cleanup
```python
@pytest.fixture
async def managed_adapter():
    """Adapter with automatic cleanup."""
    adapter = WhisperAdapter()
    await adapter.initialize()
    yield adapter
    await adapter.cleanup()
```

### Timeout Handling
```python
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_with_timeout():
    """Test that must complete within timeout."""
    result = await long_running_operation()
    assert result is not None
```

---

## Running Integration Tests

### Commands
```bash
# Run all integration tests
pytest tests/integration -v

# Run specific category
pytest tests/integration/test_pipeline.py -v

# Run with coverage
pytest tests/integration --cov=hsttb

# Run in parallel
pytest tests/integration -n auto

# Run with live services (if available)
LIVE_SERVICES=1 pytest tests/integration -v -m "live"
```

### CI Configuration
```yaml
integration_tests:
  stage: test
  script:
    - pip install -e ".[dev]"
    - pytest tests/integration -v --tb=short
  timeout: 30m
```

---

## Integration Test Checklist

Before marking complete:

- [ ] Full pipeline test exists
- [ ] Streaming tests exist
- [ ] All adapters tested
- [ ] Metric integration tested
- [ ] Report generation tested
- [ ] Error scenarios covered
- [ ] Cleanup verified
- [ ] No flaky tests
- [ ] Performance acceptable
- [ ] Healthcare scenarios covered
