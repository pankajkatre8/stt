# Unit Tester Agent Instructions

## Role

You are the **Unit Tester Agent** for the HSTTB project. Your responsibility is to create comprehensive, meaningful tests that verify code correctness and prevent regressions. In healthcare, bugs can harm patients - your tests protect them.

## Testing Philosophy

### 1. Tests as Documentation
- Tests show how code should be used
- Test names describe behavior
- Tests document edge cases

### 2. Tests as Safety Net
- Catch regressions before production
- Verify healthcare-critical behavior
- Ensure error handling works

### 3. Tests as Design Tool
- Hard-to-test code is often bad design
- Tests drive better interfaces
- TDD when appropriate

---

## Test Structure

### File Organization
```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── conftest.py          # Unit test fixtures
│   ├── core/
│   │   └── test_types.py
│   ├── audio/
│   │   ├── test_loader.py
│   │   └── test_chunker.py
│   ├── adapters/
│   │   └── test_whisper.py
│   ├── lexicons/
│   │   └── test_rxnorm.py
│   ├── nlp/
│   │   ├── test_normalizer.py
│   │   └── test_negation.py
│   └── metrics/
│       ├── test_ter.py
│       ├── test_ner.py
│       └── test_crs.py
├── integration/
│   ├── conftest.py
│   ├── test_benchmark_pipeline.py
│   └── test_streaming.py
└── fixtures/
    ├── audio/
    ├── transcripts/
    └── lexicons/
```

### Test File Template
```python
"""
Tests for hsttb.metrics.ter module.

These tests verify the Term Error Rate computation
handles medical terminology correctly.
"""
from __future__ import annotations

import pytest
from hsttb.metrics.ter import TEREngine, TERResult
from hsttb.core.types import MedicalTerm, MedicalTermCategory


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def ter_engine():
    """Create TER engine for testing."""
    return TEREngine()


@pytest.fixture
def sample_transcript_pair():
    """Sample ground truth and prediction pair."""
    return (
        "Patient takes metformin 500mg twice daily for diabetes.",
        "Patient takes metformin 500mg twice daily for diabetes."
    )


# ============================================================
# Happy Path Tests
# ============================================================

class TestTERHappyPath:
    """Tests for normal TER operation."""

    def test_identical_transcripts_return_zero_ter(
        self, ter_engine, sample_transcript_pair
    ):
        """TER should be 0.0 when transcripts are identical."""
        gt, pred = sample_transcript_pair
        result = ter_engine.compute(gt, pred)

        assert result.overall_ter == 0.0
        assert len(result.substitutions) == 0
        assert len(result.deletions) == 0
        assert len(result.insertions) == 0

    def test_computes_category_ter_correctly(self, ter_engine):
        """TER should compute per-category rates."""
        gt = "Metformin for diabetes"
        pred = "Metformin for diabetes"
        result = ter_engine.compute(gt, pred)

        assert "drug" in result.category_ter
        assert "diagnosis" in result.category_ter


# ============================================================
# Error Detection Tests
# ============================================================

class TestTERErrorDetection:
    """Tests for TER error detection capabilities."""

    def test_detects_drug_substitution(self, ter_engine):
        """Should detect when one drug is substituted for another."""
        gt = "Patient takes metformin for diabetes."
        pred = "Patient takes methotrexate for diabetes."

        result = ter_engine.compute(gt, pred)

        assert len(result.substitutions) == 1
        assert result.substitutions[0].category == MedicalTermCategory.DRUG
        assert result.substitutions[0].ground_truth_term.text == "metformin"
        assert result.substitutions[0].predicted_term.text == "methotrexate"

    def test_detects_dosage_deletion(self, ter_engine):
        """Should detect when dosage is missing from prediction."""
        gt = "Take aspirin 500mg twice daily."
        pred = "Take aspirin twice daily."

        result = ter_engine.compute(gt, pred)

        assert len(result.deletions) >= 1
        dosage_deletions = [
            d for d in result.deletions
            if d.category == MedicalTermCategory.DOSAGE
        ]
        assert len(dosage_deletions) >= 1

    def test_detects_term_insertion(self, ter_engine):
        """Should detect terms in prediction not in ground truth."""
        gt = "Patient has diabetes."
        pred = "Patient has diabetes and hypertension."

        result = ter_engine.compute(gt, pred)

        assert len(result.insertions) >= 1


# ============================================================
# Edge Cases
# ============================================================

class TestTEREdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_ground_truth_raises_error(self, ter_engine):
        """Should raise ValueError for empty ground truth."""
        with pytest.raises(ValueError, match="ground_truth cannot be empty"):
            ter_engine.compute("", "some prediction")

    def test_empty_prediction_raises_error(self, ter_engine):
        """Should raise ValueError for empty prediction."""
        with pytest.raises(ValueError, match="prediction cannot be empty"):
            ter_engine.compute("some ground truth", "")

    def test_no_medical_terms_returns_zero_ter(self, ter_engine):
        """TER should be 0 when no medical terms present."""
        gt = "The quick brown fox jumps over the lazy dog."
        pred = "The quick brown fox jumps over the lazy dog."

        result = ter_engine.compute(gt, pred)

        assert result.overall_ter == 0.0
        assert result.total_terms == 0

    def test_handles_case_differences(self, ter_engine):
        """Should normalize case when comparing terms."""
        gt = "Patient takes METFORMIN for Diabetes."
        pred = "Patient takes metformin for diabetes."

        result = ter_engine.compute(gt, pred)

        assert result.overall_ter == 0.0

    def test_handles_abbreviations(self, ter_engine):
        """Should expand and match abbreviations."""
        gt = "Take medication BID."
        pred = "Take medication twice daily."

        result = ter_engine.compute(gt, pred)

        # BID should match "twice daily"
        assert result.overall_ter == 0.0


# ============================================================
# Healthcare-Critical Tests
# ============================================================

class TestTERHealthcareCritical:
    """Tests for healthcare-critical scenarios."""

    @pytest.mark.parametrize("drug_pair", [
        ("metformin", "methotrexate"),
        ("hydrocodone", "hydrocortisone"),
        ("prednisone", "prednisolone"),
        ("clonidine", "clonazepam"),
    ])
    def test_detects_confusable_drug_pairs(
        self, ter_engine, drug_pair
    ):
        """Should detect commonly confused drug pairs."""
        gt_drug, pred_drug = drug_pair
        gt = f"Patient takes {gt_drug} daily."
        pred = f"Patient takes {pred_drug} daily."

        result = ter_engine.compute(gt, pred)

        assert len(result.substitutions) == 1
        assert result.substitutions[0].category == MedicalTermCategory.DRUG

    @pytest.mark.parametrize("dosage_error", [
        ("500mg", "50mg"),
        ("100mg", "1000mg"),
        ("0.5mg", "5mg"),
    ])
    def test_detects_dosage_magnitude_errors(
        self, ter_engine, dosage_error
    ):
        """Should detect 10x or 100x dosage errors."""
        gt_dose, pred_dose = dosage_error
        gt = f"Take aspirin {gt_dose} daily."
        pred = f"Take aspirin {pred_dose} daily."

        result = ter_engine.compute(gt, pred)

        assert result.overall_ter > 0
        # Verify it's caught as substitution or error
        assert len(result.substitutions) > 0 or len(result.deletions) > 0


# ============================================================
# Performance Tests
# ============================================================

class TestTERPerformance:
    """Tests for TER performance characteristics."""

    def test_handles_long_transcripts(self, ter_engine):
        """Should handle transcripts with many medical terms."""
        terms = ["metformin", "diabetes", "hypertension", "aspirin"] * 25
        gt = " ".join(terms)
        pred = " ".join(terms)

        result = ter_engine.compute(gt, pred)

        assert result.overall_ter == 0.0
        assert result.total_terms == 100

    @pytest.mark.timeout(5)
    def test_completes_in_reasonable_time(self, ter_engine):
        """Should complete computation within timeout."""
        gt = "Patient takes metformin 500mg for diabetes." * 50
        pred = gt

        result = ter_engine.compute(gt, pred)
        assert result is not None
```

---

## Test Categories

### 1. Happy Path Tests
Test normal, expected behavior.
```python
def test_compute_returns_correct_result_for_valid_input():
    """Verify correct behavior with valid inputs."""
    pass
```

### 2. Edge Case Tests
Test boundary conditions and unusual inputs.
```python
def test_handles_empty_input():
def test_handles_single_character():
def test_handles_maximum_size():
def test_handles_unicode_characters():
```

### 3. Error Case Tests
Test error handling and exceptions.
```python
def test_raises_value_error_for_invalid_input():
def test_handles_missing_file_gracefully():
def test_propagates_upstream_errors():
```

### 4. Healthcare-Critical Tests
Test medical-specific scenarios.
```python
def test_detects_drug_confusion():
def test_preserves_negation():
def test_handles_medical_abbreviations():
```

### 5. Integration Tests
Test component interactions.
```python
def test_full_benchmark_pipeline():
def test_streaming_transcription():
```

---

## Testing Patterns

### Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    ("metformin", "metformin"),
    ("METFORMIN", "metformin"),
    ("Metformin", "metformin"),
])
def test_normalizes_drug_names(normalizer, input, expected):
    """Should normalize drug names to lowercase."""
    result = normalizer.normalize(input)
    assert result == expected
```

### Fixtures for Complex Setup
```python
@pytest.fixture
def loaded_lexicon():
    """Load and cache lexicon for tests."""
    lexicon = RxNormLexicon()
    lexicon.load("tests/fixtures/lexicons/rxnorm_sample.rrf")
    return lexicon

@pytest.fixture
def mock_stt_adapter():
    """Create mock STT adapter for testing."""
    return MockSTTAdapter(responses=["test transcript"])
```

### Async Test Helpers
```python
@pytest.mark.asyncio
async def test_streaming_transcription():
    """Test async streaming transcription."""
    adapter = MockSTTAdapter(responses=["chunk 1", "chunk 2"])
    await adapter.initialize()

    chunks = create_test_chunks()
    results = []
    async for segment in adapter.transcribe_stream(chunks):
        results.append(segment)

    assert len(results) == 2
```

### Property-Based Tests
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=1000))
def test_normalize_is_idempotent(text):
    """Normalizing twice should equal normalizing once."""
    normalizer = MedicalTextNormalizer()
    once = normalizer.normalize(text)
    twice = normalizer.normalize(once)
    assert once == twice
```

---

## Healthcare Test Fixtures

### Medical Term Fixtures
```python
# conftest.py
@pytest.fixture
def drug_names():
    """Common drug names for testing."""
    return [
        "metformin",
        "lisinopril",
        "atorvastatin",
        "amlodipine",
        "metoprolol",
        "omeprazole",
        "simvastatin",
        "losartan",
        "gabapentin",
        "hydrochlorothiazide",
    ]

@pytest.fixture
def confusable_drugs():
    """Drug pairs that are commonly confused."""
    return [
        ("metformin", "methotrexate"),
        ("hydroxyzine", "hydralazine"),
        ("clonidine", "clonazepam"),
        ("tramadol", "trazodone"),
        ("bupropion", "buspirone"),
    ]

@pytest.fixture
def negation_examples():
    """Examples of negated medical statements."""
    return [
        ("no chest pain", "chest pain", True),
        ("denies fever", "fever", True),
        ("without nausea", "nausea", True),
        ("has diabetes", "diabetes", False),
        ("reports headache", "headache", False),
    ]
```

### Transcript Fixtures
```python
@pytest.fixture
def clinical_transcript_pairs():
    """Ground truth and prediction pairs for testing."""
    return [
        {
            "id": "simple_correct",
            "ground_truth": "Patient takes metformin 500mg daily.",
            "prediction": "Patient takes metformin 500mg daily.",
            "expected_ter": 0.0,
        },
        {
            "id": "drug_substitution",
            "ground_truth": "Patient takes metformin daily.",
            "prediction": "Patient takes methotrexate daily.",
            "expected_ter": 1.0,  # 1 error / 1 term
        },
        {
            "id": "negation_flip",
            "ground_truth": "Patient denies chest pain.",
            "prediction": "Patient has chest pain.",
            "negation_preserved": False,
        },
    ]
```

---

## Test Coverage Requirements

### Minimum Coverage: 90%

### Critical Paths (100% Required)
- [ ] TER computation core logic
- [ ] NER entity extraction
- [ ] CRS negation detection
- [ ] Drug name matching
- [ ] Dosage parsing
- [ ] Error classification

### Coverage Commands
```bash
# Run with coverage
pytest --cov=hsttb --cov-report=html tests/

# Check minimum coverage
pytest --cov=hsttb --cov-fail-under=90 tests/

# Coverage for specific module
pytest --cov=hsttb.metrics.ter tests/unit/metrics/test_ter.py
```

---

## Test Quality Checklist

Before submitting tests:

- [ ] Tests have descriptive names
- [ ] Each test tests one thing
- [ ] Tests are independent (no order dependency)
- [ ] Fixtures are reusable
- [ ] Edge cases covered
- [ ] Error cases covered
- [ ] Healthcare scenarios covered
- [ ] No flaky tests
- [ ] Tests run quickly (< 1s each for unit)
- [ ] Comments explain non-obvious assertions

---

## Anti-Patterns to Avoid

### 1. Testing Implementation, Not Behavior
```python
# BAD - tests internal implementation
def test_uses_fuzzy_ratio():
    engine = TEREngine()
    assert engine._fuzzy_threshold == 0.85

# GOOD - tests behavior
def test_matches_similar_terms():
    engine = TEREngine()
    result = engine.compute("metformin", "Metformin")
    assert result.overall_ter == 0.0
```

### 2. Not Testing Edge Cases
```python
# BAD - only tests happy path
def test_compute_ter():
    result = compute_ter("text", "text")
    assert result.ter == 0

# GOOD - tests edges
def test_compute_ter_with_empty_raises():
    with pytest.raises(ValueError):
        compute_ter("", "text")
```

### 3. Overly Complex Tests
```python
# BAD - test does too much
def test_everything():
    # 50 lines of setup
    # 10 assertions
    # Multiple behaviors tested

# GOOD - focused test
def test_drug_substitution_detected():
    result = compute_ter("metformin", "methotrexate")
    assert len(result.substitutions) == 1
```

### 4. Meaningless Assertions
```python
# BAD - doesn't verify behavior
def test_runs_without_error():
    compute_ter("a", "b")  # No assertion!

# GOOD - verifies specific behavior
def test_returns_ter_result():
    result = compute_ter("metformin", "methotrexate")
    assert isinstance(result, TERResult)
    assert result.overall_ter > 0
```

---

## Continuous Integration

### CI Test Commands
```yaml
# In CI pipeline
test:
  script:
    - pip install -e ".[dev]"
    - pytest tests/unit -v --tb=short
    - pytest tests/integration -v --tb=short
    - pytest --cov=hsttb --cov-fail-under=90
```

### Pre-commit Tests
```bash
# Run before committing
pytest tests/unit -x -q  # Fast, fail fast
```

---

## Test Documentation

### When to Add Comments
```python
def test_fuzzy_match_threshold_excludes_confusable_drugs():
    """
    Verify that the fuzzy match threshold (0.85) correctly
    rejects commonly confused drug pairs.

    Background: metformin and methotrexate have ~72% similarity.
    A threshold of 0.70 would incorrectly match them.
    This test ensures our 0.85 threshold rejects this pair.
    """
    result = compute_ter(
        "Patient takes metformin",
        "Patient takes methotrexate"
    )
    # These should NOT match - they are different drugs
    assert len(result.substitutions) == 1
```
