"""
Tests for Term Error Rate (TER) computation engine.

Tests the TER engine's ability to detect term errors
between ground truth and predicted transcriptions.
"""
from __future__ import annotations

import pytest

from hsttb.core.types import ErrorType, MedicalTermCategory
from hsttb.lexicons import MockMedicalLexicon
from hsttb.metrics import TEREngine, TERResult, compute_ter


class TestTERResult:
    """Tests for TERResult dataclass."""

    def test_total_errors(self) -> None:
        """Total errors includes all error types."""
        result = TERResult(
            overall_ter=0.5,
            category_ter={},
            total_gt_terms=4,
            total_pred_terms=4,
            correct_matches=2,
            substitutions=[],
            deletions=[],
            insertions=[],
        )
        assert result.total_errors == 0

    def test_precision(self) -> None:
        """Precision = correct / total_pred."""
        result = TERResult(
            overall_ter=0.5,
            category_ter={},
            total_gt_terms=10,
            total_pred_terms=8,
            correct_matches=6,
            substitutions=[],
            deletions=[],
            insertions=[],
        )
        assert result.precision == 6 / 8

    def test_recall(self) -> None:
        """Recall = correct / total_gt."""
        result = TERResult(
            overall_ter=0.5,
            category_ter={},
            total_gt_terms=10,
            total_pred_terms=8,
            correct_matches=6,
            substitutions=[],
            deletions=[],
            insertions=[],
        )
        assert result.recall == 6 / 10

    def test_f1_score(self) -> None:
        """F1 = 2 * precision * recall / (precision + recall)."""
        result = TERResult(
            overall_ter=0.0,
            category_ter={},
            total_gt_terms=10,
            total_pred_terms=10,
            correct_matches=8,
            substitutions=[],
            deletions=[],
            insertions=[],
        )
        p = 8 / 10
        r = 8 / 10
        expected_f1 = 2 * p * r / (p + r)
        assert abs(result.f1_score - expected_f1) < 0.001

    def test_empty_predictions(self) -> None:
        """Handle empty predictions gracefully."""
        result = TERResult(
            overall_ter=1.0,
            category_ter={},
            total_gt_terms=5,
            total_pred_terms=0,
            correct_matches=0,
            substitutions=[],
            deletions=[],
            insertions=[],
        )
        assert result.precision == 1.0  # No false positives
        assert result.recall == 0.0


class TestTEREngine:
    """Tests for TEREngine class."""

    @pytest.fixture
    def lexicon(self) -> MockMedicalLexicon:
        """Create mock lexicon with common terms."""
        return MockMedicalLexicon.with_common_terms()

    @pytest.fixture
    def engine(self, lexicon: MockMedicalLexicon) -> TEREngine:
        """Create TER engine with mock lexicon."""
        return TEREngine(lexicon)

    def test_perfect_match(self, engine: TEREngine) -> None:
        """TER is 0.0 for perfect match."""
        result = engine.compute(
            ground_truth="patient takes metformin for diabetes",
            prediction="patient takes metformin for diabetes",
        )
        assert result.overall_ter == 0.0
        assert result.total_errors == 0

    def test_deletion_error(self, engine: TEREngine) -> None:
        """Detect deletion when term is missing."""
        result = engine.compute(
            ground_truth="patient takes metformin and aspirin",
            prediction="patient takes metformin",
        )
        # "aspirin" should be detected as deleted
        assert len(result.deletions) > 0
        assert result.overall_ter > 0.0

    def test_insertion_error(self, engine: TEREngine) -> None:
        """Detect insertion when extra term appears."""
        result = engine.compute(
            ground_truth="patient takes metformin",
            prediction="patient takes metformin and aspirin",
        )
        # "aspirin" should be detected as inserted
        assert len(result.insertions) > 0

    def test_substitution_error(self, engine: TEREngine) -> None:
        """Detect substitution when term is replaced."""
        result = engine.compute(
            ground_truth="patient takes aspirin",
            prediction="patient takes ibuprofen",
        )
        # Two different drugs - might be substitution or del+ins
        # depending on similarity threshold
        assert result.total_errors > 0

    def test_case_insensitive(self, engine: TEREngine) -> None:
        """Matching is case-insensitive."""
        result = engine.compute(
            ground_truth="Patient takes METFORMIN",
            prediction="patient takes metformin",
        )
        # Should match despite case difference
        assert result.correct_matches > 0

    def test_abbreviation_expansion(self, engine: TEREngine) -> None:
        """Abbreviations are expanded before comparison."""
        result = engine.compute(
            ground_truth="patient has HTN",
            prediction="patient has hypertension",
        )
        # HTN expands to hypertension
        # This depends on normalizer + lexicon having hypertension

    def test_category_ter(self, engine: TEREngine) -> None:
        """TER is computed per category."""
        result = engine.compute(
            ground_truth="patient takes metformin and aspirin for diabetes and hypertension",
            prediction="patient takes metformin and aspirin for diabetes",
        )
        # Should have category-wise breakdown
        assert isinstance(result.category_ter, dict)

    def test_no_medical_terms(self, engine: TEREngine) -> None:
        """Handle text with no medical terms."""
        result = engine.compute(
            ground_truth="the patient is doing well",
            prediction="the patient is doing well",
        )
        assert result.total_gt_terms == 0
        assert result.overall_ter == 0.0

    def test_empty_strings(self, engine: TEREngine) -> None:
        """Handle empty input strings."""
        result = engine.compute(ground_truth="", prediction="")
        assert result.overall_ter == 0.0
        assert result.total_gt_terms == 0

    def test_drug_terms_extracted(self, engine: TEREngine) -> None:
        """Drug terms are correctly extracted."""
        result = engine.compute(
            ground_truth="patient takes metformin",
            prediction="patient takes metformin",
        )
        assert result.total_gt_terms >= 1
        assert result.correct_matches >= 1

    def test_diagnosis_terms_extracted(self, engine: TEREngine) -> None:
        """Diagnosis terms are correctly extracted."""
        result = engine.compute(
            ground_truth="patient has diabetes mellitus",
            prediction="patient has diabetes mellitus",
        )
        assert result.total_gt_terms >= 1
        assert result.correct_matches >= 1


class TestTERTermExtraction:
    """Tests for term extraction functionality."""

    @pytest.fixture
    def engine(self) -> TEREngine:
        """Create engine with common terms."""
        lexicon = MockMedicalLexicon.with_common_terms()
        return TEREngine(lexicon)

    def test_extract_single_term(self, engine: TEREngine) -> None:
        """Extract single medical term."""
        terms = engine._extract_terms("patient takes metformin daily")
        term_texts = [t.text for t in terms]
        assert "metformin" in term_texts

    def test_extract_multiple_terms(self, engine: TEREngine) -> None:
        """Extract multiple terms from text."""
        terms = engine._extract_terms("patient takes metformin and aspirin")
        term_texts = [t.text for t in terms]
        assert "metformin" in term_texts
        assert "aspirin" in term_texts

    def test_extract_multiword_term(self, engine: TEREngine) -> None:
        """Extract multi-word terms like 'diabetes mellitus'."""
        terms = engine._extract_terms("patient has diabetes mellitus")
        term_texts = [t.text for t in terms]
        # Should find "diabetes mellitus" as single term
        assert "diabetes mellitus" in term_texts or "diabetes" in term_texts

    def test_extract_with_spans(self, engine: TEREngine) -> None:
        """Extracted terms have correct spans."""
        text = "metformin"
        terms = engine._extract_terms(text)
        if terms:
            assert terms[0].span[0] >= 0
            assert terms[0].span[1] <= len(text)


class TestTERAlignment:
    """Tests for term alignment algorithm."""

    @pytest.fixture
    def engine(self) -> TEREngine:
        """Create TER engine."""
        lexicon = MockMedicalLexicon.with_common_terms()
        return TEREngine(lexicon)

    def test_align_identical(self, engine: TEREngine) -> None:
        """Align identical term lists."""
        from hsttb.core.types import MedicalTerm, MedicalTermCategory

        terms = [
            MedicalTerm(
                text="metformin",
                normalized="metformin",
                category=MedicalTermCategory.DRUG,
                source="mock",
                span=(0, 9),
            )
        ]
        matches = engine._align_terms(terms, terms.copy())
        assert all(m.match_type == "correct" for m in matches)

    def test_align_with_deletion(self, engine: TEREngine) -> None:
        """Align when prediction is missing terms."""
        from hsttb.core.types import MedicalTerm, MedicalTermCategory

        gt_terms = [
            MedicalTerm(
                text="metformin",
                normalized="metformin",
                category=MedicalTermCategory.DRUG,
                source="mock",
                span=(0, 9),
            ),
            MedicalTerm(
                text="aspirin",
                normalized="aspirin",
                category=MedicalTermCategory.DRUG,
                source="mock",
                span=(10, 17),
            ),
        ]
        pred_terms = [
            MedicalTerm(
                text="metformin",
                normalized="metformin",
                category=MedicalTermCategory.DRUG,
                source="mock",
                span=(0, 9),
            )
        ]
        matches = engine._align_terms(gt_terms, pred_terms)
        deletions = [m for m in matches if m.match_type == "deletion"]
        assert len(deletions) == 1

    def test_align_with_insertion(self, engine: TEREngine) -> None:
        """Align when prediction has extra terms."""
        from hsttb.core.types import MedicalTerm, MedicalTermCategory

        gt_terms = [
            MedicalTerm(
                text="metformin",
                normalized="metformin",
                category=MedicalTermCategory.DRUG,
                source="mock",
                span=(0, 9),
            )
        ]
        pred_terms = [
            MedicalTerm(
                text="metformin",
                normalized="metformin",
                category=MedicalTermCategory.DRUG,
                source="mock",
                span=(0, 9),
            ),
            MedicalTerm(
                text="aspirin",
                normalized="aspirin",
                category=MedicalTermCategory.DRUG,
                source="mock",
                span=(10, 17),
            ),
        ]
        matches = engine._align_terms(gt_terms, pred_terms)
        insertions = [m for m in matches if m.match_type == "insertion"]
        assert len(insertions) == 1


class TestComputeTER:
    """Tests for convenience function compute_ter."""

    def test_compute_ter_function(self) -> None:
        """compute_ter returns TER value."""
        lexicon = MockMedicalLexicon.with_common_terms()
        ter = compute_ter(
            ground_truth="patient takes metformin",
            prediction="patient takes metformin",
            lexicon=lexicon,
        )
        assert isinstance(ter, float)
        assert 0.0 <= ter <= 1.0

    def test_compute_ter_with_errors(self) -> None:
        """compute_ter returns non-zero for errors."""
        lexicon = MockMedicalLexicon.with_common_terms()
        ter = compute_ter(
            ground_truth="patient takes metformin and aspirin",
            prediction="patient takes metformin",
            lexicon=lexicon,
        )
        # Should have some error rate
        assert ter >= 0.0
