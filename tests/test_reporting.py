"""
Tests for report generation module.

Tests the report generator's ability to produce reports
in multiple formats.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hsttb.core.types import (
    BenchmarkResult,
    BenchmarkSummary,
    CRSResult,
    ErrorType,
    MedicalTerm,
    MedicalTermCategory,
    NERResult,
    TermError,
    TERResult,
)
from hsttb.reporting import ReportConfig, ReportGenerator, generate_report

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_ter_result() -> TERResult:
    """Create sample TER result."""
    return TERResult(
        overall_ter=0.1,
        category_ter={"drug": 0.1},
        total_terms=10,
    )


@pytest.fixture
def sample_ner_result() -> NERResult:
    """Create sample NER result."""
    return NERResult(
        precision=0.9,
        recall=0.85,
        f1_score=0.875,
        entity_distortion_rate=0.05,
        entity_omission_rate=0.1,
    )


@pytest.fixture
def sample_crs_result() -> CRSResult:
    """Create sample CRS result."""
    return CRSResult(
        composite_score=0.92,
        semantic_similarity=0.95,
        entity_continuity=0.9,
        negation_consistency=0.9,
        context_drift_rate=0.02,
    )


@pytest.fixture
def sample_benchmark_result(
    sample_ter_result: TERResult,
    sample_ner_result: NERResult,
    sample_crs_result: CRSResult,
) -> BenchmarkResult:
    """Create sample benchmark result."""
    return BenchmarkResult(
        audio_id="test_audio_001",
        ter=sample_ter_result,
        ner=sample_ner_result,
        crs=sample_crs_result,
        transcript_ground_truth="patient takes metformin for diabetes",
        transcript_predicted="patient takes metformin for diabetes",
        streaming_profile="ideal",
        adapter_name="mock",
    )


@pytest.fixture
def sample_summary(sample_benchmark_result: BenchmarkResult) -> BenchmarkSummary:
    """Create sample benchmark summary."""
    return BenchmarkSummary(
        total_files=1,
        avg_ter=0.1,
        avg_ner_f1=0.875,
        avg_crs=0.92,
        results=[sample_benchmark_result],
        streaming_profile="ideal",
        adapter_name="mock",
    )


# ==============================================================================
# Report Config Tests
# ==============================================================================


class TestReportConfig:
    """Tests for ReportConfig."""

    def test_default_config(self) -> None:
        """Default config enables all reports."""
        config = ReportConfig()

        assert config.generate_json is True
        assert config.generate_csv is True
        assert config.generate_html is True
        assert config.generate_clinical_risk is True

    def test_custom_config(self) -> None:
        """Config accepts custom values."""
        config = ReportConfig(
            generate_json=True,
            generate_csv=False,
            generate_html=False,
            generate_clinical_risk=True,
        )

        assert config.generate_json is True
        assert config.generate_csv is False


# ==============================================================================
# Report Generator Tests
# ==============================================================================


class TestReportGenerator:
    """Tests for ReportGenerator."""

    @pytest.fixture
    def generator(self, tmp_path: Path) -> ReportGenerator:
        """Create report generator."""
        return ReportGenerator(tmp_path)

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """Generator creates output directory."""
        output_dir = tmp_path / "reports" / "nested"
        _ = ReportGenerator(output_dir)

        assert output_dir.exists()

    def test_generate_all(
        self, generator: ReportGenerator, sample_summary: BenchmarkSummary
    ) -> None:
        """Generate all report types."""
        generated = generator.generate_all(sample_summary)

        assert "json" in generated
        assert "csv" in generated
        assert "html" in generated
        assert "clinical_risk" in generated

    def test_generate_json(
        self, generator: ReportGenerator, sample_summary: BenchmarkSummary
    ) -> None:
        """Generate JSON report."""
        path = generator.generate_json(sample_summary)

        assert path.exists()
        assert path.suffix == ".json"

        with open(path) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "summary" in data
        assert "results" in data
        assert data["summary"]["total_files"] == 1

    def test_generate_csv(
        self, generator: ReportGenerator, sample_summary: BenchmarkSummary
    ) -> None:
        """Generate CSV report."""
        path = generator.generate_csv(sample_summary)

        assert path.exists()
        assert path.suffix == ".csv"

        content = path.read_text()
        lines = content.strip().split("\n")

        # Header + 1 data row
        assert len(lines) == 2
        assert "audio_id" in lines[0]
        assert "test_audio_001" in lines[1]

    def test_generate_html(
        self, generator: ReportGenerator, sample_summary: BenchmarkSummary
    ) -> None:
        """Generate HTML report."""
        path = generator.generate_html(sample_summary)

        assert path.exists()
        assert path.suffix == ".html"

        content = path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "HSTTB Benchmark Report" in content
        assert "test_audio_001" in content

    def test_generate_clinical_risk(
        self, generator: ReportGenerator, sample_summary: BenchmarkSummary
    ) -> None:
        """Generate clinical risk report."""
        path = generator.generate_clinical_risk(sample_summary)

        assert path.exists()
        assert path.suffix == ".json"

        with open(path) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "total_files" in data
        assert "items" in data


class TestClinicalRiskAnalysis:
    """Tests for clinical risk analysis."""

    @pytest.fixture
    def summary_with_drug_error(
        self,
        sample_ner_result: NERResult,
        sample_crs_result: CRSResult,
    ) -> BenchmarkSummary:
        """Create summary with drug substitution error."""
        gt_term = MedicalTerm(
            text="metformin",
            normalized="metformin",
            category=MedicalTermCategory.DRUG,
            source="mock",
            span=(0, 9),
        )
        pred_term = MedicalTerm(
            text="methotrexate",
            normalized="methotrexate",
            category=MedicalTermCategory.DRUG,
            source="mock",
            span=(0, 12),
        )
        error = TermError(
            error_type=ErrorType.SUBSTITUTION,
            category=MedicalTermCategory.DRUG,
            ground_truth_term=gt_term,
            predicted_term=pred_term,
        )

        ter_result = TERResult(
            overall_ter=0.2,
            category_ter={"drug": 0.2},
            total_terms=5,
            substitutions=[error],
        )

        result = BenchmarkResult(
            audio_id="test",
            ter=ter_result,
            ner=sample_ner_result,
            crs=sample_crs_result,
            transcript_ground_truth="patient takes metformin",
            transcript_predicted="patient takes methotrexate",
            streaming_profile="ideal",
            adapter_name="mock",
        )

        return BenchmarkSummary(
            total_files=1,
            avg_ter=0.2,
            avg_ner_f1=0.875,
            avg_crs=0.92,
            results=[result],
            streaming_profile="ideal",
            adapter_name="mock",
        )

    def test_detects_drug_substitution(
        self, tmp_path: Path, summary_with_drug_error: BenchmarkSummary
    ) -> None:
        """Detect critical drug substitution errors."""
        generator = ReportGenerator(tmp_path)
        risk_report = generator._analyze_clinical_risks(summary_with_drug_error)

        assert risk_report.critical_count >= 1
        assert any(i.risk_type == "drug_substitution" for i in risk_report.items)

    def test_risk_report_structure(
        self, tmp_path: Path, summary_with_drug_error: BenchmarkSummary
    ) -> None:
        """Risk report has correct structure."""
        generator = ReportGenerator(tmp_path)
        risk_report = generator._analyze_clinical_risks(summary_with_drug_error)

        assert hasattr(risk_report, "timestamp")
        assert hasattr(risk_report, "total_files")
        assert hasattr(risk_report, "critical_count")
        assert hasattr(risk_report, "items")


# ==============================================================================
# Score Class Tests
# ==============================================================================


class TestScoreClasses:
    """Tests for score classification."""

    @pytest.fixture
    def generator(self, tmp_path: Path) -> ReportGenerator:
        """Create report generator."""
        return ReportGenerator(tmp_path)

    def test_ter_score_good(self, generator: ReportGenerator) -> None:
        """Low TER is good."""
        css_class = generator._get_score_class(0.05, lower_is_better=True)
        assert css_class == "score-good"

    def test_ter_score_bad(self, generator: ReportGenerator) -> None:
        """High TER is bad."""
        css_class = generator._get_score_class(0.5, lower_is_better=True)
        assert css_class == "score-bad"

    def test_ner_score_good(self, generator: ReportGenerator) -> None:
        """High NER F1 is good."""
        css_class = generator._get_score_class(0.95, lower_is_better=False)
        assert css_class == "score-good"

    def test_ner_score_bad(self, generator: ReportGenerator) -> None:
        """Low NER F1 is bad."""
        css_class = generator._get_score_class(0.5, lower_is_better=False)
        assert css_class == "score-bad"


# ==============================================================================
# Factory Function Tests
# ==============================================================================


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_basic_usage(
        self, tmp_path: Path, sample_summary: BenchmarkSummary
    ) -> None:
        """Basic usage of generate_report."""
        generated = generate_report(sample_summary, tmp_path)

        assert "json" in generated
        assert all(p.exists() for p in generated.values())

    def test_custom_config(
        self, tmp_path: Path, sample_summary: BenchmarkSummary
    ) -> None:
        """generate_report with custom config."""
        generated = generate_report(
            sample_summary,
            tmp_path,
            generate_csv=False,
            generate_html=False,
        )

        assert "json" in generated
        assert "csv" not in generated
        assert "html" not in generated


# ==============================================================================
# Empty Results Tests
# ==============================================================================


class TestEmptyResults:
    """Tests for handling empty results."""

    def test_empty_summary(self, tmp_path: Path) -> None:
        """Handle empty summary."""
        summary = BenchmarkSummary(
            total_files=0,
            avg_ter=0.0,
            avg_ner_f1=0.0,
            avg_crs=0.0,
            results=[],
            streaming_profile="ideal",
            adapter_name="mock",
        )

        generator = ReportGenerator(tmp_path)
        generated = generator.generate_all(summary)

        assert all(p.exists() for p in generated.values())

    def test_json_with_no_results(self, tmp_path: Path) -> None:
        """JSON report with no results."""
        summary = BenchmarkSummary(
            total_files=0,
            avg_ter=0.0,
            avg_ner_f1=0.0,
            avg_crs=0.0,
            results=[],
        )

        generator = ReportGenerator(tmp_path)
        path = generator.generate_json(summary)

        with open(path) as f:
            data = json.load(f)

        assert data["summary"]["total_files"] == 0
        assert len(data["results"]) == 0
