"""
Report generation for benchmark results.

This module provides report generation in multiple formats:
- JSON: Machine-readable detailed results
- CSV: Tabular data for spreadsheet analysis
- HTML: Human-readable summary reports
- Clinical Risk: Identifies critical transcription errors

Example:
    >>> from hsttb.reporting.generator import ReportGenerator
    >>> generator = ReportGenerator(output_dir=Path("results"))
    >>> generator.generate_all(summary)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hsttb.core.types import BenchmarkResult, BenchmarkSummary, MedicalTermCategory


@dataclass
class ReportConfig:
    """
    Configuration for report generation.

    Attributes:
        generate_json: Generate JSON report.
        generate_csv: Generate CSV report.
        generate_html: Generate HTML report.
        generate_clinical_risk: Generate clinical risk report.
        include_detailed_errors: Include detailed error information.
        include_segments: Include segment-level scores.
    """

    generate_json: bool = True
    generate_csv: bool = True
    generate_html: bool = True
    generate_clinical_risk: bool = True
    include_detailed_errors: bool = True
    include_segments: bool = False


@dataclass
class ClinicalRiskItem:
    """
    A clinical risk item identified in transcription.

    Attributes:
        risk_level: Severity level (critical, high, medium, low).
        risk_type: Type of risk (drug_substitution, negation_flip, etc.).
        audio_id: Source audio file.
        description: Human-readable description.
        original: Original/ground truth value.
        predicted: Predicted value.
        category: Medical category if applicable.
    """

    risk_level: str
    risk_type: str
    audio_id: str
    description: str
    original: str | None = None
    predicted: str | None = None
    category: str | None = None


@dataclass
class ClinicalRiskReport:
    """
    Clinical risk report.

    Attributes:
        timestamp: Report generation time.
        total_files: Number of files analyzed.
        critical_count: Number of critical risks.
        high_count: Number of high risks.
        medium_count: Number of medium risks.
        low_count: Number of low risks.
        items: List of risk items.
    """

    timestamp: str
    total_files: int
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    items: list[ClinicalRiskItem] = field(default_factory=list)

    @property
    def total_risks(self) -> int:
        """Total number of risks."""
        return self.critical_count + self.high_count + self.medium_count + self.low_count


class ReportGenerator:
    """
    Generate reports from benchmark results.

    Produces reports in multiple formats for different use cases:
    - JSON for programmatic access
    - CSV for spreadsheet analysis
    - HTML for human viewing
    - Clinical Risk for safety analysis

    Attributes:
        output_dir: Directory for generated reports.
        config: Report generation configuration.

    Example:
        >>> generator = ReportGenerator(Path("results"))
        >>> generator.generate_all(summary)
        >>> # Creates: results.json, results.csv, report.html, clinical_risk.json
    """

    def __init__(
        self,
        output_dir: Path | str,
        config: ReportConfig | None = None,
    ) -> None:
        """
        Initialize the report generator.

        Args:
            output_dir: Directory for generated reports.
            config: Report generation configuration.
        """
        self.output_dir = Path(output_dir)
        self.config = config or ReportConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, summary: BenchmarkSummary) -> dict[str, Path]:
        """
        Generate all configured reports.

        Args:
            summary: Benchmark summary to report on.

        Returns:
            Dictionary mapping report type to output path.
        """
        generated: dict[str, Path] = {}

        if self.config.generate_json:
            generated["json"] = self.generate_json(summary)

        if self.config.generate_csv:
            generated["csv"] = self.generate_csv(summary)

        if self.config.generate_html:
            generated["html"] = self.generate_html(summary)

        if self.config.generate_clinical_risk:
            generated["clinical_risk"] = self.generate_clinical_risk(summary)

        return generated

    def generate_json(self, summary: BenchmarkSummary) -> Path:
        """
        Generate JSON report.

        Args:
            summary: Benchmark summary.

        Returns:
            Path to generated JSON file.
        """
        output_path = self.output_dir / "results.json"

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files": summary.total_files,
                "avg_ter": summary.avg_ter,
                "avg_ner_f1": summary.avg_ner_f1,
                "avg_crs": summary.avg_crs,
                "streaming_profile": summary.streaming_profile,
                "adapter_name": summary.adapter_name,
            },
            "results": [self._result_to_dict(r) for r in summary.results],
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return output_path

    def generate_csv(self, summary: BenchmarkSummary) -> Path:
        """
        Generate CSV report.

        Args:
            summary: Benchmark summary.

        Returns:
            Path to generated CSV file.
        """
        output_path = self.output_dir / "results.csv"

        # Build CSV content
        headers = [
            "audio_id",
            "ter",
            "ner_f1",
            "crs",
            "streaming_profile",
            "adapter_name",
        ]

        lines = [",".join(headers)]

        for result in summary.results:
            row = [
                result.audio_id,
                f"{result.ter.overall_ter:.4f}",
                f"{result.ner.f1_score:.4f}",
                f"{result.crs.composite_score:.4f}",
                result.streaming_profile,
                result.adapter_name,
            ]
            lines.append(",".join(row))

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return output_path

    def generate_html(self, summary: BenchmarkSummary) -> Path:
        """
        Generate HTML summary report.

        Args:
            summary: Benchmark summary.

        Returns:
            Path to generated HTML file.
        """
        output_path = self.output_dir / "report.html"

        html = self._build_html_report(summary)

        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def generate_clinical_risk(self, summary: BenchmarkSummary) -> Path:
        """
        Generate clinical risk report.

        Identifies critical transcription errors that could
        impact patient safety.

        Args:
            summary: Benchmark summary.

        Returns:
            Path to generated risk report.
        """
        output_path = self.output_dir / "clinical_risk.json"

        risk_report = self._analyze_clinical_risks(summary)

        with open(output_path, "w") as f:
            json.dump(asdict(risk_report), f, indent=2, default=str)

        return output_path

    def _result_to_dict(self, result: BenchmarkResult) -> dict[str, Any]:
        """Convert BenchmarkResult to dictionary."""
        data: dict[str, Any] = {
            "audio_id": result.audio_id,
            "ter": {
                "overall_ter": result.ter.overall_ter,
                "category_ter": result.ter.category_ter,
                "total_terms": result.ter.total_terms,
            },
            "ner": {
                "precision": result.ner.precision,
                "recall": result.ner.recall,
                "f1_score": result.ner.f1_score,
                "entity_distortion_rate": result.ner.entity_distortion_rate,
                "entity_omission_rate": result.ner.entity_omission_rate,
            },
            "crs": {
                "composite_score": result.crs.composite_score,
                "semantic_similarity": result.crs.semantic_similarity,
                "entity_continuity": result.crs.entity_continuity,
                "negation_consistency": result.crs.negation_consistency,
            },
            "transcript_ground_truth": result.transcript_ground_truth,
            "transcript_predicted": result.transcript_predicted,
            "streaming_profile": result.streaming_profile,
            "adapter_name": result.adapter_name,
        }

        if self.config.include_detailed_errors:
            data["errors"] = {
                "substitutions": len(result.ter.substitutions),
                "deletions": len(result.ter.deletions),
                "insertions": len(result.ter.insertions),
            }

        return data

    def _build_html_report(self, summary: BenchmarkSummary) -> str:
        """Build HTML report content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build results table rows
        result_rows = []
        for result in summary.results:
            ter_class = self._get_score_class(result.ter.overall_ter, lower_is_better=True)
            ner_class = self._get_score_class(result.ner.f1_score, lower_is_better=False)
            crs_class = self._get_score_class(result.crs.composite_score, lower_is_better=False)

            result_rows.append(f"""
                <tr>
                    <td>{result.audio_id}</td>
                    <td class="{ter_class}">{result.ter.overall_ter:.2%}</td>
                    <td class="{ner_class}">{result.ner.f1_score:.2%}</td>
                    <td class="{crs_class}">{result.crs.composite_score:.2%}</td>
                </tr>
            """)

        results_table = "\n".join(result_rows)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lunagen STT Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2 {{ color: #333; }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card-title {{ font-size: 14px; color: #666; margin-bottom: 8px; }}
        .card-value {{ font-size: 28px; font-weight: bold; color: #333; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f8f8; font-weight: 600; }}
        .score-good {{ color: #22c55e; }}
        .score-warning {{ color: #f59e0b; }}
        .score-bad {{ color: #ef4444; }}
        .timestamp {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>Lunagen STT Benchmark Report</h1>
    <p class="timestamp">Generated: {timestamp}</p>
    <p>Streaming Profile: <strong>{summary.streaming_profile}</strong> | Adapter: <strong>{summary.adapter_name}</strong></p>

    <h2>Summary</h2>
    <div class="summary-cards">
        <div class="card">
            <div class="card-title">Total Files</div>
            <div class="card-value">{summary.total_files}</div>
        </div>
        <div class="card">
            <div class="card-title">Average TER</div>
            <div class="card-value">{summary.avg_ter:.2%}</div>
        </div>
        <div class="card">
            <div class="card-title">Average NER F1</div>
            <div class="card-value">{summary.avg_ner_f1:.2%}</div>
        </div>
        <div class="card">
            <div class="card-title">Average CRS</div>
            <div class="card-value">{summary.avg_crs:.2%}</div>
        </div>
    </div>

    <h2>Results</h2>
    <table>
        <thead>
            <tr>
                <th>Audio ID</th>
                <th>TER</th>
                <th>NER F1</th>
                <th>CRS</th>
            </tr>
        </thead>
        <tbody>
            {results_table}
        </tbody>
    </table>
</body>
</html>"""

        return html

    def _get_score_class(self, score: float, lower_is_better: bool) -> str:
        """Get CSS class based on score quality."""
        if lower_is_better:
            if score < 0.1:
                return "score-good"
            elif score < 0.3:
                return "score-warning"
            return "score-bad"
        else:
            if score > 0.9:
                return "score-good"
            elif score > 0.7:
                return "score-warning"
            return "score-bad"

    def _analyze_clinical_risks(self, summary: BenchmarkSummary) -> ClinicalRiskReport:
        """Analyze clinical risks in transcription results."""
        items: list[ClinicalRiskItem] = []

        for result in summary.results:
            # Drug substitution errors (critical)
            for error in result.ter.substitutions:
                if error.category == MedicalTermCategory.DRUG:
                    gt_text = error.ground_truth_term.text if error.ground_truth_term else "?"
                    pred_text = error.predicted_term.text if error.predicted_term else "?"
                    items.append(
                        ClinicalRiskItem(
                            risk_level="critical",
                            risk_type="drug_substitution",
                            audio_id=result.audio_id,
                            description=f"Drug name substitution: '{gt_text}' -> '{pred_text}'",
                            original=gt_text,
                            predicted=pred_text,
                            category="drug",
                        )
                    )

            # Drug deletions (critical)
            for error in result.ter.deletions:
                if error.category == MedicalTermCategory.DRUG:
                    gt_text = error.ground_truth_term.text if error.ground_truth_term else "?"
                    items.append(
                        ClinicalRiskItem(
                            risk_level="critical",
                            risk_type="drug_omission",
                            audio_id=result.audio_id,
                            description=f"Drug name omitted: '{gt_text}'",
                            original=gt_text,
                            predicted=None,
                            category="drug",
                        )
                    )

            # Dosage errors (high)
            for error in result.ter.substitutions + result.ter.deletions:
                if error.category == MedicalTermCategory.DOSAGE:
                    gt_text = error.ground_truth_term.text if error.ground_truth_term else "?"
                    pred_text = error.predicted_term.text if error.predicted_term else None
                    items.append(
                        ClinicalRiskItem(
                            risk_level="high",
                            risk_type="dosage_error",
                            audio_id=result.audio_id,
                            description=f"Dosage error: '{gt_text}' -> '{pred_text or 'omitted'}'",
                            original=gt_text,
                            predicted=pred_text,
                            category="dosage",
                        )
                    )

            # Negation flips (high)
            for seg_score in result.crs.segment_scores:
                if seg_score.negation_flips > 0:
                    items.append(
                        ClinicalRiskItem(
                            risk_level="high",
                            risk_type="negation_flip",
                            audio_id=result.audio_id,
                            description=f"Negation flip in segment: '{seg_score.ground_truth_text[:50]}...'",
                            original=seg_score.ground_truth_text,
                            predicted=seg_score.predicted_text,
                            category=None,
                        )
                    )

        # Count by severity
        critical = sum(1 for i in items if i.risk_level == "critical")
        high = sum(1 for i in items if i.risk_level == "high")
        medium = sum(1 for i in items if i.risk_level == "medium")
        low = sum(1 for i in items if i.risk_level == "low")

        return ClinicalRiskReport(
            timestamp=datetime.now().isoformat(),
            total_files=summary.total_files,
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            items=items,
        )


def generate_report(
    summary: BenchmarkSummary,
    output_dir: Path | str,
    **kwargs: object,
) -> dict[str, Path]:
    """
    Convenience function to generate all reports.

    Args:
        summary: Benchmark summary.
        output_dir: Output directory.
        **kwargs: Additional config options.

    Returns:
        Dictionary mapping report type to path.
    """
    config = ReportConfig(**kwargs)  # type: ignore[arg-type]
    generator = ReportGenerator(output_dir, config)
    return generator.generate_all(summary)
