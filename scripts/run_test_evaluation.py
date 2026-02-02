#!/usr/bin/env python3
"""
Run HSTTB evaluation on all test conversations.

Evaluates all generated test cases and produces comprehensive reports
showing how the benchmarking system performs.

Usage:
    python scripts/run_test_evaluation.py

Output:
    test_data/conversations/results/
    ├── summary_report.json     - Overall statistics
    ├── detailed_results.json   - Per-case results
    ├── category_reports/       - Category-specific reports
    │   ├── 01_perfect_transcriptions.json
    │   ├── 02_drug_name_errors.json
    │   └── ...
    └── evaluation_report.md    - Human-readable report
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


@dataclass
class EvaluationResult:
    """Result for a single test case."""
    case_id: str
    category: str
    name: str
    severity: str
    expected_ter_range: tuple[float, float]

    # Quality scores
    quality_composite: float | None = None
    quality_recommendation: str | None = None
    perplexity_score: float | None = None
    grammar_score: float | None = None
    entity_validity_score: float | None = None
    coherence_score: float | None = None

    # Transcription error detection
    transcription_error_score: float | None = None
    potential_errors_count: int = 0
    spelling_inconsistencies_count: int = 0
    known_terms_count: int = 0

    # Clinical risk
    clinical_risk_score: float | None = None
    clinical_risk_level: str | None = None
    clinical_recommendation: str | None = None

    # Reference-based (if ground truth)
    ter_score: float | None = None
    wer_score: float | None = None
    ner_f1: float | None = None

    # Meta
    evaluation_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    # Detected issues
    detected_drug_errors: list[dict] = field(default_factory=list)
    detected_dosage_issues: list[str] = field(default_factory=list)
    clinical_concerns: list[str] = field(default_factory=list)


def load_test_cases(test_dir: Path) -> list[dict]:
    """Load all test cases from directory."""
    cases = []

    for category_dir in sorted(test_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        if category_dir.name in ('results', '__pycache__'):
            continue

        for case_dir in sorted(category_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            try:
                gt_file = case_dir / "ground_truth.txt"
                trans_file = case_dir / "transcribed.txt"
                meta_file = case_dir / "metadata.json"

                if not all(f.exists() for f in [gt_file, trans_file, meta_file]):
                    continue

                cases.append({
                    "id": case_dir.name,
                    "category": category_dir.name,
                    "ground_truth": gt_file.read_text(),
                    "transcribed": trans_file.read_text(),
                    "metadata": json.loads(meta_file.read_text()),
                })
            except Exception as e:
                print(f"Error loading {case_dir}: {e}")

    return cases


def evaluate_case(case: dict, engines: dict) -> EvaluationResult:
    """Evaluate a single test case."""
    meta = case["metadata"]

    result = EvaluationResult(
        case_id=case["id"],
        category=case["category"],
        name=meta.get("name", ""),
        severity=meta.get("severity", "unknown"),
        expected_ter_range=tuple(meta.get("expected_ter_range", [0, 1])),
    )

    start_time = time.time()

    try:
        # Quality evaluation (reference-free)
        quality_engine = engines.get("quality")
        if quality_engine:
            quality_result = quality_engine.compute(case["transcribed"])

            result.quality_composite = quality_result.composite_score
            result.quality_recommendation = quality_result.recommendation
            result.perplexity_score = quality_result.perplexity_score
            result.grammar_score = quality_result.grammar_score
            result.entity_validity_score = quality_result.entity_validity_score
            result.coherence_score = quality_result.coherence_score

            # Transcription error detection
            result.transcription_error_score = quality_result.transcription_error_score
            result.potential_errors_count = len(quality_result.potential_transcription_errors)
            result.spelling_inconsistencies_count = len(quality_result.spelling_inconsistencies)
            result.known_terms_count = len(quality_result.known_terms_found)

            # Clinical risk
            result.clinical_risk_score = quality_result.clinical_risk_score
            result.clinical_risk_level = quality_result.clinical_risk_level
            result.clinical_recommendation = quality_result.clinical_recommendation

            # Detected issues
            result.detected_drug_errors = quality_result.potential_transcription_errors
            result.detected_dosage_issues = quality_result.dosage_issues
            result.clinical_concerns = quality_result.clinical_concerns

        # Reference-based evaluation (TER)
        ter_engine = engines.get("ter")
        if ter_engine:
            ter_result = ter_engine.compute(case["ground_truth"], case["transcribed"])
            result.ter_score = ter_result.overall_ter

        # WER calculation
        wer_func = engines.get("wer")
        if wer_func:
            wer_result = wer_func(case["ground_truth"], case["transcribed"])
            result.wer_score = wer_result

        # NER evaluation
        ner_engine = engines.get("ner")
        if ner_engine:
            try:
                ner_result = ner_engine.compute(case["ground_truth"], case["transcribed"])
                result.ner_f1 = ner_result.f1_score
            except Exception as e:
                result.errors.append(f"NER error: {e}")

    except Exception as e:
        result.errors.append(str(e))

    result.evaluation_time_ms = (time.time() - start_time) * 1000
    return result


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 1.0 if hyp_words else 0.0

    # Simple Levenshtein distance
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n] / m


def generate_report(results: list[EvaluationResult], output_dir: Path) -> dict:
    """Generate comprehensive evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    total = len(results)
    by_category = {}
    by_severity = {}

    for r in results:
        # By category
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)

        # By severity
        if r.severity not in by_severity:
            by_severity[r.severity] = []
        by_severity[r.severity].append(r)

    # Summary statistics
    summary = {
        "evaluation_date": datetime.now().isoformat(),
        "total_cases": total,
        "categories_evaluated": len(by_category),
        "overall_metrics": {},
        "by_severity": {},
        "by_category": {},
        "detection_accuracy": {},
    }

    # Overall metrics (excluding None values)
    quality_scores = [r.quality_composite for r in results if r.quality_composite is not None]
    ter_scores = [r.ter_score for r in results if r.ter_score is not None]
    wer_scores = [r.wer_score for r in results if r.wer_score is not None]
    clinical_scores = [r.clinical_risk_score for r in results if r.clinical_risk_score is not None]
    trans_error_scores = [r.transcription_error_score for r in results if r.transcription_error_score is not None]

    if quality_scores:
        summary["overall_metrics"]["avg_quality_score"] = round(sum(quality_scores) / len(quality_scores), 4)
    if ter_scores:
        summary["overall_metrics"]["avg_ter"] = round(sum(ter_scores) / len(ter_scores), 4)
    if wer_scores:
        summary["overall_metrics"]["avg_wer"] = round(sum(wer_scores) / len(wer_scores), 4)
    if clinical_scores:
        summary["overall_metrics"]["avg_clinical_risk_score"] = round(sum(clinical_scores) / len(clinical_scores), 4)
    if trans_error_scores:
        summary["overall_metrics"]["avg_transcription_error_score"] = round(sum(trans_error_scores) / len(trans_error_scores), 4)

    # By severity statistics
    for severity, cases in by_severity.items():
        quality = [r.quality_composite for r in cases if r.quality_composite is not None]
        ter = [r.ter_score for r in cases if r.ter_score is not None]
        clinical = [r.clinical_risk_score for r in cases if r.clinical_risk_score is not None]

        summary["by_severity"][severity] = {
            "count": len(cases),
            "avg_quality": round(sum(quality) / len(quality), 4) if quality else None,
            "avg_ter": round(sum(ter) / len(ter), 4) if ter else None,
            "avg_clinical_risk": round(sum(clinical) / len(clinical), 4) if clinical else None,
        }

    # By category statistics
    for category, cases in by_category.items():
        quality = [r.quality_composite for r in cases if r.quality_composite is not None]
        ter = [r.ter_score for r in cases if r.ter_score is not None]
        clinical = [r.clinical_risk_score for r in cases if r.clinical_risk_score is not None]
        trans_err = [r.transcription_error_score for r in cases if r.transcription_error_score is not None]
        errors_detected = sum(r.potential_errors_count for r in cases)

        summary["by_category"][category] = {
            "count": len(cases),
            "avg_quality": round(sum(quality) / len(quality), 4) if quality else None,
            "avg_ter": round(sum(ter) / len(ter), 4) if ter else None,
            "avg_clinical_risk": round(sum(clinical) / len(clinical), 4) if clinical else None,
            "avg_trans_error_score": round(sum(trans_err) / len(trans_err), 4) if trans_err else None,
            "total_errors_detected": errors_detected,
        }

    # Detection accuracy analysis
    # For error categories, check if we detected errors
    error_categories = [
        "02_drug_name_errors",
        "03_dosage_errors",
        "04_negation_flips",
        "05_medical_condition_errors",
        "07_spelling_inconsistencies",
        "08_multiple_errors",
    ]

    detected_correctly = 0
    total_error_cases = 0

    for category in error_categories:
        if category in by_category:
            for r in by_category[category]:
                total_error_cases += 1
                # Consider detected if quality score is low or errors were found
                if (r.quality_composite and r.quality_composite < 0.8) or \
                   r.potential_errors_count > 0 or \
                   (r.ter_score and r.ter_score > 0.05):
                    detected_correctly += 1

    if total_error_cases > 0:
        summary["detection_accuracy"]["error_detection_rate"] = round(detected_correctly / total_error_cases, 4)
        summary["detection_accuracy"]["total_error_cases"] = total_error_cases
        summary["detection_accuracy"]["detected_correctly"] = detected_correctly

    # Perfect transcription accuracy
    perfect_cases = by_category.get("01_perfect_transcriptions", [])
    if perfect_cases:
        perfect_detected = sum(1 for r in perfect_cases
                               if r.quality_composite and r.quality_composite >= 0.8
                               and r.potential_errors_count == 0)
        summary["detection_accuracy"]["perfect_accuracy"] = round(perfect_detected / len(perfect_cases), 4)

    # Write summary report
    summary_file = output_dir / "summary_report.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    # Write detailed results
    detailed_results = []
    for r in results:
        result_dict = {
            "case_id": r.case_id,
            "category": r.category,
            "name": r.name,
            "severity": r.severity,
            "expected_ter_range": r.expected_ter_range,
            "quality_composite": r.quality_composite,
            "quality_recommendation": r.quality_recommendation,
            "perplexity_score": r.perplexity_score,
            "grammar_score": r.grammar_score,
            "entity_validity_score": r.entity_validity_score,
            "coherence_score": r.coherence_score,
            "transcription_error_score": r.transcription_error_score,
            "potential_errors_count": r.potential_errors_count,
            "spelling_inconsistencies_count": r.spelling_inconsistencies_count,
            "known_terms_count": r.known_terms_count,
            "clinical_risk_score": r.clinical_risk_score,
            "clinical_risk_level": r.clinical_risk_level,
            "clinical_recommendation": r.clinical_recommendation,
            "ter_score": r.ter_score,
            "wer_score": r.wer_score,
            "ner_f1": r.ner_f1,
            "evaluation_time_ms": r.evaluation_time_ms,
            "errors": r.errors,
            "detected_drug_errors": r.detected_drug_errors,
            "detected_dosage_issues": r.detected_dosage_issues,
            "clinical_concerns": r.clinical_concerns,
        }
        detailed_results.append(result_dict)

    detailed_file = output_dir / "detailed_results.json"
    detailed_file.write_text(json.dumps(detailed_results, indent=2))

    # Write category reports
    category_dir = output_dir / "category_reports"
    category_dir.mkdir(exist_ok=True)

    for category, cases in by_category.items():
        cat_results = [r for r in detailed_results if r["category"] == category]
        cat_file = category_dir / f"{category}.json"
        cat_file.write_text(json.dumps(cat_results, indent=2))

    # Generate markdown report
    report_md = generate_markdown_report(summary, results, by_category, by_severity)
    report_file = output_dir / "evaluation_report.md"
    report_file.write_text(report_md)

    return summary


def generate_markdown_report(
    summary: dict,
    results: list[EvaluationResult],
    by_category: dict,
    by_severity: dict,
) -> str:
    """Generate human-readable markdown report."""

    md = f"""# HSTTB Evaluation Report

**Generated:** {summary['evaluation_date']}
**Total Test Cases:** {summary['total_cases']}
**Categories Evaluated:** {summary['categories_evaluated']}

---

## Executive Summary

"""

    # Overall metrics
    metrics = summary.get("overall_metrics", {})
    if metrics:
        md += "### Overall Performance\n\n"
        md += "| Metric | Value |\n"
        md += "|--------|-------|\n"
        for key, value in metrics.items():
            display_key = key.replace("_", " ").title()
            if value is not None:
                if "score" in key.lower() or "ter" in key.lower() or "wer" in key.lower():
                    md += f"| {display_key} | {value:.2%} |\n"
                else:
                    md += f"| {display_key} | {value} |\n"
        md += "\n"

    # Detection accuracy
    detection = summary.get("detection_accuracy", {})
    if detection:
        md += "### Detection Accuracy\n\n"
        if "error_detection_rate" in detection:
            md += f"- **Error Detection Rate:** {detection['error_detection_rate']:.1%} "
            md += f"({detection['detected_correctly']}/{detection['total_error_cases']} error cases detected)\n"
        if "perfect_accuracy" in detection:
            md += f"- **Perfect Transcription Accuracy:** {detection['perfect_accuracy']:.1%}\n"
        md += "\n"

    # Results by severity
    md += "---\n\n## Results by Severity\n\n"
    md += "| Severity | Count | Avg Quality | Avg TER | Avg Clinical Risk |\n"
    md += "|----------|-------|-------------|---------|-------------------|\n"

    severity_order = ["none", "low", "medium", "high", "critical"]
    for sev in severity_order:
        if sev in summary.get("by_severity", {}):
            data = summary["by_severity"][sev]
            quality = f"{data['avg_quality']:.2%}" if data['avg_quality'] else "N/A"
            ter = f"{data['avg_ter']:.2%}" if data['avg_ter'] else "N/A"
            clinical = f"{data['avg_clinical_risk']:.2%}" if data['avg_clinical_risk'] else "N/A"
            md += f"| {sev.upper()} | {data['count']} | {quality} | {ter} | {clinical} |\n"

    md += "\n"

    # Results by category
    md += "---\n\n## Results by Category\n\n"

    for category in sorted(by_category.keys()):
        cat_data = summary.get("by_category", {}).get(category, {})
        cases = by_category[category]

        md += f"### {category.replace('_', ' ').title()}\n\n"
        md += f"**Cases:** {len(cases)}\n\n"

        if cat_data:
            md += "| Metric | Value |\n"
            md += "|--------|-------|\n"
            if cat_data.get("avg_quality"):
                md += f"| Avg Quality Score | {cat_data['avg_quality']:.2%} |\n"
            if cat_data.get("avg_ter"):
                md += f"| Avg TER | {cat_data['avg_ter']:.2%} |\n"
            if cat_data.get("avg_clinical_risk"):
                md += f"| Avg Clinical Risk | {cat_data['avg_clinical_risk']:.2%} |\n"
            if cat_data.get("avg_trans_error_score"):
                md += f"| Avg Transcription Error Score | {cat_data['avg_trans_error_score']:.2%} |\n"
            if cat_data.get("total_errors_detected"):
                md += f"| Total Errors Detected | {cat_data['total_errors_detected']} |\n"
            md += "\n"

        # Individual case results table
        md += "<details>\n<summary>Individual Case Results</summary>\n\n"
        md += "| Case ID | Name | TER | Quality | Clinical Risk | Errors Detected |\n"
        md += "|---------|------|-----|---------|---------------|----------------|\n"

        for r in cases:
            ter = f"{r.ter_score:.2%}" if r.ter_score is not None else "N/A"
            quality = f"{r.quality_composite:.2%}" if r.quality_composite else "N/A"
            clinical = f"{r.clinical_risk_score:.2%}" if r.clinical_risk_score else "N/A"
            md += f"| {r.case_id} | {r.name[:30]}... | {ter} | {quality} | {clinical} | {r.potential_errors_count} |\n"

        md += "\n</details>\n\n"

    # Critical findings
    md += "---\n\n## Critical Findings\n\n"

    critical_cases = by_severity.get("critical", [])
    if critical_cases:
        md += f"### Critical Severity Cases ({len(critical_cases)} total)\n\n"
        for r in critical_cases:
            status = "DETECTED" if (r.quality_composite and r.quality_composite < 0.7) or r.potential_errors_count > 0 else "MISSED"
            md += f"- **{r.name}** ({r.case_id}): Quality={r.quality_composite:.2%}, "
            md += f"Clinical Risk={r.clinical_risk_score:.2%}, Status={status}\n"
            if r.clinical_concerns:
                for concern in r.clinical_concerns[:2]:
                    md += f"  - {concern}\n"
        md += "\n"

    # Recommendations
    md += "---\n\n## Recommendations\n\n"

    # Analyze detection gaps
    missed_critical = [r for r in critical_cases
                       if r.quality_composite and r.quality_composite >= 0.7
                       and r.potential_errors_count == 0]

    if missed_critical:
        md += f"### Areas for Improvement\n\n"
        md += f"- **{len(missed_critical)} critical cases** were not detected with current thresholds\n"
        md += "- Consider adjusting detection sensitivity for:\n"
        categories_missed = set(r.category for r in missed_critical)
        for cat in categories_missed:
            md += f"  - {cat.replace('_', ' ')}\n"
        md += "\n"

    # Performance summary
    md += "### Performance Summary\n\n"

    error_rate = detection.get("error_detection_rate", 0)
    if error_rate >= 0.9:
        md += "- System shows **excellent** error detection capability\n"
    elif error_rate >= 0.7:
        md += "- System shows **good** error detection, with room for improvement\n"
    else:
        md += "- System needs **significant improvement** in error detection\n"

    md += "\n---\n\n*Report generated by HSTTB Evaluation Framework*\n"

    return md


def main():
    """Run evaluation on all test cases."""
    print("HSTTB Test Conversation Evaluation")
    print("=" * 50)

    # Setup paths
    test_dir = project_root / "test_data" / "conversations"

    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        print("Run generate_test_conversations.py first.")
        return

    # Load test cases
    print("\nLoading test cases...")
    cases = load_test_cases(test_dir)
    print(f"Loaded {len(cases)} test cases")

    # Initialize evaluation engines
    print("\nInitializing evaluation engines...")
    engines = {}

    # Quality engine
    try:
        from hsttb.metrics.quality import QualityEngine
        engines["quality"] = QualityEngine(
            use_perplexity=False,  # Skip for faster evaluation
            use_grammar=False,
            use_embedding_drift=False,
            use_confidence_variance=False,
        )
        print("  - Quality engine: OK")
    except Exception as e:
        print(f"  - Quality engine: FAILED ({e})")

    # TER engine
    try:
        from hsttb.metrics.ter import TEREngine
        from hsttb.lexicons.sqlite_lexicon import SQLiteMedicalLexicon

        lexicon = SQLiteMedicalLexicon()
        lexicon.load("auto")
        engines["ter"] = TEREngine(lexicon)
        print("  - TER engine: OK")
    except Exception as e:
        print(f"  - TER engine: FAILED ({e})")

    # WER function
    engines["wer"] = compute_wer
    print("  - WER function: OK")

    # NER engine (optional, may not be installed)
    try:
        from hsttb.metrics.ner import NEREngine
        from hsttb.nlp.scispacy_ner import SciSpacyNERPipeline
        pipeline = SciSpacyNERPipeline()
        engines["ner"] = NEREngine(pipeline)
        print("  - NER engine: OK")
    except Exception as e:
        print(f"  - NER engine: SKIPPED ({e})")

    # Run evaluation
    print("\nEvaluating test cases...")
    results = []
    start_time = time.time()

    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']}: ", end="", flush=True)
        result = evaluate_case(case, engines)
        results.append(result)

        if result.errors:
            print(f"ERRORS ({len(result.errors)})")
        elif result.quality_composite:
            print(f"Quality={result.quality_composite:.1%}, TER={result.ter_score:.1%}")
        else:
            print("DONE")

    total_time = time.time() - start_time
    print(f"\nEvaluation completed in {total_time:.1f} seconds")

    # Generate report
    print("\nGenerating reports...")
    output_dir = test_dir / "results"
    summary = generate_report(results, output_dir)

    print(f"\nReports written to: {output_dir}")
    print(f"\nKey files:")
    print(f"  - summary_report.json    : Overall statistics")
    print(f"  - detailed_results.json  : Per-case detailed results")
    print(f"  - evaluation_report.md   : Human-readable report")
    print(f"  - category_reports/      : Category-specific results")

    # Print quick summary
    print(f"\n{'=' * 50}")
    print("QUICK SUMMARY")
    print(f"{'=' * 50}")

    metrics = summary.get("overall_metrics", {})
    if metrics.get("avg_quality_score"):
        print(f"Average Quality Score: {metrics['avg_quality_score']:.2%}")
    if metrics.get("avg_ter"):
        print(f"Average TER: {metrics['avg_ter']:.2%}")
    if metrics.get("avg_clinical_risk_score"):
        print(f"Average Clinical Risk Score: {metrics['avg_clinical_risk_score']:.2%}")

    detection = summary.get("detection_accuracy", {})
    if detection.get("error_detection_rate"):
        print(f"Error Detection Rate: {detection['error_detection_rate']:.2%}")


if __name__ == "__main__":
    main()
