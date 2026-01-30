# Phase 6: Reporting & Hardening

**Duration**: Week 7
**Status**: Pending
**Objective**: Implement reporting, dashboards, and production hardening.

## Phase Overview

This final phase delivers actionable outputs: reports for stakeholders, dashboards for analysis, and production-quality code with comprehensive tests.

## Task List

| Task ID | Title | Status | Complexity | Blocks |
|---------|-------|--------|------------|--------|
| TASK-6P01 | JSON Report Generator | pending | Medium | TASK-6P03 |
| TASK-6P02 | Parquet Export | pending | Small | TASK-6P03 |
| TASK-6P03 | Clinical Risk Report | pending | Large | TASK-6P04 |
| TASK-6P04 | FastAPI Dashboard | pending | Large | TASK-6P05 |
| TASK-6P05 | Model Comparison Views | pending | Medium | TASK-6P06 |
| TASK-6P06 | Final Testing & Hardening | pending | Large | None |

## Dependencies

- **Requires**: Phase 5 complete (benchmark runner)
- **Blocks**: Production release

## Key Files

```
src/hsttb/
└── reporting/
    ├── generator.py     # TASK-6P01, TASK-6P02
    ├── clinical_risk.py # TASK-6P03
    └── dashboard.py     # TASK-6P04, TASK-6P05
```

## Report Types

### 1. JSON Report
Complete benchmark results in structured format.

### 2. Parquet Export
Analysis-friendly columnar format for pandas/spark.

### 3. Clinical Risk Report
Prioritizes errors by clinical impact:
- **Critical**: Drug substitutions, dosage errors
- **High**: Negation flips, diagnosis errors
- **Medium**: Anatomy confusion, timing errors

### 4. Dashboard
Interactive web interface for:
- Model comparison
- Metric drill-down
- Error analysis
- Regression tracking

## Report Structure

```json
{
  "benchmark_id": "run_001",
  "timestamp": "2024-01-01T00:00:00Z",
  "config": {...},
  "summary": {
    "total_files": 100,
    "avg_ter": 0.05,
    "avg_ner_f1": 0.92,
    "avg_crs": 0.88
  },
  "models_compared": [...],
  "clinical_risk": {
    "critical": [...],
    "high": [...],
    "medium": [...]
  },
  "results": [...]
}
```

## Dashboard Endpoints

| Endpoint | Purpose |
|----------|---------|
| GET /models | List benchmarked models |
| GET /compare | Compare model metrics |
| GET /results/{id} | Get run details |
| GET /clinical-risk/{id} | Get risk report |
| GET /trends | Show metric trends |

## Healthcare Considerations

- Clinical risk report is essential for stakeholders
- Drug/dosage errors must be prominently displayed
- Reports should be actionable (not just numbers)
- Dashboard access may need authentication

## Acceptance Criteria

- [ ] JSON report is valid and complete
- [ ] Parquet export is queryable
- [ ] Clinical risk report identifies critical errors
- [ ] Dashboard runs and displays data
- [ ] Model comparison works correctly
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] mypy --strict passes
- [ ] Security review complete
- [ ] Documentation complete

## Production Checklist

- [ ] Error handling is comprehensive
- [ ] Logging is appropriate (no PHI)
- [ ] Performance is acceptable
- [ ] Security is verified
- [ ] Documentation is complete
- [ ] README updated
- [ ] CHANGELOG updated
- [ ] Version bumped to 1.0.0
