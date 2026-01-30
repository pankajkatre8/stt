# Phase 5: Evaluation Orchestration

**Duration**: Week 6
**Status**: Pending
**Objective**: Implement benchmark runner, SRS metric, and experiment tracking.

## Phase Overview

This phase brings all metrics together into a unified benchmark runner. It also implements the SRS (Streaming Robustness Score) which validates that benchmark results reflect model quality, not streaming artifacts.

## Task List

| Task ID | Title | Status | Complexity | Blocks |
|---------|-------|--------|------------|--------|
| TASK-5E01 | Ground Truth Loader | pending | Medium | TASK-5E02 |
| TASK-5E02 | Benchmark Runner Core | pending | Large | TASK-5E03+ |
| TASK-5E03 | Session-Level Evaluation | pending | Medium | TASK-5E04 |
| TASK-5E04 | Segment-Level Evaluation | pending | Medium | TASK-5E05 |
| TASK-5E05 | SRS Computation | pending | Large | TASK-5E06 |
| TASK-5E06 | MLflow Integration | pending | Medium | None |

## Dependencies

- **Requires**: Phases 2, 3, 4 complete (all metrics)
- **Blocks**: Phase 6 (reporting)

## Key Files

```
src/hsttb/
├── evaluation/
│   ├── runner.py        # TASK-5E02
│   ├── session.py       # TASK-5E03
│   ├── segment.py       # TASK-5E04
│   └── ground_truth.py  # TASK-5E01
├── metrics/
│   └── srs.py           # TASK-5E05
└── tracking/
    └── mlflow.py        # TASK-5E06
```

## SRS Metric (Critical)

### Purpose
Streaming Robustness Score validates that quality differences are due to the model, not streaming simulation.

### Computation
```
SRS = Score(realtime_profile) / Score(ideal_profile)
```

### Interpretation
- SRS ≈ 1.0: Model is robust to streaming
- SRS < 0.9: Model degrades under streaming
- Use to compare models fairly

## Benchmark Flow

```
1. Load audio files
2. Load ground truth transcripts
3. For each (audio, ground_truth):
   a. Stream audio through chunker
   b. Transcribe with STT adapter
   c. Compute TER, NER, CRS
   d. Aggregate results
4. Compute summary statistics
5. Log to MLflow
6. Generate report
```

## Healthcare Considerations

- Benchmark should identify clinically critical failures
- SRS helps ensure fair model comparison
- Results must be reproducible
- Audit trail for experiment tracking

## Acceptance Criteria

- [ ] Ground truth loader handles various formats
- [ ] Benchmark runner processes directory of files
- [ ] Session-level metrics computed
- [ ] Segment-level metrics computed
- [ ] SRS correctly compares profiles
- [ ] MLflow tracks experiments
- [ ] Results are reproducible (same seed = same results)
- [ ] Errors don't crash entire benchmark
- [ ] Unit and integration tests complete
