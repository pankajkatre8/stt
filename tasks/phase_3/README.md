# Phase 3: NER Engine (Named Entity Recognition)

**Duration**: Week 4
**Status**: Pending
**Objective**: Implement medical entity extraction and NER accuracy metrics.

## Phase Overview

NER accuracy measures whether the STT output preserves structured medical entities correctly. Even if text sounds right, entities must be accurate for clinical use.

## Task List

| Task ID | Title | Status | Complexity | Blocks |
|---------|-------|--------|------------|--------|
| TASK-3N01 | scispaCy Integration | pending | Medium | TASK-3N02 |
| TASK-3N02 | medspacy Integration | pending | Medium | TASK-3N03 |
| TASK-3N03 | Medical NER Pipeline | pending | Large | TASK-3N04 |
| TASK-3N04 | Entity Alignment Logic | pending | Large | TASK-3N05 |
| TASK-3N05 | NER Metrics Computation | pending | Large | TASK-3N06 |
| TASK-3N06 | Per-Entity-Type Breakdown | pending | Medium | None |

## Dependencies

- **Requires**: Phase 1 complete (types)
- **Blocks**: Phase 4 (CRS), Phase 5 (orchestration)

## Key Files

```
src/hsttb/
├── nlp/
│   ├── medical_ner.py      # TASK-3N01, TASK-3N02, TASK-3N03
│   └── entity_alignment.py  # TASK-3N04
└── metrics/
    └── ner.py              # TASK-3N05, TASK-3N06
```

## Entity Types

| Type | Examples | Clinical Impact |
|------|----------|-----------------|
| DRUG | metformin, aspirin | Medication errors |
| DOSAGE | 500mg, twice daily | Overdose/underdose |
| SYMPTOM | chest pain, fever | Diagnosis |
| DIAGNOSIS | diabetes, hypertension | Treatment |
| ANATOMY | left arm, chest | Procedure accuracy |
| LAB_VALUE | glucose 120 | Monitoring |

## Healthcare Considerations

- Entity span drift must be handled
- Partial matches need careful scoring
- Negated entities must be flagged
- Entity distortion is often worse than omission

## Acceptance Criteria

- [ ] scispaCy models loaded and working
- [ ] medspacy context detection working
- [ ] Medical NER extracts all entity types
- [ ] Entity alignment handles span drift
- [ ] Precision/Recall/F1 computed correctly
- [ ] Distortion and omission rates computed
- [ ] Per-entity-type breakdown available
- [ ] Unit tests with >90% coverage
