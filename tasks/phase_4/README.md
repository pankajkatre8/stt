# Phase 4: CRS Engine (Context Retention Score)

**Duration**: Week 5
**Status**: Pending
**Objective**: Implement context retention scoring for streaming continuity.

## Phase Overview

CRS measures whether the STT system preserves semantic and clinical context across streaming segments. This is the most complex metric because it involves multiple sub-scores.

## Task List

| Task ID | Title | Status | Complexity | Blocks |
|---------|-------|--------|------------|--------|
| TASK-4R01 | Sentence Embeddings Setup | pending | Medium | TASK-4R02 |
| TASK-4R02 | Semantic Similarity Engine | pending | Medium | TASK-4R05 |
| TASK-4R03 | Entity Continuity Tracker | pending | Large | TASK-4R05 |
| TASK-4R04 | Negation Consistency Checker | pending | Large | TASK-4R05 |
| TASK-4R05 | Temporal Consistency Checker | pending | Medium | TASK-4R06 |
| TASK-4R06 | Context Drift Detection | pending | Medium | TASK-4R07 |
| TASK-4R07 | CRS Composite Scoring | pending | Large | TASK-4R08 |
| TASK-4R08 | Segment-Level CRS | pending | Medium | None |

## Dependencies

- **Requires**: Phase 3 complete (NER for entity tracking)
- **Blocks**: Phase 5 (orchestration)

## Key Files

```
src/hsttb/
├── nlp/
│   ├── negation.py           # TASK-4R04
│   └── temporal.py           # TASK-4R05
└── metrics/
    ├── semantic_similarity.py # TASK-4R01, TASK-4R02
    ├── entity_continuity.py   # TASK-4R03
    ├── context_drift.py       # TASK-4R06
    └── crs.py                 # TASK-4R07, TASK-4R08
```

## CRS Components

| Component | Weight | Purpose |
|-----------|--------|---------|
| Semantic Similarity | 0.4 | Overall meaning preserved |
| Entity Continuity | 0.3 | Entities tracked across segments |
| Negation Consistency | 0.3 | Negations not flipped |

## Streaming Challenges

- **Chunk boundaries**: Entities may span chunks
- **Context loss**: Earlier context forgotten
- **Negation scope**: "no chest pain" split incorrectly
- **Entity drift**: Same entity changes attributes

## Healthcare Considerations

- Negation flips are dangerous ("no pain" → "pain")
- Temporal markers affect treatment timing
- Entity continuity affects care coordination
- Context drift can cause repeated information

## Acceptance Criteria

- [ ] Sentence embeddings computed efficiently
- [ ] Semantic similarity accurate for medical text
- [ ] Entity continuity tracked across segments
- [ ] Negation consistency detected
- [ ] Temporal consistency checked
- [ ] Context drift rate computed
- [ ] Composite CRS correctly weighted
- [ ] Segment-level scores available
- [ ] Unit tests with >90% coverage
