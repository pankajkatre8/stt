# Phase 2: TER Engine (Term Error Rate)

**Duration**: Week 3
**Status**: Pending
**Objective**: Implement medical term extraction and Term Error Rate computation.

## Phase Overview

TER measures the accuracy of medical terminology transcription. This is critical for healthcare because errors in drug names, dosages, or diagnoses can harm patients.

## Task List

| Task ID | Title | Status | Complexity | Blocks |
|---------|-------|--------|------------|--------|
| TASK-2T01 | Medical Lexicon Interface | pending | Medium | TASK-2T02+ |
| TASK-2T02 | RxNorm Loader | pending | Large | TASK-2T05 |
| TASK-2T03 | SNOMED CT Loader | pending | Large | TASK-2T05 |
| TASK-2T04 | ICD-10 Loader | pending | Medium | TASK-2T05 |
| TASK-2T05 | Unified Lexicon | pending | Medium | TASK-2T06 |
| TASK-2T06 | Text Normalizer | pending | Medium | TASK-2T07 |
| TASK-2T07 | Medical Term Extractor | pending | Large | TASK-2T08 |
| TASK-2T08 | TER Computation Engine | pending | Large | None |

## Dependencies

- **Requires**: Phase 1 complete (types, config)
- **Blocks**: Phase 5 (orchestration)

## Key Files

```
src/hsttb/
├── lexicons/
│   ├── base.py           # TASK-2T01
│   ├── rxnorm.py         # TASK-2T02
│   ├── snomed.py         # TASK-2T03
│   ├── icd10.py          # TASK-2T04
│   └── unified.py        # TASK-2T05
├── nlp/
│   └── normalizer.py     # TASK-2T06
└── metrics/
    ├── term_extractor.py # TASK-2T07
    └── ter.py            # TASK-2T08
```

## Healthcare Considerations

- Drug name accuracy is CRITICAL (life-threatening errors)
- Dosage errors can cause overdose/underdose
- Diagnosis errors affect treatment decisions
- Fuzzy matching must be tuned carefully

## Acceptance Criteria

- [ ] Can load RxNorm, SNOMED, ICD-10 data
- [ ] Unified lookup across all lexicons
- [ ] Text normalization handles abbreviations, case, plurals
- [ ] Term extraction finds medical terms in text
- [ ] TER computation identifies substitutions, deletions, insertions
- [ ] Category-wise TER breakdown available
- [ ] Unit tests with >90% coverage
