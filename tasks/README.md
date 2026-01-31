# Task Management

This directory contains task breakdowns for the HSTTB project.

## Structure

```
tasks/
├── README.md           # This file
├── phase_1/            # Foundation & Infrastructure
├── phase_2/            # TER Engine
├── phase_3/            # NER Engine
├── phase_4/            # CRS Engine
├── phase_5/            # Orchestration
├── phase_6/            # Reporting & Hardening
└── phase_7/            # Multi-Adapter & Enhanced UI
```

## Task Naming Convention

```
TASK-[PHASE][COMPONENT][NUMBER].md

Examples:
- TASK-1A01.md  → Phase 1, Audio, Task 01
- TASK-2T03.md  → Phase 2, TER, Task 03
- TASK-3N02.md  → Phase 3, NER, Task 02
```

### Component Codes

| Code | Component |
|------|-----------|
| A | Audio |
| S | STT/Adapters |
| C | Core/Config |
| T | TER |
| N | NER |
| R | CRS (Retention) |
| E | Evaluation |
| P | Reporting |
| X | Cross-cutting |

## Task Status

| Status | Meaning |
|--------|---------|
| `pending` | Not started |
| `in_progress` | Currently being worked on |
| `review` | Code complete, under review |
| `blocked` | Waiting on dependency |
| `completed` | Done and verified |

## Workflow

1. **Planner**: Creates task files with requirements
2. **Developer**: Picks up task, marks `in_progress`
3. **Developer**: Implements, creates PR
4. **Code Reviewer**: Reviews code
5. **Tester**: Verifies tests
6. **Developer**: Marks `completed`

## Quick Reference

### View All Tasks
```bash
find tasks -name "TASK-*.md" -exec grep -l "status: pending" {} \;
```

### View In-Progress Tasks
```bash
find tasks -name "TASK-*.md" -exec grep -l "status: in_progress" {} \;
```

## Phase Overview

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1 | 12 | ✅ Complete |
| Phase 2 | 8 | ✅ Complete |
| Phase 3 | 6 | ✅ Complete |
| Phase 4 | 8 | ✅ Complete |
| Phase 5 | 6 | ✅ Complete |
| Phase 6 | 6 | ✅ Complete |
| Phase 7 | 17 | ✅ Complete |
| **Total** | **63** | **All Complete** |

## Phase 7 Tasks (Multi-Adapter & Enhanced UI)

| Task | Description | Status |
|------|-------------|--------|
| 7-01 | WhisperAdapter implementation | ✅ Complete |
| 7-02 | GeminiAdapter implementation | ✅ Complete |
| 7-03 | DeepgramAdapter implementation | ✅ Complete |
| 7-04 | ElevenLabsTTSGenerator | ✅ Complete |
| 7-05 | NLP Pipeline Registry | ✅ Complete |
| 7-06 | SciSpaCy NER Pipeline | ✅ Complete |
| 7-07 | Biomedical NER Pipeline | ✅ Complete |
| 7-08 | MedSpaCy NER Pipeline | ✅ Complete |
| 7-09 | MultiNLPEvaluator | ✅ Complete |
| 7-10 | AudioHandler | ✅ Complete |
| 7-11 | WebSocketHandler | ✅ Complete |
| 7-12 | Audio Upload API | ✅ Complete |
| 7-13 | Multi-Model API | ✅ Complete |
| 7-14 | Web UI Audio Input | ✅ Complete |
| 7-15 | Radar Chart Visualization | ✅ Complete |
| 7-16 | Diff View Component | ✅ Complete |
| 7-17 | Test Suite for Phase 7 | ✅ Complete |
