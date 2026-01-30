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
└── phase_6/            # Reporting & Hardening
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
| Phase 1 | 12 | Pending |
| Phase 2 | 8 | Pending |
| Phase 3 | 6 | Pending |
| Phase 4 | 8 | Pending |
| Phase 5 | 6 | Pending |
| Phase 6 | 6 | Pending |
| **Total** | **46** | |
