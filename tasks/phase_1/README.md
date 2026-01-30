# Phase 1: Foundation & Infrastructure

**Duration**: Week 1-2
**Status**: Pending
**Objective**: Set up project infrastructure, core types, audio streaming, and STT adapter interface.

## Phase Overview

This phase establishes the foundation for the entire project. All subsequent phases depend on these components being correct and well-designed.

## Task List

| Task ID | Title | Status | Complexity | Blocks |
|---------|-------|--------|------------|--------|
| TASK-1C01 | Project Setup | pending | Small | All |
| TASK-1C02 | Core Types | pending | Medium | TASK-1C03+ |
| TASK-1C03 | Configuration System | pending | Medium | TASK-1A01+ |
| TASK-1C04 | Exception Hierarchy | pending | Small | TASK-1A01+ |
| TASK-1A01 | Audio Loader | pending | Medium | TASK-1A02 |
| TASK-1A02 | Streaming Chunker | pending | Large | TASK-1A03 |
| TASK-1A03 | Streaming Profiles | pending | Medium | TASK-1S01 |
| TASK-1S01 | STT Adapter Interface | pending | Medium | TASK-1S02+ |
| TASK-1S02 | Mock STT Adapter | pending | Small | TASK-1S03 |
| TASK-1S03 | Whisper Adapter | pending | Large | TASK-1X01 |
| TASK-1X01 | CLI Basic | pending | Medium | None |
| TASK-1X02 | Unit Tests Phase 1 | pending | Large | None |

## Dependency Graph

```
TASK-1C01 (Project Setup)
    │
    ├──► TASK-1C02 (Core Types)
    │        │
    │        ├──► TASK-1C03 (Configuration)
    │        │        │
    │        │        └──► TASK-1A01 (Audio Loader)
    │        │                  │
    │        │                  └──► TASK-1A02 (Streaming Chunker)
    │        │                            │
    │        │                            └──► TASK-1A03 (Streaming Profiles)
    │        │                                      │
    │        │                                      └──► TASK-1S01 (Adapter Interface)
    │        │                                                │
    │        │                                                ├──► TASK-1S02 (Mock Adapter)
    │        │                                                │
    │        │                                                └──► TASK-1S03 (Whisper Adapter)
    │        │
    │        └──► TASK-1C04 (Exceptions)
    │
    └──► TASK-1X01 (CLI) ──► TASK-1X02 (Tests)
```

## Deliverables

At the end of Phase 1:

- [ ] Project is pip-installable
- [ ] All core types defined with full type hints
- [ ] Configuration loads from YAML
- [ ] Audio files can be loaded and chunked
- [ ] Streaming profiles control chunk behavior
- [ ] STT adapter interface defined
- [ ] Mock adapter available for testing
- [ ] Basic Whisper adapter working
- [ ] CLI can run basic transcription
- [ ] Unit tests for all components (>90% coverage)

## Quality Gates

Before Phase 2 can start:

- [ ] All Phase 1 tasks marked complete
- [ ] All unit tests pass
- [ ] mypy --strict passes
- [ ] Code review complete
- [ ] Documentation complete
- [ ] memory.md updated

## Notes

- Focus on correctness over features
- Ensure streaming is truly streaming (not buffering)
- Make profiles deterministic for reproducibility
- Keep adapter interface minimal and clean
