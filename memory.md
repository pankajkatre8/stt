# Memory - HSTTB Project Context

> **Purpose**: This file preserves context across sessions and before compaction.
> **Update**: Before every context compaction and at end of each session.
> **Read**: At the start of every new session.

---

## Current State

### Project Phase
- **Current Phase**: Phase 1 - Foundation & Infrastructure
- **Phase Status**: COMPLETE
- **Next Phase**: Phase 2 - TER Engine

### Active Work
- **In Progress**: None (Phase 1 complete)
- **Blocked**: None
- **Completed Recently**: CLI implementation, audio tests, adapter tests

### Last Updated
- **Date**: 2026-01-31
- **By**: Claude
- **Session**: Phase 1 complete - 106 tests passing

---

## Project Overview (Quick Reference)

### What We're Building
Healthcare Streaming STT Benchmarking Framework - a model-agnostic evaluation system for healthcare speech-to-text with three core metrics:
1. **TER** - Term Error Rate (medical term accuracy)
2. **NER** - Named Entity Recognition accuracy
3. **CRS** - Context Retention Score (streaming continuity)

### Why It Matters
- Mission critical for company's future sales
- Healthcare STT errors can harm patients
- Must deliver beyond expectations
- Code quality must be top-notch

### Key Technical Decisions
1. Adapter pattern for STT integration
2. Streaming profiles for reproducible benchmarks
3. Composite scoring with configurable weights
4. Medical lexicons (RxNorm, SNOMED, ICD-10)

---

## Phase Status

| Phase | Name | Status | Progress |
|-------|------|--------|----------|
| 0 | Planning | ✅ Complete | 100% |
| 1 | Foundation | ✅ Complete | 100% |
| 2 | TER Engine | ⏳ Not Started | 0% |
| 3 | NER Engine | ⏳ Not Started | 0% |
| 4 | CRS Engine | ⏳ Not Started | 0% |
| 5 | Orchestration | ⏳ Not Started | 0% |
| 6 | Reporting | ⏳ Not Started | 0% |

---

## Files Created

### Documentation
- [x] `CLAUDE.md` - Project context
- [x] `plan.md` - High-level plan
- [x] `development_phases.md` - Detailed phases
- [x] `changelog.md` - Change tracking
- [x] `memory.md` - This file

### Agent Instructions
- [x] `.claude/planner.md`
- [x] `.claude/developer.md`
- [x] `.claude/code_reviewer.md`
- [x] `.claude/unit_tester.md`
- [x] `.claude/architect.md`
- [x] `.claude/security_reviewer.md`
- [x] `.claude/integration_tester.md`

### Source Code (Phase 1 Complete)
- [x] `src/hsttb/__init__.py`
- [x] `src/hsttb/cli.py` - CLI with transcribe, profiles, adapters, info, simulate
- [x] `src/hsttb/core/__init__.py`
- [x] `src/hsttb/core/types.py` - Core data types and enums
- [x] `src/hsttb/core/config.py` - Configuration system with profiles
- [x] `src/hsttb/core/exceptions.py` - Exception hierarchy
- [x] `src/hsttb/audio/__init__.py`
- [x] `src/hsttb/audio/loader.py` - Audio file loading
- [x] `src/hsttb/audio/chunker.py` - Streaming simulation
- [x] `src/hsttb/adapters/__init__.py`
- [x] `src/hsttb/adapters/base.py` - STTAdapter interface
- [x] `src/hsttb/adapters/registry.py` - Adapter factory
- [x] `src/hsttb/adapters/mock_adapter.py` - Mock adapters
- [x] `tests/unit/core/test_types.py` - 36 type tests
- [x] `tests/test_adapters.py` - 30 adapter tests
- [x] `tests/test_audio.py` - 40 audio tests
- **Total: 106 tests passing**

---

## Next Steps (Prioritized)

### Immediate (This Session or Next)
1. **Create project structure**
   - Initialize `pyproject.toml`
   - Create `src/hsttb/` directory
   - Set up `requirements.txt`

2. **Implement core types** (`src/hsttb/core/types.py`)
   - AudioChunk dataclass
   - TranscriptSegment dataclass
   - Entity dataclass
   - MedicalTerm dataclass

3. **Implement configuration** (`src/hsttb/core/config.py`)
   - StreamingProfile
   - EvaluationConfig
   - YAML loading

### Short Term (Phase 1)
4. Audio loader (`src/hsttb/audio/loader.py`)
5. Streaming chunker (`src/hsttb/audio/chunker.py`)
6. Streaming profiles (`src/hsttb/audio/profiles.py`)
7. STT adapter interface (`src/hsttb/adapters/base.py`)
8. Mock adapter for testing (`src/hsttb/adapters/mock_adapter.py`)

### Medium Term (Phase 2-4)
9. Medical lexicon loaders
10. TER engine implementation
11. NER engine implementation
12. CRS engine implementation

---

## Technical Decisions Log

### Decided
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Python version | 3.10+ | Match types, async features |
| Type system | Pydantic | Validation + serialization |
| STT integration | Adapter pattern | Model-agnostic |
| Fuzzy matching | rapidfuzz | Fast, medical-aware |
| NER | scispaCy + medspacy | Healthcare-specific |
| Embeddings | sentence-transformers | Good balance |

### Pending
| Decision | Options | Notes |
|----------|---------|-------|
| Primary STT for testing | Whisper, Deepgram | Need to evaluate |
| Lexicon data source | UMLS, custom | License considerations |
| Dashboard framework | Streamlit, FastAPI | TBD |

---

## Key Context (Don't Lose)

### Critical Requirements
1. **Medical accuracy is paramount** - errors can harm patients
2. **Reproducibility required** - benchmarks must be deterministic
3. **Streaming-first design** - not just batch processing
4. **Model-agnostic** - must work with any STT provider

### Healthcare-Specific Concerns
1. PHI must never be logged
2. Drug name errors are critical
3. Negation preservation is essential
4. Dosage errors are dangerous

### Quality Standards
1. 90%+ test coverage required
2. Complete type hints mandatory
3. All public functions need docstrings
4. Security review for all code

---

## Open Questions

1. **Data**: What test audio data is available?
2. **Lexicons**: Do we have UMLS license?
3. **STT**: Which provider to integrate first?
4. **Compliance**: Specific HIPAA requirements?
5. **Timeline**: Hard deadline for MVP?

---

## Session Handoff Template

When ending a session, update this section:

```markdown
### Session End: [DATE]

**Completed This Session**:
- [List of completed items]

**In Progress**:
- [Items started but not finished]

**Blocked/Issues**:
- [Any blockers encountered]

**Next Session Should**:
1. [First priority]
2. [Second priority]
3. [Third priority]

**Important Context**:
- [Any critical information for next session]
```

---

## Recovery Instructions

If context is lost, here's how to recover:

1. **Read these files in order**:
   - `memory.md` (this file) - Current state
   - `CLAUDE.md` - Project overview
   - `changelog.md` - Recent changes
   - `plan.md` - High-level plan
   - `development_phases.md` - Implementation details

2. **Check task files**:
   - `tasks/` directory for active tasks
   - Look for `status: in_progress`

3. **Review recent code**:
   - `git log --oneline -20` (if git initialized)
   - Check most recently modified files

4. **Resume work**:
   - Pick up from "Next Steps" section
   - Follow agent instructions in `.claude/`

---

## Contact Points

- **Project Documentation**: This repo
- **Issue Tracking**: TBD
- **Communication**: TBD

---

*Last compaction: Never*
*Sessions since compaction: 1*
