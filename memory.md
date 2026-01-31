# Memory - HSTTB Project Context

> **Purpose**: This file preserves context across sessions and before compaction.
> **Update**: Before every context compaction and at end of each session.
> **Read**: At the start of every new session.

---

## Current State

### Project Phase
- **Current Phase**: Phase 7 - Multi-Adapter & Enhanced UI
- **Phase Status**: COMPLETE
- **Next Phase**: Production deployment / Integration testing

### Active Work
- **In Progress**: None
- **Blocked**: None
- **Completed Recently**:
  - All STT adapters (Whisper, Gemini, Deepgram)
  - ElevenLabs TTS generator
  - NLP pipeline registry with factory pattern
  - MedSpaCy NER pipeline
  - Multi-NLP evaluator
  - Audio upload/recording UI
  - WebSocket streaming transcription
  - Web API endpoints
  - Comprehensive test suite for new components

### Last Updated
- **Date**: 2026-01-31
- **By**: Claude
- **Session**: Phase 7 complete - Multi-adapter support, enhanced UI, 120+ new tests

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
| 2 | TER Engine | ✅ Complete | 100% |
| 3 | NER Engine | ✅ Complete | 100% |
| 4 | CRS Engine | ✅ Complete | 100% |
| 5 | Orchestration | ✅ Complete | 100% |
| 6 | Reporting | ✅ Complete | 100% |
| 7 | Multi-Adapter & Enhanced UI | ✅ Complete | 100% |

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

### Source Code (Phase 1 + Phase 2 Complete)
**Phase 1 - Foundation:**
- [x] `src/hsttb/__init__.py`
- [x] `src/hsttb/cli.py` - CLI commands
- [x] `src/hsttb/core/` - types, config, exceptions
- [x] `src/hsttb/audio/` - loader, chunker
- [x] `src/hsttb/adapters/` - base, registry, mocks

**Phase 2 - TER Engine (Complete):**
- [x] `src/hsttb/lexicons/base.py` - MedicalLexicon interface
- [x] `src/hsttb/lexicons/mock_lexicon.py` - Mock with drugs/diagnoses
- [x] `src/hsttb/lexicons/unified.py` - Multi-source lookup
- [x] `src/hsttb/nlp/normalizer.py` - MedicalTextNormalizer
- [x] `src/hsttb/metrics/ter.py` - TEREngine, compute_ter

**Phase 3 - NER Engine (Complete):**
- [x] `src/hsttb/nlp/ner_pipeline.py` - NERPipeline interface, MockNERPipeline
- [x] `src/hsttb/nlp/entity_alignment.py` - EntityAligner, alignment algorithms
- [x] `src/hsttb/metrics/ner.py` - NEREngine, accuracy computation

**Phase 4 - CRS Engine (Complete):**
- [x] `src/hsttb/metrics/semantic_similarity.py` - Token/embedding similarity
- [x] `src/hsttb/metrics/entity_continuity.py` - Entity continuity tracking
- [x] `src/hsttb/nlp/negation.py` - Negation detection
- [x] `src/hsttb/metrics/crs.py` - CRS computation engine

**Phase 5 - Orchestration (Complete):**
- [x] `src/hsttb/evaluation/__init__.py` - Evaluation module
- [x] `src/hsttb/evaluation/runner.py` - BenchmarkRunner
- [x] `src/hsttb/metrics/srs.py` - SRS computation engine

**Phase 6 - Reporting (Complete):**
- [x] `src/hsttb/reporting/__init__.py` - Reporting module
- [x] `src/hsttb/reporting/generator.py` - ReportGenerator

**Phase 7 - Multi-Adapter & Enhanced UI (Complete):**
- [x] `src/hsttb/adapters/whisper_adapter.py` - Local Whisper STT
- [x] `src/hsttb/adapters/gemini_adapter.py` - Google Cloud Speech STT
- [x] `src/hsttb/adapters/deepgram_adapter.py` - Deepgram STT
- [x] `src/hsttb/adapters/elevenlabs_tts.py` - ElevenLabs TTS generator
- [x] `src/hsttb/nlp/registry.py` - NLP pipeline registry
- [x] `src/hsttb/nlp/medspacy_ner.py` - MedSpaCy NER pipeline
- [x] `src/hsttb/metrics/multi_nlp.py` - Multi-NLP evaluator
- [x] `src/hsttb/webapp/audio_handler.py` - Audio upload handling
- [x] `src/hsttb/webapp/websocket_handler.py` - WebSocket streaming

**Tests (Previous Phases):**
- [x] `tests/unit/core/test_types.py` - 36 type tests
- [x] `tests/test_adapters.py` - 30 adapter tests
- [x] `tests/test_audio.py` - 40 audio tests
- [x] `tests/test_lexicons.py` - 32 lexicon tests
- [x] `tests/test_ter.py` - 25 TER tests
- [x] `tests/test_ner.py` - 34 NER tests
- [x] `tests/test_crs.py` - 33 CRS tests
- [x] `tests/test_evaluation.py` - 22 evaluation tests
- [x] `tests/test_reporting.py` - 18 reporting tests
- **Subtotal: 270 tests**

**Tests (Phase 7):**
- [x] `tests/test_nlp_registry.py` - 25 NLP registry tests
- [x] `tests/test_multi_nlp.py` - 30 multi-NLP tests
- [x] `tests/test_audio_handler.py` - 28 audio handler tests
- [x] `tests/test_new_adapters.py` - 22 new adapter tests
- [x] `tests/test_websocket_handler.py` - 18 WebSocket tests
- **Subtotal: 123 new tests**
- **Total: ~393 tests**

---

## Next Steps (Prioritized)

### Immediate (This Session or Next)
1. **Run test suite** to verify all tests pass
   - `pytest tests/ -v --tb=short`
   - Address any test failures

2. **Integration testing**
   - Test full pipeline with real audio files
   - Verify WebSocket streaming works end-to-end
   - Test multi-adapter comparison

3. **Documentation updates**
   - Update README with new features
   - Create user guide for new UI features
   - Add API documentation for new endpoints

### Short Term
4. **Production deployment preparation**
   - Docker containerization
   - Environment variable configuration
   - Logging and monitoring setup

5. **Performance optimization**
   - Profile NLP pipeline performance
   - Optimize audio streaming buffer sizes
   - Add caching for repeated transcriptions

### Medium Term
6. **Additional adapters** (as needed)
   - Azure Speech Services
   - AWS Transcribe Medical
   - AssemblyAI

7. **Enhanced UI features**
   - Real-time waveform visualization
   - Transcript editing
   - Annotation tools

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
