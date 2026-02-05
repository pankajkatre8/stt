# Memory - Lunagen STT Benchmarking Tool Project Context

> **Purpose**: This file preserves context across sessions and before compaction.
> **Update**: Before every context compaction and at end of each session.
> **Read**: At the start of every new session.

---

## Current State

### Project Phase
- **Current Phase**: Phase 9 - Stellicare Integration
- **Phase Status**: Stellicare WSS Integration COMPLETE
- **Next Phase**: Production deployment / Integration testing

### Active Work
- **In Progress**: None
- **Blocked**: None
- **Completed Recently**:
  - **Stellicare WSS Integration** (NEW - Lunagen STT pipeline)
    - StellicareConfig - Pydantic config with env var overrides
    - stream_audio_to_stellicare - WSS streaming with pipe-delimited protocol
    - refine_transcript - REST API refinement
    - process_files_sequentially - Multi-file sequential processing
    - StellicareWebSocketHandler - Browser-to-backend WS bridge
    - New UI tab: Stellicare with multi-file WAV upload, progress tracking, live transcript
    - Endpoints: WS /ws/stellicare, POST /api/stellicare/refine, GET /api/stellicare/config
  - **Clinical Risk Scoring** (NEW - prioritizes clinical safety over fluency)
    - EntityAssertionAnalyzer - tracks affirmed/negated/uncertain status
    - ClinicalContradictionDetector - soft/hard contradiction detection
    - DosagePlausibilityChecker - medication dose validation
    - ClinicalRiskScorer - combines all signals with clinical weighting
    - Risk levels: LOW, MEDIUM, HIGH, CRITICAL
    - Clinical recommendations: ACCEPT, ACCEPT_WITH_REVIEW, NEEDS_REVIEW, REJECT
  - QualityEngine with composite scoring
  - Perplexity scorer (GPT-2 based fluency)
  - Grammar checker (language-tool-python)
  - Medical coherence validator
  - Contradiction detection with question filtering
  - Embedding drift analysis
  - Confidence variance scoring
  - Speech rate validation (hallucination/missing content detection)
  - Quality UI with 7 component metrics and visualizations
  - Speech rate UI section after transcription
  - NLP model availability indicators
  - Fixed MedSpaCy negation detection bug
  - Removed mock NLP model from production

### Last Updated
- **Date**: 2026-02-05
- **By**: Claude
- **Session**: Phase 9 complete - Stellicare WSS integration, Lunagen rebranding

---

## Project Overview (Quick Reference)

### What We're Building
Lunagen STT Benchmarking Tool - a model-agnostic evaluation system for healthcare speech-to-text with three core metrics:
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
| 8 | Reference-Free Quality Metrics | ✅ Complete | 100% |
| 9 | Stellicare Integration | ✅ Complete | 100% |

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

**Phase 8 - Reference-Free Quality Metrics (Complete):**
- [x] `src/hsttb/metrics/quality.py` - QualityEngine with composite scoring
- [x] `src/hsttb/metrics/perplexity.py` - GPT-2 based fluency scorer
- [x] `src/hsttb/metrics/grammar.py` - Grammar checker wrapper
- [x] `src/hsttb/metrics/medical_coherence.py` - Drug-condition validation
- [x] `src/hsttb/metrics/contradiction.py` - Internal contradiction detection
- [x] `src/hsttb/metrics/embedding_drift.py` - Semantic stability analysis
- [x] `src/hsttb/metrics/confidence_variance.py` - Token confidence analysis
- [x] `src/hsttb/metrics/speech_rate.py` - Speech rate validation

**Clinical Risk Scoring (Complete):**
- [x] `src/hsttb/metrics/clinical_risk.py` - Clinical risk scoring engine
- [x] `src/hsttb/metrics/entity_assertion.py` - Entity assertion tracking
- [x] `src/hsttb/metrics/clinical_contradiction.py` - Soft/hard contradiction detection
- [x] `src/hsttb/metrics/dosage_plausibility.py` - Medication dosage validation

**Dynamic Medical Terminology (Complete):**
- [x] `src/hsttb/metrics/medical_terms.py` - Central medical terms provider
  - Loads from SQLite lexicon or embedded fallback
  - Provides drugs, conditions, symptoms, dosage ranges
  - All quality metrics now use this provider instead of hardcoded lists

**Medical Terminology APIs (Complete):**
- [x] `src/hsttb/lexicons/api_fetcher.py` - RxNorm/OpenFDA/ICD-10 API client
- [x] `src/hsttb/lexicons/sqlite_lexicon.py` - SQLite-backed lexicon storage
- [x] `src/hsttb/lexicons/dynamic_lexicon.py` - API-based lexicon with caching

**Scripts & Docker (Complete):**
- [x] `scripts/startup.py` - Startup script with lexicon initialization
- [x] `scripts/dev.sh` - Local development setup script
- [x] `Dockerfile` - Updated with httpx and startup script
- [x] `docker-compose.yml` - Updated with lexicon volume persistence

**Stellicare Integration (Complete):**
- [x] `src/hsttb/webapp/stellicare_client.py` - Stellicare WSS + refine client
- [x] `src/hsttb/webapp/stellicare_handler.py` - WebSocket handler
- [x] `src/hsttb/webapp/static/lunagen-logo.png` - Lunagen branding

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
1. **Stellicare end-to-end testing**
   - Verify WSS streaming with live Stellicare endpoint
   - Test multi-file sequential upload and progress tracking
   - Validate transcript refinement via REST API

2. **Run test suite** to verify all tests pass
   - `pytest tests/ -v --tb=short`
   - Add tests for Stellicare client and handler
   - Address any test failures

3. **Integration testing**
   - Test Stellicare tab UI with real WAV files
   - Verify WebSocket bridge (browser to Stellicare)
   - Test multi-adapter comparison

### Short Term
4. **Production deployment preparation**
   - Docker containerization with Stellicare config
   - Environment variable configuration for Stellicare endpoints
   - Logging and monitoring setup

5. **Documentation updates**
   - Update README with Lunagen branding and Stellicare features
   - Create user guide for Stellicare tab
   - Add API documentation for new endpoints

### Medium Term
6. **Performance optimization**
   - Profile NLP pipeline performance
   - Optimize audio streaming buffer sizes
   - Add caching for repeated transcriptions

7. **Additional adapters** (as needed)
   - Azure Speech Services
   - AWS Transcribe Medical
   - AssemblyAI

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
