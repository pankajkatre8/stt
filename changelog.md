# Changelog

All notable changes to the HSTTB project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Initial project documentation and planning
- CLAUDE.md project context file
- Agent instruction files in `.claude/` directory
- `plan.md` - High-level project plan
- `development_phases.md` - Detailed phase breakdown
- `memory.md` - Context preservation for session continuity
- `tasks/` directory structure for task tracking

### Phase 1 Implementation (Foundation)
- **TASK-1C01**: Project setup complete
  - `pyproject.toml` with all dependencies
  - Source directory structure `src/hsttb/`
  - Test directory structure `tests/`
  - Development tools configured (ruff, mypy, pytest)

- **TASK-1C02**: Core types implemented (`src/hsttb/core/types.py`)
  - Enums: EntityLabel, MedicalTermCategory, MatchType, ErrorType
  - AudioChunk, TranscriptSegment dataclasses
  - Entity, MedicalTerm dataclasses
  - TermError, EntityMatch dataclasses
  - TERResult, NERResult, CRSResult, SRSResult dataclasses
  - BenchmarkResult, BenchmarkSummary dataclasses
  - 36 unit tests passing

- **TASK-1C03**: Configuration system implemented (`src/hsttb/core/config.py`)
  - AudioConfig, ChunkingConfig, NetworkConfig, VADConfig
  - StreamingProfile with validation
  - EvaluationConfig with TER/NER/CRS sub-configs
  - Built-in profiles: ideal, realtime_mobile, realtime_clinical, high_latency
  - YAML loading/saving utilities

- **TASK-1C04**: Exception hierarchy implemented (`src/hsttb/core/exceptions.py`)
  - HSSTBError base class
  - AudioError, AudioLoadError, AudioFormatError
  - STTAdapterError, STTConnectionError, STTTranscriptionError
  - LexiconError, LexiconLoadError, LexiconLookupError
  - MetricComputationError, TERComputationError, NERComputationError, CRSComputationError
  - EvaluationError, BenchmarkError, ReportGenerationError

- **TASK-1A01**: Audio loader implemented (`src/hsttb/audio/loader.py`)
  - AudioLoader class with format conversion
  - Support for WAV, FLAC, OGG, MP3
  - Resampling and mono conversion
  - Checksum generation for reproducibility

- **TASK-1A02**: Streaming chunker implemented (`src/hsttb/audio/chunker.py`)
  - StreamingChunker class with deterministic behavior
  - Configurable chunk size, jitter, overlap
  - Network delay simulation
  - Async streaming interface

- **TASK-1S01**: STT adapter interface implemented (`src/hsttb/adapters/`)
  - Abstract STTAdapter base class (`base.py`)
  - Adapter registry with factory pattern (`registry.py`)
  - MockSTTAdapter for testing (`mock_adapter.py`)
  - FailingMockAdapter for error handling tests
  - 30 unit tests for adapter module

- **TASK-1CLI**: CLI implementation complete (`src/hsttb/cli.py`)
  - `transcribe` - Transcribe audio with streaming simulation
  - `profiles` - List available streaming profiles
  - `adapters` - List registered STT adapters
  - `info` - Show audio file information
  - `simulate` - Preview chunk boundaries

### Phase 1 Complete
- 106 unit tests passing
- All core modules implemented (types, config, exceptions, audio, adapters)
- CLI operational with mock adapters

### Phase 2 Implementation (TER Engine)
- **TASK-2L01**: Medical lexicons implemented (`src/hsttb/lexicons/`)
  - LexiconEntry, LexiconSource base types
  - MedicalLexicon abstract interface
  - MockMedicalLexicon with 30+ common drugs and diagnoses
  - UnifiedMedicalLexicon for multi-source lookup
  - 32 unit tests for lexicon module

- **TASK-2N01**: Text normalizer implemented (`src/hsttb/nlp/`)
  - MedicalTextNormalizer with configurable options
  - 70+ medical abbreviation expansions
  - Dosage normalization (500mg → 500 mg)
  - Number word conversion
  - normalize_for_ter utility function

- **TASK-2T01**: TER computation engine implemented (`src/hsttb/metrics/`)
  - TEREngine class for Term Error Rate computation
  - Term extraction using lexicon matching
  - Term alignment with greedy matching
  - Detection of substitutions, deletions, insertions
  - Category-wise TER breakdown
  - 25 unit tests for TER module

### Phase 2 Complete
- 163 unit tests passing
- Medical lexicons with drugs/diagnoses
- Text normalization with medical abbreviations
- TER engine with term extraction and alignment

### Phase 3 Implementation (NER Engine)
- **TASK-3N01**: NER Pipeline implemented (`src/hsttb/nlp/ner_pipeline.py`)
  - NERPipeline abstract interface
  - NERPipelineConfig for pipeline configuration
  - MockNERPipeline with pattern-based extraction
  - Support for DRUG, DIAGNOSIS, SYMPTOM, ANATOMY, PROCEDURE, LAB_VALUE, DOSAGE
  - Negation detection for medical context
  - `with_common_patterns()` factory with 60+ patterns
  - `with_custom_patterns()` for user-defined patterns

- **TASK-3E01**: Entity alignment implemented (`src/hsttb/nlp/entity_alignment.py`)
  - EntityAligner class for matching gold/pred entities
  - AlignmentConfig with configurable strategies
  - SpanMatchStrategy: EXACT, PARTIAL, BOUNDARY
  - Span IOU computation for overlap detection
  - Text similarity for fuzzy matching
  - Greedy alignment algorithm

- **TASK-3A01**: NER accuracy engine implemented (`src/hsttb/metrics/ner.py`)
  - NEREngine class for accuracy computation
  - NEREngineConfig for engine configuration
  - Precision, Recall, F1 computation
  - Entity distortion rate, omission rate
  - Per-entity-type metrics breakdown
  - `compute_ner_accuracy()`, `compute_entity_f1()` utilities

### Phase 3 Complete
- 197 unit tests passing (34 new NER tests)
- NER pipeline with pattern-based entity extraction
- Entity alignment with configurable strategies
- NER accuracy engine with precision/recall/F1

### Phase 4 Implementation (CRS Engine)
- **TASK-4S01**: Semantic similarity engine implemented (`src/hsttb/metrics/semantic_similarity.py`)
  - SemanticSimilarityEngine abstract interface
  - TokenBasedSimilarity using Jaccard/n-gram/LCS
  - EmbeddingBasedSimilarity (requires sentence-transformers)
  - Segment-wise and average similarity computation

- **TASK-4E01**: Entity continuity tracker implemented (`src/hsttb/metrics/entity_continuity.py`)
  - EntityContinuityTracker for cross-segment tracking
  - DiscontinuityType: disappearance, conflict, label_change, negation_flip
  - Entity occurrence mapping and timeline building
  - Entity preservation rate computation

- **TASK-4N01**: Negation detection implemented (`src/hsttb/nlp/negation.py`)
  - NegationDetector with rule-based patterns
  - 15+ pre-negation cues (no, not, denies, without, etc.)
  - Negation scope detection
  - Negation consistency checking between GT/pred

- **TASK-4C01**: CRS computation engine implemented (`src/hsttb/metrics/crs.py`)
  - CRSEngine combining all CRS components
  - CRSConfig with configurable weights
  - Composite score from semantic/entity/negation
  - Per-segment scores and context drift rate
  - `compute_crs()` convenience function

### Phase 4 Complete
- 230 unit tests passing (33 new CRS tests)
- Semantic similarity with token-based fallback
- Entity continuity tracking across segments
- Negation detection and consistency checking
- CRS engine with weighted composite scoring

### Phase 5 Implementation (Orchestration)
- **TASK-5R01**: Benchmark runner implemented (`src/hsttb/evaluation/runner.py`)
  - BenchmarkRunner class for evaluation orchestration
  - BenchmarkConfig for runner configuration
  - EvaluationResult for single-file results
  - Audio loading, chunking, and STT pipeline
  - Lazy initialization of metric engines
  - Text segmentation for CRS

- **TASK-5S01**: SRS computation implemented (`src/hsttb/metrics/srs.py`)
  - SRSEngine for Streaming Robustness Score
  - SRSConfig with profile settings
  - Runs evaluation under ideal and realtime conditions
  - Computes degradation per metric
  - Weighted composite SRS score
  - `compute_srs()` convenience function

### Phase 5 Complete
- 252 unit tests passing (22 new evaluation tests)
- Benchmark runner for full evaluation pipeline
- SRS engine for streaming robustness analysis

### Phase 6 Implementation (Reporting)
- **TASK-6R01**: Report generator implemented (`src/hsttb/reporting/generator.py`)
  - ReportGenerator class for multi-format reports
  - ReportConfig for customizable output
  - JSON report with detailed metrics
  - CSV report for spreadsheet analysis
  - HTML report with styled summary
  - Clinical risk report for safety analysis
  - ClinicalRiskItem and ClinicalRiskReport types
  - Drug substitution/omission detection (critical)
  - Dosage error detection (high)
  - Negation flip detection (high)
  - `generate_report()` convenience function

### Phase 6 Complete
- 270 unit tests passing (18 new reporting tests)
- Report generation in JSON, CSV, HTML formats
- Clinical risk analysis for patient safety
- All 6 phases complete

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

---

## Change Log Guidelines

### When to Update

Update this file when:
1. **Adding new features** - Document what was added
2. **Modifying existing code** - Document what changed and why
3. **Fixing bugs** - Document what was fixed
4. **Security changes** - Always document security-related changes
5. **Breaking changes** - Highlight clearly
6. **Before context compaction** - Ensure recent changes are captured

### Entry Format

```markdown
### [Category]
- Brief description of change
- Files affected: `path/to/file.py`
- Related task: TASK-XXX
- Notes: Any important context
```

### Categories

- **Added**: New features or files
- **Changed**: Changes to existing functionality
- **Deprecated**: Features to be removed in future
- **Removed**: Removed features or files
- **Fixed**: Bug fixes
- **Security**: Security-related changes

---

## Session Log

### Session: 2024-XX-XX (Initial Setup)

**Objective**: Set up project infrastructure and agent system

**Changes Made**:
1. Created project directory structure
2. Created CLAUDE.md with project context
3. Created agent instruction files
4. Created planning documents
5. Set up changelog and memory systems

**Context for Next Session**:
- Project is in initial planning phase
- No code has been written yet
- Next step: Begin Phase 1 implementation
- Review `memory.md` for current state

**Open Questions**:
- Which STT provider to integrate first?
- What medical lexicon data is available?
- Any specific healthcare compliance requirements?

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.0.1 | TBD | Initial planning and documentation |
| 0.1.0 | TBD | Phase 1 complete - Foundation |
| 0.2.0 | TBD | Phase 2 complete - TER Engine |
| 0.3.0 | TBD | Phase 3 complete - NER Engine |
| 0.4.0 | TBD | Phase 4 complete - CRS Engine |
| 0.5.0 | TBD | Phase 5 complete - Orchestration |
| 1.0.0 | TBD | Phase 6 complete - Production Ready |
