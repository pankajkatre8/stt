# CLAUDE.md - Healthcare Streaming STT Benchmarking (HSTTB)

## Project Overview

**Project Name**: Healthcare Streaming STT Benchmarking Framework (HSTTB)
**Criticality**: MISSION CRITICAL - Company's future sales depend on this
**Quality Bar**: Beyond expectations - Top-notch code quality required

### What This Project Does

A model-agnostic evaluation framework to benchmark streaming Speech-to-Text (STT) systems for healthcare applications. Measures three core metrics:
- **TER** (Term Error Rate): Medical term transcription accuracy
- **NER Accuracy**: Medical entity integrity preservation
- **CRS** (Context Retention Score): Streaming context continuity

### Why It Matters

- Healthcare STT errors can lead to clinical misinterpretation
- Drug name errors (metformin → methotrexate) are life-threatening
- Negation flips ("no chest pain" → "chest pain") change diagnosis
- This framework ensures STT systems meet healthcare safety standards

---

## Quick Reference

### Directory Structure
```
hsttb/
├── src/hsttb/           # Main source code
│   ├── core/            # Types, config, base classes
│   ├── audio/           # Audio loading, streaming simulation
│   ├── adapters/        # STT model adapters
│   ├── lexicons/        # Medical lexicon loaders
│   ├── nlp/             # NLP pipelines (NER, normalization)
│   ├── metrics/         # TER, NER, CRS, SRS engines
│   ├── evaluation/      # Benchmark orchestration
│   └── reporting/       # Reports, dashboards
├── tests/               # Unit and integration tests
├── configs/             # YAML configurations
├── data/                # Test data (not in git)
├── .claude/             # Agent instructions
├── tasks/               # Task breakdown files
└── docs/                # Documentation
```

### Key Commands
```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=hsttb

# Run linting
ruff check src/ tests/
mypy src/

# Run benchmark
python -m hsttb.cli benchmark --audio-dir data/audio --gt-dir data/ground_truth

# Start dashboard
uvicorn hsttb.reporting.dashboard:app --reload
```

### Current Phase
Check `memory.md` for current development phase and active tasks.

---

## Code Quality Standards (NON-NEGOTIABLE)

### 1. Type Safety
- ALL functions must have complete type hints
- Use `from __future__ import annotations` in every file
- Run `mypy --strict` with zero errors

### 2. Documentation
- Every public class/function needs docstring
- Use Google-style docstrings
- Include Args, Returns, Raises sections
- Add usage examples for complex functions

### 3. Testing Requirements
- Minimum 90% code coverage
- Every metric engine needs property-based tests
- Integration tests for full pipeline
- Mock external services (STT APIs)

### 4. Error Handling
- Never swallow exceptions silently
- Use custom exception hierarchy
- Log all errors with context
- Graceful degradation where possible

### 5. Security (Healthcare-Specific)
- Never log PHI (Protected Health Information)
- Sanitize all inputs
- No hardcoded credentials
- Audit trail for all operations

### 6. Performance
- Streaming operations must not buffer entire audio
- Lazy loading for lexicons
- Async where beneficial
- Profile before optimizing

---

## Architecture Decisions

### ADR-001: Adapter Pattern for STT Integration
- STT models integrated via adapters
- Standardized interface for all models
- New models don't change evaluation logic

### ADR-002: Streaming Profiles for Valid Benchmarking
- Audio replayed through controlled profiles
- Deterministic chunking for reproducibility
- SRS metric separates model quality from streaming artifacts

### ADR-003: Composite Scoring with Configurable Weights
- TER, NER, CRS combined into composite score
- Weights configurable per use case
- Category-wise breakdowns available

### ADR-004: Medical Lexicons as First-Class Citizens
- RxNorm, SNOMED CT, ICD-10 integrated
- Unified lookup interface
- Fuzzy matching with medical awareness

---

## Agent System

This project uses multiple specialized agents. See `.claude/` directory:

| Agent | File | Purpose |
|-------|------|---------|
| Planner | `.claude/planner.md` | Task planning and breakdown |
| Developer | `.claude/developer.md` | Code implementation |
| Code Reviewer | `.claude/code_reviewer.md` | Quality assurance |
| Unit Tester | `.claude/unit_tester.md` | Test creation |
| Architect | `.claude/architect.md` | Architecture decisions |
| Security Reviewer | `.claude/security_reviewer.md` | Healthcare security |
| Integration Tester | `.claude/integration_tester.md` | E2E testing |

---

## Critical Files (Must Read Before Changes)

1. `src/hsttb/core/types.py` - All type definitions
2. `src/hsttb/adapters/base.py` - STT adapter interface
3. `src/hsttb/metrics/ter.py` - TER computation (most complex)
4. `src/hsttb/metrics/crs.py` - CRS computation (most complex)
5. `memory.md` - Current state and context
6. `changelog.md` - Recent changes

---

## Dependencies

### Core
- Python 3.10+
- pydantic >= 2.0
- numpy, pandas

### Audio
- soundfile
- webrtcvad

### NLP
- scispacy
- medspacy
- sentence-transformers

### Metrics
- jiwer (WER base)
- rapidfuzz (fuzzy matching)
- seqeval (NER metrics)

### Infrastructure
- fastapi
- mlflow
- pytest

---

## Glossary

| Term | Definition |
|------|------------|
| TER | Term Error Rate - Medical term accuracy |
| NER | Named Entity Recognition |
| CRS | Context Retention Score - Streaming continuity |
| SRS | Streaming Robustness Score - Model degradation under streaming |
| STT | Speech-to-Text |
| PHI | Protected Health Information |
| RxNorm | Drug terminology standard |
| SNOMED CT | Clinical terminology standard |
| ICD-10 | Diagnosis coding standard |
| UMLS | Unified Medical Language System |

---

## Contact & Resources

- Project Plan: `plan.md`
- Development Phases: `development_phases.md`
- Change Log: `changelog.md`
- Current Memory: `memory.md`
- Task Breakdown: `tasks/` directory

---

## Before You Start

1. Read `memory.md` for current context
2. Check `changelog.md` for recent changes
3. Review active tasks in `tasks/`
4. Follow agent instructions in `.claude/`
5. Update `changelog.md` after changes
6. Update `memory.md` before context compaction

## Auto-Commit Policy

**IMPORTANT**: Auto-commit after each phase or major task completion.
- Commit after completing each phase (Phase 1, Phase 2, etc.)
- Commit after completing major task groups
- Use descriptive commit messages following conventional commits
- Continue implementation without asking for confirmation
