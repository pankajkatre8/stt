# Healthcare Streaming STT Benchmarking - Development Plan

## 1. Project Overview

### 1.1 Objective
Build a model-agnostic evaluation framework to benchmark streaming Speech-to-Text (STT) systems for healthcare applications, focusing on:
- Medical term accuracy (TER)
- Medical entity integrity (NER Accuracy)
- Context continuity in streaming speech (CRS)

### 1.2 Key Outcomes
- Objective benchmarking of current healthcare STT system
- Standardized evaluation framework for future models
- Identification of clinically critical transcription failures
- Data-driven STT model selection and optimization

---

## 2. Technical Architecture

### 2.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Audio Input Layer                             │
│  ┌──────────────────┐    ┌──────────────────────────────────────┐  │
│  │  Live Streaming  │    │  Recorded Audio (Replay as Stream)   │  │
│  └────────┬─────────┘    └─────────────────┬────────────────────┘  │
│           │                                 │                        │
│           └─────────────┬───────────────────┘                        │
│                         ▼                                            │
│              ┌─────────────────────┐                                │
│              │  Streaming Profiles  │ (chunk size, jitter, VAD)     │
│              └──────────┬──────────┘                                │
└─────────────────────────┼───────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STT Integration Layer                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                   STT Adapter Interface                         │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │ │
│  │  │ Whisper │  │ Deepgram│  │ AWS     │  │ Custom  │           │ │
│  │  │ Adapter │  │ Adapter │  │ Adapter │  │ Adapter │           │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ▼                                       │
│              Partial + Final Transcripts with Timestamps             │
└─────────────────────────────┬───────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Evaluation Engine                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│  │   TER Engine   │  │   NER Engine   │  │   CRS Engine   │        │
│  │ (Medical Term  │  │ (Entity        │  │ (Context       │        │
│  │  Error Rate)   │  │  Extraction)   │  │  Retention)    │        │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘        │
│          └───────────────────┼───────────────────┘                  │
│                              ▼                                       │
│                    Metric Aggregation                                │
└─────────────────────────────┬───────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Reporting Layer                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│  │ Model Compare  │  │ Regression     │  │ Clinical Risk  │        │
│  │ Dashboards     │  │ Analysis       │  │ Reports        │        │
│  └────────────────┘  └────────────────┘  └────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
hsttb/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic configs
│   │   └── types.py               # Type definitions
│   │
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── loader.py              # Audio file loading
│   │   ├── chunker.py             # Streaming chunk simulator
│   │   └── profiles.py            # Streaming profile definitions
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract STT adapter
│   │   ├── whisper_adapter.py
│   │   ├── deepgram_adapter.py
│   │   └── custom_adapter.py
│   │
│   ├── lexicons/
│   │   ├── __init__.py
│   │   ├── loader.py              # Lexicon loading utilities
│   │   ├── rxnorm.py              # RxNorm integration
│   │   ├── snomed.py              # SNOMED CT integration
│   │   └── icd10.py               # ICD-10 integration
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── ter.py                 # Term Error Rate
│   │   ├── ner.py                 # NER Accuracy
│   │   ├── crs.py                 # Context Retention Score
│   │   └── srs.py                 # Streaming Robustness Score
│   │
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── medical_ner.py         # Medical NER pipeline
│   │   ├── normalizer.py          # Text normalization
│   │   └── negation.py            # Negation detection
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── runner.py              # Benchmark orchestrator
│   │   ├── session.py             # Session-level evaluation
│   │   └── segment.py             # Segment-level evaluation
│   │
│   └── reporting/
│       ├── __init__.py
│       ├── generator.py           # Report generation
│       └── dashboard.py           # Dashboard API
│
├── data/
│   ├── lexicons/                  # Medical lexicon files
│   ├── test_audio/                # Test audio corpus
│   └── ground_truth/              # Annotated ground truth
│
├── configs/
│   ├── streaming_profiles/
│   │   ├── ideal_replay.yaml
│   │   ├── realtime_mobile.yaml
│   │   └── realtime_clinical.yaml
│   └── evaluation.yaml
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── scripts/
│   ├── run_benchmark.py
│   └── generate_report.py
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 3. Core Components Specification

### 3.1 Streaming Profiles (Critical for Valid Benchmarking)

```python
# configs/streaming_profiles/ideal_replay.yaml
profile_name: ideal_replay_v1
description: "Ideal conditions for baseline measurement"
audio:
  sample_rate: 16000
  channels: 1
  bit_depth: 16
chunking:
  chunk_size_ms: 1000
  chunk_jitter_ms: 0
  overlap_ms: 0
network:
  delay_ms: 0
  jitter_ms: 0
  packet_loss_rate: 0
vad:
  enabled: false
```

```python
# configs/streaming_profiles/realtime_mobile.yaml
profile_name: realtime_mobile_v1
description: "Simulates mobile network conditions"
audio:
  sample_rate: 16000
  channels: 1
  bit_depth: 16
chunking:
  chunk_size_ms: 1000
  chunk_jitter_ms: 50       # ±50ms variation
  overlap_ms: 100
network:
  delay_ms: 50
  jitter_ms: 30
  packet_loss_rate: 0.01
vad:
  enabled: true
  silence_threshold_ms: 300
```

### 3.2 STT Adapter Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

@dataclass
class TranscriptSegment:
    text: str
    is_partial: bool
    is_final: bool
    confidence: float
    start_time_ms: int
    end_time_ms: int
    word_timestamps: list[dict] | None = None

class STTAdapter(ABC):
    """Abstract base class for STT model adapters."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to STT service."""
        pass

    @abstractmethod
    async def send_audio_chunk(self, chunk: bytes, sequence_id: int) -> None:
        """Send audio chunk to STT service."""
        pass

    @abstractmethod
    async def receive_transcripts(self) -> AsyncIterator[TranscriptSegment]:
        """Yield transcript segments as they arrive."""
        pass

    @abstractmethod
    async def finalize(self) -> TranscriptSegment:
        """Signal end of audio and get final transcript."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass
```

### 3.3 TER Engine (Medical Term Error Rate)

```python
@dataclass
class TERResult:
    overall_ter: float
    category_ter: dict[str, float]  # drug, diagnosis, dosage, anatomy, procedure
    substitutions: list[tuple[str, str]]
    deletions: list[str]
    insertions: list[str]
    total_medical_terms: int

@dataclass
class MedicalTerm:
    text: str
    normalized: str
    category: str  # drug | diagnosis | dosage | anatomy | procedure
    source: str    # rxnorm | snomed | icd10
    span: tuple[int, int]
```

### 3.4 NER Accuracy Engine

```python
@dataclass
class NERResult:
    precision: float
    recall: float
    f1_score: float
    entity_distortion_rate: float
    entity_omission_rate: float
    entities_ground_truth: list[Entity]
    entities_predicted: list[Entity]
    matched: list[tuple[Entity, Entity]]
    distorted: list[tuple[Entity, Entity]]
    omitted: list[Entity]
    hallucinated: list[Entity]

@dataclass
class Entity:
    text: str
    label: str  # DRUG | DOSAGE | SYMPTOM | DIAGNOSIS | ANATOMY | LAB_VALUE
    span: tuple[int, int]
    normalized: str | None = None
```

### 3.5 Context Retention Score (CRS)

```python
@dataclass
class CRSResult:
    composite_score: float
    semantic_similarity: float
    entity_continuity: float
    negation_consistency: float
    temporal_consistency: float
    context_drift_rate: float
    segment_scores: list[SegmentCRSScore]

@dataclass
class SegmentCRSScore:
    segment_id: int
    ground_truth_text: str
    predicted_text: str
    semantic_similarity: float
    entities_preserved: int
    entities_lost: int
    negation_flips: int
```

### 3.6 Streaming Robustness Score (SRS)

```python
@dataclass
class SRSResult:
    """Measures model degradation under different streaming conditions."""
    model_id: str
    ideal_profile_score: float
    realtime_profile_score: float
    srs: float  # realtime / ideal ratio
    degradation_breakdown: dict[str, float]  # TER, NER, CRS degradation
```

---

## 4. Technology Stack

### 4.1 Core Python Dependencies

```
# requirements.txt

# Core
python>=3.10
numpy>=1.24.0
pandas>=2.0.0
pydantic>=2.0.0
orjson>=3.9.0

# Audio Processing
soundfile>=0.12.0
webrtcvad>=2.0.10
pydub>=0.25.0

# Streaming / Async
asyncio
websockets>=12.0
grpcio>=1.60.0
aiofiles>=23.0.0

# Text Processing / Alignment
jiwer>=3.0.0
rapidfuzz>=3.5.0

# Medical NLP
scispacy>=0.5.0
medspacy>=1.0.0
spacy>=3.7.0

# Sentence Embeddings (CRS)
sentence-transformers>=2.2.0

# NER Models (install separately)
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz

# Metrics & ML
scikit-learn>=1.3.0
seqeval>=1.2.0
networkx>=3.2.0

# API / Reporting
fastapi>=0.109.0
uvicorn>=0.27.0
plotly>=5.18.0

# Experiment Tracking
mlflow>=2.10.0

# Configuration
hydra-core>=1.3.0
omegaconf>=2.3.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.23.0
```

### 4.2 ML Models

| Purpose | Model | Source |
|---------|-------|--------|
| Clinical NER | `en_ner_bc5cdr_md` | scispaCy |
| Medical embeddings | `en_core_sci_md` | scispaCy |
| Negation detection | MedSpaCy context | medspacy |
| Sentence similarity | `all-MiniLM-L6-v2` | sentence-transformers |
| Biomedical similarity | `BioSentVec` (optional) | NCBI |

### 4.3 Medical Lexicons

| Lexicon | Purpose | Format |
|---------|---------|--------|
| RxNorm | Drug names | UMLS RRF |
| SNOMED CT | Clinical terms | UMLS RRF |
| ICD-10 | Diagnoses | CMS files |
| UMLS | Metathesaurus | UMLS RRF |

---

## 5. Streaming Architecture: Live vs Recorded

### 5.1 The Problem

Live streaming and replayed audio behave differently:

| Aspect | Live Streaming | Replayed Recording |
|--------|----------------|-------------------|
| Timing | Real-time, jitter | Perfectly regular |
| Chunk boundaries | OS/network dependent | Deterministic |
| Packet loss | Possible | None |
| Silence handling | Natural pauses | Artificially perfect |

### 5.2 The Solution: Controlled Streaming Profiles

```
                    ┌─────────────────────┐
                    │   Full Audio File   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Streaming Profile  │
                    │  (chunk size,       │
                    │   jitter, delay)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
         ┌─────────────────┐   ┌─────────────────┐
         │  Ideal Profile  │   │ Realtime Profile │
         │  (baseline)     │   │ (production sim) │
         └────────┬────────┘   └────────┬────────┘
                  │                     │
                  ▼                     ▼
              Score: 0.92           Score: 0.85
                  │                     │
                  └─────────┬───────────┘
                            │
                            ▼
                    SRS = 0.85 / 0.92 = 0.92
                    (Streaming Robustness Score)
```

### 5.3 Validation Strategy

To prove quality degradation is due to the model, not the streaming simulation:

1. **Lock the streaming layer**: Same code, chunker, profiles, random seed
2. **Test single model across profiles**: If quality drops, streaming sensitivity confirmed
3. **Test multiple models under same profile**: If only one degrades, model issue

### 5.4 Required Safeguards

- Audio checksum per chunk
- Deterministic chunk replay (seeded RNG)
- Timestamp alignment validation
- Partial transcript logging

---

## 6. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Deliverables:**
- [ ] Project structure and configuration
- [ ] Audio loader and streaming chunker
- [ ] Streaming profile system
- [ ] STT adapter interface + 1 concrete adapter
- [ ] Basic CLI for running benchmarks

**Key Files:**
- `src/core/config.py`
- `src/core/types.py`
- `src/audio/loader.py`
- `src/audio/chunker.py`
- `src/audio/profiles.py`
- `src/adapters/base.py`
- `src/adapters/whisper_adapter.py`

### Phase 2: TER Engine (Week 3)

**Deliverables:**
- [ ] Medical lexicon loaders (RxNorm, SNOMED, ICD-10)
- [ ] Medical term extractor
- [ ] Text normalizer (abbreviations, plurals, case)
- [ ] Term-level alignment and diff
- [ ] Category-wise TER computation

**Key Files:**
- `src/lexicons/loader.py`
- `src/lexicons/rxnorm.py`
- `src/lexicons/snomed.py`
- `src/nlp/normalizer.py`
- `src/metrics/ter.py`

### Phase 3: NER Engine (Week 4)

**Deliverables:**
- [ ] Medical NER pipeline (scispaCy + medspacy)
- [ ] Entity alignment logic (span drift handling)
- [ ] Precision/Recall/F1 computation
- [ ] Entity distortion and omission metrics

**Key Files:**
- `src/nlp/medical_ner.py`
- `src/metrics/ner.py`

### Phase 4: CRS Engine (Week 5)

**Deliverables:**
- [ ] Sentence embedding similarity
- [ ] Cross-segment entity continuity tracker
- [ ] Negation consistency checker
- [ ] Temporal consistency checker
- [ ] Composite CRS scoring

**Key Files:**
- `src/nlp/negation.py`
- `src/metrics/crs.py`

### Phase 5: Evaluation Orchestration (Week 6)

**Deliverables:**
- [ ] Benchmark runner (orchestrates full pipeline)
- [ ] Session-level and segment-level evaluation
- [ ] SRS computation (model comparison across profiles)
- [ ] MLflow integration for experiment tracking

**Key Files:**
- `src/evaluation/runner.py`
- `src/evaluation/session.py`
- `src/evaluation/segment.py`
- `src/metrics/srs.py`

### Phase 6: Reporting & Hardening (Week 7)

**Deliverables:**
- [ ] JSON/Parquet report export
- [ ] FastAPI dashboard endpoints
- [ ] Model comparison visualizations
- [ ] Clinical risk error summaries
- [ ] Unit and integration tests

**Key Files:**
- `src/reporting/generator.py`
- `src/reporting/dashboard.py`
- `tests/*`

---

## 7. Data Requirements

### 7.1 Test Audio Corpus

| Category | Requirement |
|----------|-------------|
| Duration | 10-50 hours minimum |
| Domains | SOAP notes, discharge summaries, prescriptions |
| Speakers | Multiple accents, speech speeds |
| Conditions | Clean + noise variations |
| Format | 16kHz, mono, 16-bit PCM |

### 7.2 Ground Truth Annotations

Each audio file requires:
- Verified transcript
- Medical term tags (with category)
- Entity annotations (with type)
- Negation markers
- Temporal markers

```json
{
  "audio_id": "sample_001",
  "transcript": "Patient denies chest pain. History of diabetes.",
  "medical_terms": [
    {"text": "chest pain", "category": "symptom", "span": [15, 25]},
    {"text": "diabetes", "category": "diagnosis", "span": [38, 46]}
  ],
  "entities": [
    {"text": "chest pain", "label": "SYMPTOM", "span": [15, 25], "negated": true},
    {"text": "diabetes", "label": "DIAGNOSIS", "span": [38, 46], "negated": false}
  ]
}
```

---

## 8. Benchmark Report Format

### 8.1 Model Comparison Table

| Model | Profile | TER | NER F1 | CRS | SRS |
|-------|---------|-----|--------|-----|-----|
| Model A | ideal | 4.1% | 0.92 | 0.95 | - |
| Model A | realtime | 6.8% | 0.85 | 0.81 | 0.87 |
| Model B | ideal | 4.3% | 0.91 | 0.93 | - |
| Model B | realtime | 5.1% | 0.88 | 0.87 | 0.94 |

### 8.2 Category-Wise TER Breakdown

| Category | Model A | Model B |
|----------|---------|---------|
| Drug names | 5.2% | 4.8% |
| Diagnoses | 3.1% | 3.5% |
| Dosages | 8.4% | 6.2% |
| Anatomy | 2.8% | 3.1% |
| Procedures | 4.9% | 5.3% |

### 8.3 Clinical Risk Report

| Risk Level | Error Type | Example | Count |
|------------|------------|---------|-------|
| Critical | Drug name substitution | "metformin" → "methotrexate" | 3 |
| Critical | Dosage error | "500mg" → "50mg" | 2 |
| High | Negation loss | "no chest pain" → "chest pain" | 5 |
| Medium | Anatomy confusion | "left arm" → "left leg" | 4 |

---

## 9. API Design

### 9.1 Benchmark Runner CLI

```bash
# Run benchmark with specific profile
python -m hsttb.run \
  --audio-dir data/test_audio \
  --ground-truth data/ground_truth \
  --adapter whisper \
  --profile realtime_mobile \
  --output results/run_001

# Compare models
python -m hsttb.compare \
  --results results/model_a results/model_b \
  --output reports/comparison.html
```

### 9.2 Python API

```python
from hsttb import BenchmarkRunner, WhisperAdapter
from hsttb.profiles import load_profile

# Initialize
adapter = WhisperAdapter(model="large-v3")
profile = load_profile("realtime_mobile")
runner = BenchmarkRunner(adapter, profile)

# Run evaluation
results = await runner.evaluate(
    audio_dir="data/test_audio",
    ground_truth_dir="data/ground_truth"
)

# Access metrics
print(f"TER: {results.ter.overall_ter:.2%}")
print(f"NER F1: {results.ner.f1_score:.3f}")
print(f"CRS: {results.crs.composite_score:.3f}")
```

---

## 10. Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Foundation | Project setup, streaming, adapters |
| 3 | TER Engine | Medical term extraction and scoring |
| 4 | NER Engine | Entity extraction and matching |
| 5 | CRS Engine | Context retention scoring |
| 6 | Orchestration | Runner, SRS, experiment tracking |
| 7 | Reporting | Dashboards, reports, testing |

**MVP Total: 6-7 weeks**

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Insufficient annotated data | High | Start with public datasets (MIMIC, LibriSpeech-medical) |
| Lexicon licensing (UMLS) | Medium | Apply for UMLS license early; fallback to open alternatives |
| CRS metric instability | Medium | Start with heuristic-heavy v1, iterate |
| STT adapter complexity | Low | Focus on 1-2 adapters for MVP |

---

## 12. Success Criteria

### MVP (Week 7)
- [ ] Benchmark 1 STT model successfully
- [ ] All 3 core metrics (TER, NER, CRS) operational
- [ ] Reproducible results across runs
- [ ] Basic comparison report

### Production (Week 10+)
- [ ] 3+ STT adapters integrated
- [ ] Multiple streaming profiles
- [ ] SRS metric operational
- [ ] Regression testing pipeline
- [ ] Dashboard with drill-down

---

## 13. Next Steps

1. **Week 0**: Environment setup, UMLS license application, data sourcing
2. **Week 1**: Begin Phase 1 implementation
3. **Ongoing**: Weekly progress review and metric validation
