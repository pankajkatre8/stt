# Lunagen STT Benchmarking Tool - Workflow Explainer

A detailed explanation of how the Lunagen STT Benchmarking Tool processes audio and computes healthcare-specific metrics.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Audio Input Layer](#audio-input-layer)
3. [STT Integration Layer](#stt-integration-layer)
4. [Evaluation Engine](#evaluation-engine)
5. [Reporting Layer](#reporting-layer)
6. [End-to-End Workflow](#end-to-end-workflow)
7. [Data Flow Diagrams](#data-flow-diagrams)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Lunagen STT Framework                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │    Audio     │ -> │     STT      │ -> │  Evaluation  │ -> │ Reporting │ │
│  │    Input     │    │  Integration │    │    Engine    │    │   Layer   │ │
│  │    Layer     │    │    Layer     │    │              │    │           │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                   │       │
│         v                   v                   v                   v       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ AudioChunks  │    │  Transcript  │    │  TER/NER/CRS │    │   JSON    │ │
│  │              │    │   Segments   │    │   Results    │    │ CSV/HTML  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Audio Input Layer

### Purpose

Transform raw audio files into controlled, reproducible streaming chunks that simulate real-world conditions.

### Components

#### 1. AudioLoader (`audio/loader.py`)

**Responsibility:** Load audio files and normalize format.

```
Audio File (WAV/FLAC/OGG/MP3)
           │
           v
    ┌──────────────┐
    │ AudioLoader  │
    │              │
    │ • Load file  │
    │ • Resample   │
    │ • Mono conv  │
    │ • Checksum   │
    └──────────────┘
           │
           v
    numpy array + sample_rate + checksum
```

**Key Operations:**
- **Format Detection:** Identifies audio format and codec
- **Resampling:** Converts to target sample rate (default 16kHz)
- **Channel Conversion:** Converts stereo to mono
- **Checksum Generation:** Creates deterministic hash for reproducibility

**Code Flow:**
```python
loader = AudioLoader(target_sample_rate=16000)
audio_data, sample_rate, checksum = loader.load("patient_notes.wav")

# audio_data: numpy array of samples
# sample_rate: int (e.g., 16000)
# checksum: str (MD5 hash of audio data)
```

#### 2. StreamingChunker (`audio/chunker.py`)

**Responsibility:** Simulate streaming by splitting audio into time-bounded chunks.

```
Audio Data (numpy array)
           │
           v
    ┌────────────────────┐
    │  StreamingChunker  │
    │                    │
    │  StreamingProfile  │
    │  ┌──────────────┐  │
    │  │ chunk_size   │  │
    │  │ jitter       │  │
    │  │ overlap      │  │
    │  │ network_delay│  │
    │  └──────────────┘  │
    └────────────────────┘
           │
           v
    AsyncIterator[AudioChunk]
```

**Key Operations:**
- **Chunking:** Splits audio into fixed-size segments
- **Jitter Simulation:** Adds controlled timing variability
- **Overlap Handling:** Maintains context between chunks
- **Network Simulation:** Introduces realistic delays

**Chunking Algorithm:**
```
┌─────────────────────────────────────────────────────────────┐
│                      Full Audio Signal                       │
└─────────────────────────────────────────────────────────────┘
     │          │          │          │          │
     v          v          v          v          v
┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐
│ Chunk 0 ││ Chunk 1 ││ Chunk 2 ││ Chunk 3 ││ Chunk 4 │
│ 1000ms  ││ 1000ms  ││ 1000ms  ││ 1000ms  ││ final   │
│ seq=0   ││ seq=1   ││ seq=2   ││ seq=3   ││ seq=4   │
└─────────┘└─────────┘└─────────┘└─────────┘└─────────┘
     │          │          │          │          │
     + jitter   + jitter   + jitter   + jitter   │
     ±50ms      ±50ms      ±50ms      ±50ms      │
```

#### 3. StreamingProfile (`core/config.py`)

**Responsibility:** Define streaming simulation parameters.

**Built-in Profiles:**

| Profile | Chunk Size | Jitter | Overlap | Use Case |
|---------|------------|--------|---------|----------|
| `ideal` | 1000ms | 0ms | 0ms | Baseline/reference |
| `realtime_mobile` | 1000ms | ±50ms | 100ms | Mobile networks |
| `realtime_clinical` | 500ms | ±20ms | 50ms | Clinical setting |
| `high_latency` | 1000ms | ±100ms | 100ms | Poor connectivity |

---

## STT Integration Layer

### Purpose

Abstract STT provider differences behind a unified interface for model-agnostic evaluation.

### Components

#### 1. STTAdapter Base Class (`adapters/base.py`)

**Responsibility:** Define common interface for all STT providers.

```
┌─────────────────────────────────────────┐
│            STTAdapter (ABC)             │
├─────────────────────────────────────────┤
│ Properties:                             │
│   • name: str                           │
│                                         │
│ Methods:                                │
│   • initialize() -> None                │
│   • transcribe_stream(chunks) -> segs   │
│   • transcribe_batch(chunks) -> str     │
│   • cleanup() -> None                   │
└─────────────────────────────────────────┘
           △
           │
     ┌─────┴─────┐
     │           │
┌─────────┐ ┌─────────┐
│  Mock   │ │ Whisper │
│ Adapter │ │ Adapter │
└─────────┘ └─────────┘
```

#### 2. Adapter Registry (`adapters/registry.py`)

**Responsibility:** Discover and instantiate adapters by name.

```python
# Registration
AdapterRegistry.register("whisper", WhisperAdapter)
AdapterRegistry.register("deepgram", DeepgramAdapter)

# Discovery
adapters = AdapterRegistry.list_adapters()
# ["mock", "failing_mock", "whisper", "deepgram"]

# Instantiation
adapter = AdapterRegistry.get("whisper", model_size="large")
```

#### 3. MockSTTAdapter (`adapters/mock_adapter.py`)

**Responsibility:** Testing without real STT infrastructure.

```
┌─────────────────────────────────────────┐
│           MockSTTAdapter                │
├─────────────────────────────────────────┤
│ • Preconfigured responses               │
│ • Simulates streaming behavior          │
│ • Deterministic output                  │
│ • No external dependencies              │
└─────────────────────────────────────────┘
```

### Data Flow

```
AudioChunk Stream
       │
       v
┌──────────────────┐
│   STT Adapter    │
│                  │
│ ┌──────────────┐ │
│ │   Buffer     │ │     TranscriptSegment
│ │   Manager    │ │     ┌──────────────┐
│ └──────────────┘ │ --> │ text         │
│        │         │     │ is_partial   │
│        v         │     │ is_final     │
│ ┌──────────────┐ │     │ confidence   │
│ │  STT Model   │ │     │ start_time   │
│ └──────────────┘ │     │ end_time     │
└──────────────────┘     └──────────────┘
```

---

## Evaluation Engine

### Purpose

Compute healthcare-specific metrics that capture clinically significant transcription errors.

### Metric Computation Pipeline

```
Ground Truth Text + Predicted Text
              │
              v
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ TER Engine  │   │ NER Engine  │   │ CRS Engine  │       │
│  │             │   │             │   │             │       │
│  │ • Lexicon   │   │ • NER       │   │ • Semantic  │       │
│  │   matching  │   │   Pipeline  │   │   Similarity│       │
│  │ • Term      │   │ • Entity    │   │ • Entity    │       │
│  │   alignment │   │   Alignment │   │   Continuity│       │
│  │ • Error     │   │ • Accuracy  │   │ • Negation  │       │
│  │   detection │   │   Metrics   │   │   Detector  │       │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│         │                 │                 │               │
│         v                 v                 v               │
│    TERResult         NERResult         CRSResult           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
              │
              v
       BenchmarkResult
```

### TER Engine Detail

**Purpose:** Measure medical terminology accuracy.

```
┌─────────────────────────────────────────────────────────────┐
│                       TER Engine                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Ground Truth Text          Predicted Text                  │
│         │                         │                         │
│         v                         v                         │
│  ┌─────────────┐           ┌─────────────┐                 │
│  │   Text      │           │    Text     │                 │
│  │ Normalizer  │           │  Normalizer │                 │
│  │             │           │             │                 │
│  │ • lowercase │           │ • lowercase │                 │
│  │ • abbrevs   │           │ • abbrevs   │                 │
│  │ • dosages   │           │ • dosages   │                 │
│  └──────┬──────┘           └──────┬──────┘                 │
│         │                         │                         │
│         v                         v                         │
│  ┌─────────────┐           ┌─────────────┐                 │
│  │   Term      │           │    Term     │                 │
│  │ Extraction  │           │ Extraction  │                 │
│  │             │           │             │                 │
│  │ Lexicon     │           │  Lexicon    │                 │
│  │ Matching    │           │  Matching   │                 │
│  └──────┬──────┘           └──────┬──────┘                 │
│         │                         │                         │
│         v                         v                         │
│  GT Terms: [metformin, 500mg, diabetes]                    │
│  Pred Terms: [methotrexate, 500mg, diabetes]               │
│         │                         │                         │
│         └───────────┬─────────────┘                         │
│                     v                                       │
│              ┌─────────────┐                                │
│              │    Term     │                                │
│              │  Alignment  │                                │
│              │             │                                │
│              │ Greedy      │                                │
│              │ Matching    │                                │
│              └──────┬──────┘                                │
│                     │                                       │
│                     v                                       │
│              Error Classification                           │
│              ┌─────────────────────────────────────┐       │
│              │ SUBSTITUTION: metformin→methotrexate│       │
│              │ MATCH: 500mg                        │       │
│              │ MATCH: diabetes                     │       │
│              └─────────────────────────────────────┘       │
│                     │                                       │
│                     v                                       │
│              TERResult                                      │
│              • overall_ter: 0.33                           │
│              • category_ter: {drug: 1.0, dosage: 0.0}      │
│              • substitutions: 1                            │
│              • deletions: 0                                │
│              • insertions: 0                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### NER Engine Detail

**Purpose:** Measure entity extraction accuracy.

```
┌─────────────────────────────────────────────────────────────┐
│                       NER Engine                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Ground Truth Text          Predicted Text                  │
│         │                         │                         │
│         v                         v                         │
│  ┌─────────────┐           ┌─────────────┐                 │
│  │    NER      │           │    NER      │                 │
│  │  Pipeline   │           │  Pipeline   │                 │
│  │             │           │             │                 │
│  │ Pattern-    │           │ Pattern-    │                 │
│  │ based or    │           │ based or    │                 │
│  │ ML-based    │           │ ML-based    │                 │
│  └──────┬──────┘           └──────┬──────┘                 │
│         │                         │                         │
│         v                         v                         │
│  GT Entities:               Pred Entities:                  │
│  ┌─────────────────┐       ┌─────────────────┐             │
│  │ DRUG: metformin │       │ DRUG: metformin │             │
│  │ SYMPTOM: pain   │       │ DIAGNOSIS: pain │ (wrong!)    │
│  │ DIAGNOSIS: HTN  │       │ DIAGNOSIS: HTN  │             │
│  └─────────────────┘       └─────────────────┘             │
│         │                         │                         │
│         └───────────┬─────────────┘                         │
│                     v                                       │
│              ┌─────────────┐                                │
│              │   Entity    │                                │
│              │   Aligner   │                                │
│              │             │                                │
│              │ • Span IOU  │                                │
│              │ • Text sim  │                                │
│              │ • Label req │                                │
│              └──────┬──────┘                                │
│                     │                                       │
│                     v                                       │
│              Match Classification                           │
│              ┌─────────────────────────────────────┐       │
│              │ EXACT: metformin (drug)            │       │
│              │ LABEL_MISMATCH: pain               │       │
│              │ EXACT: HTN (diagnosis)             │       │
│              └─────────────────────────────────────┘       │
│                     │                                       │
│                     v                                       │
│              NERResult                                      │
│              • precision: 0.67                             │
│              • recall: 0.67                                │
│              • f1_score: 0.67                              │
│              • entity_distortion_rate: 0.33                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### CRS Engine Detail

**Purpose:** Measure context preservation across streaming segments.

```
┌─────────────────────────────────────────────────────────────┐
│                       CRS Engine                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GT Segments              Pred Segments                     │
│  ┌──────────┐             ┌──────────┐                      │
│  │ Seg 1    │             │ Seg 1    │                      │
│  │ Seg 2    │             │ Seg 2    │                      │
│  │ Seg 3    │             │ Seg 3    │                      │
│  └──────────┘             └──────────┘                      │
│       │                        │                            │
│       │    ┌───────────────────┘                            │
│       │    │                                                │
│       v    v                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Component Computation                   │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │                                                      │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌──────────┐│   │
│  │  │   Semantic    │  │    Entity     │  │ Negation ││   │
│  │  │  Similarity   │  │  Continuity   │  │ Detector ││   │
│  │  │               │  │               │  │          ││   │
│  │  │ Token-based:  │  │ Track across  │  │ Pattern- ││   │
│  │  │ • Jaccard     │  │ segments:     │  │ based:   ││   │
│  │  │ • N-gram      │  │ • appearances │  │ • "no"   ││   │
│  │  │ • LCS         │  │ • disappear   │  │ • "deny" ││   │
│  │  │               │  │ • conflicts   │  │ • "not"  ││   │
│  │  │ Embedding:    │  │ • negation    │  │          ││   │
│  │  │ • Cosine sim  │  │   flips       │  │ Check    ││   │
│  │  └───────┬───────┘  └───────┬───────┘  │ consist- ││   │
│  │          │                  │          │ ency     ││   │
│  │          │                  │          └────┬─────┘│   │
│  │          v                  v               │      │   │
│  │     sim: 0.85         cont: 0.90            │      │   │
│  │          │                  │               │      │   │
│  │          └────────┬─────────┘               │      │   │
│  │                   │                         │      │   │
│  │                   └────────┬────────────────┘      │   │
│  │                            │                        │   │
│  └────────────────────────────┼────────────────────────┘   │
│                               │                            │
│                               v                            │
│                    Weighted Composite                      │
│             ┌────────────────────────────────┐             │
│             │ CRS = 0.4 × semantic           │             │
│             │     + 0.4 × entity_cont        │             │
│             │     + 0.2 × negation_cons      │             │
│             └────────────────────────────────┘             │
│                               │                            │
│                               v                            │
│                          CRSResult                         │
│                    • composite_score: 0.87                 │
│                    • semantic_similarity: 0.85             │
│                    • entity_continuity: 0.90               │
│                    • negation_consistency: 0.85            │
│                    • context_drift_rate: 0.02              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### SRS Engine Detail

**Purpose:** Measure model robustness under streaming conditions.

```
┌─────────────────────────────────────────────────────────────┐
│                       SRS Engine                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Same Model, Same Audio                                     │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │         │                                              │
│    v         v                                              │
│  ┌──────┐  ┌──────────┐                                    │
│  │Ideal │  │ Realtime │                                    │
│  │Profile│  │ Profile  │                                    │
│  └──┬───┘  └────┬─────┘                                    │
│     │           │                                           │
│     v           v                                           │
│  Benchmark   Benchmark                                      │
│  Result      Result                                         │
│     │           │                                           │
│     v           v                                           │
│  TER: 0.05   TER: 0.08                                     │
│  NER: 0.92   NER: 0.87                                     │
│  CRS: 0.90   CRS: 0.82                                     │
│     │           │                                           │
│     └─────┬─────┘                                           │
│           │                                                 │
│           v                                                 │
│    Compute Ratios                                           │
│    ┌────────────────────────────────┐                      │
│    │ TER degradation: (0.08-0.05)   │                      │
│    │ NER degradation: (0.92-0.87)   │                      │
│    │ CRS degradation: (0.90-0.82)   │                      │
│    │                                │                      │
│    │ SRS = realtime / ideal         │                      │
│    │     = 0.92                     │                      │
│    └────────────────────────────────┘                      │
│           │                                                 │
│           v                                                 │
│       SRSResult                                             │
│       • composite_srs: 0.92                                │
│       • degradation: {ter: 0.6, ner: 0.05, crs: 0.09}      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Reporting Layer

### Purpose

Transform evaluation results into actionable reports for different audiences.

### Report Generation Pipeline

```
BenchmarkSummary
       │
       v
┌──────────────────┐
│ ReportGenerator  │
│                  │
│  ReportConfig    │
│  ┌────────────┐  │
│  │json: true  │  │
│  │csv: true   │  │
│  │html: true  │  │
│  │risk: true  │  │
│  └────────────┘  │
└────────┬─────────┘
         │
    ┌────┼────┬────────┐
    │    │    │        │
    v    v    v        v
┌─────┐┌────┐┌────┐┌──────────┐
│JSON ││CSV ││HTML││ Clinical │
│     ││    ││    ││   Risk   │
└─────┘└────┘└────┘└──────────┘
```

### Report Types

#### JSON Report
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "summary": {
    "total_files": 10,
    "avg_ter": 0.05,
    "avg_ner_f1": 0.92,
    "avg_crs": 0.88,
    "streaming_profile": "ideal",
    "adapter_name": "whisper"
  },
  "results": [
    {
      "audio_id": "sample_001",
      "ter": { "overall_ter": 0.05, ... },
      "ner": { "precision": 0.90, ... },
      "crs": { "composite_score": 0.88, ... }
    }
  ]
}
```

#### CSV Report
```
audio_id,ter,ner_f1,crs,streaming_profile,adapter_name
sample_001,0.0500,0.9200,0.8800,ideal,whisper
sample_002,0.0800,0.8900,0.8500,ideal,whisper
```

#### HTML Report
```html
<!DOCTYPE html>
<html>
<head>
  <title>Lunagen STT Benchmark Report</title>
  <style>
    .score-good { color: green; }
    .score-bad { color: red; }
  </style>
</head>
<body>
  <h1>Lunagen STT Benchmark Report</h1>
  <div class="summary-cards">
    <div class="card">Total Files: 10</div>
    <div class="card">Avg TER: 5.0%</div>
    <div class="card">Avg NER F1: 92.0%</div>
    <div class="card">Avg CRS: 88.0%</div>
  </div>
  <table>
    <tr><th>Audio ID</th><th>TER</th><th>NER F1</th><th>CRS</th></tr>
    <tr><td>sample_001</td><td class="score-good">5%</td>...</tr>
  </table>
</body>
</html>
```

#### Clinical Risk Report
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "total_files": 10,
  "critical_count": 2,
  "high_count": 5,
  "medium_count": 3,
  "low_count": 0,
  "items": [
    {
      "risk_level": "critical",
      "risk_type": "drug_substitution",
      "audio_id": "sample_001",
      "description": "Drug name substitution: 'metformin' -> 'methotrexate'",
      "original": "metformin",
      "predicted": "methotrexate",
      "category": "drug"
    },
    {
      "risk_level": "high",
      "risk_type": "negation_flip",
      "audio_id": "sample_002",
      "description": "Negation flip in segment: 'Patient denies...'",
      "original": "Patient denies chest pain",
      "predicted": "Patient has chest pain"
    }
  ]
}
```

---

## End-to-End Workflow

### Complete Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Complete Lunagen STT Workflow                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. INPUT PREPARATION                                                   │
│  ───────────────────                                                    │
│                                                                          │
│     Audio Files              Ground Truth                               │
│     ┌─────────┐              ┌─────────┐                                │
│     │ .wav    │              │ .txt    │                                │
│     │ .flac   │              │ .json   │                                │
│     └────┬────┘              └────┬────┘                                │
│          │                        │                                     │
│          v                        │                                     │
│  2. AUDIO PROCESSING              │                                     │
│  ───────────────────              │                                     │
│                                   │                                     │
│     ┌─────────────┐               │                                     │
│     │ AudioLoader │               │                                     │
│     │ • Load      │               │                                     │
│     │ • Resample  │               │                                     │
│     │ • Normalize │               │                                     │
│     └──────┬──────┘               │                                     │
│            │                      │                                     │
│            v                      │                                     │
│     ┌─────────────────┐           │                                     │
│     │StreamingChunker │           │                                     │
│     │ • Chunk         │           │                                     │
│     │ • Add jitter    │           │                                     │
│     │ • Simulate net  │           │                                     │
│     └────────┬────────┘           │                                     │
│              │                    │                                     │
│              v                    │                                     │
│  3. STT TRANSCRIPTION             │                                     │
│  ────────────────────             │                                     │
│                                   │                                     │
│     ┌─────────────┐               │                                     │
│     │ STT Adapter │               │                                     │
│     │             │               │                                     │
│     │ • Buffer    │               │                                     │
│     │ • Transcribe│               │                                     │
│     │ • Stream    │               │                                     │
│     └──────┬──────┘               │                                     │
│            │                      │                                     │
│            v                      v                                     │
│  4. EVALUATION                                                          │
│  ─────────────                                                          │
│                                                                          │
│     ┌───────────────────────────────────────────────────┐              │
│     │              BenchmarkRunner                       │              │
│     │                                                    │              │
│     │  Predicted Text ─────────┐                        │              │
│     │  Ground Truth ───────────┼─────────┐              │              │
│     │                          │         │              │              │
│     │                          v         v              │              │
│     │                    ┌───────────────────┐          │              │
│     │                    │                   │          │              │
│     │     ┌──────────┐   │   ┌──────────┐   │          │              │
│     │     │   TER    │   │   │   NER    │   │          │              │
│     │     │  Engine  │   │   │  Engine  │   │          │              │
│     │     └────┬─────┘   │   └────┬─────┘   │          │              │
│     │          │         │        │         │          │              │
│     │          │         │        │         │          │              │
│     │     ┌────┴─────────┴────────┴─────────┴───┐      │              │
│     │     │                                      │      │              │
│     │     │  Segment Pairs ──► CRS Engine       │      │              │
│     │     │                                      │      │              │
│     │     └──────────────────┬───────────────────┘      │              │
│     │                        │                          │              │
│     │                        v                          │              │
│     │                 BenchmarkResult                   │              │
│     └────────────────────────┬──────────────────────────┘              │
│                              │                                         │
│                              v                                         │
│  5. AGGREGATION                                                        │
│  ──────────────                                                        │
│                                                                          │
│     ┌─────────────────────────────────────┐                            │
│     │         BenchmarkSummary            │                            │
│     │                                     │                            │
│     │  • total_files: 10                  │                            │
│     │  • avg_ter: 0.05                    │                            │
│     │  • avg_ner_f1: 0.92                 │                            │
│     │  • avg_crs: 0.88                    │                            │
│     │  • results: [...]                   │                            │
│     └─────────────────┬───────────────────┘                            │
│                       │                                                 │
│                       v                                                 │
│  6. REPORTING                                                          │
│  ────────────                                                          │
│                                                                          │
│     ┌─────────────────────────────────────┐                            │
│     │         ReportGenerator             │                            │
│     │                                     │                            │
│     │  ┌─────┐ ┌────┐ ┌────┐ ┌──────┐   │                            │
│     │  │JSON │ │CSV │ │HTML│ │ Risk │   │                            │
│     │  └──┬──┘ └─┬──┘ └─┬──┘ └──┬───┘   │                            │
│     │     │      │      │       │       │                            │
│     └─────┼──────┼──────┼───────┼───────┘                            │
│           │      │      │       │                                     │
│           v      v      v       v                                     │
│     ┌─────────────────────────────────┐                               │
│     │        Output Files             │                               │
│     │                                 │                               │
│     │  results.json                   │                               │
│     │  results.csv                    │                               │
│     │  report.html                    │                               │
│     │  clinical_risk.json             │                               │
│     └─────────────────────────────────┘                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagrams

### Sequence Diagram: Single File Evaluation

```
User          BenchmarkRunner    AudioLoader    Chunker    STTAdapter    Metrics
  │                 │                │            │            │           │
  │ evaluate()      │                │            │            │           │
  │────────────────>│                │            │            │           │
  │                 │                │            │            │           │
  │                 │ load()         │            │            │           │
  │                 │───────────────>│            │            │           │
  │                 │                │            │            │           │
  │                 │ audio_data     │            │            │           │
  │                 │<───────────────│            │            │           │
  │                 │                │            │            │           │
  │                 │ stream_audio() │            │            │           │
  │                 │───────────────────────────> │            │           │
  │                 │                │            │            │           │
  │                 │                │   AudioChunks           │           │
  │                 │<─────────────────────────── │            │           │
  │                 │                │            │            │           │
  │                 │ transcribe_stream()         │            │           │
  │                 │─────────────────────────────────────────>│           │
  │                 │                │            │            │           │
  │                 │                │   TranscriptSegments    │           │
  │                 │<─────────────────────────────────────────│           │
  │                 │                │            │            │           │
  │                 │ compute_ter()  │            │            │           │
  │                 │───────────────────────────────────────────────────> │
  │                 │                │            │            │           │
  │                 │ compute_ner()  │            │            │           │
  │                 │───────────────────────────────────────────────────> │
  │                 │                │            │            │           │
  │                 │ compute_crs()  │            │            │           │
  │                 │───────────────────────────────────────────────────> │
  │                 │                │            │            │           │
  │                 │                │   TER/NER/CRS Results   │           │
  │                 │<─────────────────────────────────────────────────── │
  │                 │                │            │            │           │
  │ BenchmarkResult │                │            │            │           │
  │<────────────────│                │            │            │           │
  │                 │                │            │            │           │
```

### State Diagram: Audio Chunk Lifecycle

```
                    ┌─────────────┐
                    │   Created   │
                    └──────┬──────┘
                           │
                           v
                    ┌─────────────┐
                    │   Queued    │
                    └──────┬──────┘
                           │
               ┌───────────┴───────────┐
               │                       │
               v                       v
        ┌─────────────┐         ┌─────────────┐
        │  Delayed    │         │  Immediate  │
        │  (jitter)   │         │             │
        └──────┬──────┘         └──────┬──────┘
               │                       │
               └───────────┬───────────┘
                           │
                           v
                    ┌─────────────┐
                    │   Sent to   │
                    │     STT     │
                    └──────┬──────┘
                           │
                           v
                    ┌─────────────┐
                    │ Transcribed │
                    └──────┬──────┘
                           │
               ┌───────────┴───────────┐
               │                       │
               v                       v
        ┌─────────────┐         ┌─────────────┐
        │   Partial   │         │    Final    │
        │   Result    │         │   Result    │
        └─────────────┘         └─────────────┘
```

---

## Performance Considerations

### Memory Management

```
Audio File (large)
      │
      v
┌─────────────────────────────────┐
│  Streaming (not full buffering) │
│                                 │
│  • Chunk-by-chunk processing    │
│  • Generator-based iteration    │
│  • Memory-bounded buffers       │
└─────────────────────────────────┘
      │
      v
Constant Memory Usage
```

### Lazy Initialization

```python
class BenchmarkRunner:
    def __init__(self):
        self._ter_engine = None  # Not created yet
        self._ner_engine = None
        self._crs_engine = None

    @property
    def ter_engine(self):
        if self._ter_engine is None:
            self._ter_engine = TEREngine()  # Created on first use
        return self._ter_engine
```

### Async Processing

```python
async def evaluate_batch(files: list[Path]) -> list[BenchmarkResult]:
    # Process files concurrently
    tasks = [evaluate_single(f) for f in files]
    return await asyncio.gather(*tasks)
```

---

## Error Handling

### Error Hierarchy

```
HSSTBError (base)
    │
    ├── AudioError
    │   ├── AudioLoadError
    │   └── AudioFormatError
    │
    ├── STTAdapterError
    │   ├── STTConnectionError
    │   └── STTTranscriptionError
    │
    ├── LexiconError
    │   ├── LexiconLoadError
    │   └── LexiconLookupError
    │
    ├── MetricComputationError
    │   ├── TERComputationError
    │   ├── NERComputationError
    │   └── CRSComputationError
    │
    ├── EvaluationError
    │   └── BenchmarkError
    │
    └── ReportGenerationError
```

### Graceful Degradation

```python
async def evaluate_batch(files):
    results = []
    errors = []

    for file in files:
        try:
            result = await evaluate_single(file)
            results.append(result)
        except EvaluationError as e:
            errors.append((file, e))
            # Continue with remaining files

    return results, errors
```
