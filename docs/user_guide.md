# HSTTB User Guide

Healthcare Streaming STT Benchmarking Framework - A comprehensive guide to evaluating speech-to-text systems for healthcare applications.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [CLI Reference](#cli-reference)
6. [Python API](#python-api)
7. [Metrics Explained](#metrics-explained)
8. [Streaming Profiles](#streaming-profiles)
9. [STT Adapters](#stt-adapters)
10. [Report Generation](#report-generation)
11. [Best Practices](#best-practices)

---

## Overview

HSTTB is a model-agnostic evaluation framework designed to benchmark streaming Speech-to-Text (STT) systems for healthcare applications. It measures three core metrics:

| Metric | Full Name | Purpose |
|--------|-----------|---------|
| **TER** | Term Error Rate | Medical terminology accuracy |
| **NER** | Named Entity Recognition Accuracy | Medical entity preservation |
| **CRS** | Context Retention Score | Streaming context continuity |

### Why Healthcare-Specific Benchmarking?

Standard word error rate (WER) doesn't capture healthcare-critical errors:

- **Drug substitutions**: "metformin" → "methotrexate" (life-threatening)
- **Negation flips**: "no chest pain" → "chest pain" (changes diagnosis)
- **Dosage errors**: "500mg" → "50mg" (dangerous)

HSTTB focuses specifically on these clinically significant transcription failures.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd hsttb

# Install in development mode
pip install -e ".[dev]"

# Or using uv
uv pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check version
python -c "from hsttb import __version__; print(__version__)"

# Run tests
pytest tests/ -v
```

---

## Quick Start

### 1. Basic Transcription with Streaming Simulation

```python
import asyncio
from hsttb.audio.loader import AudioLoader
from hsttb.audio.chunker import StreamingChunker
from hsttb.adapters.mock_adapter import MockSTTAdapter
from hsttb.core.config import StreamingProfile

async def transcribe_audio():
    # Load audio
    loader = AudioLoader()
    audio_data, sample_rate, _ = loader.load("patient_notes.wav")

    # Set up streaming simulation
    profile = StreamingProfile(profile_name="realtime")
    chunker = StreamingChunker(profile)

    # Set up STT adapter (use mock for testing)
    adapter = MockSTTAdapter(responses=["Patient reports chest pain."])
    await adapter.initialize()

    # Stream and transcribe
    async for chunk in chunker.stream_audio(audio_data, sample_rate):
        segments = await adapter.transcribe_stream(iter([chunk]))
        for segment in segments:
            print(f"Transcript: {segment.text}")

asyncio.run(transcribe_audio())
```

### 2. Compute TER (Term Error Rate)

```python
from hsttb.metrics.ter import TEREngine, compute_ter
from hsttb.lexicons.mock_lexicon import MockMedicalLexicon

# Quick computation
ground_truth = "Patient takes metformin 500mg for type 2 diabetes"
prediction = "Patient takes methotrexate 500mg for type 2 diabetes"

result = compute_ter(ground_truth, prediction)

print(f"TER: {result.overall_ter:.2%}")
print(f"Substitutions: {len(result.substitutions)}")
print(f"Deletions: {len(result.deletions)}")
```

### 3. Compute NER Accuracy

```python
from hsttb.metrics.ner import NEREngine, compute_ner_accuracy

ground_truth = "Patient denies chest pain. History of diabetes."
prediction = "Patient has chest pain. History of diabetes."

result = compute_ner_accuracy(ground_truth, prediction)

print(f"Precision: {result.precision:.2%}")
print(f"Recall: {result.recall:.2%}")
print(f"F1 Score: {result.f1_score:.2%}")
```

### 4. Compute CRS (Context Retention Score)

```python
from hsttb.metrics.crs import CRSEngine, compute_crs

gt_segments = [
    "Patient presents with chest pain.",
    "No prior history of cardiac issues.",
    "Prescribed aspirin 81mg daily."
]

pred_segments = [
    "Patient presents with chest pain.",
    "Prior history of cardiac issues.",  # Negation lost!
    "Prescribed aspirin 81mg daily."
]

result = compute_crs(gt_segments, pred_segments)

print(f"Composite CRS: {result.composite_score:.2%}")
print(f"Semantic Similarity: {result.semantic_similarity:.2%}")
print(f"Entity Continuity: {result.entity_continuity:.2%}")
print(f"Negation Consistency: {result.negation_consistency:.2%}")
```

### 5. Run Full Benchmark

```python
import asyncio
from pathlib import Path
from hsttb.evaluation.runner import BenchmarkRunner
from hsttb.adapters.mock_adapter import MockSTTAdapter

async def run_benchmark():
    adapter = MockSTTAdapter(responses=["Sample transcript."])
    runner = BenchmarkRunner(adapter=adapter, profile_name="ideal")

    result = await runner.evaluate_text(
        ground_truth="Patient takes metformin for diabetes.",
        audio_id="sample_001"
    )

    print(f"TER: {result.ter.overall_ter:.2%}")
    print(f"NER F1: {result.ner.f1_score:.2%}")
    print(f"CRS: {result.crs.composite_score:.2%}")

asyncio.run(run_benchmark())
```

### 6. Generate Reports

```python
from pathlib import Path
from hsttb.reporting import generate_report, ReportGenerator

# Using convenience function
reports = generate_report(
    summary=benchmark_summary,
    output_dir=Path("reports"),
    generate_json=True,
    generate_csv=True,
    generate_html=True,
    generate_clinical_risk=True
)

print(f"JSON report: {reports['json']}")
print(f"HTML report: {reports['html']}")
print(f"Clinical risk: {reports['clinical_risk']}")
```

---

## Core Concepts

### Audio Processing Pipeline

```
Audio File → Loader → Chunker → STT Adapter → Transcript Segments
                ↓
         Streaming Profile
         (chunk size, jitter, delay)
```

### Evaluation Pipeline

```
Ground Truth + Prediction
        ↓
   ┌────┴────┐
   ↓    ↓    ↓
  TER  NER  CRS
   ↓    ↓    ↓
   └────┬────┘
        ↓
  BenchmarkResult
        ↓
  Report Generation
```

### Key Data Types

| Type | Description | Location |
|------|-------------|----------|
| `AudioChunk` | Audio data with metadata | `core/types.py` |
| `TranscriptSegment` | STT output segment | `core/types.py` |
| `Entity` | Medical entity with span | `core/types.py` |
| `MedicalTerm` | Lexicon-matched term | `core/types.py` |
| `TERResult` | Term error metrics | `core/types.py` |
| `NERResult` | Entity accuracy metrics | `core/types.py` |
| `CRSResult` | Context retention metrics | `core/types.py` |
| `BenchmarkResult` | Combined metrics | `core/types.py` |

---

## CLI Reference

### Available Commands

```bash
# Transcribe audio file
hsttb transcribe audio.wav --adapter mock --profile ideal

# List streaming profiles
hsttb profiles

# List registered adapters
hsttb adapters

# Show audio file information
hsttb info audio.wav

# Preview chunk boundaries
hsttb simulate audio.wav --profile realtime_mobile
```

### Command Options

#### `transcribe`

```bash
hsttb transcribe <audio_file> [OPTIONS]

Options:
  --adapter TEXT    STT adapter name (default: mock)
  --profile TEXT    Streaming profile (default: ideal)
  --output TEXT     Output file for transcript
  --verbose         Show detailed output
```

#### `profiles`

```bash
hsttb profiles

# Output:
# ideal: Ideal conditions for baseline measurement
# realtime_mobile: Mobile network simulation
# realtime_clinical: Clinical environment simulation
# high_latency: High latency network simulation
```

---

## Python API

### Audio Module

```python
from hsttb.audio.loader import AudioLoader
from hsttb.audio.chunker import StreamingChunker

# Load audio
loader = AudioLoader(target_sample_rate=16000)
audio_data, sample_rate, checksum = loader.load("audio.wav")

# Get audio info
info = loader.get_info("audio.wav")
print(f"Duration: {info['duration_ms']}ms")
print(f"Channels: {info['channels']}")

# Stream audio
chunker = StreamingChunker(profile, seed=42)  # Deterministic
async for chunk in chunker.stream_audio(audio_data, sample_rate):
    print(f"Chunk {chunk.sequence_id}: {chunk.duration_ms}ms")
```

### Metrics Module

```python
from hsttb.metrics.ter import TEREngine
from hsttb.metrics.ner import NEREngine
from hsttb.metrics.crs import CRSEngine
from hsttb.metrics.srs import SRSEngine

# TER Engine
ter_engine = TEREngine()
ter_result = ter_engine.compute(ground_truth, prediction)

# NER Engine
ner_engine = NEREngine()
ner_result = ner_engine.compute(ground_truth, prediction)

# CRS Engine
crs_engine = CRSEngine()
crs_result = crs_engine.compute(gt_segments, pred_segments)

# SRS Engine (Streaming Robustness)
srs_engine = SRSEngine()
srs_result = srs_engine.compute(
    ideal_result=ideal_benchmark,
    realtime_result=realtime_benchmark
)
```

### Lexicons Module

```python
from hsttb.lexicons.mock_lexicon import MockMedicalLexicon
from hsttb.lexicons.unified import UnifiedMedicalLexicon

# Single lexicon
lexicon = MockMedicalLexicon()
entry = lexicon.lookup("metformin")
print(f"Category: {entry.category}")
print(f"Source: {entry.source}")

# Unified lookup across multiple sources
unified = UnifiedMedicalLexicon()
unified.add_source("mock", MockMedicalLexicon())
entry = unified.lookup("diabetes")
```

### Evaluation Module

```python
from hsttb.evaluation.runner import BenchmarkRunner, BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    compute_ter=True,
    compute_ner=True,
    compute_crs=True
)

# Run benchmark
runner = BenchmarkRunner(adapter, profile_name="ideal", config=config)
result = await runner.evaluate_text(ground_truth, audio_id="test")

# Access results
print(f"TER: {result.ter.overall_ter}")
print(f"NER F1: {result.ner.f1_score}")
print(f"CRS: {result.crs.composite_score}")
```

### Reporting Module

```python
from hsttb.reporting import ReportGenerator, ReportConfig

# Configure reports
config = ReportConfig(
    generate_json=True,
    generate_csv=True,
    generate_html=True,
    generate_clinical_risk=True,
    include_detailed_errors=True
)

# Generate reports
generator = ReportGenerator(output_dir, config)
reports = generator.generate_all(benchmark_summary)

# Access individual reports
json_path = generator.generate_json(summary)
csv_path = generator.generate_csv(summary)
html_path = generator.generate_html(summary)
risk_path = generator.generate_clinical_risk(summary)
```

---

## Metrics Explained

### TER (Term Error Rate)

Measures accuracy of medical terminology transcription.

**Formula:**
```
TER = (Substitutions + Deletions + Insertions) / Total Ground Truth Terms
```

**Categories tracked:**
- Drug names (e.g., metformin, aspirin)
- Diagnoses (e.g., diabetes, hypertension)
- Dosages (e.g., 500mg, twice daily)
- Anatomy (e.g., left arm, chest)
- Procedures (e.g., MRI, blood test)

**Interpretation:**
- TER = 0%: Perfect medical term transcription
- TER < 5%: Excellent
- TER 5-10%: Good
- TER > 10%: Needs improvement

### NER Accuracy

Measures entity extraction quality using precision, recall, and F1.

**Metrics:**
- **Precision**: Correct entities / Total predicted entities
- **Recall**: Correct entities / Total ground truth entities
- **F1 Score**: Harmonic mean of precision and recall
- **Entity Distortion Rate**: Partially correct entities
- **Entity Omission Rate**: Missing entities

**Entity Types:**
- DRUG, DIAGNOSIS, SYMPTOM, ANATOMY, PROCEDURE, LAB_VALUE, DOSAGE

### CRS (Context Retention Score)

Measures how well streaming transcription preserves context.

**Components:**
1. **Semantic Similarity** (40%): Overall meaning preservation
2. **Entity Continuity** (40%): Entity consistency across segments
3. **Negation Consistency** (20%): Negation preservation

**Formula:**
```
CRS = 0.4 × SemanticSim + 0.4 × EntityCont + 0.2 × NegationCons
```

**Additional Metrics:**
- **Context Drift Rate**: How much quality degrades over segments

### SRS (Streaming Robustness Score)

Measures model degradation under streaming conditions.

**Formula:**
```
SRS = Realtime Score / Ideal Score
```

**Interpretation:**
- SRS = 1.0: No degradation under streaming
- SRS > 0.9: Robust to streaming
- SRS < 0.8: Sensitive to streaming conditions

---

## Streaming Profiles

### Built-in Profiles

| Profile | Chunk Size | Jitter | Use Case |
|---------|------------|--------|----------|
| `ideal` | 1000ms | 0ms | Baseline measurement |
| `realtime_mobile` | 1000ms | ±50ms | Mobile network simulation |
| `realtime_clinical` | 500ms | ±20ms | Clinical environment |
| `high_latency` | 1000ms | ±100ms | Poor network conditions |

### Custom Profiles

```python
from hsttb.core.config import StreamingProfile, ChunkingConfig, AudioConfig

profile = StreamingProfile(
    profile_name="custom_profile",
    description="Custom streaming configuration",
    audio=AudioConfig(
        sample_rate=16000,
        channels=1,
        bit_depth=16
    ),
    chunking=ChunkingConfig(
        chunk_size_ms=500,
        chunk_jitter_ms=30,
        overlap_ms=50
    )
)
```

### YAML Configuration

```yaml
# configs/my_profile.yaml
profile_name: my_custom_profile
description: "Custom profile for testing"
audio:
  sample_rate: 16000
  channels: 1
  bit_depth: 16
chunking:
  chunk_size_ms: 750
  chunk_jitter_ms: 25
  overlap_ms: 100
network:
  delay_ms: 50
  jitter_ms: 20
```

---

## STT Adapters

### Adapter Interface

All STT adapters implement the `STTAdapter` base class:

```python
from hsttb.adapters.base import STTAdapter

class CustomAdapter(STTAdapter):
    @property
    def name(self) -> str:
        return "custom"

    async def initialize(self) -> None:
        # Load models, connect to services
        pass

    async def transcribe_stream(self, audio_stream):
        # Process streaming audio
        async for chunk in audio_stream:
            yield TranscriptSegment(...)

    async def cleanup(self) -> None:
        # Release resources
        pass
```

### Built-in Adapters

1. **MockSTTAdapter**: For testing without real STT
2. **FailingMockAdapter**: For error handling tests

### Registering Custom Adapters

```python
from hsttb.adapters.registry import AdapterRegistry

# Register adapter
AdapterRegistry.register("my_adapter", MyCustomAdapter)

# Get adapter
adapter = AdapterRegistry.get("my_adapter")

# List available adapters
adapters = AdapterRegistry.list_adapters()
```

---

## Report Generation

### JSON Report

Contains detailed metrics in machine-readable format:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "summary": {
    "total_files": 10,
    "avg_ter": 0.05,
    "avg_ner_f1": 0.92,
    "avg_crs": 0.88
  },
  "results": [...]
}
```

### CSV Report

Tabular format for spreadsheet analysis:

```csv
audio_id,ter,ner_f1,crs,streaming_profile,adapter_name
sample_001,0.05,0.92,0.88,ideal,mock
sample_002,0.08,0.89,0.85,ideal,mock
```

### HTML Report

Human-readable summary with styled tables and color-coded scores.

### Clinical Risk Report

Identifies critical transcription errors:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "total_files": 10,
  "critical_count": 2,
  "high_count": 5,
  "items": [
    {
      "risk_level": "critical",
      "risk_type": "drug_substitution",
      "audio_id": "sample_001",
      "description": "Drug name substitution: 'metformin' -> 'methotrexate'",
      "original": "metformin",
      "predicted": "methotrexate"
    }
  ]
}
```

---

## Best Practices

### 1. Use Deterministic Streaming

Always use a fixed seed for reproducible benchmarks:

```python
chunker = StreamingChunker(profile, seed=42)
```

### 2. Test Under Multiple Profiles

Compare model performance across conditions:

```python
for profile in ["ideal", "realtime_mobile", "high_latency"]:
    runner = BenchmarkRunner(adapter, profile_name=profile)
    result = await runner.evaluate_text(ground_truth, audio_id)
    print(f"{profile}: TER={result.ter.overall_ter:.2%}")
```

### 3. Focus on Clinical Risk

Prioritize fixing critical errors first:

```python
risk_report = generator._analyze_clinical_risks(summary)
critical = [i for i in risk_report.items if i.risk_level == "critical"]
for item in critical:
    print(f"CRITICAL: {item.description}")
```

### 4. Monitor Category-Specific TER

Drug and dosage errors are most critical:

```python
ter_result = ter_engine.compute(gt, pred)
print(f"Drug TER: {ter_result.category_ter.get('drug', 0):.2%}")
print(f"Dosage TER: {ter_result.category_ter.get('dosage', 0):.2%}")
```

### 5. Use SRS for Model Comparison

SRS helps identify streaming-sensitive models:

```python
srs_result = srs_engine.compute(ideal_result, realtime_result)
if srs_result.composite_srs < 0.85:
    print("Warning: Model degrades significantly under streaming")
```

---

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure PYTHONPATH includes src directory
export PYTHONPATH=src:$PYTHONPATH
```

**Missing dependencies:**
```bash
# Install all dependencies
pip install -e ".[dev]"
```

**Test failures:**
```bash
# Run with verbose output
pytest tests/ -v --tb=long
```

### Getting Help

- Check the [CLAUDE.md](../CLAUDE.md) for project context
- Review [changelog.md](../changelog.md) for recent changes
- See [development_phases.md](../development_phases.md) for technical details

---

## Appendix

### Supported Audio Formats

- WAV (recommended)
- FLAC
- OGG
- MP3

### Medical Lexicon Sources

Currently supported via mock implementations:
- Drug names (30+ common medications)
- Diagnoses (20+ common conditions)

Future support planned for:
- RxNorm (drug terminology)
- SNOMED CT (clinical terms)
- ICD-10 (diagnosis codes)
- UMLS (unified medical language)

### Performance Tips

1. Use lazy initialization for metric engines
2. Process files in batches for large datasets
3. Use async operations for I/O-bound tasks
4. Cache lexicon lookups for repeated terms
