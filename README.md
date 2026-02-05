# Lunagen Speech-to-Text Benchmarking Tool

A model-agnostic evaluation framework to benchmark streaming Speech-to-Text (STT) systems for healthcare applications. Built for [Lunagen](https://www.lunagen.ai/).

## Why Healthcare-Specific Benchmarking?

Standard word error rate (WER) doesn't capture healthcare-critical errors:

- **Drug substitutions**: "metformin" → "methotrexate" (life-threatening)
- **Negation flips**: "no chest pain" → "chest pain" (changes diagnosis)
- **Dosage errors**: "500mg" → "50mg" (dangerous)

This tool focuses specifically on clinically significant transcription failures.

## Core Metrics

| Metric | Full Name | Purpose |
|--------|-----------|---------|
| **TER** | Term Error Rate | Medical terminology accuracy |
| **NER** | Named Entity Recognition Accuracy | Medical entity preservation |
| **CRS** | Context Retention Score | Streaming context continuity |
| **SRS** | Streaming Robustness Score | Model degradation under streaming |

## Features

- **Multi-Adapter STT Support**: Whisper (local), Deepgram, Google Cloud Speech, and Stellicare
- **Stellicare Integration**: Stream WAV files to Stellicare WSS for transcription, refine via REST API
- **Reference-Free Quality Metrics**: Perplexity, grammar, medical coherence, contradiction detection
- **Clinical Risk Scoring**: Entity assertion, dosage plausibility, clinical contradiction detection
- **Web Dashboard**: Upload, record, TTS generation, Stellicare pipeline, and evaluation in one UI
- **Medical Lexicons**: RxNorm, SNOMED CT, ICD-10 with SQLite caching and API fetching
- **NLP Pipelines**: scispaCy, MedSpaCy with configurable registry

## Quick Start

### Installation

```bash
# Clone and install
git clone <repo-url>
cd hsttb
pip install -e ".[dev,api]"
```

### Start the Web Application

```bash
uvicorn hsttb.webapp.app:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

### Tabs

| Tab | Purpose |
|-----|---------|
| **Upload** | Upload a WAV file for STT transcription via local adapters |
| **Record** | Record audio from microphone |
| **Generate TTS** | Generate audio from text using ElevenLabs |
| **History** | View TTS generation history |
| **Text Only** | Evaluate transcription accuracy from pasted text |
| **Stellicare** | Stream WAV files to Stellicare WSS, refine, and evaluate |

### Stellicare Pipeline

1. Upload one or more WAV files in the **Stellicare** tab
2. Click **Process with Stellicare** — files stream sequentially to the Stellicare WSS endpoint
3. View live transcript progress as each file completes
4. Click **Refine Transcript** to run through Stellicare's refinement API
5. Click **Use for Evaluation** to benchmark against ground truth

### CLI Benchmark

```bash
python -m hsttb.cli benchmark --audio-dir data/audio --gt-dir data/ground_truth
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=hsttb

# Run linting
ruff check src/ tests/
mypy src/
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STELLICARE_WSS_URL` | `wss://dev-lunagen.com/transcript/ws/audio?token=123456` | Stellicare WSS endpoint |
| `STELLICARE_REFINE_URL` | `https://dev-lunagen.com/transcript/refine/test` | Stellicare refine API |
| `STELLICARE_CHUNK_SIZE` | `4096` | Audio chunk size in bytes |
| `STELLICARE_CONNECTION_TIMEOUT` | `30` | WSS connection timeout (seconds) |
| `STELLICARE_READ_TIMEOUT` | `120` | Read timeout (seconds) |

## Architecture

```
Browser                    Backend                        Stellicare
  |                           |                               |
  |-- Upload WAV files ------>|                               |
  |-- WS /ws/stellicare ----->|                               |
  |-- {start, file_ids} ----->|-- WSS stream audio ---------> |
  |<-- {phrase, text} --------|<-- INTERIM|text --------------|
  |<-- {file_complete} -------|                               |
  |<-- {all_complete} --------|                               |
  |                           |                               |
  |-- POST /api/stellicare/ ->|-- PUT /transcript/refine ---->|
  |    refine                 |<-- refined transcript --------|
  |<-- refined transcript ----|                               |
```

## Project Structure

```
src/hsttb/
├── core/            # Types, config, exceptions
├── audio/           # Audio loading, streaming simulation
├── adapters/        # STT model adapters (Whisper, Deepgram, Gemini)
├── lexicons/        # Medical lexicon loaders (RxNorm, SNOMED, ICD-10)
├── nlp/             # NLP pipelines (NER, normalization, negation)
├── metrics/         # TER, NER, CRS, SRS, quality, clinical risk
├── evaluation/      # Benchmark orchestration
├── reporting/       # Reports, dashboards
└── webapp/          # FastAPI web application
    ├── app.py                 # Main FastAPI app with all endpoints
    ├── audio_handler.py       # Audio upload handling
    ├── websocket_handler.py   # Internal STT WebSocket handler
    ├── stellicare_client.py   # Stellicare WSS + refine API client
    ├── stellicare_handler.py  # Stellicare WebSocket handler
    ├── static/                # JS, CSS, logo
    └── templates/             # HTML templates
```

## License

Proprietary - Lunagen AI
