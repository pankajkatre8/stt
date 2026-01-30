# Healthcare STT Benchmarking - Development Phases

## Overview

This document breaks down the implementation into detailed development phases with specific tasks, deliverables, and acceptance criteria.

---

## Phase 1: Foundation & Infrastructure
**Duration: Week 1-2**

### 1.1 Project Setup (Days 1-2)

#### Tasks
- [ ] Initialize Python project with `pyproject.toml`
- [ ] Set up virtual environment and dependency management
- [ ] Configure linting (ruff), formatting (black), type checking (mypy)
- [ ] Set up pre-commit hooks
- [ ] Create directory structure
- [ ] Initialize git repository with `.gitignore`

#### Deliverables
```
hsttb/
├── src/hsttb/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── config.py
│       └── types.py
├── tests/
├── configs/
├── pyproject.toml
├── requirements.txt
└── README.md
```

#### Acceptance Criteria
- [ ] `pip install -e .` works
- [ ] `pytest` runs (even with no tests)
- [ ] Import `from hsttb import __version__` works

---

### 1.2 Core Types & Configuration (Days 2-3)

#### Tasks
- [ ] Define Pydantic models for configuration
- [ ] Define core data types (TranscriptSegment, AudioChunk, etc.)
- [ ] Implement configuration loading (YAML + environment variables)
- [ ] Set up logging infrastructure

#### Key Files

**`src/hsttb/core/types.py`**
```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class EntityLabel(str, Enum):
    DRUG = "DRUG"
    DOSAGE = "DOSAGE"
    SYMPTOM = "SYMPTOM"
    DIAGNOSIS = "DIAGNOSIS"
    ANATOMY = "ANATOMY"
    LAB_VALUE = "LAB_VALUE"
    PROCEDURE = "PROCEDURE"

class MedicalTermCategory(str, Enum):
    DRUG = "drug"
    DIAGNOSIS = "diagnosis"
    DOSAGE = "dosage"
    ANATOMY = "anatomy"
    PROCEDURE = "procedure"

@dataclass
class AudioChunk:
    data: bytes
    sequence_id: int
    timestamp_ms: int
    duration_ms: int
    is_final: bool = False

@dataclass
class TranscriptSegment:
    text: str
    is_partial: bool
    is_final: bool
    confidence: float
    start_time_ms: int
    end_time_ms: int
    word_timestamps: Optional[list[dict]] = None

@dataclass
class Entity:
    text: str
    label: EntityLabel
    span: tuple[int, int]
    normalized: Optional[str] = None
    negated: bool = False

@dataclass
class MedicalTerm:
    text: str
    normalized: str
    category: MedicalTermCategory
    source: str  # rxnorm, snomed, icd10
    span: tuple[int, int]
```

**`src/hsttb/core/config.py`**
```python
from pydantic import BaseModel
from pathlib import Path

class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16

class ChunkingConfig(BaseModel):
    chunk_size_ms: int = 1000
    chunk_jitter_ms: int = 0
    overlap_ms: int = 0

class StreamingProfile(BaseModel):
    profile_name: str
    description: str = ""
    audio: AudioConfig = AudioConfig()
    chunking: ChunkingConfig = ChunkingConfig()

class EvaluationConfig(BaseModel):
    ter_weight: float = 0.4
    ner_weight: float = 0.3
    crs_weight: float = 0.3
```

#### Acceptance Criteria
- [ ] Config loads from YAML file
- [ ] All types have proper type hints
- [ ] Pydantic validation works for invalid inputs

---

### 1.3 Audio Loading & Streaming Simulator (Days 3-5)

#### Tasks
- [ ] Implement audio file loader (WAV, MP3, FLAC support)
- [ ] Implement streaming chunk simulator
- [ ] Implement streaming profiles (ideal, realtime, noisy)
- [ ] Add deterministic replay (seeded randomness for jitter)
- [ ] Add audio checksum generation

#### Key Files

**`src/hsttb/audio/loader.py`**
```python
import soundfile as sf
import numpy as np
from pathlib import Path

class AudioLoader:
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def load(self, file_path: Path) -> tuple[np.ndarray, int]:
        """Load audio file and resample if needed."""
        data, sample_rate = sf.read(file_path)
        # Resample if needed
        # Convert to mono if stereo
        return data, sample_rate

    def get_checksum(self, file_path: Path) -> str:
        """Generate deterministic checksum for audio file."""
        pass
```

**`src/hsttb/audio/chunker.py`**
```python
import asyncio
from typing import AsyncIterator
from hsttb.core.types import AudioChunk
from hsttb.core.config import StreamingProfile

class StreamingChunker:
    def __init__(self, profile: StreamingProfile, seed: int = 42):
        self.profile = profile
        self.rng = np.random.default_rng(seed)

    async def stream_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> AsyncIterator[AudioChunk]:
        """Yield audio chunks according to streaming profile."""
        chunk_samples = int(
            self.profile.chunking.chunk_size_ms * sample_rate / 1000
        )

        position = 0
        sequence_id = 0

        while position < len(audio_data):
            # Apply jitter if configured
            jitter = self._get_jitter()

            chunk = audio_data[position:position + chunk_samples]

            yield AudioChunk(
                data=chunk.tobytes(),
                sequence_id=sequence_id,
                timestamp_ms=int(position / sample_rate * 1000),
                duration_ms=self.profile.chunking.chunk_size_ms,
                is_final=(position + chunk_samples >= len(audio_data))
            )

            # Simulate network delay
            if jitter > 0:
                await asyncio.sleep(jitter / 1000)

            position += chunk_samples
            sequence_id += 1

    def _get_jitter(self) -> int:
        """Get jitter value based on profile."""
        if self.profile.chunking.chunk_jitter_ms == 0:
            return 0
        return int(self.rng.uniform(
            -self.profile.chunking.chunk_jitter_ms,
            self.profile.chunking.chunk_jitter_ms
        ))
```

**`src/hsttb/audio/profiles.py`**
```python
from pathlib import Path
import yaml
from hsttb.core.config import StreamingProfile

BUILTIN_PROFILES = {
    "ideal": StreamingProfile(
        profile_name="ideal_replay_v1",
        description="Ideal conditions for baseline",
        chunking=ChunkingConfig(chunk_size_ms=1000, chunk_jitter_ms=0)
    ),
    "realtime_mobile": StreamingProfile(
        profile_name="realtime_mobile_v1",
        description="Mobile network simulation",
        chunking=ChunkingConfig(chunk_size_ms=1000, chunk_jitter_ms=50)
    ),
    "realtime_clinical": StreamingProfile(
        profile_name="realtime_clinical_v1",
        description="Clinical environment simulation",
        chunking=ChunkingConfig(chunk_size_ms=500, chunk_jitter_ms=20)
    )
}

def load_profile(name_or_path: str) -> StreamingProfile:
    """Load streaming profile by name or from YAML file."""
    if name_or_path in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name_or_path]

    path = Path(name_or_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
            return StreamingProfile(**data)

    raise ValueError(f"Unknown profile: {name_or_path}")
```

#### Acceptance Criteria
- [ ] Load WAV file and convert to 16kHz mono
- [ ] Chunk audio with configurable sizes
- [ ] Same seed produces identical chunk sequence
- [ ] Jitter varies chunk timing within bounds

---

### 1.4 STT Adapter Interface (Days 5-7)

#### Tasks
- [ ] Define abstract STT adapter interface
- [ ] Implement Whisper adapter (local)
- [ ] Implement mock adapter for testing
- [ ] Add adapter registry

#### Key Files

**`src/hsttb/adapters/base.py`**
```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from hsttb.core.types import AudioChunk, TranscriptSegment

class STTAdapter(ABC):
    """Abstract base class for STT model adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name for identification."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the adapter (load models, etc.)."""
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptSegment]:
        """
        Process streaming audio and yield transcript segments.

        Args:
            audio_stream: Async iterator of audio chunks

        Yields:
            TranscriptSegment for each partial/final result
        """
        pass

    @abstractmethod
    async def transcribe_file(self, file_path: str) -> str:
        """Transcribe complete audio file (for comparison)."""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass
```

**`src/hsttb/adapters/whisper_adapter.py`**
```python
import whisper
from hsttb.adapters.base import STTAdapter

class WhisperAdapter(STTAdapter):
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None

    @property
    def name(self) -> str:
        return f"whisper-{self.model_size}"

    async def initialize(self) -> None:
        self.model = whisper.load_model(self.model_size)

    async def transcribe_stream(self, audio_stream):
        # Whisper doesn't natively support streaming
        # Buffer chunks and transcribe periodically
        buffer = []
        async for chunk in audio_stream:
            buffer.append(chunk.data)

            # Transcribe every N chunks or on final
            if len(buffer) >= 5 or chunk.is_final:
                audio_data = b"".join(buffer)
                result = self.model.transcribe(audio_data)

                yield TranscriptSegment(
                    text=result["text"],
                    is_partial=not chunk.is_final,
                    is_final=chunk.is_final,
                    confidence=1.0,  # Whisper doesn't provide confidence
                    start_time_ms=0,
                    end_time_ms=0
                )

                if not chunk.is_final:
                    buffer = buffer[-2:]  # Keep overlap

    async def transcribe_file(self, file_path: str) -> str:
        result = self.model.transcribe(file_path)
        return result["text"]
```

**`src/hsttb/adapters/mock_adapter.py`**
```python
class MockSTTAdapter(STTAdapter):
    """Mock adapter for testing."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    async def initialize(self) -> None:
        pass

    async def transcribe_stream(self, audio_stream):
        async for chunk in audio_stream:
            if chunk.is_final:
                yield TranscriptSegment(
                    text=self.responses[self.call_count % len(self.responses)],
                    is_partial=False,
                    is_final=True,
                    confidence=0.95,
                    start_time_ms=0,
                    end_time_ms=chunk.timestamp_ms + chunk.duration_ms
                )
                self.call_count += 1
```

#### Acceptance Criteria
- [ ] Abstract interface defined with async methods
- [ ] Whisper adapter transcribes audio file
- [ ] Mock adapter works for testing
- [ ] Adapters are discoverable via registry

---

### 1.5 Basic CLI (Days 7-8)

#### Tasks
- [ ] Create CLI entry point
- [ ] Implement `transcribe` command
- [ ] Implement `list-profiles` command
- [ ] Add progress indicators

#### Key Files

**`src/hsttb/cli.py`**
```python
import click
import asyncio
from hsttb.audio.loader import AudioLoader
from hsttb.audio.chunker import StreamingChunker
from hsttb.audio.profiles import load_profile
from hsttb.adapters import get_adapter

@click.group()
def cli():
    """Healthcare STT Benchmarking Tool"""
    pass

@cli.command()
@click.argument("audio_file")
@click.option("--adapter", default="whisper", help="STT adapter to use")
@click.option("--profile", default="ideal", help="Streaming profile")
def transcribe(audio_file: str, adapter: str, profile: str):
    """Transcribe an audio file using streaming simulation."""
    asyncio.run(_transcribe(audio_file, adapter, profile))

async def _transcribe(audio_file: str, adapter_name: str, profile_name: str):
    loader = AudioLoader()
    audio_data, sample_rate = loader.load(audio_file)

    profile = load_profile(profile_name)
    chunker = StreamingChunker(profile)

    adapter = get_adapter(adapter_name)
    await adapter.initialize()

    audio_stream = chunker.stream_audio(audio_data, sample_rate)

    async for segment in adapter.transcribe_stream(audio_stream):
        if segment.is_final:
            click.echo(f"[FINAL] {segment.text}")
        else:
            click.echo(f"[PARTIAL] {segment.text}")

@cli.command()
def list_profiles():
    """List available streaming profiles."""
    from hsttb.audio.profiles import BUILTIN_PROFILES
    for name, profile in BUILTIN_PROFILES.items():
        click.echo(f"{name}: {profile.description}")

if __name__ == "__main__":
    cli()
```

#### Acceptance Criteria
- [ ] `hsttb transcribe audio.wav` works
- [ ] `hsttb list-profiles` shows available profiles
- [ ] Progress indication during transcription

---

## Phase 2: TER Engine (Medical Term Error Rate)
**Duration: Week 3**

### 2.1 Medical Lexicon Loaders (Days 1-2)

#### Tasks
- [ ] Implement UMLS RRF file parser
- [ ] Implement RxNorm loader (drug names)
- [ ] Implement SNOMED CT loader (clinical terms)
- [ ] Implement ICD-10 loader (diagnoses)
- [ ] Create unified lexicon interface
- [ ] Add caching for loaded lexicons

#### Key Files

**`src/hsttb/lexicons/base.py`**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LexiconEntry:
    term: str
    normalized: str
    code: str
    category: str
    source: str
    synonyms: list[str]

class MedicalLexicon(ABC):
    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def lookup(self, term: str) -> LexiconEntry | None:
        pass

    @abstractmethod
    def contains(self, term: str) -> bool:
        pass

    @abstractmethod
    def get_category(self, term: str) -> str | None:
        pass
```

**`src/hsttb/lexicons/rxnorm.py`**
```python
from hsttb.lexicons.base import MedicalLexicon, LexiconEntry
import sqlite3
from pathlib import Path

class RxNormLexicon(MedicalLexicon):
    def __init__(self):
        self.entries: dict[str, LexiconEntry] = {}
        self._normalized_index: dict[str, str] = {}

    def load(self, path: str) -> None:
        """Load RxNorm from UMLS RRF files or SQLite."""
        # Parse RXNCONSO.RRF for drug names
        # Build lookup indices
        pass

    def lookup(self, term: str) -> LexiconEntry | None:
        normalized = self._normalize(term)
        if normalized in self._normalized_index:
            key = self._normalized_index[normalized]
            return self.entries[key]
        return None

    def _normalize(self, term: str) -> str:
        """Normalize term for matching."""
        return term.lower().strip()
```

**`src/hsttb/lexicons/unified.py`**
```python
class UnifiedMedicalLexicon:
    """Combines multiple lexicons for comprehensive lookup."""

    def __init__(self):
        self.lexicons: dict[str, MedicalLexicon] = {}

    def add_lexicon(self, name: str, lexicon: MedicalLexicon) -> None:
        self.lexicons[name] = lexicon

    def lookup(self, term: str) -> LexiconEntry | None:
        for lexicon in self.lexicons.values():
            entry = lexicon.lookup(term)
            if entry:
                return entry
        return None

    def identify_medical_terms(self, text: str) -> list[MedicalTerm]:
        """Extract all medical terms from text."""
        # Use lexicon matching + NLP
        pass
```

#### Acceptance Criteria
- [ ] Load RxNorm and look up "metformin"
- [ ] Load SNOMED and look up "diabetes mellitus"
- [ ] Unified lookup across all lexicons
- [ ] Caching reduces reload time

---

### 2.2 Text Normalizer (Days 2-3)

#### Tasks
- [ ] Implement case normalization
- [ ] Implement plural handling
- [ ] Implement abbreviation expansion
- [ ] Implement number/dosage normalization
- [ ] Handle common medical abbreviations

#### Key Files

**`src/hsttb/nlp/normalizer.py`**
```python
import re

class MedicalTextNormalizer:
    # Common medical abbreviations
    ABBREVIATIONS = {
        "mg": "milligram",
        "ml": "milliliter",
        "bp": "blood pressure",
        "hr": "heart rate",
        "hx": "history",
        "dx": "diagnosis",
        "rx": "prescription",
        "prn": "as needed",
        "bid": "twice daily",
        "tid": "three times daily",
        "qid": "four times daily",
    }

    def normalize(self, text: str) -> str:
        """Apply all normalizations."""
        text = self.normalize_case(text)
        text = self.normalize_whitespace(text)
        text = self.expand_abbreviations(text)
        text = self.normalize_numbers(text)
        return text

    def normalize_case(self, text: str) -> str:
        return text.lower()

    def normalize_whitespace(self, text: str) -> str:
        return " ".join(text.split())

    def expand_abbreviations(self, text: str) -> str:
        for abbrev, expansion in self.ABBREVIATIONS.items():
            pattern = rf"\b{abbrev}\b"
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text

    def normalize_numbers(self, text: str) -> str:
        # "five hundred" -> "500"
        # "5mg" -> "5 mg"
        pass

    def normalize_for_comparison(
        self,
        term1: str,
        term2: str
    ) -> tuple[str, str]:
        """Normalize two terms for comparison."""
        return self.normalize(term1), self.normalize(term2)
```

#### Acceptance Criteria
- [ ] "Metformin" == "metformin" after normalization
- [ ] "500mg" == "500 mg" after normalization
- [ ] "BP" expands to "blood pressure"
- [ ] Plural handling: "tablets" matches "tablet"

---

### 2.3 Medical Term Extractor (Days 3-4)

#### Tasks
- [ ] Implement term boundary detection
- [ ] Implement lexicon-based term identification
- [ ] Integrate with scispaCy for additional coverage
- [ ] Categorize terms (drug, diagnosis, etc.)

#### Key Files

**`src/hsttb/nlp/term_extractor.py`**
```python
import spacy
from hsttb.lexicons.unified import UnifiedMedicalLexicon
from hsttb.core.types import MedicalTerm

class MedicalTermExtractor:
    def __init__(self, lexicon: UnifiedMedicalLexicon):
        self.lexicon = lexicon
        self.nlp = spacy.load("en_core_sci_md")

    def extract_terms(self, text: str) -> list[MedicalTerm]:
        """Extract medical terms from text."""
        terms = []

        # Lexicon-based extraction
        terms.extend(self._extract_via_lexicon(text))

        # NLP-based extraction (scispaCy)
        terms.extend(self._extract_via_nlp(text))

        # Deduplicate and merge
        return self._deduplicate(terms)

    def _extract_via_lexicon(self, text: str) -> list[MedicalTerm]:
        """Find terms by lexicon matching."""
        terms = []
        # Sliding window approach
        # Match against lexicon
        return terms

    def _extract_via_nlp(self, text: str) -> list[MedicalTerm]:
        """Find terms using scispaCy NER."""
        doc = self.nlp(text)
        terms = []
        for ent in doc.ents:
            entry = self.lexicon.lookup(ent.text)
            if entry:
                terms.append(MedicalTerm(
                    text=ent.text,
                    normalized=entry.normalized,
                    category=entry.category,
                    source=entry.source,
                    span=(ent.start_char, ent.end_char)
                ))
        return terms

    def _deduplicate(self, terms: list[MedicalTerm]) -> list[MedicalTerm]:
        """Remove duplicate/overlapping terms."""
        pass
```

#### Acceptance Criteria
- [ ] Extract "metformin 500mg" as drug + dosage
- [ ] Extract "type 2 diabetes" as diagnosis
- [ ] Handle overlapping terms correctly
- [ ] Return span positions for alignment

---

### 2.4 TER Computation Engine (Days 4-6)

#### Tasks
- [ ] Implement term-level alignment
- [ ] Implement substitution detection
- [ ] Implement deletion detection
- [ ] Implement insertion detection
- [ ] Compute category-wise TER
- [ ] Generate detailed error report

#### Key Files

**`src/hsttb/metrics/ter.py`**
```python
from dataclasses import dataclass
from rapidfuzz import fuzz
from hsttb.core.types import MedicalTerm, MedicalTermCategory
from hsttb.nlp.term_extractor import MedicalTermExtractor

@dataclass
class TermError:
    error_type: str  # substitution, deletion, insertion
    ground_truth_term: MedicalTerm | None
    predicted_term: MedicalTerm | None
    category: MedicalTermCategory
    similarity_score: float = 0.0

@dataclass
class TERResult:
    overall_ter: float
    category_ter: dict[str, float]
    total_terms: int
    substitutions: list[TermError]
    deletions: list[TermError]
    insertions: list[TermError]

class TEREngine:
    def __init__(
        self,
        term_extractor: MedicalTermExtractor,
        fuzzy_threshold: float = 0.85
    ):
        self.extractor = term_extractor
        self.fuzzy_threshold = fuzzy_threshold

    def compute(
        self,
        ground_truth: str,
        prediction: str
    ) -> TERResult:
        """Compute Term Error Rate between ground truth and prediction."""
        gt_terms = self.extractor.extract_terms(ground_truth)
        pred_terms = self.extractor.extract_terms(prediction)

        # Align terms
        alignment = self._align_terms(gt_terms, pred_terms)

        # Categorize errors
        substitutions = []
        deletions = []
        insertions = []

        for gt_term, pred_term, match_type in alignment:
            if match_type == "substitution":
                substitutions.append(TermError(
                    error_type="substitution",
                    ground_truth_term=gt_term,
                    predicted_term=pred_term,
                    category=gt_term.category,
                    similarity_score=self._similarity(gt_term, pred_term)
                ))
            elif match_type == "deletion":
                deletions.append(TermError(
                    error_type="deletion",
                    ground_truth_term=gt_term,
                    predicted_term=None,
                    category=gt_term.category
                ))
            elif match_type == "insertion":
                insertions.append(TermError(
                    error_type="insertion",
                    ground_truth_term=None,
                    predicted_term=pred_term,
                    category=pred_term.category
                ))

        # Compute TER
        total_errors = len(substitutions) + len(deletions) + len(insertions)
        overall_ter = total_errors / len(gt_terms) if gt_terms else 0.0

        # Category-wise TER
        category_ter = self._compute_category_ter(
            gt_terms, substitutions, deletions, insertions
        )

        return TERResult(
            overall_ter=overall_ter,
            category_ter=category_ter,
            total_terms=len(gt_terms),
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions
        )

    def _align_terms(
        self,
        gt_terms: list[MedicalTerm],
        pred_terms: list[MedicalTerm]
    ) -> list[tuple]:
        """Align ground truth and predicted terms."""
        # Use position-aware fuzzy matching
        pass

    def _similarity(self, term1: MedicalTerm, term2: MedicalTerm) -> float:
        """Compute similarity between two terms."""
        return fuzz.ratio(term1.normalized, term2.normalized) / 100

    def _compute_category_ter(self, gt_terms, subs, dels, ins) -> dict[str, float]:
        """Compute TER per category."""
        pass
```

#### Acceptance Criteria
- [ ] Detect "metformin" → "methotrexate" as substitution
- [ ] Detect missing "500mg" as deletion
- [ ] Detect extra "daily" as insertion
- [ ] Category-wise TER computed correctly
- [ ] TER = 0 for identical transcripts

---

## Phase 3: NER Accuracy Engine
**Duration: Week 4**

### 3.1 Medical NER Pipeline (Days 1-2)

#### Tasks
- [ ] Set up scispaCy with medical models
- [ ] Set up medspacy for additional features
- [ ] Implement entity extraction pipeline
- [ ] Add entity normalization

#### Key Files

**`src/hsttb/nlp/medical_ner.py`**
```python
import spacy
import medspacy
from hsttb.core.types import Entity, EntityLabel

class MedicalNERPipeline:
    def __init__(self):
        # Load scispaCy model
        self.nlp = spacy.load("en_ner_bc5cdr_md")

        # Add medspacy components
        self.nlp.add_pipe("medspacy_context")

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract medical entities from text."""
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            label = self._map_label(ent.label_)
            if label:
                entities.append(Entity(
                    text=ent.text,
                    label=label,
                    span=(ent.start_char, ent.end_char),
                    normalized=self._normalize_entity(ent),
                    negated=self._is_negated(ent)
                ))

        return entities

    def _map_label(self, spacy_label: str) -> EntityLabel | None:
        """Map spaCy labels to our entity types."""
        mapping = {
            "CHEMICAL": EntityLabel.DRUG,
            "DISEASE": EntityLabel.DIAGNOSIS,
            # Add more mappings
        }
        return mapping.get(spacy_label)

    def _normalize_entity(self, ent) -> str:
        """Normalize entity text."""
        return ent.text.lower().strip()

    def _is_negated(self, ent) -> bool:
        """Check if entity is negated using medspacy context."""
        return getattr(ent._, "is_negated", False)
```

#### Acceptance Criteria
- [ ] Extract DRUG entities from clinical text
- [ ] Extract DIAGNOSIS entities
- [ ] Detect negation ("no chest pain")
- [ ] Entity spans are accurate

---

### 3.2 Entity Alignment & Matching (Days 2-3)

#### Tasks
- [ ] Implement span-based entity alignment
- [ ] Handle span drift (off-by-N characters)
- [ ] Implement fuzzy entity matching
- [ ] Detect partial matches

#### Key Files

**`src/hsttb/metrics/entity_alignment.py`**
```python
from dataclasses import dataclass
from hsttb.core.types import Entity
from rapidfuzz import fuzz

@dataclass
class EntityMatch:
    ground_truth: Entity
    predicted: Entity | None
    match_type: str  # exact, partial, distorted, missing
    similarity: float

class EntityAligner:
    def __init__(
        self,
        span_tolerance: int = 5,
        text_threshold: float = 0.8
    ):
        self.span_tolerance = span_tolerance
        self.text_threshold = text_threshold

    def align(
        self,
        gt_entities: list[Entity],
        pred_entities: list[Entity]
    ) -> list[EntityMatch]:
        """Align ground truth and predicted entities."""
        matches = []
        used_pred = set()

        for gt_ent in gt_entities:
            best_match = self._find_best_match(
                gt_ent, pred_entities, used_pred
            )
            matches.append(best_match)
            if best_match.predicted:
                used_pred.add(id(best_match.predicted))

        # Add hallucinated entities (in pred but not in gt)
        for pred_ent in pred_entities:
            if id(pred_ent) not in used_pred:
                matches.append(EntityMatch(
                    ground_truth=None,
                    predicted=pred_ent,
                    match_type="hallucinated",
                    similarity=0.0
                ))

        return matches

    def _find_best_match(
        self,
        gt_ent: Entity,
        pred_entities: list[Entity],
        used: set
    ) -> EntityMatch:
        """Find best matching predicted entity."""
        best_score = 0
        best_pred = None

        for pred_ent in pred_entities:
            if id(pred_ent) in used:
                continue

            # Check label match
            if gt_ent.label != pred_ent.label:
                continue

            # Check span proximity
            if not self._spans_overlap(gt_ent.span, pred_ent.span):
                continue

            # Check text similarity
            score = fuzz.ratio(gt_ent.text, pred_ent.text) / 100
            if score > best_score:
                best_score = score
                best_pred = pred_ent

        if best_pred is None:
            return EntityMatch(gt_ent, None, "missing", 0.0)
        elif best_score >= self.text_threshold:
            return EntityMatch(gt_ent, best_pred, "exact", best_score)
        else:
            return EntityMatch(gt_ent, best_pred, "distorted", best_score)

    def _spans_overlap(
        self,
        span1: tuple[int, int],
        span2: tuple[int, int]
    ) -> bool:
        """Check if spans overlap within tolerance."""
        s1_start, s1_end = span1
        s2_start, s2_end = span2

        return (
            abs(s1_start - s2_start) <= self.span_tolerance or
            abs(s1_end - s2_end) <= self.span_tolerance or
            (s1_start <= s2_start <= s1_end) or
            (s2_start <= s1_start <= s2_end)
        )
```

#### Acceptance Criteria
- [ ] Align entities with exact span match
- [ ] Handle span drift up to N characters
- [ ] Detect distorted entities
- [ ] Detect hallucinated entities

---

### 3.3 NER Metrics Computation (Days 3-5)

#### Tasks
- [ ] Implement precision computation
- [ ] Implement recall computation
- [ ] Implement F1 score
- [ ] Implement entity distortion rate
- [ ] Implement entity omission rate
- [ ] Generate per-entity-type breakdown

#### Key Files

**`src/hsttb/metrics/ner.py`**
```python
from dataclasses import dataclass
from hsttb.core.types import Entity
from hsttb.nlp.medical_ner import MedicalNERPipeline
from hsttb.metrics.entity_alignment import EntityAligner, EntityMatch

@dataclass
class NERResult:
    precision: float
    recall: float
    f1_score: float
    entity_distortion_rate: float
    entity_omission_rate: float
    per_type_metrics: dict[str, dict[str, float]]
    matches: list[EntityMatch]

class NEREngine:
    def __init__(self):
        self.ner_pipeline = MedicalNERPipeline()
        self.aligner = EntityAligner()

    def compute(
        self,
        ground_truth: str,
        prediction: str
    ) -> NERResult:
        """Compute NER accuracy metrics."""
        gt_entities = self.ner_pipeline.extract_entities(ground_truth)
        pred_entities = self.ner_pipeline.extract_entities(prediction)

        matches = self.aligner.align(gt_entities, pred_entities)

        # Count match types
        exact = sum(1 for m in matches if m.match_type == "exact")
        distorted = sum(1 for m in matches if m.match_type == "distorted")
        missing = sum(1 for m in matches if m.match_type == "missing")
        hallucinated = sum(1 for m in matches if m.match_type == "hallucinated")

        # Compute metrics
        tp = exact + distorted  # Consider distorted as partial match
        fp = hallucinated
        fn = missing

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        distortion_rate = distorted / len(gt_entities) if gt_entities else 0.0
        omission_rate = missing / len(gt_entities) if gt_entities else 0.0

        # Per-type metrics
        per_type = self._compute_per_type_metrics(matches, gt_entities)

        return NERResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            entity_distortion_rate=distortion_rate,
            entity_omission_rate=omission_rate,
            per_type_metrics=per_type,
            matches=matches
        )

    def _compute_per_type_metrics(
        self,
        matches: list[EntityMatch],
        gt_entities: list[Entity]
    ) -> dict[str, dict[str, float]]:
        """Compute metrics per entity type."""
        pass
```

#### Acceptance Criteria
- [ ] Precision/Recall/F1 computed correctly
- [ ] Distortion rate identifies altered entities
- [ ] Omission rate identifies missing entities
- [ ] Per-entity-type breakdown available
- [ ] Perfect transcript yields F1 = 1.0

---

## Phase 4: Context Retention Score (CRS)
**Duration: Week 5**

### 4.1 Semantic Similarity Engine (Days 1-2)

#### Tasks
- [ ] Set up sentence-transformers
- [ ] Implement segment embedding
- [ ] Implement similarity computation
- [ ] Add biomedical embedding option

#### Key Files

**`src/hsttb/metrics/semantic_similarity.py`**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSimilarityEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def compute_segment_similarities(
        self,
        gt_segments: list[str],
        pred_segments: list[str]
    ) -> list[float]:
        """Compute similarity for each segment pair."""
        similarities = []
        for gt, pred in zip(gt_segments, pred_segments):
            similarities.append(self.compute_similarity(gt, pred))
        return similarities
```

#### Acceptance Criteria
- [ ] Similarity between identical texts = 1.0
- [ ] Semantically similar texts have high score
- [ ] Unrelated texts have low score

---

### 4.2 Entity Continuity Tracker (Days 2-3)

#### Tasks
- [ ] Track entities across segments
- [ ] Detect entity appearance/disappearance
- [ ] Detect conflicting entity attributes
- [ ] Build entity graph

#### Key Files

**`src/hsttb/metrics/entity_continuity.py`**
```python
from dataclasses import dataclass
import networkx as nx
from hsttb.core.types import Entity

@dataclass
class EntityOccurrence:
    entity: Entity
    segment_id: int
    context: str

@dataclass
class ContinuityResult:
    continuity_score: float
    entities_tracked: int
    discontinuities: list[dict]
    entity_graph: nx.DiGraph

class EntityContinuityTracker:
    def __init__(self):
        self.graph = nx.DiGraph()

    def track(
        self,
        segments: list[str],
        entities_per_segment: list[list[Entity]]
    ) -> ContinuityResult:
        """Track entity continuity across segments."""
        # Build entity occurrence map
        occurrences: dict[str, list[EntityOccurrence]] = {}

        for seg_id, (segment, entities) in enumerate(
            zip(segments, entities_per_segment)
        ):
            for entity in entities:
                key = self._entity_key(entity)
                if key not in occurrences:
                    occurrences[key] = []
                occurrences[key].append(EntityOccurrence(
                    entity=entity,
                    segment_id=seg_id,
                    context=segment
                ))

        # Analyze continuity
        discontinuities = []
        for key, occs in occurrences.items():
            issues = self._check_continuity(occs)
            discontinuities.extend(issues)

        # Build graph
        self._build_graph(occurrences)

        # Compute score
        total = sum(len(occs) for occs in occurrences.values())
        issues = len(discontinuities)
        score = 1.0 - (issues / total) if total > 0 else 1.0

        return ContinuityResult(
            continuity_score=score,
            entities_tracked=len(occurrences),
            discontinuities=discontinuities,
            entity_graph=self.graph
        )

    def _entity_key(self, entity: Entity) -> str:
        """Generate key for entity grouping."""
        return f"{entity.label}:{entity.normalized}"

    def _check_continuity(
        self,
        occurrences: list[EntityOccurrence]
    ) -> list[dict]:
        """Check for continuity issues."""
        issues = []
        # Check for attribute conflicts
        # Check for unexpected disappearances
        return issues

    def _build_graph(self, occurrences: dict) -> None:
        """Build entity relationship graph."""
        pass
```

#### Acceptance Criteria
- [ ] Track entity across multiple segments
- [ ] Detect entity attribute changes
- [ ] Detect unexpected entity loss
- [ ] Continuity score reflects issues

---

### 4.3 Negation & Temporal Consistency (Days 3-4)

#### Tasks
- [ ] Implement negation detection
- [ ] Track negation consistency across segments
- [ ] Implement temporal marker detection
- [ ] Track temporal consistency

#### Key Files

**`src/hsttb/nlp/negation.py`**
```python
import medspacy
from medspacy.context import ConTextComponent

class NegationDetector:
    def __init__(self):
        self.nlp = medspacy.load()
        self.nlp.add_pipe("medspacy_context")

    def detect_negations(self, text: str) -> list[dict]:
        """Detect negated entities in text."""
        doc = self.nlp(text)
        negations = []

        for ent in doc.ents:
            if ent._.is_negated:
                negations.append({
                    "entity": ent.text,
                    "span": (ent.start_char, ent.end_char),
                    "context": self._get_context(doc, ent)
                })

        return negations

    def check_negation_consistency(
        self,
        gt_negations: list[dict],
        pred_negations: list[dict]
    ) -> dict:
        """Check if negations are preserved."""
        flips = []

        for gt_neg in gt_negations:
            # Check if same entity is negated in prediction
            found = False
            for pred_neg in pred_negations:
                if self._entities_match(gt_neg, pred_neg):
                    found = True
                    break

            if not found:
                flips.append({
                    "entity": gt_neg["entity"],
                    "issue": "negation_lost"
                })

        return {
            "negation_flips": len(flips),
            "details": flips
        }

    def _entities_match(self, e1: dict, e2: dict) -> bool:
        """Check if two entity references match."""
        return e1["entity"].lower() == e2["entity"].lower()

    def _get_context(self, doc, ent, window: int = 5) -> str:
        """Get surrounding context for entity."""
        start = max(0, ent.start - window)
        end = min(len(doc), ent.end + window)
        return doc[start:end].text
```

#### Acceptance Criteria
- [ ] Detect "no chest pain" as negated
- [ ] Detect "denies fever" as negated
- [ ] Track negation consistency across segments
- [ ] Flag negation flips

---

### 4.4 CRS Computation Engine (Days 4-6)

#### Tasks
- [ ] Combine semantic similarity
- [ ] Combine entity continuity
- [ ] Combine negation consistency
- [ ] Compute composite CRS score
- [ ] Compute context drift rate

#### Key Files

**`src/hsttb/metrics/crs.py`**
```python
from dataclasses import dataclass
from hsttb.metrics.semantic_similarity import SemanticSimilarityEngine
from hsttb.metrics.entity_continuity import EntityContinuityTracker
from hsttb.nlp.negation import NegationDetector
from hsttb.nlp.medical_ner import MedicalNERPipeline

@dataclass
class SegmentCRSScore:
    segment_id: int
    ground_truth_text: str
    predicted_text: str
    semantic_similarity: float
    entities_preserved: int
    entities_lost: int
    negation_flips: int

@dataclass
class CRSResult:
    composite_score: float
    semantic_similarity: float
    entity_continuity: float
    negation_consistency: float
    context_drift_rate: float
    segment_scores: list[SegmentCRSScore]

class CRSEngine:
    def __init__(
        self,
        semantic_weight: float = 0.4,
        entity_weight: float = 0.3,
        negation_weight: float = 0.3
    ):
        self.semantic_engine = SemanticSimilarityEngine()
        self.continuity_tracker = EntityContinuityTracker()
        self.negation_detector = NegationDetector()
        self.ner_pipeline = MedicalNERPipeline()

        self.semantic_weight = semantic_weight
        self.entity_weight = entity_weight
        self.negation_weight = negation_weight

    def compute(
        self,
        gt_segments: list[str],
        pred_segments: list[str]
    ) -> CRSResult:
        """Compute Context Retention Score."""
        # Semantic similarity
        semantic_scores = self.semantic_engine.compute_segment_similarities(
            gt_segments, pred_segments
        )
        avg_semantic = sum(semantic_scores) / len(semantic_scores)

        # Entity continuity
        gt_entities = [
            self.ner_pipeline.extract_entities(seg)
            for seg in gt_segments
        ]
        pred_entities = [
            self.ner_pipeline.extract_entities(seg)
            for seg in pred_segments
        ]

        gt_continuity = self.continuity_tracker.track(gt_segments, gt_entities)
        pred_continuity = self.continuity_tracker.track(pred_segments, pred_entities)

        entity_score = self._compare_continuity(gt_continuity, pred_continuity)

        # Negation consistency
        negation_score = self._compute_negation_consistency(
            gt_segments, pred_segments
        )

        # Composite score
        composite = (
            self.semantic_weight * avg_semantic +
            self.entity_weight * entity_score +
            self.negation_weight * negation_score
        )

        # Context drift
        drift_rate = self._compute_drift_rate(semantic_scores)

        # Segment-level scores
        segment_scores = self._compute_segment_scores(
            gt_segments, pred_segments, semantic_scores
        )

        return CRSResult(
            composite_score=composite,
            semantic_similarity=avg_semantic,
            entity_continuity=entity_score,
            negation_consistency=negation_score,
            context_drift_rate=drift_rate,
            segment_scores=segment_scores
        )

    def _compare_continuity(self, gt, pred) -> float:
        """Compare entity continuity between gt and pred."""
        pass

    def _compute_negation_consistency(
        self,
        gt_segments: list[str],
        pred_segments: list[str]
    ) -> float:
        """Compute negation consistency score."""
        total_negations = 0
        preserved = 0

        for gt_seg, pred_seg in zip(gt_segments, pred_segments):
            gt_negs = self.negation_detector.detect_negations(gt_seg)
            pred_negs = self.negation_detector.detect_negations(pred_seg)

            total_negations += len(gt_negs)
            result = self.negation_detector.check_negation_consistency(
                gt_negs, pred_negs
            )
            preserved += len(gt_negs) - result["negation_flips"]

        return preserved / total_negations if total_negations > 0 else 1.0

    def _compute_drift_rate(self, scores: list[float]) -> float:
        """Compute context drift rate."""
        if len(scores) < 2:
            return 0.0

        # Drift = degradation over segments
        drifts = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        negative_drifts = [d for d in drifts if d > 0]

        return sum(negative_drifts) / len(scores)

    def _compute_segment_scores(
        self, gt_segments, pred_segments, semantic_scores
    ) -> list[SegmentCRSScore]:
        """Compute per-segment CRS scores."""
        pass
```

#### Acceptance Criteria
- [ ] Composite CRS computed from all components
- [ ] Semantic similarity weighted correctly
- [ ] Entity continuity weighted correctly
- [ ] Negation consistency weighted correctly
- [ ] Context drift rate identifies degradation

---

## Phase 5: Evaluation Orchestration
**Duration: Week 6**

### 5.1 Benchmark Runner (Days 1-2)

#### Tasks
- [ ] Implement main benchmark orchestrator
- [ ] Handle audio file discovery
- [ ] Handle ground truth loading
- [ ] Coordinate STT → metrics pipeline
- [ ] Handle errors gracefully

#### Key Files

**`src/hsttb/evaluation/runner.py`**
```python
import asyncio
from pathlib import Path
from dataclasses import dataclass
from hsttb.audio.loader import AudioLoader
from hsttb.audio.chunker import StreamingChunker
from hsttb.adapters.base import STTAdapter
from hsttb.metrics.ter import TEREngine, TERResult
from hsttb.metrics.ner import NEREngine, NERResult
from hsttb.metrics.crs import CRSEngine, CRSResult

@dataclass
class BenchmarkResult:
    audio_id: str
    ter: TERResult
    ner: NERResult
    crs: CRSResult
    transcript_ground_truth: str
    transcript_predicted: str
    streaming_profile: str
    adapter_name: str

@dataclass
class BenchmarkSummary:
    total_files: int
    avg_ter: float
    avg_ner_f1: float
    avg_crs: float
    results: list[BenchmarkResult]

class BenchmarkRunner:
    def __init__(
        self,
        adapter: STTAdapter,
        streaming_profile: str = "ideal"
    ):
        self.adapter = adapter
        self.profile_name = streaming_profile

        self.audio_loader = AudioLoader()
        self.chunker = StreamingChunker(load_profile(streaming_profile))

        self.ter_engine = TEREngine()
        self.ner_engine = NEREngine()
        self.crs_engine = CRSEngine()

    async def evaluate(
        self,
        audio_dir: Path,
        ground_truth_dir: Path
    ) -> BenchmarkSummary:
        """Run benchmark on all audio files."""
        await self.adapter.initialize()

        audio_files = list(audio_dir.glob("*.wav"))
        results = []

        for audio_file in audio_files:
            try:
                result = await self._evaluate_single(
                    audio_file, ground_truth_dir
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

        await self.adapter.cleanup()

        return self._summarize(results)

    async def _evaluate_single(
        self,
        audio_file: Path,
        ground_truth_dir: Path
    ) -> BenchmarkResult:
        """Evaluate single audio file."""
        # Load audio
        audio_data, sample_rate = self.audio_loader.load(audio_file)

        # Load ground truth
        gt_file = ground_truth_dir / f"{audio_file.stem}.txt"
        ground_truth = gt_file.read_text()

        # Stream to STT
        audio_stream = self.chunker.stream_audio(audio_data, sample_rate)

        segments = []
        async for segment in self.adapter.transcribe_stream(audio_stream):
            if segment.is_final:
                segments.append(segment.text)

        prediction = " ".join(segments)

        # Compute metrics
        ter_result = self.ter_engine.compute(ground_truth, prediction)
        ner_result = self.ner_engine.compute(ground_truth, prediction)

        # For CRS, we need segment-level comparison
        gt_segments = self._segment_ground_truth(ground_truth, len(segments))
        crs_result = self.crs_engine.compute(gt_segments, segments)

        return BenchmarkResult(
            audio_id=audio_file.stem,
            ter=ter_result,
            ner=ner_result,
            crs=crs_result,
            transcript_ground_truth=ground_truth,
            transcript_predicted=prediction,
            streaming_profile=self.profile_name,
            adapter_name=self.adapter.name
        )

    def _segment_ground_truth(
        self,
        text: str,
        num_segments: int
    ) -> list[str]:
        """Split ground truth into segments for CRS."""
        # Split by sentence or by length
        pass

    def _summarize(self, results: list[BenchmarkResult]) -> BenchmarkSummary:
        """Compute summary statistics."""
        return BenchmarkSummary(
            total_files=len(results),
            avg_ter=sum(r.ter.overall_ter for r in results) / len(results),
            avg_ner_f1=sum(r.ner.f1_score for r in results) / len(results),
            avg_crs=sum(r.crs.composite_score for r in results) / len(results),
            results=results
        )
```

#### Acceptance Criteria
- [ ] Benchmark runs on directory of audio files
- [ ] All metrics computed for each file
- [ ] Summary statistics computed
- [ ] Errors don't crash entire run

---

### 5.2 SRS Computation (Days 2-3)

#### Tasks
- [ ] Run same model under different profiles
- [ ] Compute score ratios
- [ ] Generate degradation breakdown

#### Key Files

**`src/hsttb/metrics/srs.py`**
```python
from dataclasses import dataclass
from hsttb.evaluation.runner import BenchmarkRunner, BenchmarkSummary

@dataclass
class SRSResult:
    model_name: str
    ideal_scores: dict[str, float]
    realtime_scores: dict[str, float]
    srs: float
    degradation: dict[str, float]

class SRSEngine:
    async def compute(
        self,
        adapter,
        audio_dir,
        ground_truth_dir
    ) -> SRSResult:
        """Compute Streaming Robustness Score."""
        # Run with ideal profile
        ideal_runner = BenchmarkRunner(adapter, "ideal")
        ideal_results = await ideal_runner.evaluate(audio_dir, ground_truth_dir)

        # Run with realtime profile
        realtime_runner = BenchmarkRunner(adapter, "realtime_mobile")
        realtime_results = await realtime_runner.evaluate(audio_dir, ground_truth_dir)

        # Compute ratios
        ideal_scores = {
            "ter": ideal_results.avg_ter,
            "ner_f1": ideal_results.avg_ner_f1,
            "crs": ideal_results.avg_crs
        }

        realtime_scores = {
            "ter": realtime_results.avg_ter,
            "ner_f1": realtime_results.avg_ner_f1,
            "crs": realtime_results.avg_crs
        }

        # SRS = weighted ratio
        composite_ideal = self._composite(ideal_scores)
        composite_realtime = self._composite(realtime_scores)
        srs = composite_realtime / composite_ideal if composite_ideal > 0 else 0

        # Degradation breakdown
        degradation = {
            metric: (ideal_scores[metric] - realtime_scores[metric]) / ideal_scores[metric]
            for metric in ideal_scores
            if ideal_scores[metric] > 0
        }

        return SRSResult(
            model_name=adapter.name,
            ideal_scores=ideal_scores,
            realtime_scores=realtime_scores,
            srs=srs,
            degradation=degradation
        )

    def _composite(self, scores: dict) -> float:
        """Compute composite score (higher is better)."""
        # Invert TER (lower is better)
        return (1 - scores["ter"]) * 0.4 + scores["ner_f1"] * 0.3 + scores["crs"] * 0.3
```

#### Acceptance Criteria
- [ ] Same model tested under both profiles
- [ ] SRS ratio computed correctly
- [ ] Degradation breakdown available
- [ ] Results are reproducible

---

### 5.3 MLflow Integration (Days 3-4)

#### Tasks
- [ ] Set up MLflow tracking
- [ ] Log benchmark runs
- [ ] Log metrics and artifacts
- [ ] Enable experiment comparison

#### Key Files

**`src/hsttb/evaluation/tracking.py`**
```python
import mlflow
from hsttb.evaluation.runner import BenchmarkSummary

class ExperimentTracker:
    def __init__(self, experiment_name: str = "hsttb-benchmark"):
        mlflow.set_experiment(experiment_name)

    def log_benchmark(
        self,
        summary: BenchmarkSummary,
        adapter_name: str,
        profile_name: str,
        tags: dict = None
    ):
        """Log benchmark results to MLflow."""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("adapter", adapter_name)
            mlflow.log_param("streaming_profile", profile_name)
            mlflow.log_param("total_files", summary.total_files)

            # Log metrics
            mlflow.log_metric("avg_ter", summary.avg_ter)
            mlflow.log_metric("avg_ner_f1", summary.avg_ner_f1)
            mlflow.log_metric("avg_crs", summary.avg_crs)

            # Log tags
            if tags:
                mlflow.set_tags(tags)

            # Log detailed results as artifact
            self._log_detailed_results(summary)

    def _log_detailed_results(self, summary: BenchmarkSummary):
        """Log detailed results as JSON artifact."""
        pass
```

#### Acceptance Criteria
- [ ] Benchmark runs logged to MLflow
- [ ] Metrics visible in MLflow UI
- [ ] Experiments can be compared
- [ ] Artifacts downloadable

---

## Phase 6: Reporting & Hardening
**Duration: Week 7**

### 6.1 Report Generation (Days 1-2)

#### Tasks
- [ ] Generate JSON reports
- [ ] Generate Parquet files for analysis
- [ ] Generate HTML summary reports
- [ ] Generate clinical risk reports

#### Key Files

**`src/hsttb/reporting/generator.py`**
```python
import json
from pathlib import Path
from dataclasses import asdict
import pandas as pd
from hsttb.evaluation.runner import BenchmarkSummary

class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, summary: BenchmarkSummary):
        """Generate all report formats."""
        self.generate_json(summary)
        self.generate_parquet(summary)
        self.generate_html(summary)
        self.generate_clinical_risk_report(summary)

    def generate_json(self, summary: BenchmarkSummary):
        """Generate JSON report."""
        output_path = self.output_dir / "results.json"
        with open(output_path, "w") as f:
            json.dump(asdict(summary), f, indent=2, default=str)

    def generate_parquet(self, summary: BenchmarkSummary):
        """Generate Parquet for analysis."""
        rows = []
        for result in summary.results:
            rows.append({
                "audio_id": result.audio_id,
                "ter": result.ter.overall_ter,
                "ner_f1": result.ner.f1_score,
                "crs": result.crs.composite_score,
                "profile": result.streaming_profile,
                "adapter": result.adapter_name
            })

        df = pd.DataFrame(rows)
        df.to_parquet(self.output_dir / "results.parquet")

    def generate_html(self, summary: BenchmarkSummary):
        """Generate HTML summary report."""
        pass

    def generate_clinical_risk_report(self, summary: BenchmarkSummary):
        """Generate clinical risk-focused report."""
        # Identify critical errors
        critical_errors = []

        for result in summary.results:
            # Drug name errors
            for error in result.ter.substitutions:
                if error.category == "drug":
                    critical_errors.append({
                        "risk_level": "critical",
                        "type": "drug_substitution",
                        "original": error.ground_truth_term.text,
                        "predicted": error.predicted_term.text,
                        "audio_id": result.audio_id
                    })

            # Negation flips
            for seg_score in result.crs.segment_scores:
                if seg_score.negation_flips > 0:
                    critical_errors.append({
                        "risk_level": "high",
                        "type": "negation_flip",
                        "segment": seg_score.ground_truth_text,
                        "audio_id": result.audio_id
                    })

        output_path = self.output_dir / "clinical_risk.json"
        with open(output_path, "w") as f:
            json.dump(critical_errors, f, indent=2)
```

#### Acceptance Criteria
- [ ] JSON report generated
- [ ] Parquet file for analysis
- [ ] Clinical risk report identifies critical errors
- [ ] Reports are readable and useful

---

### 6.2 Dashboard API (Days 2-3)

#### Tasks
- [ ] Create FastAPI endpoints
- [ ] Implement model comparison endpoint
- [ ] Implement drill-down endpoints
- [ ] Add basic visualizations

#### Key Files

**`src/hsttb/reporting/dashboard.py`**
```python
from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd

app = FastAPI(title="HSTTB Dashboard")

RESULTS_DIR = Path("results")

@app.get("/models")
async def list_models():
    """List all benchmarked models."""
    models = set()
    for f in RESULTS_DIR.glob("*/results.parquet"):
        df = pd.read_parquet(f)
        models.update(df["adapter"].unique())
    return {"models": list(models)}

@app.get("/compare")
async def compare_models(models: str = None, profile: str = "ideal"):
    """Compare metrics across models."""
    # Load and aggregate results
    pass

@app.get("/results/{run_id}")
async def get_run_results(run_id: str):
    """Get detailed results for a run."""
    result_path = RESULTS_DIR / run_id / "results.json"
    if not result_path.exists():
        raise HTTPException(404, "Run not found")

    import json
    with open(result_path) as f:
        return json.load(f)

@app.get("/clinical-risk/{run_id}")
async def get_clinical_risk(run_id: str):
    """Get clinical risk report for a run."""
    pass
```

#### Acceptance Criteria
- [ ] API endpoints work
- [ ] Model comparison returns data
- [ ] Detailed results retrievable
- [ ] Clinical risk endpoint works

---

### 6.3 Testing (Days 3-5)

#### Tasks
- [ ] Write unit tests for all engines
- [ ] Write integration tests
- [ ] Set up CI pipeline
- [ ] Achieve >80% coverage

#### Key Files

**`tests/unit/test_ter.py`**
```python
import pytest
from hsttb.metrics.ter import TEREngine

@pytest.fixture
def ter_engine():
    return TEREngine()

def test_identical_transcripts(ter_engine):
    """TER should be 0 for identical transcripts."""
    text = "Patient takes metformin 500mg daily for diabetes."
    result = ter_engine.compute(text, text)
    assert result.overall_ter == 0.0

def test_drug_substitution(ter_engine):
    """Should detect drug name substitution."""
    gt = "Patient takes metformin for diabetes."
    pred = "Patient takes methotrexate for diabetes."
    result = ter_engine.compute(gt, pred)
    assert len(result.substitutions) == 1
    assert result.substitutions[0].category == "drug"

def test_dosage_deletion(ter_engine):
    """Should detect dosage deletion."""
    gt = "Take aspirin 500mg twice daily."
    pred = "Take aspirin twice daily."
    result = ter_engine.compute(gt, pred)
    assert len(result.deletions) >= 1
```

**`tests/integration/test_benchmark.py`**
```python
import pytest
from pathlib import Path
from hsttb.evaluation.runner import BenchmarkRunner
from hsttb.adapters.mock_adapter import MockSTTAdapter

@pytest.fixture
def mock_adapter():
    return MockSTTAdapter(responses=["Patient takes metformin daily."])

@pytest.mark.asyncio
async def test_full_benchmark(mock_adapter, tmp_path):
    """Test full benchmark pipeline."""
    # Create test audio and ground truth
    # ...

    runner = BenchmarkRunner(mock_adapter, "ideal")
    summary = await runner.evaluate(tmp_path / "audio", tmp_path / "gt")

    assert summary.total_files > 0
    assert 0 <= summary.avg_ter <= 1
    assert 0 <= summary.avg_ner_f1 <= 1
    assert 0 <= summary.avg_crs <= 1
```

#### Acceptance Criteria
- [ ] Unit tests for TER, NER, CRS engines
- [ ] Integration test for full pipeline
- [ ] CI runs tests on push
- [ ] Coverage report generated

---

## Summary: Phase Checklist

| Phase | Week | Status | Key Deliverable |
|-------|------|--------|-----------------|
| 1. Foundation | 1-2 | ⬜ | Streaming infrastructure + STT adapter |
| 2. TER Engine | 3 | ⬜ | Medical term error rate computation |
| 3. NER Engine | 4 | ⬜ | Entity extraction and accuracy metrics |
| 4. CRS Engine | 5 | ⬜ | Context retention scoring |
| 5. Orchestration | 6 | ⬜ | Benchmark runner + SRS + tracking |
| 6. Reporting | 7 | ⬜ | Reports + dashboard + testing |

---

## Dependencies Between Phases

```
Phase 1 (Foundation)
    │
    ├──► Phase 2 (TER) ──────────────────┐
    │                                     │
    ├──► Phase 3 (NER) ──────────────────┼──► Phase 5 (Orchestration)
    │                                     │         │
    └──► Phase 4 (CRS) ──────────────────┘         │
                                                    │
                                                    ▼
                                          Phase 6 (Reporting)
```

**Critical Path**: Phase 1 → Phase 5 → Phase 6

Phases 2, 3, 4 can be parallelized after Phase 1 is complete.
