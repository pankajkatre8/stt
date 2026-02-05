# Lunagen STT Benchmarking Tool - Metrics Explanation

> **Lunagen Speech-to-Text Benchmarking Tool**
>
> This document explains each evaluation metric, how it's calculated, and why it matters for healthcare transcription quality.

---

## Table of Contents

1. [Overview](#overview)
2. [Reference-Based Metrics](#reference-based-metrics)
   - [WER (Word Error Rate)](#wer-word-error-rate)
   - [CER (Character Error Rate)](#cer-character-error-rate)
   - [TER (Term Error Rate)](#ter-term-error-rate)
   - [NER (Named Entity Recognition)](#ner-named-entity-recognition)
   - [CRS (Context Retention Score)](#crs-context-retention-score)
3. [Reference-Free Quality Metrics](#reference-free-quality-metrics)
   - [Perplexity (Fluency)](#perplexity-fluency)
   - [Grammar Score](#grammar-score)
   - [Entity Validity](#entity-validity)
   - [Medical Coherence](#medical-coherence)
   - [Contradiction Detection](#contradiction-detection)
   - [Embedding Drift (Semantic Stability)](#embedding-drift-semantic-stability)
   - [Confidence Variance](#confidence-variance)
4. [Speech Rate Validation](#speech-rate-validation)
5. [Composite Scoring](#composite-scoring)
6. [Clinical Risk Assessment](#clinical-risk-assessment)

---

## Overview

HSTTB provides two evaluation modes:

| Mode | Requires Ground Truth | Use Case |
|------|----------------------|----------|
| **Reference-Based** | Yes | Compare transcription against known correct text |
| **Reference-Free** | No | Evaluate transcription quality without ground truth |

**Why Both Modes?**
- Reference-based metrics tell you *accuracy* (is it correct?)
- Reference-free metrics tell you *quality* (does it look right?)
- Combined analysis provides comprehensive evaluation

---

## Reference-Based Metrics

These metrics require ground truth (the correct transcription) for comparison.

### WER (Word Error Rate)

**What it measures**: Overall word-level transcription accuracy.

**Formula**:
```
WER = (S + D + I) / N

Where:
  S = Substitutions (wrong words)
  D = Deletions (missing words)
  I = Insertions (extra words)
  N = Total words in ground truth
```

**Example**:
```
Ground Truth: "Patient takes metformin twice daily"
Predicted:    "Patient takes methotrexate twice"

Analysis:
  - "metformin" → "methotrexate" = 1 substitution
  - "daily" missing = 1 deletion

WER = (1 + 1 + 0) / 5 = 0.40 = 40%
```

**Interpretation**:
| WER | Quality |
|-----|---------|
| 0-5% | Excellent |
| 5-15% | Good |
| 15-30% | Acceptable |
| >30% | Poor |

**Limitations**: WER treats all words equally. "Metformin" → "methotrexate" and "the" → "a" have the same weight, but the medical impact is vastly different.

---

### CER (Character Error Rate)

**What it measures**: Character-level transcription accuracy.

**Formula**:
```
CER = (S + D + I) / N

Where:
  S = Character substitutions
  D = Character deletions
  I = Character insertions
  N = Total characters in ground truth
```

**Example**:
```
Ground Truth: "metformin"
Predicted:    "metforman"

Analysis:
  - 'i' → 'a' = 1 substitution

CER = 1 / 9 = 0.11 = 11%
```

**When to use CER over WER**:
- Detecting spelling errors
- Evaluating phonetically similar mistakes
- More granular analysis of transcription quality

---

### TER (Term Error Rate)

**What it measures**: Accuracy of **medical terminology** transcription.

**Why it matters**: In healthcare, not all errors are equal. "Metformin" → "methotrexate" is a critical drug substitution that could harm patients, while "the" → "a" has no clinical impact.

**Formula**:
```
TER = (Term_Substitutions + Term_Deletions + Term_Insertions) / Total_GT_Terms

Where terms are medical entities:
  - Drug names (metformin, lisinopril, etc.)
  - Conditions (diabetes, hypertension, etc.)
  - Dosages (500mg, twice daily, etc.)
  - Procedures (MRI, blood test, etc.)
```

**How Terms are Extracted**:
1. Text is normalized (lowercase, abbreviation expansion)
2. Medical lexicon matching identifies terms
3. NER pipeline extracts entities
4. Terms are aligned between ground truth and predicted

**Example**:
```
Ground Truth: "Patient takes metformin 500mg for diabetes"
Predicted:    "Patient takes methotrexate 500mg for diabetes"

Medical Terms (GT): [metformin, 500mg, diabetes]
Medical Terms (Pred): [methotrexate, 500mg, diabetes]

Analysis:
  - metformin → methotrexate = 1 substitution (CRITICAL)
  - 500mg = correct
  - diabetes = correct

TER = 1 / 3 = 0.33 = 33%
Term Accuracy = 67%
```

**Category Breakdown**:
TER is computed per category:
- **Drug TER**: Errors in medication names
- **Dosage TER**: Errors in dosage values
- **Condition TER**: Errors in diagnoses
- **Procedure TER**: Errors in procedures

**Clinical Risk Levels**:
| Error Type | Risk Level | Example |
|------------|------------|---------|
| Drug substitution | CRITICAL | metformin → methotrexate |
| Drug omission | HIGH | Missing medication name |
| Dosage error | HIGH | 500mg → 5000mg |
| Negation flip | HIGH | "no pain" → "pain" |
| Condition change | MEDIUM | diabetes → hypertension |

---

### NER (Named Entity Recognition)

**What it measures**: How well medical entities are preserved in transcription.

**Entities Detected**:
| Label | Examples |
|-------|----------|
| DRUG | metformin, lisinopril, aspirin |
| CONDITION | diabetes, hypertension, COPD |
| SYMPTOM | chest pain, fatigue, nausea |
| PROCEDURE | MRI, blood test, surgery |
| ANATOMY | heart, lungs, liver |
| LAB_VALUE | glucose 120, A1C 7.2 |
| DOSAGE | 500mg, twice daily |

**Metrics Computed**:

**Precision**: Of entities found in predicted text, how many are correct?
```
Precision = True_Positives / (True_Positives + False_Positives)
```

**Recall**: Of entities in ground truth, how many were found?
```
Recall = True_Positives / (True_Positives + False_Negatives)
```

**F1 Score**: Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Entity Distortion Rate**: Percentage of entities that were changed
```
EDR = Distorted_Entities / Total_GT_Entities
```

**Entity Omission Rate**: Percentage of entities that were missed
```
EOR = Missing_Entities / Total_GT_Entities
```

**Example**:
```
Ground Truth: "Patient has diabetes and takes metformin"
Predicted:    "Patient has hypertension and takes metformin"

GT Entities: [diabetes (CONDITION), metformin (DRUG)]
Pred Entities: [hypertension (CONDITION), metformin (DRUG)]

Analysis:
  - diabetes → hypertension = distortion
  - metformin = correct match

Precision = 1/2 = 50% (metformin correct, hypertension wrong)
Recall = 1/2 = 50% (metformin found, diabetes missed)
F1 = 50%
```

---

### CRS (Context Retention Score)

**What it measures**: How well the overall meaning and context is preserved across the transcription, especially important for streaming scenarios.

**Components**:

#### 1. Semantic Similarity
Measures meaning preservation using sentence embeddings.

```
Similarity = cosine_similarity(embedding(GT), embedding(Pred))

Using sentence-transformers model: all-MiniLM-L6-v2
```

#### 2. Entity Continuity
Tracks entities across segments to detect:
- **Disappearance**: Entity mentioned, then vanishes
- **Conflict**: Same entity with different values
- **Label change**: Entity type changes
- **Negation flip**: Affirmed ↔ negated

```
Entity_Continuity = Preserved_Entities / Total_Entities
```

#### 3. Negation Consistency
Ensures negations are preserved (critical for clinical meaning).

```
Ground Truth: "Patient denies chest pain"
Predicted:    "Patient has chest pain"

This is a NEGATION FLIP - meaning completely reversed!
```

**Composite CRS Formula**:
```
CRS = (w1 × Semantic_Similarity) +
      (w2 × Entity_Continuity) +
      (w3 × Negation_Consistency)

Default weights: w1=0.4, w2=0.3, w3=0.3
```

**Context Drift Rate**:
Measures how much meaning shifts across segments:
```
Drift_Rate = Segments_With_Low_Similarity / Total_Segments
```

---

## Reference-Free Quality Metrics

These metrics evaluate transcription quality **without** needing ground truth.

### Perplexity (Fluency)

**What it measures**: How "natural" or "fluent" the text sounds to a language model.

**How it works**:
1. Feed text through GPT-2 language model
2. Compute probability of each word given previous words
3. Lower perplexity = more natural text

**Formula**:
```
Perplexity = exp(-1/N × Σ log P(word_i | context))

Where:
  N = number of words
  P(word_i | context) = probability of word given previous words
```

**Example**:
```
Text A: "Patient takes metformin for diabetes"
Perplexity: 45.2 (natural medical text)

Text B: "Patient metforman takess for diabetees"
Perplexity: 892.3 (unnatural, likely transcription errors)
```

**Score Conversion**:
```
Perplexity_Score = max(0, 1 - (perplexity - 20) / 180)

Clamped to [0, 1] range
```

| Perplexity | Fluency Score | Interpretation |
|------------|---------------|----------------|
| < 50 | > 85% | Excellent fluency |
| 50-100 | 60-85% | Good fluency |
| 100-200 | 30-60% | Moderate issues |
| > 200 | < 30% | Likely garbled |

---

### Grammar Score

**What it measures**: Grammatical correctness of the transcription.

**How it works**:
1. Analyze text with language-tool-python (LanguageTool)
2. Count grammar, spelling, and style errors
3. Filter out medical terms (don't flag "metformin" as misspelled)

**Formula**:
```
Grammar_Score = max(0, 1 - (error_count × 0.1))

Each error reduces score by 10%
```

**Error Types Detected**:
- Subject-verb agreement
- Missing articles
- Punctuation errors
- Spelling mistakes (non-medical words)
- Run-on sentences

**Medical Term Filtering**:
Known medical terms are excluded from spell-checking to avoid false positives.

---

### Entity Validity

**What it measures**: Are the medical entities in the text real/valid?

**How it works**:
1. Extract medical entities from text
2. Check each entity against medical lexicons
3. Flag unknown or misspelled entities

**Formula**:
```
Entity_Validity = Valid_Entities / Total_Entities
```

**Example**:
```
Text: "Patient takes metforman for diabeetes"

Entities found: [metforman, diabeetes]
Lexicon check:
  - metforman: NOT FOUND (should be "metformin")
  - diabeetes: NOT FOUND (should be "diabetes")

Entity_Validity = 0 / 2 = 0%
Invalid entities: [metforman, diabeetes]
```

**Lexicon Sources**:
- RxNorm (drug names)
- SNOMED CT (clinical terms)
- ICD-10 (diagnosis codes)
- Custom medical vocabulary

---

### Medical Coherence

**What it measures**: Do drug-condition pairs make clinical sense?

**How it works**:
1. Extract drugs and conditions from text
2. Check if drug-condition relationships are valid
3. Flag implausible combinations

**Known Valid Pairs** (examples):
| Drug | Valid Conditions |
|------|------------------|
| metformin | diabetes, type 2 diabetes, hyperglycemia |
| lisinopril | hypertension, heart failure |
| omeprazole | GERD, acid reflux, ulcer |
| atorvastatin | high cholesterol, hyperlipidemia |

**Formula**:
```
Coherence_Score = Valid_Pairs / Total_Pairs
```

**Example**:
```
Text: "Patient takes metformin for migraine"

Drug-condition pairs: [(metformin, migraine)]
Validation:
  - metformin for migraine: INVALID (metformin is for diabetes)

Coherence_Score = 0%
```

---

### Contradiction Detection

**What it measures**: Are there internal contradictions within the transcript?

**How it works**:
1. Track entity states throughout the text
2. Detect when same entity has conflicting states
3. Filter out questions (questions aren't contradictions)

**Types of Contradictions**:

#### Affirm-Negate Contradiction
```
"Patient has diabetes. Patient denies having diabetes."
Entity: diabetes
State 1: AFFIRMED
State 2: NEGATED
Result: CONTRADICTION DETECTED
```

#### Negate-Affirm Contradiction
```
"Patient denies chest pain. Patient reports chest pain."
Entity: chest pain
State 1: NEGATED
State 2: AFFIRMED
Result: CONTRADICTION DETECTED
```

**Question Filtering**:
Questions are not treated as contradictions:
```
"Do you have chest pain? I have chest pain."
- "Do you have chest pain?" = QUESTION (skipped)
- "I have chest pain" = AFFIRMATION
Result: No contradiction (question + answer pattern)
```

**Formula**:
```
Contradiction_Score = 1 - (Contradictions_Found × 0.2)

Each contradiction reduces score by 20%
Minimum score: 0
```

---

### Embedding Drift (Semantic Stability)

**What it measures**: How stable is the meaning across different parts of the transcript?

**Why it matters**: Transcription errors often cause sudden meaning shifts. Natural conversation has gradual topic transitions.

**How it works**:
1. Split transcript into sentences
2. Compute embedding for each sentence
3. Measure similarity between consecutive sentences
4. Flag sudden drops in similarity

**Formula**:
```
For each consecutive sentence pair (i, i+1):
  similarity[i] = cosine_similarity(embed(sent_i), embed(sent_i+1))

Mean_Similarity = average(all similarities)
Drift_Points = sentences where similarity < threshold

Embedding_Drift_Score = min(1.0, mean_similarity × 1.5 + 0.25) - anomaly_penalty
```

**Example**:
```
Sentence 1: "Patient has diabetes and takes metformin"
Sentence 2: "Blood sugar is well controlled"
Similarity: 0.72 (related topics, natural transition)

Sentence 1: "Patient has diabetes"
Sentence 2: "Purple elephant dancing"
Similarity: 0.08 (sudden drift - likely transcription error)
```

**Drift Detection Threshold**: 0.35 (below this = anomalous drift)

---

### Confidence Variance

**What it measures**: How confident is the language model about each word?

**Why it matters**: Transcription errors produce unusual word sequences that language models are uncertain about.

**How it works**:
1. Feed text through GPT-2
2. Get log probability for each token
3. Detect unusually low-probability tokens
4. Flag sudden confidence drops

**Formula**:
```
For each token:
  log_prob[i] = log P(token_i | previous_tokens)

Low_Confidence_Tokens = tokens where log_prob < -18.0
Confidence_Drops = tokens where (log_prob[i-1] - log_prob[i]) > 8.0

Confidence_Score = 1.0 - (low_conf_count + drop_count) × 0.05
```

**Example**:
```
Text: "Patient takes metformin daily"
Token log probs: [-3.2, -4.1, -5.5, -3.8]
All normal range - high confidence

Text: "Patient takes mettforrmin daaily"
Token log probs: [-3.2, -4.1, -18.5, -16.2]
                              ↑ sudden drop = transcription error likely
```

---

## Speech Rate Validation

**What it measures**: Is the word count plausible for the audio duration?

**Why it matters**:
- Too many words = possible hallucination (model generating extra text)
- Too few words = possible missing content

**Normal Speech Rates**:
| Category | Words per Minute |
|----------|------------------|
| Slow | 50-100 WPM |
| Normal | 100-180 WPM |
| Fast | 180-220 WPM |
| Implausibly Low | < 50 WPM |
| Implausibly High | > 250 WPM |

**Formula**:
```
WPM = Word_Count / (Audio_Duration_Seconds / 60)

Plausibility_Score = based on distance from optimal range (110-160 WPM)
```

**Example**:
```
Audio Duration: 60 seconds
Transcript Words: 150

WPM = 150 / 1 = 150 WPM
Category: NORMAL
Plausibility: 100%
```

**Warning Scenarios**:
```
Audio: 30 seconds, Words: 200
WPM = 400 WPM
Warning: "Too many words (200) for 30s audio. Expected at most 125. Possible hallucination."

Audio: 60 seconds, Words: 20
WPM = 20 WPM
Warning: "Too few words (20) for 60s audio. Expected at least 50. Missing content?"
```

---

## Composite Scoring

### Quality Composite Score

Combines all reference-free metrics:

```
Quality_Score =
    (perplexity_weight × Perplexity_Score) +
    (grammar_weight × Grammar_Score) +
    (entity_validity_weight × Entity_Validity) +
    (coherence_weight × Coherence_Score) +
    (contradiction_weight × Contradiction_Score) +
    (embedding_drift_weight × Drift_Score) +
    (confidence_weight × Confidence_Score)

Default weights:
  - Perplexity: 0.15
  - Grammar: 0.15
  - Entity Validity: 0.20
  - Coherence: 0.15
  - Contradiction: 0.15
  - Embedding Drift: 0.10
  - Confidence: 0.10
```

### Overall Evaluation Score

When both ground truth and quality metrics are available:

```
Overall_Score = average(
    WER_Accuracy,      # 1 - WER
    TER_Accuracy,      # 1 - TER
    NER_F1,
    CRS_Score,
    Quality_Score
)
```

### Recommendation Thresholds

| Score | Recommendation | Meaning |
|-------|----------------|---------|
| ≥ 0.80 | ACCEPT | Safe for clinical use |
| 0.50-0.79 | REVIEW | Human review recommended |
| < 0.50 | REJECT | Likely contains errors |

---

## Clinical Risk Assessment

HSTTB identifies specific high-risk errors:

### Critical Risk (Immediate Danger)
- **Drug name substitution**: metformin → methotrexate
- **Drug name with similar sound**: Celebrex → Celexa

### High Risk (Significant Impact)
- **Drug omission**: Medication not transcribed
- **Dosage error**: 500mg → 5000mg (10x overdose)
- **Negation flip**: "no allergies" → "allergies"
- **Frequency error**: "once daily" → "once weekly"

### Medium Risk (Clinical Relevance)
- **Condition substitution**: diabetes → hypertension
- **Procedure confusion**: MRI → CT scan
- **Anatomy error**: left arm → right arm

### Low Risk (Documentation Impact)
- **Minor word changes**: the → a
- **Punctuation differences**
- **Filler word omissions**: um, uh

---

## Summary Table

| Metric | Type | Range | Optimal | Purpose |
|--------|------|-------|---------|---------|
| WER | Reference | 0-100%+ | <5% | Word-level accuracy |
| CER | Reference | 0-100%+ | <5% | Character-level accuracy |
| TER | Reference | 0-100% | <10% | Medical term accuracy |
| NER F1 | Reference | 0-100% | >90% | Entity preservation |
| CRS | Reference | 0-100% | >85% | Context retention |
| Perplexity | Quality | 0-100% | >80% | Text fluency |
| Grammar | Quality | 0-100% | >90% | Grammatical correctness |
| Entity Validity | Quality | 0-100% | 100% | Valid medical terms |
| Coherence | Quality | 0-100% | >90% | Clinical plausibility |
| Contradiction | Quality | 0-100% | 100% | Internal consistency |
| Embedding Drift | Quality | 0-100% | >80% | Semantic stability |
| Confidence | Quality | 0-100% | >90% | LM confidence |
| Speech Rate | Quality | 0-100% | >90% | Duration plausibility |

---

## References

- Levenshtein Distance: https://en.wikipedia.org/wiki/Levenshtein_distance
- Word Error Rate: https://en.wikipedia.org/wiki/Word_error_rate
- Perplexity: https://en.wikipedia.org/wiki/Perplexity
- Sentence-BERT: https://arxiv.org/abs/1908.10084
- MedSpaCy: https://github.com/medspacy/medspacy
- scispaCy: https://allenai.github.io/scispacy/

---

*Document Version: 1.0*
*Last Updated: 2026-02-02*
