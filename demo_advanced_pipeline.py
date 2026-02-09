"""
Advanced Reference-Free Transcription Analysis Pipeline (PoC).

This script implements the "Hybrid Robust Architecture" discussed for HSSTB.
It moves beyond simple regex/grammar checks and uses AI models to:
1. REPAIR broken text (Flan-T5).
2. DETECT missing words (PubMedBERT).
3. VALIDATE clinical logic (Vector Matching).

Models Used:
- google/flan-t5-base (Text Correction)
- microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext (Gap Detection)
- sentence-transformers/all-MiniLM-L6-v2 (Medical Term Matching)
"""

import logging
import re
import torch
import numpy as np
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
class Config:
    REPAIR_MODEL = "google/flan-t5-base"
    GAP_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    VECTOR_MODEL = "all-MiniLM-L6-v2"
    
    # Valid medical terms reference (Simulated Database)
    KNOWN_CONDITIONS = ["diabetes", "hypertension", "chest pain", "asthma"]
    KNOWN_DRUGS = ["metformin", "lisinopril", "aspirin", "albuterol"]

# --- MODULE 1: SEMANTIC REPAIR (The "Fixer") ---
class SemanticRepair:
    def __init__(self):
        logger.info("Loading Repair Model (Flan-T5)...")
        # Load Model and Tokenizer directly (More robust than pipeline)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.REPAIR_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(Config.REPAIR_MODEL)

    def fix(self, text: str) -> str:
        """Fixes grammar, stutters, and slang using an LLM."""
        prompt = f"Fix grammar and medical terms: {text}"
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate output
        outputs = self.model.generate(
            **inputs, 
            max_length=128, 
            num_beams=5,             # Use beam search for better quality
            early_stopping=True
        )
        
        cleaned_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log if changes were made
        if cleaned_text.lower() != text.lower():
            logger.info(f"Refined: '{text}' -> '{cleaned_text}'")
            
        return cleaned_text
# --- MODULE 2: GAP DETECTOR (The "Missing Word Hunter") ---
class GapDetector:
    def __init__(self):
        logger.info("Loading Gap Detector (PubMedBERT)...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.GAP_MODEL)
        self.model = AutoModelForMaskedLM.from_pretrained(Config.GAP_MODEL)
        self.fill_mask = pipeline("fill-mask", model=Config.GAP_MODEL, tokenizer=Config.GAP_MODEL)

    def check_for_missing_drug(self, text: str) -> dict:
        """
        Checks if a drug is missing after verbs like 'take' or 'prescribe'.
        Returns {'detected': bool, 'suggestion': str}
        """
        # 1. Regex Heuristic: Look for "take [preposition]" pattern
        # This catches "I take for diabetes"
        suspicious_pattern = re.compile(r"\b(take|takes|taking|prescribed?)\s+(for|with)\b", re.IGNORECASE)
        match = suspicious_pattern.search(text)

        if match:
            # 2. AI Validation: Insert mask and ask BERT
            verb, prep = match.group(1), match.group(2)
            masked_sentence = text.replace(f"{verb} {prep}", f"{verb} [MASK] {prep}")
            
            logger.info(f"Gap detected. Probing: '{masked_sentence}'")
            
            predictions = self.fill_mask(masked_sentence)
            
            # Check if top predictions are drugs/medications
            for pred in predictions[:3]: # Top 3 guesses
                token = pred['token_str']
                score = pred['score']
                
                # If BERT predicts these words, it confirms a drug is missing
                if token in ["medication", "medicine", "insulin", "metformin", "pills"]:
                    return {
                        "detected": True,
                        "type": "MISSING_MEDICATION",
                        "confidence": score,
                        "context": f"Missing object between '{verb}' and '{prep}'"
                    }
        
        return {"detected": False}

# --- MODULE 3: LOGIC VALIDATOR (Vector Check) ---
class VectorLogicChecker:
    def __init__(self):
        logger.info("Loading Vector Model (SBERT)...")
        self.model = SentenceTransformer(Config.VECTOR_MODEL)
        
        # Encode our "Database" once
        self.drug_embeddings = self.model.encode(Config.KNOWN_DRUGS)
        self.cond_embeddings = self.model.encode(Config.KNOWN_CONDITIONS)

 # Inside class VectorLogicChecker

    def validate(self, text: str):
        doc_embedding = self.model.encode(text)
        
        # --- DRUG CHECK ---
        drug_scores = util.cos_sim(doc_embedding, self.drug_embeddings)[0]
        best_drug_idx = torch.argmax(drug_scores)
        best_drug_score = float(drug_scores[best_drug_idx])
        detected_drug = Config.KNOWN_DRUGS[best_drug_idx]
        
        # IMPROVEMENT: The "Second Chance" Logic
        # If vector score is okay-ish (0.25+) but not perfect, check string similarity
        if 0.25 < best_drug_score < 0.4:
            from difflib import SequenceMatcher
            # Check if any word in text looks like the detected drug
            for word in text.split():
                similarity = SequenceMatcher(None, word.lower(), detected_drug).ratio()
                if similarity > 0.8: # If spelling is 80% close (e.g. "metaformin")
                    best_drug_score = 0.85 # Boost confidence!
                    break

        final_drug = detected_drug if best_drug_score > 0.4 else None

        # --- CONDITION CHECK (Same Logic) ---
        cond_scores = util.cos_sim(doc_embedding, self.cond_embeddings)[0]
        best_cond_idx = torch.argmax(cond_scores)
        best_cond_score = float(cond_scores[best_cond_idx])
        detected_cond = Config.KNOWN_CONDITIONS[best_cond_idx]

        return {
            "found_drug": final_drug,
            "found_condition": detected_cond if best_cond_score > 0.4 else None,
            "drug_confidence": best_drug_score,
            "condition_confidence": best_cond_score
        }

# --- THE ORCHESTRATOR ---
class RobustPipeline:
    def __init__(self):
        self.repair = SemanticRepair()
        self.gap_detector = GapDetector()
        self.logic = VectorLogicChecker()

    def process(self, raw_transcript: str):
        print(f"\n{'='*60}")
        print(f"INPUT:  \"{raw_transcript}\"")
        print(f"{'='*60}")

        # Step 1: Semantic Repair
        clean_text = self.repair.fix(raw_transcript)
        print(f"STEP 1 [Repair]:  \"{clean_text}\"")

        # Step 2: Gap Detection
        gap_result = self.gap_detector.check_for_missing_drug(clean_text)
        if gap_result['detected']:
            print(f"STEP 2 [Gap]:     CRITICAL ALERT! {gap_result['type']}")
            print(f"                  (AI Confidence: {gap_result['confidence']:.2f})")
        else:
            print(f"STEP 2 [Gap]:     No obvious gaps detected.")

        # Step 3: Logic/Entity Check
        logic_result = self.logic.validate(clean_text)
        print(f"STEP 3 [Logic]:   Entities Detected:")
        print(f"                  - Drug: {logic_result['found_drug']} (Conf: {logic_result['drug_confidence']:.2f})")
        print(f"                  - Cond: {logic_result['found_condition']} (Conf: {logic_result['condition_confidence']:.2f})")

        # Final Verdict
        if gap_result['detected']:
            return "FLAG: REVIEW_REQUIRED (Missing Context)"
        if logic_result['found_drug'] and logic_result['found_condition']:
            return "PASS: CLINICALLY VALID"
        
        return "PASS: LOW RISK (Casual/Incomplete)"

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    pipeline = RobustPipeline()

    # TEST CASE 1: The "Messy" One (Slang + Typos)
    # "sugar" -> Diabetes, "metaforminnn" -> Metformin
    pipeline.process("I have sugar issues and take metaforminnn for it.")

    # TEST CASE 2: The "Missing Word" One
    # "take for" -> Missing Drug
    pipeline.process("I have diabetes so I take for it daily.")

    # TEST CASE 3: The "Broken Grammar" One
    # "Pain chest" -> Chest Pain, "denies" handling
    pipeline.process("Pain chest I have... actually denies that.")