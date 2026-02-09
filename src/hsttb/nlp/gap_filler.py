"""
Gap Filler & Missing Word Predictor.

Uses Masked Language Models (PubMedBERT) to predict missing words based on context.
Designed to handle reference-free 'Missing Context' errors.
"""
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

class GapFiller:
    """
    Predicts missing words in medical transcripts using masked language modeling.
    Uses 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext' by default.
    """
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        self.model_name = model_name
        self._pipeline = None
        
    def _ensure_loaded(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                logger.info(f"Loading GapFiller model: {self.model_name}")
                self._pipeline = pipeline("fill-mask", model=self.model_name)
            except ImportError:
                logger.error("Transformers library not found.")
                raise

    def predict_gap(self, text: str, prev_word: str, next_word: str, top_k: int = 5) -> Dict:
        """
        Insert a mask between prev_word and next_word and predict the missing term.
        
        Args:
            text: The full transcript.
            prev_word: Word immediately before the suspected gap.
            next_word: Word immediately after the suspected gap.
            
        Returns:
            Dict containing predicted tokens and confidence scores.
        """
        self._ensure_loaded()
        
        # Construct probe sentence
        # Example: "I take [MASK] for diabetes"
        gap_context = f"{prev_word} {next_word}"
        masked_context = f"{prev_word} [MASK] {next_word}"
        
        # We replace only the first occurrence for simplicity in this logic
        # In production, use indices for precision
        if gap_context not in text:
            return {"detected": False, "reason": "Context not found"}
            
        probe_sentence = text.replace(gap_context, masked_context, 1)
        
        try:
            results = self._pipeline(probe_sentence, top_k=top_k)
        except Exception as e:
            logger.warning(f"Gap prediction failed: {e}")
            return {"detected": False, "reason": "Model error"}

        # Analyze predictions
        predictions = []
        is_medical_gap = False
        
        medical_triggers = {"medication", "drug", "insulin", "metformin", "lisinopril", "pain", "dose"}
        
        for res in results:
            token = res['token_str'].strip()
            score = res['score']
            predictions.append((token, score))
            
            if token in medical_triggers or score > 0.5:
                is_medical_gap = True

        return {
            "detected": True,
            "probe_text": probe_sentence,
            "top_predictions": predictions,
            "likely_missing_type": "Medical/Clinical" if is_medical_gap else "Grammatical",
            "confidence": predictions[0][1] if predictions else 0.0
        }