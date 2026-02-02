"""
Intelligent Medical Term Matcher.

Provides efficient database-backed term lookups and fuzzy matching
for detecting potential transcription errors.

Features:
- Lazy database queries (not loading entire table in memory)
- Fuzzy matching for misspelled terms
- Context-aware error detection
- Phonetic matching (Soundex/Metaphone)

Example:
    >>> from hsttb.metrics.term_matcher import MedicalTermMatcher
    >>> matcher = MedicalTermMatcher()
    >>> errors = matcher.find_potential_errors("Patient takes paracetol 500mg")
    >>> for err in errors:
    ...     print(f"{err.found_term} -> possibly meant {err.suggested_term}")
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path.home() / ".hsttb" / "medical_lexicon.db"


@dataclass
class PotentialError:
    """A potential transcription error detected."""

    found_term: str  # The term found in text
    suggested_term: str  # The likely intended term
    confidence: float  # 0-1 confidence score
    similarity: float  # Fuzzy match similarity
    match_type: str  # "fuzzy", "phonetic", "context"
    position: tuple[int, int]  # Start, end position in text
    context: str  # Surrounding text
    source: str  # Database source (ICD10, RxNorm, etc.)


@dataclass
class TermMatchResult:
    """Result of term matching analysis."""

    text: str
    known_terms_found: list[dict] = field(default_factory=list)
    potential_errors: list[PotentialError] = field(default_factory=list)
    unrecognized_medical_like: list[str] = field(default_factory=list)
    error_score: float = 1.0  # 1.0 = no errors, lower = more errors


class MedicalTermMatcher:
    """
    Intelligent medical term matcher with fuzzy matching.

    Uses lazy database queries instead of loading all terms into memory.
    Detects potential transcription errors using:
    - Levenshtein distance (fuzzy matching)
    - Phonetic matching (Soundex)
    - Context-based inference

    Example:
        >>> matcher = MedicalTermMatcher()
        >>> result = matcher.analyze("Patient has diabetis and takes metforman")
        >>> for err in result.potential_errors:
        ...     print(f"'{err.found_term}' might be '{err.suggested_term}'")
    """

    # Medical-like word patterns (might be misspelled terms)
    MEDICAL_PATTERNS = [
        r"\b\w+itis\b",  # inflammation: bronchitis, arthritis
        r"\b\w+emia\b",  # blood conditions: anemia, leukemia
        r"\b\w+osis\b",  # conditions: cirrhosis, fibrosis
        r"\b\w+pathy\b",  # disease: neuropathy, myopathy
        r"\b\w+ectomy\b",  # surgical removal: appendectomy
        r"\b\w+plasty\b",  # surgical repair: angioplasty
        r"\b\w+scopy\b",  # examination: endoscopy
        r"\b\w+gram\b",  # recording: electrocardiogram
        r"\b\w+ine\b",  # drugs: metformin, aspirin (many end in -ine)
        r"\b\w+ol\b",  # drugs: metoprolol, atenolol
        r"\b\w+an\b",  # drugs: lisinopril variations
        r"\b\w+in\b",  # drugs: aspirin, insulin
        r"\b\w+ide\b",  # drugs: hydrochlorothiazide
        r"\b\w+ate\b",  # drugs: atorvastatin
        r"\b\w+pril\b",  # ACE inhibitors: lisinopril
        r"\b\w+sartan\b",  # ARBs: losartan
        r"\b\w+statin\b",  # statins: atorvastatin
        r"\b\w+zole\b",  # antifungals/PPIs: omeprazole
        r"\b\w+cillin\b",  # antibiotics: amoxicillin
        r"\b\w+mycin\b",  # antibiotics: azithromycin
    ]

    # Common drug name prefixes
    DRUG_PREFIXES = [
        "met", "lis", "ator", "omep", "amlo", "gaba", "hydro",
        "pant", "sertr", "losar", "pred", "warfa", "aspir",
        "ibupro", "aceta", "amoxi", "azith", "cipro",
    ]

    def __init__(self, db_path: Path | None = None, similarity_threshold: float = 0.75):
        """
        Initialize matcher.

        Args:
            db_path: Path to SQLite database.
            similarity_threshold: Minimum similarity for fuzzy matches (0-1).
        """
        self._db_path = db_path or DB_PATH
        self._similarity_threshold = similarity_threshold
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.MEDICAL_PATTERNS]

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def lookup_term(self, term: str) -> dict | None:
        """
        Look up a single term in the database.

        Efficient single-query lookup instead of loading all terms.

        Args:
            term: Term to look up.

        Returns:
            Dict with term info or None if not found.
        """
        normalized = term.lower().strip()

        conn = self._get_connection()
        try:
            # Check terms table (drugs)
            cursor = conn.execute(
                "SELECT term, code, category, source FROM terms WHERE normalized = ?",
                (normalized,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "term": row["term"],
                    "code": row["code"],
                    "category": row["category"],
                    "source": row["source"],
                }

            # Check medical_terms table (ICD-10, SNOMED, etc.)
            cursor = conn.execute(
                "SELECT term, code, category, source FROM medical_terms WHERE normalized = ?",
                (normalized,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "term": row["term"],
                    "code": row["code"],
                    "category": row["category"],
                    "source": row["source"],
                }

            return None

        finally:
            conn.close()

    def is_known_term(self, term: str) -> bool:
        """Check if term exists in database (efficient query)."""
        return self.lookup_term(term) is not None

    def find_similar_terms(
        self,
        term: str,
        limit: int = 5,
        min_similarity: float | None = None,
    ) -> list[tuple[str, float, str]]:
        """
        Find similar terms using fuzzy matching.

        Args:
            term: Term to find matches for.
            limit: Maximum matches to return.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of (term, similarity, source) tuples.
        """
        if min_similarity is None:
            min_similarity = self._similarity_threshold

        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning("rapidfuzz not installed, fuzzy matching unavailable")
            return []

        normalized = term.lower().strip()
        matches = []

        conn = self._get_connection()
        try:
            # Search drugs first (smaller table, more likely for drug names)
            cursor = conn.execute(
                "SELECT DISTINCT term, source FROM terms WHERE category = 'drug'"
            )
            for row in cursor.fetchall():
                db_term = row["term"].lower()
                # Quick length check to skip obviously different terms
                if abs(len(db_term) - len(normalized)) > 3:
                    continue
                similarity = fuzz.ratio(normalized, db_term) / 100.0
                if similarity >= min_similarity:
                    matches.append((row["term"], similarity, row["source"]))

            # If searching for condition-like terms, check medical_terms
            if len(normalized) > 5:  # Only for longer terms
                # Use LIKE for prefix matching to reduce candidates
                prefix = normalized[:3]
                cursor = conn.execute(
                    """
                    SELECT DISTINCT term, source FROM medical_terms
                    WHERE normalized LIKE ? AND category = 'diagnosis'
                    LIMIT 1000
                    """,
                    (f"{prefix}%",)
                )
                for row in cursor.fetchall():
                    db_term = row["term"].lower()
                    if abs(len(db_term) - len(normalized)) > 5:
                        continue
                    similarity = fuzz.ratio(normalized, db_term) / 100.0
                    if similarity >= min_similarity:
                        matches.append((row["term"], similarity, row["source"]))

        finally:
            conn.close()

        # Sort by similarity and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]

    def soundex(self, term: str) -> str:
        """
        Generate Soundex code for phonetic matching.

        Soundex encodes similar-sounding words to the same code.
        """
        if not term:
            return ""

        term = term.upper()
        # Keep first letter
        soundex = term[0]

        # Encoding map
        codes = {
            "BFPV": "1",
            "CGJKQSXZ": "2",
            "DT": "3",
            "L": "4",
            "MN": "5",
            "R": "6",
        }

        prev_code = ""
        for char in term[1:]:
            code = ""
            for key, val in codes.items():
                if char in key:
                    code = val
                    break

            if code and code != prev_code:
                soundex += code
                prev_code = code

            if len(soundex) >= 4:
                break

        # Pad with zeros if needed
        return soundex.ljust(4, "0")[:4]

    def find_phonetic_matches(self, term: str, limit: int = 5) -> list[tuple[str, str]]:
        """
        Find terms that sound similar using Soundex.

        Args:
            term: Term to find phonetic matches for.
            limit: Maximum matches to return.

        Returns:
            List of (term, source) tuples.
        """
        target_soundex = self.soundex(term)
        matches = []

        conn = self._get_connection()
        try:
            # Check drugs
            cursor = conn.execute("SELECT DISTINCT term, source FROM terms WHERE category = 'drug'")
            for row in cursor.fetchall():
                if self.soundex(row["term"]) == target_soundex:
                    if row["term"].lower() != term.lower():  # Don't match itself
                        matches.append((row["term"], row["source"]))

        finally:
            conn.close()

        return matches[:limit]

    def extract_medical_like_words(self, text: str) -> list[tuple[str, int, int]]:
        """
        Extract words that look like medical terms.

        Args:
            text: Text to analyze.

        Returns:
            List of (word, start, end) tuples.
        """
        found = []
        seen = set()

        for pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                word = match.group()
                if word.lower() not in seen:
                    found.append((word, match.start(), match.end()))
                    seen.add(word.lower())

        return found

    def analyze(self, text: str) -> TermMatchResult:
        """
        Analyze text for known terms and potential errors.

        This is the main method for detecting transcription issues.

        Args:
            text: Text to analyze.

        Returns:
            TermMatchResult with findings.
        """
        known_terms = []
        potential_errors = []
        unrecognized = []

        # Extract medical-like words
        medical_words = self.extract_medical_like_words(text)

        # Also extract words near drug-like prefixes
        for prefix in self.DRUG_PREFIXES:
            pattern = re.compile(rf"\b({prefix}\w+)\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                word = match.group(1)
                if word.lower() not in [w[0].lower() for w in medical_words]:
                    medical_words.append((word, match.start(), match.end()))

        # Check each medical-like word
        for word, start, end in medical_words:
            # Get context
            ctx_start = max(0, start - 50)
            ctx_end = min(len(text), end + 50)
            context = text[ctx_start:ctx_end]

            # Look up in database
            term_info = self.lookup_term(word)

            if term_info:
                # Known term found
                known_terms.append({
                    "term": word,
                    "position": (start, end),
                    "info": term_info,
                })
            else:
                # Not found - check for similar terms
                similar = self.find_similar_terms(word, limit=3)

                if similar:
                    best_match = similar[0]
                    potential_errors.append(PotentialError(
                        found_term=word,
                        suggested_term=best_match[0],
                        confidence=best_match[1],
                        similarity=best_match[1],
                        match_type="fuzzy",
                        position=(start, end),
                        context=context,
                        source=best_match[2],
                    ))
                else:
                    # Try phonetic matching
                    phonetic = self.find_phonetic_matches(word, limit=1)
                    if phonetic:
                        potential_errors.append(PotentialError(
                            found_term=word,
                            suggested_term=phonetic[0][0],
                            confidence=0.6,  # Lower confidence for phonetic
                            similarity=0.6,
                            match_type="phonetic",
                            position=(start, end),
                            context=context,
                            source=phonetic[0][1],
                        ))
                    else:
                        # Unrecognized medical-like word
                        unrecognized.append(word)

        # Calculate error score
        total_medical = len(medical_words)
        if total_medical > 0:
            error_count = len(potential_errors)
            error_score = max(0.0, 1.0 - (error_count / total_medical))
        else:
            error_score = 1.0

        return TermMatchResult(
            text=text,
            known_terms_found=known_terms,
            potential_errors=potential_errors,
            unrecognized_medical_like=unrecognized,
            error_score=error_score,
        )

    def _get_stemmer(self):
        """Get or create Porter Stemmer instance."""
        if not hasattr(self, "_stemmer"):
            try:
                from nltk.stem import PorterStemmer
                self._stemmer = PorterStemmer()
            except ImportError:
                logger.warning("nltk not available, using fallback stemmer")
                self._stemmer = None
        return self._stemmer

    def _normalize_for_comparison(self, word: str) -> str:
        """
        Normalize a word for comparison, handling plural/singular forms.

        Uses NLTK Porter Stemmer for proper linguistic stemming.

        Args:
            word: Word to normalize.

        Returns:
            Stemmed form of the word.
        """
        word = word.lower().strip()

        stemmer = self._get_stemmer()
        if stemmer:
            return stemmer.stem(word)

        # Fallback: basic suffix removal if nltk unavailable
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("es") and len(word) > 3:
            return word[:-2]
        if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
            return word[:-1]

        return word

    def find_inconsistencies(self, text: str) -> list[dict]:
        """
        Find terms that appear correctly in some places but incorrectly in others.

        This detects context-based errors like:
        - "paracetamol" in one sentence but "paracetol" in another

        Note: Plural/singular variations (headache/headaches) are NOT flagged
        as inconsistencies - they are valid grammatical variations.

        Args:
            text: Full text to analyze.

        Returns:
            List of inconsistency findings.
        """
        # First pass: find all known terms
        known_terms = set()
        known_terms_normalized = {}  # normalized -> original
        word_pattern = re.compile(r"\b[a-zA-Z]{4,}\b")

        for match in word_pattern.finditer(text):
            word = match.group()
            if self.is_known_term(word):
                word_lower = word.lower()
                known_terms.add(word_lower)
                # Store both original and normalized form
                normalized = self._normalize_for_comparison(word_lower)
                known_terms_normalized[normalized] = word_lower

        # Second pass: find similar words that might be misspellings
        inconsistencies = []

        try:
            from rapidfuzz import fuzz
        except ImportError:
            return []

        for match in word_pattern.finditer(text):
            word = match.group()
            word_lower = word.lower()
            word_normalized = self._normalize_for_comparison(word_lower)

            # Skip if it's a known term
            if word_lower in known_terms:
                continue

            # Skip if it's a known term after normalization (plural/singular variant)
            if word_normalized in known_terms_normalized:
                continue

            # Check if similar to any known term in this text
            for known in known_terms:
                known_normalized = self._normalize_for_comparison(known)

                # Skip if normalized forms match (same word, different inflection)
                if word_normalized == known_normalized:
                    continue

                similarity = fuzz.ratio(word_lower, known) / 100.0
                if 0.7 <= similarity < 1.0:
                    # Double-check: not just a plural/singular difference
                    # by comparing normalized forms
                    normalized_similarity = fuzz.ratio(word_normalized, known_normalized) / 100.0
                    if normalized_similarity >= 1.0:
                        # Same word after normalization, skip
                        continue

                    # Might be a misspelling
                    inconsistencies.append({
                        "found": word,
                        "expected": known,
                        "similarity": similarity,
                        "position": (match.start(), match.end()),
                        "type": "inconsistent_spelling",
                    })

        return inconsistencies


# Singleton instance
_matcher: MedicalTermMatcher | None = None


def get_term_matcher() -> MedicalTermMatcher:
    """Get singleton matcher instance."""
    global _matcher
    if _matcher is None:
        _matcher = MedicalTermMatcher()
    return _matcher


def analyze_transcription_errors(text: str) -> TermMatchResult:
    """Convenience function to analyze text for errors."""
    return get_term_matcher().analyze(text)


def find_spelling_inconsistencies(text: str) -> list[dict]:
    """Convenience function to find inconsistencies."""
    return get_term_matcher().find_inconsistencies(text)
