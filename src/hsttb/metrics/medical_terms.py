"""
Dynamic Medical Terminology Provider.

Provides medical terms (drugs, conditions, symptoms) dynamically loaded
from the SQLite lexicon rather than hardcoded lists.

Example:
    >>> from hsttb.metrics.medical_terms import get_medical_terms
    >>> terms = get_medical_terms()
    >>> drugs = terms.get_drugs()
    >>> conditions = terms.get_conditions()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DosageRange:
    """Dosage range for a medication."""
    drug: str
    min_dose: float
    max_dose: float
    common_doses: list[float]
    unit: str
    typical_frequencies: list[str]
    unusual_frequencies: list[str] = field(default_factory=list)


@dataclass
class DrugConditionPair:
    """Valid drug-condition indication pair."""
    drug: str
    condition: str
    is_valid: bool = True


class MedicalTermsProvider:
    """
    Dynamic provider for medical terminology.

    Supports two modes:
    1. Lazy database queries (efficient for single lookups)
    2. Bulk loading (for modules that need full term lists)

    The is_* methods use lazy database queries.
    The get_* methods load terms into memory (use sparingly).
    """

    # Database path
    DB_PATH = Path.home() / ".hsttb" / "medical_lexicon.db"

    def __init__(self, lazy_mode: bool = True) -> None:
        """
        Initialize provider.

        Args:
            lazy_mode: If True, use database queries for lookups instead
                      of loading all terms into memory. Default True.
        """
        self._drugs: set[str] | None = None
        self._conditions: set[str] | None = None
        self._symptoms: set[str] | None = None
        self._dosage_ranges: dict[str, DosageRange] | None = None
        self._drug_indications: dict[str, set[str]] | None = None
        self._lexicon: Any = None
        self._loaded = False
        self._lazy_mode = lazy_mode
        self._db_available = self.DB_PATH.exists()

    def _ensure_loaded(self) -> None:
        """Ensure terminology is loaded."""
        if self._loaded:
            return

        self._drugs = set()
        self._conditions = set()
        self._symptoms = set()
        self._dosage_ranges = {}
        self._drug_indications = {}

        # Always load embedded fallback first (for dosage ranges)
        self._load_embedded_fallback()

        # Then try to load from SQLite database directly
        try:
            self._load_from_database_direct()
            logger.info(
                f"Loaded {len(self._drugs)} drugs, "
                f"{len(self._conditions)} conditions from database"
            )
        except Exception as e:
            logger.debug(f"Could not load from database: {e}")

            # Fallback: try via SQLite lexicon class if yaml is available
            try:
                import importlib.util
                if importlib.util.find_spec("yaml") is not None:
                    from hsttb.lexicons.sqlite_lexicon import SQLiteMedicalLexicon

                    lexicon = SQLiteMedicalLexicon()
                    lexicon.load("auto")
                    self._lexicon = lexicon
                    self._load_from_lexicon(lexicon)
                    logger.info(f"Loaded via lexicon class")
            except Exception as e2:
                logger.debug(f"Could not load via lexicon: {e2}")

        self._loaded = True

    def _load_from_database_direct(self) -> None:
        """Load terms directly from SQLite database without lexicon class."""
        import sqlite3
        from pathlib import Path

        db_path = Path.home() / ".hsttb" / "medical_lexicon.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Load drugs from 'terms' table
        try:
            cursor.execute("SELECT DISTINCT term FROM terms WHERE category = 'drug'")
            for row in cursor.fetchall():
                self._drugs.add(row[0].lower())
        except Exception:
            pass

        # Load conditions from 'terms' table
        try:
            cursor.execute("SELECT DISTINCT term FROM terms WHERE category = 'diagnosis'")
            for row in cursor.fetchall():
                self._conditions.add(row[0].lower())
        except Exception:
            pass

        # Load from 'medical_terms' table (ICD-10, ICD-9, SNOMED, UMLS)
        try:
            cursor.execute("SELECT DISTINCT term FROM medical_terms WHERE category = 'diagnosis'")
            for row in cursor.fetchall():
                self._conditions.add(row[0].lower())

            cursor.execute("SELECT DISTINCT term FROM medical_terms WHERE category = 'clinical_term'")
            for row in cursor.fetchall():
                self._symptoms.add(row[0].lower())

            cursor.execute("SELECT DISTINCT term FROM medical_terms WHERE category = 'concept'")
            for row in cursor.fetchall():
                self._conditions.add(row[0].lower())
        except Exception:
            pass

        # Load drug indications
        try:
            cursor.execute("""
                SELECT DISTINCT drug_normalized, condition_normalized
                FROM drug_indications
                WHERE drug_normalized IS NOT NULL AND condition_normalized IS NOT NULL
            """)
            for row in cursor.fetchall():
                drug = row[0].lower()
                indication = row[1].lower()
                if drug not in self._drug_indications:
                    self._drug_indications[drug] = set()
                self._drug_indications[drug].add(indication)
        except Exception:
            pass

        conn.close()

        # Add symptoms from conditions if not loaded
        if not self._symptoms:
            self._symptoms = self._conditions.copy()

        # Load dosage ranges
        self._load_dosage_ranges()

    def _load_from_lexicon(self, lexicon: Any) -> None:
        """Load terms from SQLite lexicon."""
        try:
            # Get all entries from both tables
            conn = lexicon._get_connection()
            cursor = conn.cursor()

            # Load drugs from 'terms' table (RxNorm data)
            cursor.execute("SELECT DISTINCT term FROM terms WHERE category = 'drug'")
            for row in cursor.fetchall():
                self._drugs.add(row[0].lower())

            # Load conditions from 'terms' table
            cursor.execute("SELECT DISTINCT term FROM terms WHERE category = 'diagnosis'")
            for row in cursor.fetchall():
                self._conditions.add(row[0].lower())

            # Also load from 'medical_terms' table (ICD-10, ICD-9, SNOMED, UMLS)
            try:
                cursor.execute("SELECT DISTINCT term FROM medical_terms WHERE category = 'diagnosis'")
                for row in cursor.fetchall():
                    self._conditions.add(row[0].lower())

                # Load clinical terms as symptoms
                cursor.execute("SELECT DISTINCT term FROM medical_terms WHERE category = 'clinical_term'")
                for row in cursor.fetchall():
                    self._symptoms.add(row[0].lower())

                # Load UMLS concepts as conditions too
                cursor.execute("SELECT DISTINCT term FROM medical_terms WHERE category = 'concept'")
                for row in cursor.fetchall():
                    self._conditions.add(row[0].lower())

            except Exception:
                # medical_terms table might not exist yet
                pass

            # Load drug indications if available
            try:
                cursor.execute("""
                    SELECT DISTINCT drug_normalized, condition_normalized
                    FROM drug_indications
                    WHERE drug_normalized IS NOT NULL AND condition_normalized IS NOT NULL
                """)
                for row in cursor.fetchall():
                    drug = row[0].lower()
                    indication = row[1].lower()
                    if drug not in self._drug_indications:
                        self._drug_indications[drug] = set()
                    self._drug_indications[drug].add(indication)
            except Exception:
                pass

            conn.close()

            # Add symptoms from conditions (many overlap)
            if not self._symptoms:
                self._symptoms = self._conditions.copy()

            # Load dosage ranges from embedded data (API doesn't provide this)
            self._load_dosage_ranges()

            logger.info(
                f"Loaded from lexicon: {len(self._drugs)} drugs, "
                f"{len(self._conditions)} conditions, {len(self._symptoms)} symptoms"
            )

        except Exception as e:
            logger.warning(f"Error loading from lexicon: {e}")
            self._load_embedded_fallback()

    def _load_embedded_fallback(self) -> None:
        """Load minimal embedded fallback data."""
        # Minimal drug list - these are fetched from APIs normally
        # Only used as fallback when lexicon unavailable
        self._drugs = set()
        self._conditions = set()
        self._symptoms = set()

        # Load dosage ranges (these require medical knowledge, not in APIs)
        self._load_dosage_ranges()

        # Extract drug names from dosage ranges
        for drug in self._dosage_ranges.keys():
            self._drugs.add(drug)

    def _load_dosage_ranges(self) -> None:
        """
        Load dosage ranges.

        Note: Dosage ranges require medical knowledge and are not available
        from standard APIs. These are based on clinical guidelines.
        """
        # Dosage data based on standard clinical guidelines
        # This is medical knowledge that cannot be fetched from APIs
        dosage_data = [
            ("amlodipine", 2.5, 10, [5, 10], "mg", ["once daily", "daily"]),
            ("metformin", 250, 2550, [500, 850, 1000], "mg", ["once daily", "twice daily", "bid"]),
            ("lisinopril", 2.5, 40, [5, 10, 20], "mg", ["once daily", "daily"]),
            ("atorvastatin", 10, 80, [10, 20, 40], "mg", ["once daily", "at night", "daily"]),
            ("aspirin", 75, 325, [81, 100, 325], "mg", ["once daily", "daily", "as needed"]),
            ("omeprazole", 10, 40, [20, 40], "mg", ["once daily", "twice daily", "daily"]),
            ("metoprolol", 12.5, 400, [25, 50, 100], "mg", ["once daily", "twice daily", "bid"]),
            ("losartan", 25, 100, [25, 50, 100], "mg", ["once daily", "twice daily"]),
            ("gabapentin", 100, 3600, [100, 300, 600], "mg", ["three times daily", "tid", "twice daily"]),
            ("sertraline", 25, 200, [50, 100], "mg", ["once daily", "daily"]),
            ("prednisone", 1, 80, [5, 10, 20, 40], "mg", ["once daily", "daily"]),
            ("warfarin", 1, 15, [2, 2.5, 5], "mg", ["once daily", "daily"]),
            ("levothyroxine", 12.5, 300, [25, 50, 75, 100, 125], "mcg", ["once daily", "daily"]),
            ("hydrochlorothiazide", 12.5, 50, [12.5, 25], "mg", ["once daily", "daily"]),
            ("furosemide", 20, 600, [20, 40, 80], "mg", ["once daily", "twice daily"]),
            ("insulin", 1, 200, [10, 20, 30, 40], "units", ["daily", "twice daily", "with meals"]),
        ]

        for data in dosage_data:
            drug, min_d, max_d, common, unit, freqs = data
            self._dosage_ranges[drug] = DosageRange(
                drug=drug,
                min_dose=min_d,
                max_dose=max_d,
                common_doses=common,
                unit=unit,
                typical_frequencies=freqs,
            )
            # Also add drug to drugs set
            self._drugs.add(drug)

    def get_drugs(self) -> set[str]:
        """Get all drug names."""
        self._ensure_loaded()
        return self._drugs.copy()

    def get_conditions(self) -> set[str]:
        """Get all condition/diagnosis names."""
        self._ensure_loaded()
        return self._conditions.copy()

    def get_symptoms(self) -> set[str]:
        """Get all symptom names."""
        self._ensure_loaded()
        return self._symptoms.copy()

    def get_all_medical_terms(self) -> set[str]:
        """Get all medical terms (drugs + conditions + symptoms)."""
        self._ensure_loaded()
        return self._drugs | self._conditions | self._symptoms

    def get_dosage_range(self, drug: str) -> DosageRange | None:
        """Get dosage range for a drug."""
        self._ensure_loaded()
        return self._dosage_ranges.get(drug.lower())

    def get_all_dosage_ranges(self) -> dict[str, DosageRange]:
        """Get all dosage ranges."""
        self._ensure_loaded()
        return self._dosage_ranges.copy()

    def get_drug_indications(self, drug: str) -> set[str]:
        """Get valid indications for a drug."""
        self._ensure_loaded()
        return self._drug_indications.get(drug.lower(), set()).copy()

    def is_valid_drug_condition_pair(self, drug: str, condition: str) -> bool | None:
        """
        Check if drug-condition pair is valid.

        Returns:
            True if valid pair, False if invalid, None if unknown.
        """
        self._ensure_loaded()
        drug_lower = drug.lower()
        condition_lower = condition.lower()

        indications = self._drug_indications.get(drug_lower)
        if indications is None:
            return None  # Unknown drug

        # Check if condition matches any indication
        for indication in indications:
            if condition_lower in indication or indication in condition_lower:
                return True

        return None  # Unknown relationship

    def _query_database(self, term: str, category: str | None = None) -> bool:
        """
        Query database directly for a term (lazy lookup).

        More efficient than loading all terms into memory.
        Uses indexed 'normalized' column for fast lookups.
        """
        if not self._db_available:
            return False

        import sqlite3
        term_lower = term.lower().strip()

        try:
            conn = sqlite3.connect(str(self.DB_PATH))
            cursor = conn.cursor()

            if category == "drug":
                # Only check terms table for drugs
                cursor.execute(
                    "SELECT 1 FROM terms WHERE normalized = ? AND category = 'drug' LIMIT 1",
                    (term_lower,)
                )
            elif category in ("diagnosis", "condition"):
                # Check both tables for conditions/diagnoses
                # terms table
                cursor.execute(
                    "SELECT 1 FROM terms WHERE normalized = ? AND category = 'diagnosis' LIMIT 1",
                    (term_lower,)
                )
                if cursor.fetchone():
                    conn.close()
                    return True
                # medical_terms table (diagnosis + concept categories)
                cursor.execute(
                    "SELECT 1 FROM medical_terms WHERE normalized = ? AND category IN ('diagnosis', 'concept') LIMIT 1",
                    (term_lower,)
                )
            elif category == "clinical_term":
                cursor.execute(
                    "SELECT 1 FROM medical_terms WHERE normalized = ? AND category = 'clinical_term' LIMIT 1",
                    (term_lower,)
                )
            else:
                # Check all - first terms table, then medical_terms
                cursor.execute(
                    "SELECT 1 FROM terms WHERE normalized = ? LIMIT 1",
                    (term_lower,)
                )
                if cursor.fetchone():
                    conn.close()
                    return True
                cursor.execute(
                    "SELECT 1 FROM medical_terms WHERE normalized = ? LIMIT 1",
                    (term_lower,)
                )

            result = cursor.fetchone() is not None
            conn.close()
            return result

        except Exception as e:
            logger.debug(f"Database query failed: {e}")
            return False

    def is_drug(self, term: str) -> bool:
        """Check if term is a known drug (uses lazy database query)."""
        if self._lazy_mode and self._db_available:
            return self._query_database(term, "drug")
        self._ensure_loaded()
        return term.lower() in self._drugs

    def is_condition(self, term: str) -> bool:
        """Check if term is a known condition (uses lazy database query)."""
        if self._lazy_mode and self._db_available:
            return self._query_database(term, "diagnosis")
        self._ensure_loaded()
        return term.lower() in self._conditions

    def is_symptom(self, term: str) -> bool:
        """Check if term is a known symptom (uses lazy database query)."""
        if self._lazy_mode and self._db_available:
            return self._query_database(term, "clinical_term")
        self._ensure_loaded()
        return term.lower() in self._symptoms

    def is_medical_term(self, term: str) -> bool:
        """Check if term is any known medical term (uses lazy database query)."""
        if self._lazy_mode and self._db_available:
            return self._query_database(term)
        self._ensure_loaded()
        term_lower = term.lower()
        return term_lower in self._drugs or term_lower in self._conditions or term_lower in self._symptoms

    def find_terms_in_text(self, text: str) -> dict[str, list[str]]:
        """
        Find all medical terms in text.

        Returns:
            Dict with keys 'drugs', 'conditions', 'symptoms' containing found terms.
        """
        self._ensure_loaded()
        text_lower = text.lower()

        found = {
            "drugs": [],
            "conditions": [],
            "symptoms": [],
        }

        for drug in self._drugs:
            if drug in text_lower:
                found["drugs"].append(drug)

        for condition in self._conditions:
            if condition in text_lower:
                found["conditions"].append(condition)

        for symptom in self._symptoms:
            if symptom in text_lower and symptom not in found["conditions"]:
                found["symptoms"].append(symptom)

        return found

    def reload(self) -> None:
        """Force reload of terminology."""
        self._loaded = False
        self._ensure_loaded()


# Singleton instance
_provider: MedicalTermsProvider | None = None


def get_medical_terms() -> MedicalTermsProvider:
    """Get singleton medical terms provider."""
    global _provider
    if _provider is None:
        _provider = MedicalTermsProvider()
    return _provider


def reset_medical_terms() -> None:
    """Reset the medical terms provider (for testing)."""
    global _provider
    _provider = None
