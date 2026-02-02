"""
SQLite-backed medical lexicon for fast local retrieval.

Fetches medical terms from APIs and stores them in a local SQLite
database for efficient lookups. Database auto-refreshes after 30 days.

Example:
    >>> from hsttb.lexicons.sqlite_lexicon import SQLiteMedicalLexicon
    >>> lexicon = SQLiteMedicalLexicon()
    >>> lexicon.load()  # Fetches from API if needed, else uses local DB
    >>> entry = lexicon.lookup("metformin")
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from hsttb.lexicons.base import (
    LexiconEntry,
    LexiconSource,
    LexiconStats,
    MedicalLexicon,
)

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path.home() / ".hsttb" / "medical_lexicon.db"


class SQLiteMedicalLexicon(MedicalLexicon):
    """
    SQLite-backed medical lexicon.

    Stores medical terms in a local SQLite database for fast lookups.
    Automatically fetches from RxNorm, OpenFDA, and ICD-10 APIs if
    the database is empty or stale.

    Attributes:
        db_path: Path to the SQLite database file.
        refresh_days: Days before refreshing from APIs.

    Example:
        >>> lexicon = SQLiteMedicalLexicon()
        >>> lexicon.load()  # Auto-fetches if needed
        >>> entry = lexicon.lookup("metformin")
        >>> print(f"Found: {entry.term} ({entry.code})")
    """

    # SQL schema
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    );

    CREATE TABLE IF NOT EXISTS terms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        term TEXT NOT NULL,
        normalized TEXT NOT NULL,
        code TEXT,
        category TEXT NOT NULL,
        source TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(normalized)
    );

    CREATE TABLE IF NOT EXISTS synonyms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        term_id INTEGER NOT NULL,
        synonym TEXT NOT NULL,
        normalized TEXT NOT NULL,
        FOREIGN KEY (term_id) REFERENCES terms(id) ON DELETE CASCADE,
        UNIQUE(normalized)
    );

    CREATE TABLE IF NOT EXISTS drug_indications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        drug_normalized TEXT NOT NULL,
        condition_normalized TEXT NOT NULL,
        source TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(drug_normalized, condition_normalized)
    );

    CREATE INDEX IF NOT EXISTS idx_terms_normalized ON terms(normalized);
    CREATE INDEX IF NOT EXISTS idx_terms_category ON terms(category);
    CREATE INDEX IF NOT EXISTS idx_synonyms_normalized ON synonyms(normalized);
    CREATE INDEX IF NOT EXISTS idx_indications_drug ON drug_indications(drug_normalized);
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        refresh_days: int = 30,
        drug_limit: int = 5000,
        condition_limit: int = 5000,
    ) -> None:
        """
        Initialize SQLite lexicon.

        Args:
            db_path: Path to SQLite database file.
            refresh_days: Days before refreshing from APIs.
            drug_limit: Maximum drugs to fetch from APIs.
            condition_limit: Maximum conditions to fetch from APIs.
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._refresh_days = refresh_days
        self._drug_limit = drug_limit
        self._condition_limit = condition_limit
        self._is_loaded = False
        self._load_time_ms = 0.0
        self._conn: sqlite3.Connection | None = None

        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def source(self) -> LexiconSource:
        """Return the lexicon source identifier."""
        return LexiconSource.CUSTOM

    @property
    def is_loaded(self) -> bool:
        """Check if the lexicon is loaded."""
        return self._is_loaded

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()

    def _get_metadata(self, key: str) -> str | None:
        """Get metadata value."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            return row["value"] if row else None

    def _set_metadata(self, key: str, value: str) -> None:
        """Set metadata value."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    def _needs_refresh(self) -> bool:
        """Check if database needs refresh from APIs."""
        last_refresh = self._get_metadata("last_refresh")
        if not last_refresh:
            return True

        try:
            last_date = datetime.fromisoformat(last_refresh)
            return datetime.now() > last_date + timedelta(days=self._refresh_days)
        except ValueError:
            return True

    def _get_term_count(self) -> int:
        """Get number of terms in database."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM terms")
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    def load(self, path: str = "auto") -> None:
        """
        Load lexicon from database, fetching from APIs if needed.

        Args:
            path: Load mode:
                - "auto": Use DB if fresh, else fetch from APIs
                - "refresh": Force refresh from APIs
                - "local": Use local DB only, no API calls
        """
        start = time.perf_counter()

        # Initialize database
        self._init_db()

        # Check if we need to fetch from APIs
        needs_data = self._get_term_count() < 100
        needs_refresh = path == "refresh" or (path == "auto" and self._needs_refresh())

        if needs_data or needs_refresh:
            if path != "local":
                logger.info("Fetching medical terms from APIs...")
                try:
                    asyncio.run(self._fetch_and_store())
                except RuntimeError:
                    # Already in async context
                    self._load_embedded_data()
            else:
                logger.info("Local mode: using embedded data")
                self._load_embedded_data()
        else:
            logger.info(f"Using existing database with {self._get_term_count()} terms")

        self._is_loaded = True
        self._load_time_ms = (time.perf_counter() - start) * 1000

        logger.info(f"Lexicon loaded in {self._load_time_ms:.0f}ms")

    async def load_async(self, path: str = "auto") -> None:
        """
        Load lexicon asynchronously.

        Args:
            path: Load mode (see load() for options).
        """
        start = time.perf_counter()

        self._init_db()

        needs_data = self._get_term_count() < 100
        needs_refresh = path == "refresh" or (path == "auto" and self._needs_refresh())

        if needs_data or needs_refresh:
            if path != "local":
                logger.info("Fetching medical terms from APIs...")
                await self._fetch_and_store()
            else:
                self._load_embedded_data()
        else:
            logger.info(f"Using existing database with {self._get_term_count()} terms")

        self._is_loaded = True
        self._load_time_ms = (time.perf_counter() - start) * 1000

    async def _fetch_and_store(self) -> None:
        """Fetch from APIs and store in database."""
        try:
            from hsttb.lexicons.api_fetcher import MedicalTermFetcher

            fetcher = MedicalTermFetcher()

            try:
                drugs, conditions = await fetcher.fetch_all(
                    drug_limit=self._drug_limit,
                    condition_limit=self._condition_limit,
                    use_cache=True,
                )

                # Store in database
                with self._get_connection() as conn:
                    # Clear existing data
                    conn.execute("DELETE FROM synonyms")
                    conn.execute("DELETE FROM terms")

                    # Insert drugs
                    for drug in drugs:
                        self._insert_term(
                            conn,
                            term=drug.name,
                            code=drug.rxcui or drug.ndc or "",
                            category="drug",
                            source="RxNorm",
                            synonyms=drug.brand_names,
                        )

                    # Insert conditions from API
                    for condition in conditions:
                        self._insert_term(
                            conn,
                            term=condition.name,
                            code=condition.code,
                            category="diagnosis",
                            source=condition.code_system,
                            synonyms=condition.synonyms,
                        )

                    # If no conditions from API, load embedded conditions
                    if len(conditions) == 0:
                        logger.info("No conditions from API, loading embedded conditions...")
                        embedded_conditions = self._get_embedded_conditions()
                        for term, code, category, source, synonyms in embedded_conditions:
                            self._insert_term(conn, term, code, category, source, synonyms)
                        condition_count = len(embedded_conditions)
                    else:
                        condition_count = len(conditions)

                    conn.commit()

                # Update metadata
                self._set_metadata("last_refresh", datetime.now().isoformat())
                self._set_metadata("drug_count", str(len(drugs)))
                self._set_metadata("condition_count", str(condition_count))

                logger.info(
                    f"Stored {len(drugs)} drugs and {condition_count} conditions"
                )

            finally:
                await fetcher.close()

        except ImportError as e:
            logger.warning(f"API fetcher not available: {e}")
            self._load_embedded_data()
        except Exception as e:
            logger.error(f"Failed to fetch from APIs: {e}")
            self._load_embedded_data()

    def _insert_term(
        self,
        conn: sqlite3.Connection,
        term: str,
        code: str,
        category: str,
        source: str,
        synonyms: list[str] | None = None,
    ) -> int | None:
        """Insert a term into the database."""
        normalized = self.normalize_term(term)

        try:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO terms (term, normalized, code, category, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (term, normalized, code, category, source),
            )

            if cursor.lastrowid and synonyms:
                for synonym in synonyms:
                    syn_normalized = self.normalize_term(synonym)
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO synonyms (term_id, synonym, normalized)
                        VALUES (?, ?, ?)
                        """,
                        (cursor.lastrowid, synonym, syn_normalized),
                    )

            return cursor.lastrowid

        except sqlite3.IntegrityError:
            return None

    def _load_embedded_data(self) -> None:
        """Load embedded minimal dataset as fallback."""
        logger.info("Loading embedded medical terms...")

        drug_count = 0
        condition_count = 0

        with self._get_connection() as conn:
            # Common drugs (top 50 most prescribed in US)
            drugs = [
                ("metformin", "6809", "drug", "RxNorm", ["Glucophage"]),
                ("lisinopril", "29046", "drug", "RxNorm", ["Prinivil", "Zestril"]),
                ("atorvastatin", "83367", "drug", "RxNorm", ["Lipitor"]),
                ("levothyroxine", "10582", "drug", "RxNorm", ["Synthroid", "Levoxyl"]),
                ("amlodipine", "17767", "drug", "RxNorm", ["Norvasc"]),
                ("metoprolol", "6918", "drug", "RxNorm", ["Lopressor", "Toprol"]),
                ("omeprazole", "7646", "drug", "RxNorm", ["Prilosec"]),
                ("simvastatin", "36567", "drug", "RxNorm", ["Zocor"]),
                ("losartan", "52175", "drug", "RxNorm", ["Cozaar"]),
                ("albuterol", "435", "drug", "RxNorm", ["Ventolin", "ProAir"]),
                ("gabapentin", "25480", "drug", "RxNorm", ["Neurontin"]),
                ("hydrochlorothiazide", "5487", "drug", "RxNorm", ["HCTZ", "Microzide"]),
                ("sertraline", "36437", "drug", "RxNorm", ["Zoloft"]),
                ("acetaminophen", "161", "drug", "RxNorm", ["Tylenol"]),
                ("ibuprofen", "5640", "drug", "RxNorm", ["Advil", "Motrin"]),
                ("aspirin", "1191", "drug", "RxNorm", ["Bayer"]),
                ("prednisone", "8640", "drug", "RxNorm", []),
                ("fluoxetine", "4493", "drug", "RxNorm", ["Prozac"]),
                ("pantoprazole", "40790", "drug", "RxNorm", ["Protonix"]),
                ("escitalopram", "321988", "drug", "RxNorm", ["Lexapro"]),
                ("montelukast", "88249", "drug", "RxNorm", ["Singulair"]),
                ("rosuvastatin", "301542", "drug", "RxNorm", ["Crestor"]),
                ("bupropion", "42347", "drug", "RxNorm", ["Wellbutrin"]),
                ("furosemide", "4603", "drug", "RxNorm", ["Lasix"]),
                ("tramadol", "10689", "drug", "RxNorm", ["Ultram"]),
                ("trazodone", "10737", "drug", "RxNorm", ["Desyrel"]),
                ("duloxetine", "72625", "drug", "RxNorm", ["Cymbalta"]),
                ("amoxicillin", "723", "drug", "RxNorm", ["Amoxil"]),
                ("azithromycin", "18631", "drug", "RxNorm", ["Zithromax", "Z-pack"]),
                ("ciprofloxacin", "2551", "drug", "RxNorm", ["Cipro"]),
                ("clopidogrel", "32968", "drug", "RxNorm", ["Plavix"]),
                ("warfarin", "11289", "drug", "RxNorm", ["Coumadin"]),
                ("insulin", "5856", "drug", "RxNorm", ["Humulin", "Novolin"]),
                ("glipizide", "25789", "drug", "RxNorm", ["Glucotrol"]),
                ("alprazolam", "596", "drug", "RxNorm", ["Xanax"]),
                ("lorazepam", "6470", "drug", "RxNorm", ["Ativan"]),
                ("clonazepam", "2598", "drug", "RxNorm", ["Klonopin"]),
                ("oxycodone", "7804", "drug", "RxNorm", ["OxyContin"]),
                ("hydrocodone", "5489", "drug", "RxNorm", ["Vicodin", "Norco"]),
                ("morphine", "7052", "drug", "RxNorm", ["MS Contin"]),
                ("fentanyl", "4337", "drug", "RxNorm", ["Duragesic"]),
                ("pregabalin", "187832", "drug", "RxNorm", ["Lyrica"]),
                ("venlafaxine", "39786", "drug", "RxNorm", ["Effexor"]),
                ("citalopram", "2556", "drug", "RxNorm", ["Celexa"]),
                ("quetiapine", "51272", "drug", "RxNorm", ["Seroquel"]),
                ("aripiprazole", "89013", "drug", "RxNorm", ["Abilify"]),
                ("methylphenidate", "6901", "drug", "RxNorm", ["Ritalin", "Concerta"]),
                ("amphetamine", "725", "drug", "RxNorm", ["Adderall"]),
                ("doxycycline", "3640", "drug", "RxNorm", ["Vibramycin"]),
                ("methotrexate", "6851", "drug", "RxNorm", ["Trexall"]),
            ]

            for term, code, category, source, synonyms in drugs:
                if self._insert_term(conn, term, code, category, source, synonyms):
                    drug_count += 1

            # Common conditions (ICD-10)
            conditions = [
                ("diabetes mellitus", "E11", "diagnosis", "ICD10", ["diabetes", "DM", "type 2 diabetes"]),
                ("essential hypertension", "I10", "diagnosis", "ICD10", ["hypertension", "high blood pressure", "HTN"]),
                ("hyperlipidemia", "E78.5", "diagnosis", "ICD10", ["high cholesterol"]),
                ("major depressive disorder", "F32", "diagnosis", "ICD10", ["depression", "MDD"]),
                ("generalized anxiety disorder", "F41.1", "diagnosis", "ICD10", ["anxiety", "GAD"]),
                ("chronic obstructive pulmonary disease", "J44", "diagnosis", "ICD10", ["COPD"]),
                ("asthma", "J45", "diagnosis", "ICD10", []),
                ("coronary artery disease", "I25.10", "diagnosis", "ICD10", ["CAD", "heart disease"]),
                ("heart failure", "I50", "diagnosis", "ICD10", ["CHF", "congestive heart failure"]),
                ("atrial fibrillation", "I48", "diagnosis", "ICD10", ["afib", "AF"]),
                ("chronic kidney disease", "N18", "diagnosis", "ICD10", ["CKD"]),
                ("osteoarthritis", "M19", "diagnosis", "ICD10", ["OA", "degenerative joint disease"]),
                ("rheumatoid arthritis", "M06", "diagnosis", "ICD10", ["RA"]),
                ("hypothyroidism", "E03", "diagnosis", "ICD10", []),
                ("hyperthyroidism", "E05", "diagnosis", "ICD10", []),
                ("gastroesophageal reflux disease", "K21", "diagnosis", "ICD10", ["GERD", "acid reflux"]),
                ("pneumonia", "J18", "diagnosis", "ICD10", []),
                ("urinary tract infection", "N39.0", "diagnosis", "ICD10", ["UTI"]),
                ("migraine", "G43", "diagnosis", "ICD10", []),
                ("seizure disorder", "G40", "diagnosis", "ICD10", ["epilepsy"]),
                ("anemia", "D64", "diagnosis", "ICD10", []),
                ("obesity", "E66", "diagnosis", "ICD10", []),
                ("sleep apnea", "G47.3", "diagnosis", "ICD10", []),
                ("back pain", "M54", "diagnosis", "ICD10", []),
                ("neuropathy", "G62", "diagnosis", "ICD10", ["peripheral neuropathy"]),
                ("stroke", "I63", "diagnosis", "ICD10", ["CVA", "cerebrovascular accident"]),
                ("deep vein thrombosis", "I82", "diagnosis", "ICD10", ["DVT"]),
                ("pulmonary embolism", "I26", "diagnosis", "ICD10", ["PE"]),
                ("cirrhosis", "K74", "diagnosis", "ICD10", []),
                ("hepatitis", "B19", "diagnosis", "ICD10", []),
                ("type 1 diabetes", "E10", "diagnosis", "ICD10", ["T1DM", "juvenile diabetes"]),
                ("acute myocardial infarction", "I21", "diagnosis", "ICD10", ["heart attack", "MI"]),
                ("angina pectoris", "I20", "diagnosis", "ICD10", ["chest pain", "angina"]),
                ("chronic pain", "G89", "diagnosis", "ICD10", []),
                ("fibromyalgia", "M79.7", "diagnosis", "ICD10", []),
                ("bipolar disorder", "F31", "diagnosis", "ICD10", []),
                ("schizophrenia", "F20", "diagnosis", "ICD10", []),
                ("PTSD", "F43.1", "diagnosis", "ICD10", ["post-traumatic stress disorder"]),
                ("ADHD", "F90", "diagnosis", "ICD10", ["attention deficit hyperactivity disorder"]),
                ("dementia", "F03", "diagnosis", "ICD10", []),
                ("alzheimer disease", "G30", "diagnosis", "ICD10", ["alzheimers"]),
                ("parkinson disease", "G20", "diagnosis", "ICD10", ["parkinsons"]),
                ("multiple sclerosis", "G35", "diagnosis", "ICD10", ["MS"]),
                ("breast cancer", "C50", "diagnosis", "ICD10", []),
                ("lung cancer", "C34", "diagnosis", "ICD10", []),
                ("prostate cancer", "C61", "diagnosis", "ICD10", []),
                ("colon cancer", "C18", "diagnosis", "ICD10", ["colorectal cancer"]),
                ("leukemia", "C95", "diagnosis", "ICD10", []),
                ("lymphoma", "C85", "diagnosis", "ICD10", []),
                ("sepsis", "A41", "diagnosis", "ICD10", []),
            ]

            for term, code, category, source, synonyms in conditions:
                if self._insert_term(conn, term, code, category, source, synonyms):
                    condition_count += 1

            conn.commit()

        self._set_metadata("last_refresh", datetime.now().isoformat())
        self._set_metadata("source", "embedded")
        self._set_metadata("drug_count", str(drug_count))
        self._set_metadata("condition_count", str(condition_count))

        logger.info(f"Loaded {drug_count} drugs and {condition_count} conditions from embedded data")

    def _get_embedded_conditions(self) -> list[tuple[str, str, str, str, list[str]]]:
        """Get embedded conditions data for supplementing API results."""
        return [
            ("diabetes mellitus", "E11", "diagnosis", "ICD10", ["diabetes", "DM", "type 2 diabetes"]),
            ("essential hypertension", "I10", "diagnosis", "ICD10", ["hypertension", "high blood pressure", "HTN"]),
            ("hyperlipidemia", "E78.5", "diagnosis", "ICD10", ["high cholesterol"]),
            ("major depressive disorder", "F32", "diagnosis", "ICD10", ["depression", "MDD"]),
            ("generalized anxiety disorder", "F41.1", "diagnosis", "ICD10", ["anxiety", "GAD"]),
            ("chronic obstructive pulmonary disease", "J44", "diagnosis", "ICD10", ["COPD"]),
            ("asthma", "J45", "diagnosis", "ICD10", []),
            ("coronary artery disease", "I25.10", "diagnosis", "ICD10", ["CAD", "heart disease"]),
            ("heart failure", "I50", "diagnosis", "ICD10", ["CHF", "congestive heart failure"]),
            ("atrial fibrillation", "I48", "diagnosis", "ICD10", ["afib", "AF"]),
            ("chronic kidney disease", "N18", "diagnosis", "ICD10", ["CKD"]),
            ("osteoarthritis", "M19", "diagnosis", "ICD10", ["OA", "degenerative joint disease"]),
            ("rheumatoid arthritis", "M06", "diagnosis", "ICD10", ["RA"]),
            ("hypothyroidism", "E03", "diagnosis", "ICD10", []),
            ("hyperthyroidism", "E05", "diagnosis", "ICD10", []),
            ("gastroesophageal reflux disease", "K21", "diagnosis", "ICD10", ["GERD", "acid reflux"]),
            ("pneumonia", "J18", "diagnosis", "ICD10", []),
            ("urinary tract infection", "N39.0", "diagnosis", "ICD10", ["UTI"]),
            ("migraine", "G43", "diagnosis", "ICD10", []),
            ("seizure disorder", "G40", "diagnosis", "ICD10", ["epilepsy"]),
            ("anemia", "D64", "diagnosis", "ICD10", []),
            ("obesity", "E66", "diagnosis", "ICD10", []),
            ("sleep apnea", "G47.3", "diagnosis", "ICD10", []),
            ("back pain", "M54", "diagnosis", "ICD10", []),
            ("neuropathy", "G62", "diagnosis", "ICD10", ["peripheral neuropathy"]),
            ("stroke", "I63", "diagnosis", "ICD10", ["CVA", "cerebrovascular accident"]),
            ("deep vein thrombosis", "I82", "diagnosis", "ICD10", ["DVT"]),
            ("pulmonary embolism", "I26", "diagnosis", "ICD10", ["PE"]),
            ("cirrhosis", "K74", "diagnosis", "ICD10", []),
            ("hepatitis", "B19", "diagnosis", "ICD10", []),
            ("type 1 diabetes", "E10", "diagnosis", "ICD10", ["T1DM", "juvenile diabetes"]),
            ("acute myocardial infarction", "I21", "diagnosis", "ICD10", ["heart attack", "MI"]),
            ("angina pectoris", "I20", "diagnosis", "ICD10", ["chest pain", "angina"]),
            ("chronic pain", "G89", "diagnosis", "ICD10", []),
            ("fibromyalgia", "M79.7", "diagnosis", "ICD10", []),
            ("bipolar disorder", "F31", "diagnosis", "ICD10", []),
            ("schizophrenia", "F20", "diagnosis", "ICD10", []),
            ("PTSD", "F43.1", "diagnosis", "ICD10", ["post-traumatic stress disorder"]),
            ("ADHD", "F90", "diagnosis", "ICD10", ["attention deficit hyperactivity disorder"]),
            ("dementia", "F03", "diagnosis", "ICD10", []),
            ("alzheimer disease", "G30", "diagnosis", "ICD10", ["alzheimers"]),
            ("parkinson disease", "G20", "diagnosis", "ICD10", ["parkinsons"]),
            ("multiple sclerosis", "G35", "diagnosis", "ICD10", ["MS"]),
            ("breast cancer", "C50", "diagnosis", "ICD10", []),
            ("lung cancer", "C34", "diagnosis", "ICD10", []),
            ("prostate cancer", "C61", "diagnosis", "ICD10", []),
            ("colon cancer", "C18", "diagnosis", "ICD10", ["colorectal cancer"]),
            ("leukemia", "C95", "diagnosis", "ICD10", []),
            ("lymphoma", "C85", "diagnosis", "ICD10", []),
            ("sepsis", "A41", "diagnosis", "ICD10", []),
        ]

    def lookup(self, term: str) -> LexiconEntry | None:
        """
        Look up a term in the lexicon.

        Uses indexed SQLite queries for fast retrieval.

        Args:
            term: Term to look up.

        Returns:
            LexiconEntry if found, None otherwise.
        """
        if not term or not term.strip():
            return None

        normalized = self.normalize_term(term)

        with self._get_connection() as conn:
            # First try direct term lookup
            cursor = conn.execute(
                """
                SELECT term, normalized, code, category, source
                FROM terms WHERE normalized = ?
                """,
                (normalized,),
            )
            row = cursor.fetchone()

            if not row:
                # Try synonym lookup
                cursor = conn.execute(
                    """
                    SELECT t.term, t.normalized, t.code, t.category, t.source
                    FROM terms t
                    JOIN synonyms s ON t.id = s.term_id
                    WHERE s.normalized = ?
                    """,
                    (normalized,),
                )
                row = cursor.fetchone()

            if row:
                # Get synonyms
                cursor = conn.execute(
                    """
                    SELECT synonym FROM synonyms s
                    JOIN terms t ON s.term_id = t.id
                    WHERE t.normalized = ?
                    """,
                    (row["normalized"],),
                )
                synonyms = tuple(r["synonym"] for r in cursor.fetchall())

                # Validate required fields before creating entry
                if not row["term"] or not row["normalized"]:
                    return None

                return LexiconEntry(
                    term=row["term"],
                    normalized=row["normalized"],
                    code=row["code"] or "UNKNOWN",
                    category=row["category"] or "unknown",
                    source=LexiconSource(row["source"]) if row["source"] in [s.value for s in LexiconSource] else LexiconSource.CUSTOM,
                    synonyms=synonyms,
                )

        return None

    def extract_terms(self, text: str) -> list[LexiconEntry]:
        """
        Extract all known medical terms from text.

        Args:
            text: Text to search.

        Returns:
            List of LexiconEntry for all found terms.
        """
        found: list[LexiconEntry] = []
        text_lower = text.lower()
        seen: set[str] = set()

        with self._get_connection() as conn:
            # Get all terms
            cursor = conn.execute(
                "SELECT term, normalized FROM terms"
            )

            for row in cursor.fetchall():
                # Skip empty terms
                if not row["term"] or not row["normalized"]:
                    continue

                term_lower = row["term"].lower()
                normalized = row["normalized"]

                if term_lower in text_lower and normalized not in seen:
                    entry = self.lookup(row["term"])
                    if entry:
                        found.append(entry)
                        seen.add(normalized)

            # Also check synonyms
            cursor = conn.execute(
                """
                SELECT s.synonym, t.normalized as term_normalized
                FROM synonyms s
                JOIN terms t ON s.term_id = t.id
                """
            )

            for row in cursor.fetchall():
                syn_lower = row["synonym"].lower()
                term_normalized = row["term_normalized"]

                if syn_lower in text_lower and term_normalized not in seen:
                    entry = self.lookup(row["synonym"])
                    if entry:
                        found.append(entry)
                        seen.add(term_normalized)

        return found

    def contains(self, term: str) -> bool:
        """Check if term exists in lexicon."""
        return self.lookup(term) is not None

    def get_category(self, term: str) -> str | None:
        """Get the category of a term."""
        entry = self.lookup(term)
        return entry.category if entry else None

    def get_stats(self) -> LexiconStats | None:
        """Get statistics about the lexicon."""
        if not self._is_loaded:
            return None

        with self._get_connection() as conn:
            # Count by category
            cursor = conn.execute(
                """
                SELECT category, COUNT(*) as cnt
                FROM terms
                GROUP BY category
                """
            )
            categories = {row["category"]: row["cnt"] for row in cursor.fetchall()}

            # Total count
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM terms")
            total = cursor.fetchone()["cnt"]

        return LexiconStats(
            entry_count=total,
            source=self.source,
            categories=categories,
            load_time_ms=self._load_time_ms,
        )

    def search(
        self,
        query: str,
        category: str | None = None,
        limit: int = 100,
    ) -> list[LexiconEntry]:
        """
        Search for terms matching a query.

        Args:
            query: Search query (partial match).
            category: Optional category filter ("drug" or "diagnosis").
            limit: Maximum results to return.

        Returns:
            List of matching LexiconEntry objects.
        """
        results: list[LexiconEntry] = []
        query_pattern = f"%{query.lower()}%"

        with self._get_connection() as conn:
            if category:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT term FROM terms
                    WHERE normalized LIKE ? AND category = ?
                    LIMIT ?
                    """,
                    (query_pattern, category, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT term FROM terms
                    WHERE normalized LIKE ?
                    LIMIT ?
                    """,
                    (query_pattern, limit),
                )

            for row in cursor.fetchall():
                entry = self.lookup(row["term"])
                if entry:
                    results.append(entry)

        return results

    def add_drug_indication(
        self,
        drug: str,
        condition: str,
        source: str = "manual",
    ) -> bool:
        """
        Add a drug-indication relationship.

        Args:
            drug: Drug name.
            condition: Condition/indication name.
            source: Source of the relationship.

        Returns:
            True if added, False if already exists.
        """
        drug_norm = self.normalize_term(drug)
        condition_norm = self.normalize_term(condition)

        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO drug_indications (drug_normalized, condition_normalized, source)
                    VALUES (?, ?, ?)
                    """,
                    (drug_norm, condition_norm, source),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def is_valid_indication(self, drug: str, condition: str) -> bool:
        """
        Check if a drug-condition pair is a valid indication.

        Args:
            drug: Drug name.
            condition: Condition name.

        Returns:
            True if valid indication, False otherwise.
        """
        drug_norm = self.normalize_term(drug)
        condition_norm = self.normalize_term(condition)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 1 FROM drug_indications
                WHERE drug_normalized = ? AND condition_normalized = ?
                """,
                (drug_norm, condition_norm),
            )
            return cursor.fetchone() is not None

    def get_drug_indications(self, drug: str) -> list[str]:
        """
        Get all known indications for a drug.

        Args:
            drug: Drug name.

        Returns:
            List of condition names.
        """
        drug_norm = self.normalize_term(drug)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT t.term
                FROM drug_indications di
                JOIN terms t ON di.condition_normalized = t.normalized
                WHERE di.drug_normalized = ?
                """,
                (drug_norm,),
            )
            return [row["term"] for row in cursor.fetchall()]

    def export_to_json(self, path: Path | str) -> None:
        """
        Export lexicon to JSON file.

        Args:
            path: Output file path.
        """
        import json

        data = {
            "drugs": [],
            "conditions": [],
            "indications": [],
        }

        with self._get_connection() as conn:
            # Export drugs
            cursor = conn.execute(
                "SELECT * FROM terms WHERE category = 'drug'"
            )
            for row in cursor.fetchall():
                syn_cursor = conn.execute(
                    "SELECT synonym FROM synonyms WHERE term_id = ?",
                    (row["id"],),
                )
                synonyms = [r["synonym"] for r in syn_cursor.fetchall()]

                data["drugs"].append({
                    "name": row["term"],
                    "code": row["code"],
                    "source": row["source"],
                    "synonyms": synonyms,
                })

            # Export conditions
            cursor = conn.execute(
                "SELECT * FROM terms WHERE category = 'diagnosis'"
            )
            for row in cursor.fetchall():
                syn_cursor = conn.execute(
                    "SELECT synonym FROM synonyms WHERE term_id = ?",
                    (row["id"],),
                )
                synonyms = [r["synonym"] for r in syn_cursor.fetchall()]

                data["conditions"].append({
                    "name": row["term"],
                    "code": row["code"],
                    "source": row["source"],
                    "synonyms": synonyms,
                })

            # Export indications
            cursor = conn.execute("SELECT * FROM drug_indications")
            for row in cursor.fetchall():
                data["indications"].append({
                    "drug": row["drug_normalized"],
                    "condition": row["condition_normalized"],
                    "source": row["source"],
                })

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported lexicon to {path}")


# ============================================================================
# Convenience Functions
# ============================================================================

_default_lexicon: SQLiteMedicalLexicon | None = None


def get_sqlite_lexicon() -> SQLiteMedicalLexicon:
    """
    Get the default SQLite medical lexicon (singleton).

    Returns:
        Loaded SQLiteMedicalLexicon.
    """
    global _default_lexicon

    if _default_lexicon is None:
        _default_lexicon = SQLiteMedicalLexicon()
        _default_lexicon.load("auto")

    return _default_lexicon


async def get_sqlite_lexicon_async() -> SQLiteMedicalLexicon:
    """
    Get the default SQLite medical lexicon asynchronously.

    Returns:
        Loaded SQLiteMedicalLexicon.
    """
    global _default_lexicon

    if _default_lexicon is None:
        _default_lexicon = SQLiteMedicalLexicon()
        await _default_lexicon.load_async("auto")

    return _default_lexicon
