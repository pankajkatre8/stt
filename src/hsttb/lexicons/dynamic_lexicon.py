"""
Dynamic medical lexicon that fetches from APIs.

Provides a MedicalLexicon implementation that:
1. Attempts to fetch from RxNorm, OpenFDA, ICD-10 APIs
2. Caches results locally for 30 days
3. Falls back to embedded minimal dataset if APIs unavailable

Example:
    >>> from hsttb.lexicons.dynamic_lexicon import DynamicMedicalLexicon
    >>> lexicon = DynamicMedicalLexicon()
    >>> await lexicon.load_async()  # or lexicon.load() for sync
    >>> entry = lexicon.lookup("metformin")
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from hsttb.lexicons.base import (
    LexiconEntry,
    LexiconSource,
    LexiconStats,
    MedicalLexicon,
)

logger = logging.getLogger(__name__)


class DynamicMedicalLexicon(MedicalLexicon):
    """
    Medical lexicon that fetches terms from public APIs.

    Automatically fetches and caches drug names from RxNorm/OpenFDA
    and diagnosis codes from ICD-10-CM. Falls back to a minimal
    embedded dataset if APIs are unavailable.

    Attributes:
        entries: Dictionary of normalized term to LexiconEntry.
        drug_count: Number of drugs loaded.
        condition_count: Number of conditions loaded.

    Example:
        >>> lexicon = DynamicMedicalLexicon()
        >>> lexicon.load("auto")  # Fetch from APIs or use cache
        >>> entry = lexicon.lookup("diabetes")
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        drug_limit: int = 2000,
        condition_limit: int = 2000,
    ) -> None:
        """
        Initialize dynamic lexicon.

        Args:
            cache_dir: Directory for caching API responses.
            drug_limit: Maximum drugs to fetch.
            condition_limit: Maximum conditions to fetch.
        """
        self._cache_dir = cache_dir
        self._drug_limit = drug_limit
        self._condition_limit = condition_limit
        self._entries: dict[str, LexiconEntry] = {}
        self._is_loaded = False
        self._load_time_ms = 0.0
        self._drug_count = 0
        self._condition_count = 0
        self._source_info: str = "not loaded"

    @property
    def source(self) -> LexiconSource:
        """Return the lexicon source identifier."""
        return LexiconSource.CUSTOM

    @property
    def is_loaded(self) -> bool:
        """Check if the lexicon is loaded."""
        return self._is_loaded

    @property
    def drug_count(self) -> int:
        """Number of drugs loaded."""
        return self._drug_count

    @property
    def condition_count(self) -> int:
        """Number of conditions loaded."""
        return self._condition_count

    def load(self, path: str = "auto") -> None:
        """
        Load lexicon data synchronously.

        Args:
            path: Load mode:
                - "auto": Try API first, fall back to embedded
                - "api": Force API fetch
                - "embedded": Use embedded data only
                - "cache": Use cached data only
        """
        start = time.perf_counter()

        try:
            asyncio.run(self.load_async(path))
        except RuntimeError:
            # Already in async context, use sync fallback
            self._load_sync(path)

        self._load_time_ms = (time.perf_counter() - start) * 1000

    async def load_async(self, path: str = "auto") -> None:
        """
        Load lexicon data asynchronously.

        Args:
            path: Load mode (see load() for options).
        """
        start = time.perf_counter()

        if path == "embedded":
            self._load_embedded()
        elif path == "cache":
            await self._load_from_cache_only()
        elif path in ("auto", "api"):
            await self._load_from_api(force_fetch=(path == "api"))
        else:
            # Treat as file path for compatibility
            self._load_embedded()

        self._load_time_ms = (time.perf_counter() - start) * 1000
        self._is_loaded = True

        logger.info(
            f"Loaded {len(self._entries)} terms "
            f"({self._drug_count} drugs, {self._condition_count} conditions) "
            f"from {self._source_info} in {self._load_time_ms:.0f}ms"
        )

    def _load_sync(self, path: str) -> None:
        """Synchronous loading fallback."""
        if path == "embedded" or path == "auto":
            self._load_embedded()
        else:
            self._load_embedded()

    async def _load_from_api(self, force_fetch: bool = False) -> None:
        """Load from APIs with caching."""
        try:
            from hsttb.lexicons.api_fetcher import MedicalTermFetcher

            fetcher = MedicalTermFetcher(cache_dir=self._cache_dir)

            try:
                drugs, conditions = await fetcher.fetch_all(
                    drug_limit=self._drug_limit,
                    condition_limit=self._condition_limit,
                    use_cache=not force_fetch,
                )

                # Add drugs
                for drug in drugs:
                    self._add_drug(drug)

                # Add conditions
                for condition in conditions:
                    self._add_condition(condition)

                self._source_info = "RxNorm/OpenFDA/ICD-10"

            finally:
                await fetcher.close()

        except ImportError as e:
            logger.warning(f"API fetcher not available ({e}), using embedded data")
            self._load_embedded()
        except Exception as e:
            logger.warning(f"API fetch failed ({e}), using embedded data")
            self._load_embedded()

        # If we got very few results, supplement with embedded
        if self._drug_count < 50 or self._condition_count < 20:
            logger.info("Supplementing with embedded data")
            self._load_embedded(supplement=True)

    async def _load_from_cache_only(self) -> None:
        """Load from cache without API calls."""
        try:
            from hsttb.lexicons.api_fetcher import MedicalTermFetcher

            fetcher = MedicalTermFetcher(cache_dir=self._cache_dir)

            try:
                # This will only use cache, won't make API calls if cache exists
                drugs, conditions = await fetcher.fetch_all(
                    drug_limit=self._drug_limit,
                    condition_limit=self._condition_limit,
                    use_cache=True,
                )

                for drug in drugs:
                    self._add_drug(drug)

                for condition in conditions:
                    self._add_condition(condition)

                self._source_info = "cached API data"

            finally:
                await fetcher.close()

        except Exception as e:
            logger.warning(f"Cache load failed ({e}), using embedded data")
            self._load_embedded()

    def _add_drug(self, drug) -> None:
        """Add a drug entry from API data."""
        normalized = self.normalize_term(drug.name)

        entry = LexiconEntry(
            term=drug.name,
            normalized=normalized,
            code=drug.rxcui or drug.ndc or "",
            category="drug",
            source=LexiconSource.RXNORM,
            synonyms=tuple(drug.brand_names) if drug.brand_names else (),
        )

        self._entries[normalized] = entry
        self._drug_count += 1

        # Also index brand names
        for brand in drug.brand_names:
            brand_norm = self.normalize_term(brand)
            if brand_norm not in self._entries:
                self._entries[brand_norm] = entry

    def _add_condition(self, condition) -> None:
        """Add a condition entry from API data."""
        normalized = self.normalize_term(condition.name)

        source = (
            LexiconSource.ICD10 if condition.code_system == "ICD10"
            else LexiconSource.SNOMED
        )

        entry = LexiconEntry(
            term=condition.name,
            normalized=normalized,
            code=condition.code,
            category="diagnosis",
            source=source,
            synonyms=tuple(condition.synonyms) if condition.synonyms else (),
        )

        self._entries[normalized] = entry
        self._condition_count += 1

        # Also index synonyms
        for synonym in condition.synonyms:
            syn_norm = self.normalize_term(synonym)
            if syn_norm not in self._entries:
                self._entries[syn_norm] = entry

    def _load_embedded(self, supplement: bool = False) -> None:
        """
        Load embedded minimal dataset.

        This provides a fallback when APIs are unavailable.
        """
        if not supplement:
            self._entries.clear()
            self._drug_count = 0
            self._condition_count = 0

        # Common drugs (top 100 most prescribed in US)
        drugs = [
            ("metformin", "6809", ("Glucophage",)),
            ("lisinopril", "29046", ("Prinivil", "Zestril")),
            ("atorvastatin", "83367", ("Lipitor",)),
            ("levothyroxine", "10582", ("Synthroid", "Levoxyl")),
            ("amlodipine", "17767", ("Norvasc",)),
            ("metoprolol", "6918", ("Lopressor", "Toprol")),
            ("omeprazole", "7646", ("Prilosec",)),
            ("simvastatin", "36567", ("Zocor",)),
            ("losartan", "52175", ("Cozaar",)),
            ("albuterol", "435", ("Ventolin", "ProAir")),
            ("gabapentin", "25480", ("Neurontin",)),
            ("hydrochlorothiazide", "5487", ("HCTZ", "Microzide")),
            ("sertraline", "36437", ("Zoloft",)),
            ("acetaminophen", "161", ("Tylenol",)),
            ("ibuprofen", "5640", ("Advil", "Motrin")),
            ("aspirin", "1191", ("Bayer",)),
            ("prednisone", "8640", ()),
            ("fluoxetine", "4493", ("Prozac",)),
            ("pantoprazole", "40790", ("Protonix",)),
            ("escitalopram", "321988", ("Lexapro",)),
            ("montelukast", "88249", ("Singulair",)),
            ("rosuvastatin", "301542", ("Crestor",)),
            ("bupropion", "42347", ("Wellbutrin",)),
            ("furosemide", "4603", ("Lasix",)),
            ("tramadol", "10689", ("Ultram",)),
            ("trazodone", "10737", ("Desyrel",)),
            ("duloxetine", "72625", ("Cymbalta",)),
            ("amoxicillin", "723", ("Amoxil",)),
            ("azithromycin", "18631", ("Zithromax", "Z-pack")),
            ("ciprofloxacin", "2551", ("Cipro",)),
            ("clopidogrel", "32968", ("Plavix",)),
            ("warfarin", "11289", ("Coumadin",)),
            ("insulin", "5856", ("Humulin", "Novolin")),
            ("glipizide", "25789", ("Glucotrol",)),
            ("alprazolam", "596", ("Xanax",)),
            ("lorazepam", "6470", ("Ativan",)),
            ("clonazepam", "2598", ("Klonopin",)),
            ("oxycodone", "7804", ("OxyContin",)),
            ("hydrocodone", "5489", ("Vicodin", "Norco")),
            ("morphine", "7052", ("MS Contin",)),
            ("fentanyl", "4337", ("Duragesic",)),
            ("pregabalin", "187832", ("Lyrica",)),
            ("venlafaxine", "39786", ("Effexor",)),
            ("citalopram", "2556", ("Celexa",)),
            ("quetiapine", "51272", ("Seroquel",)),
            ("aripiprazole", "89013", ("Abilify",)),
            ("methylphenidate", "6901", ("Ritalin", "Concerta")),
            ("amphetamine", "725", ("Adderall",)),
            ("doxycycline", "3640", ("Vibramycin",)),
            ("methotrexate", "6851", ("Trexall",)),
        ]

        for name, code, synonyms in drugs:
            normalized = self.normalize_term(name)
            if normalized not in self._entries:
                entry = LexiconEntry(
                    term=name,
                    normalized=normalized,
                    code=code,
                    category="drug",
                    source=LexiconSource.RXNORM,
                    synonyms=synonyms,
                )
                self._entries[normalized] = entry
                self._drug_count += 1

                for syn in synonyms:
                    syn_norm = self.normalize_term(syn)
                    if syn_norm not in self._entries:
                        self._entries[syn_norm] = entry

        # Common conditions (ICD-10)
        conditions = [
            ("diabetes mellitus", "E11", ("diabetes", "DM", "type 2 diabetes")),
            ("essential hypertension", "I10", ("hypertension", "high blood pressure", "HTN")),
            ("hyperlipidemia", "E78.5", ("high cholesterol",)),
            ("major depressive disorder", "F32", ("depression", "MDD")),
            ("generalized anxiety disorder", "F41.1", ("anxiety", "GAD")),
            ("chronic obstructive pulmonary disease", "J44", ("COPD",)),
            ("asthma", "J45", ()),
            ("coronary artery disease", "I25.10", ("CAD", "heart disease")),
            ("heart failure", "I50", ("CHF", "congestive heart failure")),
            ("atrial fibrillation", "I48", ("afib", "AF")),
            ("chronic kidney disease", "N18", ("CKD",)),
            ("osteoarthritis", "M19", ("OA", "degenerative joint disease")),
            ("rheumatoid arthritis", "M06", ("RA",)),
            ("hypothyroidism", "E03", ()),
            ("hyperthyroidism", "E05", ()),
            ("gastroesophageal reflux disease", "K21", ("GERD", "acid reflux")),
            ("pneumonia", "J18", ()),
            ("urinary tract infection", "N39.0", ("UTI",)),
            ("migraine", "G43", ()),
            ("seizure disorder", "G40", ("epilepsy",)),
            ("anemia", "D64", ()),
            ("obesity", "E66", ()),
            ("sleep apnea", "G47.3", ()),
            ("back pain", "M54", ()),
            ("neuropathy", "G62", ("peripheral neuropathy",)),
            ("stroke", "I63", ("CVA", "cerebrovascular accident")),
            ("deep vein thrombosis", "I82", ("DVT",)),
            ("pulmonary embolism", "I26", ("PE",)),
            ("cirrhosis", "K74", ()),
            ("hepatitis", "B19", ()),
        ]

        for name, code, synonyms in conditions:
            normalized = self.normalize_term(name)
            if normalized not in self._entries:
                entry = LexiconEntry(
                    term=name,
                    normalized=normalized,
                    code=code,
                    category="diagnosis",
                    source=LexiconSource.ICD10,
                    synonyms=synonyms,
                )
                self._entries[normalized] = entry
                self._condition_count += 1

                for syn in synonyms:
                    syn_norm = self.normalize_term(syn)
                    if syn_norm not in self._entries:
                        self._entries[syn_norm] = entry

        if not supplement:
            self._source_info = "embedded data"

    def lookup(self, term: str) -> LexiconEntry | None:
        """
        Look up a term in the lexicon.

        Args:
            term: Term to look up.

        Returns:
            LexiconEntry if found, None otherwise.
        """
        normalized = self.normalize_term(term)
        return self._entries.get(normalized)

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

        # Get unique entries
        unique_entries: dict[str, LexiconEntry] = {}
        for entry in self._entries.values():
            if entry.normalized not in unique_entries:
                unique_entries[entry.normalized] = entry

        # Search for each term
        for entry in unique_entries.values():
            term_lower = entry.term.lower()
            if term_lower in text_lower and term_lower not in seen:
                found.append(entry)
                seen.add(term_lower)
                continue

            for synonym in entry.synonyms:
                syn_lower = synonym.lower()
                if syn_lower in text_lower and entry.normalized not in seen:
                    found.append(entry)
                    seen.add(entry.normalized)
                    break

        return found

    def contains(self, term: str) -> bool:
        """Check if term exists in lexicon."""
        normalized = self.normalize_term(term)
        return normalized in self._entries

    def get_category(self, term: str) -> str | None:
        """Get the category of a term."""
        entry = self.lookup(term)
        return entry.category if entry else None

    def get_stats(self) -> LexiconStats | None:
        """Get statistics about the lexicon."""
        if not self._is_loaded:
            return None

        categories: dict[str, int] = {}
        unique_entries = set(self._entries.values())
        for entry in unique_entries:
            categories[entry.category] = categories.get(entry.category, 0) + 1

        return LexiconStats(
            entry_count=len(unique_entries),
            source=self.source,
            categories=categories,
            load_time_ms=self._load_time_ms,
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def get_dynamic_lexicon(
    cache_dir: Path | None = None,
    drug_limit: int = 2000,
    condition_limit: int = 2000,
) -> DynamicMedicalLexicon:
    """
    Get a loaded dynamic medical lexicon.

    Args:
        cache_dir: Cache directory path.
        drug_limit: Max drugs to load.
        condition_limit: Max conditions to load.

    Returns:
        Loaded DynamicMedicalLexicon.
    """
    lexicon = DynamicMedicalLexicon(
        cache_dir=cache_dir,
        drug_limit=drug_limit,
        condition_limit=condition_limit,
    )
    lexicon.load("auto")
    return lexicon


async def get_dynamic_lexicon_async(
    cache_dir: Path | None = None,
    drug_limit: int = 2000,
    condition_limit: int = 2000,
) -> DynamicMedicalLexicon:
    """
    Get a loaded dynamic medical lexicon asynchronously.

    Args:
        cache_dir: Cache directory path.
        drug_limit: Max drugs to load.
        condition_limit: Max conditions to load.

    Returns:
        Loaded DynamicMedicalLexicon.
    """
    lexicon = DynamicMedicalLexicon(
        cache_dir=cache_dir,
        drug_limit=drug_limit,
        condition_limit=condition_limit,
    )
    await lexicon.load_async("auto")
    return lexicon
