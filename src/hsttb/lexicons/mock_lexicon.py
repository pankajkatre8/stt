"""
Mock medical lexicon for testing.

This module provides a mock lexicon implementation that can be
preloaded with test data for unit testing the TER engine.

Example:
    >>> lexicon = MockMedicalLexicon()
    >>> lexicon.add_entry("metformin", "6809", "drug")
    >>> entry = lexicon.lookup("metformin")
    >>> assert entry.category == "drug"
"""
from __future__ import annotations

import time

from hsttb.lexicons.base import (
    LexiconEntry,
    LexiconSource,
    LexiconStats,
    MedicalLexicon,
)


class MockMedicalLexicon(MedicalLexicon):
    """
    Mock medical lexicon for testing.

    Can be initialized with predefined entries or built
    dynamically using add_entry().

    Attributes:
        entries: Dictionary of normalized term to LexiconEntry.

    Example:
        >>> lexicon = MockMedicalLexicon.with_common_drugs()
        >>> entry = lexicon.lookup("aspirin")
        >>> assert entry is not None
    """

    def __init__(
        self,
        entries: dict[str, LexiconEntry] | None = None,
        source: LexiconSource = LexiconSource.MOCK,
    ) -> None:
        """
        Initialize mock lexicon.

        Args:
            entries: Optional pre-populated entries.
            source: Source identifier for entries.
        """
        self._entries: dict[str, LexiconEntry] = entries or {}
        self._source = source
        self._is_loaded = False
        self._load_time_ms = 0.0

    @property
    def source(self) -> LexiconSource:
        """Return the lexicon source identifier."""
        return self._source

    @property
    def is_loaded(self) -> bool:
        """Check if the lexicon is loaded."""
        return self._is_loaded

    def load(self, path: str) -> None:  # noqa: ARG002
        """
        Simulate loading from path.

        For mock, this just marks as loaded.

        Args:
            path: Ignored for mock lexicon.
        """
        start = time.perf_counter()
        # Mock lexicon doesn't actually load from file
        self._is_loaded = True
        self._load_time_ms = (time.perf_counter() - start) * 1000

    def add_entry(
        self,
        term: str,
        code: str,
        category: str,
        synonyms: tuple[str, ...] | None = None,
    ) -> None:
        """
        Add an entry to the lexicon.

        Args:
            term: Term text.
            code: Standard code.
            category: Medical category.
            synonyms: Optional synonyms.
        """
        normalized = self.normalize_term(term)
        entry = LexiconEntry(
            term=term,
            normalized=normalized,
            code=code,
            category=category,
            source=self._source,
            synonyms=synonyms or (),
        )
        self._entries[normalized] = entry

        # Also index synonyms
        for synonym in entry.synonyms:
            self._entries[self.normalize_term(synonym)] = entry

        self._is_loaded = True

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

        Searches the text for all terms in the lexicon.

        Args:
            text: Text to search.

        Returns:
            List of LexiconEntry for all found terms.
        """
        found_entries: list[LexiconEntry] = []
        text_lower = text.lower()
        seen_terms: set[str] = set()

        # Get unique entries (avoid duplicates from synonyms)
        unique_entries: dict[str, LexiconEntry] = {}
        for entry in self._entries.values():
            if entry.normalized not in unique_entries:
                unique_entries[entry.normalized] = entry

        # Search for each term in the text
        for entry in unique_entries.values():
            # Check main term
            term_lower = entry.term.lower()
            if term_lower in text_lower and term_lower not in seen_terms:
                found_entries.append(entry)
                seen_terms.add(term_lower)
                continue

            # Check synonyms
            for synonym in entry.synonyms:
                syn_lower = synonym.lower()
                if syn_lower in text_lower and entry.normalized not in seen_terms:
                    found_entries.append(entry)
                    seen_terms.add(entry.normalized)
                    break

        return found_entries

    def contains(self, term: str) -> bool:
        """
        Check if term exists in lexicon.

        Args:
            term: Term to check.

        Returns:
            True if term exists.
        """
        normalized = self.normalize_term(term)
        return normalized in self._entries

    def get_category(self, term: str) -> str | None:
        """
        Get the category of a term.

        Args:
            term: Term to categorize.

        Returns:
            Category string if found.
        """
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
            source=self._source,
            categories=categories,
            load_time_ms=self._load_time_ms,
        )

    @classmethod
    def with_common_drugs(cls) -> MockMedicalLexicon:
        """
        Create a mock lexicon with common drug names.

        Returns:
            MockMedicalLexicon with pre-populated drugs.
        """
        lexicon = cls(source=LexiconSource.RXNORM)

        # Common drugs for testing - comprehensive list including commonly confused drugs
        drugs = [
            # Diabetes medications
            ("metformin", "6809", ("Glucophage", "metformin hydrochloride")),
            ("glipizide", "25789", ("Glucotrol",)),
            ("glyburide", "25793", ("DiaBeta", "Micronase")),
            ("insulin", "5856", ()),
            # Cardiovascular
            ("aspirin", "1191", ("acetylsalicylic acid", "ASA")),
            ("lisinopril", "29046", ("Prinivil", "Zestril")),
            ("atorvastatin", "83367", ("Lipitor",)),
            ("amlodipine", "17767", ("Norvasc",)),
            ("metoprolol", "6918", ("Lopressor", "Toprol")),
            ("atenolol", "1202", ("Tenormin",)),
            ("losartan", "52175", ("Cozaar",)),
            ("hydrochlorothiazide", "5487", ("HCTZ", "Microzide")),
            ("warfarin", "11289", ("Coumadin",)),
            ("clopidogrel", "32968", ("Plavix",)),
            # GI medications
            ("omeprazole", "7646", ("Prilosec",)),
            ("pantoprazole", "40790", ("Protonix",)),
            # Pain/neuro
            ("gabapentin", "25480", ("Neurontin",)),
            ("pregabalin", "187832", ("Lyrica",)),
            ("prednisone", "8640", ()),
            ("ibuprofen", "5640", ("Advil", "Motrin")),
            ("acetaminophen", "161", ("Tylenol", "paracetamol")),
            ("tramadol", "10689", ("Ultram",)),
            ("oxycodone", "7804", ("OxyContin", "Percocet")),
            # Antibiotics
            ("amoxicillin", "723", ("Amoxil",)),
            ("azithromycin", "18631", ("Zithromax", "Z-pack")),
            ("ciprofloxacin", "2551", ("Cipro",)),
            ("doxycycline", "3640", ()),
            # Commonly confused/sound-alike drugs (critical for STT evaluation!)
            ("methotrexate", "6851", ("Trexall", "Rheumatrex")),  # sounds like metformin
            ("hydroxychloroquine", "5521", ("Plaquenil",)),
            ("sulfasalazine", "9524", ("Azulfidine",)),
            ("lamotrigine", "28439", ("Lamictal",)),
            ("labetalol", "6135", ()),
            ("levetiracetam", "39998", ("Keppra",)),
            ("clonidine", "2599", ("Catapres",)),
            ("clonazepam", "2598", ("Klonopin",)),  # sounds like clonidine
            # Psych medications
            ("sertraline", "36437", ("Zoloft",)),
            ("fluoxetine", "4493", ("Prozac",)),
            ("escitalopram", "321988", ("Lexapro",)),
            ("duloxetine", "72625", ("Cymbalta",)),
            ("alprazolam", "596", ("Xanax",)),
            ("lorazepam", "6470", ("Ativan",)),
        ]

        for term, code, synonyms in drugs:
            lexicon.add_entry(term, code, "drug", synonyms)

        return lexicon

    @classmethod
    def with_common_diagnoses(cls) -> MockMedicalLexicon:
        """
        Create a mock lexicon with common diagnoses.

        Returns:
            MockMedicalLexicon with pre-populated diagnoses.
        """
        lexicon = cls(source=LexiconSource.SNOMED)

        # Common diagnoses for testing
        diagnoses = [
            ("diabetes mellitus", "73211009", ("diabetes", "DM")),
            ("hypertension", "38341003", ("high blood pressure", "HTN")),
            ("hyperlipidemia", "55822004", ("high cholesterol",)),
            ("chronic kidney disease", "709044004", ("CKD",)),
            ("coronary artery disease", "53741008", ("CAD",)),
            ("heart failure", "84114007", ("CHF", "congestive heart failure")),
            ("atrial fibrillation", "49436004", ("afib", "AF")),
            ("chronic obstructive pulmonary disease", "13645005", ("COPD",)),
            ("asthma", "195967001", ()),
            ("depression", "35489007", ("major depressive disorder",)),
            ("anxiety", "48694002", ("anxiety disorder",)),
            ("osteoarthritis", "396275006", ("OA", "degenerative joint disease")),
            ("pneumonia", "233604007", ()),
            ("urinary tract infection", "68566005", ("UTI",)),
            ("gastroesophageal reflux disease", "235595009", ("GERD", "acid reflux")),
        ]

        for term, code, synonyms in diagnoses:
            lexicon.add_entry(term, code, "diagnosis", synonyms)

        return lexicon

    @classmethod
    def with_common_terms(cls) -> MockMedicalLexicon:
        """
        Create a mock lexicon combining drugs and diagnoses.

        Returns:
            MockMedicalLexicon with common medical terms.
        """
        lexicon = cls(source=LexiconSource.MOCK)

        # Merge drugs - copy all key mappings including synonyms
        drugs_lexicon = cls.with_common_drugs()
        for key, entry in drugs_lexicon._entries.items():
            new_entry = LexiconEntry(
                term=entry.term,
                normalized=entry.normalized,
                code=entry.code,
                category=entry.category,
                source=LexiconSource.MOCK,
                synonyms=entry.synonyms,
            )
            lexicon._entries[key] = new_entry

        # Merge diagnoses - copy all key mappings including synonyms
        diagnoses_lexicon = cls.with_common_diagnoses()
        for key, entry in diagnoses_lexicon._entries.items():
            new_entry = LexiconEntry(
                term=entry.term,
                normalized=entry.normalized,
                code=entry.code,
                category=entry.category,
                source=LexiconSource.MOCK,
                synonyms=entry.synonyms,
            )
            lexicon._entries[key] = new_entry

        lexicon._is_loaded = True
        return lexicon
