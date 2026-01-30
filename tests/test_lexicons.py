"""
Tests for medical lexicon module.

Tests the lexicon interface, mock implementations, and unified lookup.
"""
from __future__ import annotations

import pytest

from hsttb.lexicons import (
    LexiconEntry,
    LexiconSource,
    MockMedicalLexicon,
    UnifiedMedicalLexicon,
)


class TestLexiconEntry:
    """Tests for LexiconEntry dataclass."""

    def test_valid_entry(self) -> None:
        """Create valid lexicon entry."""
        entry = LexiconEntry(
            term="Metformin",
            normalized="metformin",
            code="6809",
            category="drug",
            source=LexiconSource.RXNORM,
            synonyms=("Glucophage",),
        )

        assert entry.term == "Metformin"
        assert entry.normalized == "metformin"
        assert entry.code == "6809"
        assert entry.category == "drug"
        assert entry.source == LexiconSource.RXNORM
        assert entry.synonyms == ("Glucophage",)

    def test_empty_term_raises(self) -> None:
        """Empty term raises ValueError."""
        with pytest.raises(ValueError, match="term cannot be empty"):
            LexiconEntry(
                term="",
                normalized="test",
                code="123",
                category="drug",
                source=LexiconSource.MOCK,
            )

    def test_empty_normalized_raises(self) -> None:
        """Empty normalized raises ValueError."""
        with pytest.raises(ValueError, match="normalized cannot be empty"):
            LexiconEntry(
                term="test",
                normalized="",
                code="123",
                category="drug",
                source=LexiconSource.MOCK,
            )

    def test_empty_code_raises(self) -> None:
        """Empty code raises ValueError."""
        with pytest.raises(ValueError, match="code cannot be empty"):
            LexiconEntry(
                term="test",
                normalized="test",
                code="",
                category="drug",
                source=LexiconSource.MOCK,
            )

    def test_entry_is_frozen(self) -> None:
        """Entry is immutable."""
        entry = LexiconEntry(
            term="test",
            normalized="test",
            code="123",
            category="drug",
            source=LexiconSource.MOCK,
        )

        with pytest.raises(AttributeError):
            entry.term = "modified"  # type: ignore[misc]


class TestLexiconSource:
    """Tests for LexiconSource enum."""

    def test_source_values(self) -> None:
        """All expected sources exist."""
        assert LexiconSource.RXNORM.value == "rxnorm"
        assert LexiconSource.SNOMED.value == "snomed"
        assert LexiconSource.ICD10.value == "icd10"
        assert LexiconSource.CUSTOM.value == "custom"
        assert LexiconSource.MOCK.value == "mock"


class TestMockMedicalLexicon:
    """Tests for MockMedicalLexicon implementation."""

    @pytest.fixture
    def lexicon(self) -> MockMedicalLexicon:
        """Create empty mock lexicon."""
        return MockMedicalLexicon()

    def test_initialization(self, lexicon: MockMedicalLexicon) -> None:
        """Mock lexicon initializes correctly."""
        assert lexicon.source == LexiconSource.MOCK
        assert lexicon.is_loaded is False

    def test_add_entry(self, lexicon: MockMedicalLexicon) -> None:
        """Add entry to lexicon."""
        lexicon.add_entry("metformin", "6809", "drug")

        assert lexicon.is_loaded is True
        assert lexicon.contains("metformin") is True
        assert lexicon.contains("Metformin") is True  # case-insensitive

    def test_lookup(self, lexicon: MockMedicalLexicon) -> None:
        """Lookup returns correct entry."""
        lexicon.add_entry("metformin", "6809", "drug", ("Glucophage",))

        entry = lexicon.lookup("metformin")
        assert entry is not None
        assert entry.term == "metformin"
        assert entry.code == "6809"
        assert entry.category == "drug"

        # Lookup by synonym
        entry2 = lexicon.lookup("Glucophage")
        assert entry2 is not None
        assert entry2.term == "metformin"

    def test_lookup_not_found(self, lexicon: MockMedicalLexicon) -> None:
        """Lookup returns None for unknown term."""
        assert lexicon.lookup("unknown") is None

    def test_contains(self, lexicon: MockMedicalLexicon) -> None:
        """Contains checks term existence."""
        lexicon.add_entry("aspirin", "1191", "drug")

        assert lexicon.contains("aspirin") is True
        assert lexicon.contains("ASPIRIN") is True
        assert lexicon.contains("unknown") is False

    def test_get_category(self, lexicon: MockMedicalLexicon) -> None:
        """Get category returns correct category."""
        lexicon.add_entry("metformin", "6809", "drug")
        lexicon.add_entry("diabetes", "73211009", "diagnosis")

        assert lexicon.get_category("metformin") == "drug"
        assert lexicon.get_category("diabetes") == "diagnosis"
        assert lexicon.get_category("unknown") is None

    def test_load(self, lexicon: MockMedicalLexicon) -> None:
        """Load marks lexicon as loaded."""
        assert lexicon.is_loaded is False
        lexicon.load("/fake/path")
        assert lexicon.is_loaded is True

    def test_get_stats(self, lexicon: MockMedicalLexicon) -> None:
        """Get stats returns correct statistics."""
        lexicon.add_entry("metformin", "6809", "drug")
        lexicon.add_entry("aspirin", "1191", "drug")
        lexicon.add_entry("diabetes", "73211009", "diagnosis")

        stats = lexicon.get_stats()
        assert stats is not None
        assert stats.entry_count == 3
        assert stats.source == LexiconSource.MOCK
        assert stats.categories["drug"] == 2
        assert stats.categories["diagnosis"] == 1

    def test_get_stats_not_loaded(self) -> None:
        """Get stats returns None if not loaded."""
        lexicon = MockMedicalLexicon()
        assert lexicon.get_stats() is None

    def test_with_common_drugs(self) -> None:
        """Create lexicon with common drugs."""
        lexicon = MockMedicalLexicon.with_common_drugs()

        assert lexicon.is_loaded is True
        assert lexicon.source == LexiconSource.RXNORM

        # Check some common drugs
        assert lexicon.contains("metformin") is True
        assert lexicon.contains("aspirin") is True
        assert lexicon.contains("lisinopril") is True

        # Check synonym lookup
        assert lexicon.contains("Glucophage") is True
        assert lexicon.contains("Tylenol") is True

    def test_with_common_diagnoses(self) -> None:
        """Create lexicon with common diagnoses."""
        lexicon = MockMedicalLexicon.with_common_diagnoses()

        assert lexicon.is_loaded is True
        assert lexicon.source == LexiconSource.SNOMED

        # Check some common diagnoses
        assert lexicon.contains("diabetes mellitus") is True
        assert lexicon.contains("hypertension") is True
        assert lexicon.contains("COPD") is True

        # Check abbreviations
        assert lexicon.contains("HTN") is True
        assert lexicon.contains("CAD") is True

    def test_with_common_terms(self) -> None:
        """Create lexicon with combined terms."""
        lexicon = MockMedicalLexicon.with_common_terms()

        # Contains both drugs and diagnoses
        assert lexicon.contains("metformin") is True
        assert lexicon.contains("diabetes mellitus") is True

        # All use MOCK source
        entry = lexicon.lookup("metformin")
        assert entry is not None
        assert entry.source == LexiconSource.MOCK


class TestUnifiedMedicalLexicon:
    """Tests for UnifiedMedicalLexicon."""

    @pytest.fixture
    def unified(self) -> UnifiedMedicalLexicon:
        """Create unified lexicon with drugs and diagnoses."""
        unified = UnifiedMedicalLexicon()
        unified.add_lexicon("drugs", MockMedicalLexicon.with_common_drugs())
        unified.add_lexicon("diagnoses", MockMedicalLexicon.with_common_diagnoses())
        return unified

    def test_initialization(self) -> None:
        """Unified lexicon initializes correctly."""
        unified = UnifiedMedicalLexicon()
        assert len(unified.lexicons) == 0

    def test_add_lexicon(self, unified: UnifiedMedicalLexicon) -> None:
        """Add lexicon to unified."""
        assert len(unified.lexicons) == 2
        assert "drugs" in unified.lexicons
        assert "diagnoses" in unified.lexicons

    def test_remove_lexicon(self, unified: UnifiedMedicalLexicon) -> None:
        """Remove lexicon from unified."""
        result = unified.remove_lexicon("drugs")
        assert result is True
        assert "drugs" not in unified.lexicons

        result = unified.remove_lexicon("nonexistent")
        assert result is False

    def test_lookup(self, unified: UnifiedMedicalLexicon) -> None:
        """Lookup searches across all lexicons."""
        # Drug lookup
        entry = unified.lookup("metformin")
        assert entry is not None
        assert entry.category == "drug"

        # Diagnosis lookup
        entry = unified.lookup("diabetes mellitus")
        assert entry is not None
        assert entry.category == "diagnosis"

        # Not found
        assert unified.lookup("unknown") is None

    def test_lookup_all(self, unified: UnifiedMedicalLexicon) -> None:
        """Lookup all returns entries from all lexicons."""
        # Add term that exists in multiple lexicons
        drugs = MockMedicalLexicon()
        drugs.add_entry("test", "123", "drug")
        diagnoses = MockMedicalLexicon()
        diagnoses.add_entry("test", "456", "diagnosis")

        unified = UnifiedMedicalLexicon()
        unified.add_lexicon("drugs", drugs)
        unified.add_lexicon("diagnoses", diagnoses)

        results = unified.lookup_all("test")
        assert len(results) == 2

    def test_contains(self, unified: UnifiedMedicalLexicon) -> None:
        """Contains checks all lexicons."""
        assert unified.contains("metformin") is True
        assert unified.contains("diabetes") is True
        assert unified.contains("unknown") is False

    def test_get_category(self, unified: UnifiedMedicalLexicon) -> None:
        """Get category from any lexicon."""
        assert unified.get_category("metformin") == "drug"
        assert unified.get_category("diabetes mellitus") == "diagnosis"
        assert unified.get_category("unknown") is None

    def test_get_source(self, unified: UnifiedMedicalLexicon) -> None:
        """Get source from entry."""
        source = unified.get_source("metformin")
        assert source == LexiconSource.RXNORM

        source = unified.get_source("diabetes mellitus")
        assert source == LexiconSource.SNOMED

        assert unified.get_source("unknown") is None

    def test_identify_terms(self, unified: UnifiedMedicalLexicon) -> None:
        """Identify medical terms in text."""
        text = "Patient has diabetes mellitus and takes metformin"

        terms = unified.identify_terms(text)

        # Should find both terms
        term_texts = [t.text.lower() for t in terms]
        assert "diabetes mellitus" in term_texts or "diabetes" in term_texts
        assert "metformin" in term_texts

    def test_get_stats(self, unified: UnifiedMedicalLexicon) -> None:
        """Get statistics from all lexicons."""
        stats = unified.get_stats()

        assert stats.lexicon_count == 2
        assert stats.total_entries > 0
        assert "drugs" in stats.lexicon_stats
        assert "diagnoses" in stats.lexicon_stats

    def test_priority_order(self) -> None:
        """Lexicons are searched in priority order."""
        # Create overlapping lexicons
        high_priority = MockMedicalLexicon()
        high_priority.add_entry("test", "HIGH", "priority")

        low_priority = MockMedicalLexicon()
        low_priority.add_entry("test", "LOW", "priority")

        unified = UnifiedMedicalLexicon()
        unified.add_lexicon("low", low_priority)
        unified.add_lexicon("high", high_priority, priority=0)  # Higher priority

        entry = unified.lookup("test")
        assert entry is not None
        assert entry.code == "HIGH"

    def test_fuzzy_lookup(self, unified: UnifiedMedicalLexicon) -> None:
        """Fuzzy lookup returns empty by default."""
        results = unified.fuzzy_lookup("metfornin")  # Typo
        assert results == []  # Default implementation returns empty


class TestLexiconNormalization:
    """Tests for term normalization."""

    def test_case_insensitive_lookup(self) -> None:
        """Lookup is case-insensitive."""
        lexicon = MockMedicalLexicon()
        lexicon.add_entry("Metformin", "6809", "drug")

        assert lexicon.lookup("metformin") is not None
        assert lexicon.lookup("METFORMIN") is not None
        assert lexicon.lookup("MetForMin") is not None

    def test_whitespace_handling(self) -> None:
        """Lookup handles whitespace."""
        lexicon = MockMedicalLexicon()
        lexicon.add_entry("diabetes mellitus", "73211009", "diagnosis")

        assert lexicon.lookup("diabetes mellitus") is not None
        assert lexicon.lookup("  diabetes mellitus  ") is not None
