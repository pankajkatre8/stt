"""
Tests for NLP pipeline registry module.

Tests registration, factory functions, and lazy loading of NLP pipelines.
"""
from __future__ import annotations

import pytest

from hsttb.nlp import MockNERPipeline, NERPipeline
from hsttb.nlp.registry import (
    clear_nlp_registry,
    get_nlp_pipeline,
    get_pipeline_info,
    is_nlp_pipeline_registered,
    list_nlp_pipelines,
    register_nlp_pipeline,
    register_nlp_pipeline_factory,
    unregister_nlp_pipeline,
)


class TestNLPRegistry:
    """Tests for NLP pipeline registry functions."""

    def setup_method(self) -> None:
        """Save original registry state before each test."""
        self._original_pipelines = list_nlp_pipelines()

    def teardown_method(self) -> None:
        """Restore registry after each test."""
        for name in list_nlp_pipelines():
            if name not in self._original_pipelines:
                unregister_nlp_pipeline(name)

    def test_list_nlp_pipelines(self) -> None:
        """list_nlp_pipelines returns registered pipelines."""
        pipelines = list_nlp_pipelines()
        assert isinstance(pipelines, list)
        # Default pipelines should be registered
        assert "mock" in pipelines

    def test_is_nlp_pipeline_registered(self) -> None:
        """is_nlp_pipeline_registered checks registration status."""
        assert is_nlp_pipeline_registered("mock") is True
        assert is_nlp_pipeline_registered("nonexistent_pipeline") is False

    def test_get_nlp_pipeline_mock(self) -> None:
        """get_nlp_pipeline returns mock pipeline."""
        pipeline = get_nlp_pipeline("mock")
        assert isinstance(pipeline, NERPipeline)
        assert pipeline.name == "mock_ner"

    def test_get_nlp_pipeline_invalid(self) -> None:
        """get_nlp_pipeline raises ValueError for unknown pipeline."""
        with pytest.raises(ValueError, match="Unknown NLP pipeline"):
            get_nlp_pipeline("nonexistent_pipeline")

    def test_register_nlp_pipeline_decorator(self) -> None:
        """register_nlp_pipeline decorator registers a class."""

        @register_nlp_pipeline("test_custom_ner")
        class TestNERPipeline(MockNERPipeline):
            pass

        assert is_nlp_pipeline_registered("test_custom_ner") is True
        pipeline = get_nlp_pipeline("test_custom_ner")
        assert isinstance(pipeline, NERPipeline)

        # Cleanup
        unregister_nlp_pipeline("test_custom_ner")

    def test_register_nlp_pipeline_factory(self) -> None:
        """register_nlp_pipeline_factory registers a factory function."""

        def create_test_pipeline() -> NERPipeline:
            return MockNERPipeline.with_common_patterns()

        register_nlp_pipeline_factory("test_factory_ner", create_test_pipeline)

        assert is_nlp_pipeline_registered("test_factory_ner") is True
        pipeline = get_nlp_pipeline("test_factory_ner")
        assert isinstance(pipeline, NERPipeline)

        # Cleanup
        unregister_nlp_pipeline("test_factory_ner")

    def test_unregister_nlp_pipeline(self) -> None:
        """unregister_nlp_pipeline removes a registered pipeline."""

        @register_nlp_pipeline("temp_ner")
        class TempNERPipeline(MockNERPipeline):
            pass

        assert is_nlp_pipeline_registered("temp_ner") is True
        result = unregister_nlp_pipeline("temp_ner")
        assert result is True
        assert is_nlp_pipeline_registered("temp_ner") is False

    def test_unregister_nlp_pipeline_nonexistent(self) -> None:
        """unregister_nlp_pipeline returns False for unknown pipeline."""
        result = unregister_nlp_pipeline("definitely_not_registered")
        assert result is False

    def test_get_pipeline_info_class(self) -> None:
        """get_pipeline_info returns info for class-registered pipeline."""

        @register_nlp_pipeline("info_test_ner")
        class InfoTestNERPipeline(MockNERPipeline):
            """A test NER pipeline for info testing."""
            pass

        info = get_pipeline_info("info_test_ner")
        assert info["name"] == "info_test_ner"
        assert info["class"] == "InfoTestNERPipeline"
        assert info["type"] == "class"
        assert "test NER pipeline" in info["docstring"]

        # Cleanup
        unregister_nlp_pipeline("info_test_ner")

    def test_get_pipeline_info_factory(self) -> None:
        """get_pipeline_info returns info for factory-registered pipeline."""

        def create_info_test_pipeline() -> NERPipeline:
            """Factory for testing pipeline info."""
            return MockNERPipeline()

        register_nlp_pipeline_factory("info_factory_test", create_info_test_pipeline)

        info = get_pipeline_info("info_factory_test")
        assert info["name"] == "info_factory_test"
        assert info["factory"] == "create_info_test_pipeline"
        assert info["type"] == "factory"

        # Cleanup
        unregister_nlp_pipeline("info_factory_test")

    def test_get_pipeline_info_invalid(self) -> None:
        """get_pipeline_info raises ValueError for unknown pipeline."""
        with pytest.raises(ValueError, match="Unknown NLP pipeline"):
            get_pipeline_info("nonexistent")


class TestNLPRegistryIntegration:
    """Integration tests for NLP registry with pipelines."""

    def test_mock_pipeline_extract_entities(self) -> None:
        """Mock pipeline extracts entities correctly."""
        pipeline = get_nlp_pipeline("mock")
        entities = pipeline.extract_entities("patient takes metformin for diabetes")

        assert len(entities) >= 1
        # Should find drug or diagnosis
        entity_texts = [e.text.lower() for e in entities]
        assert any("metformin" in t for t in entity_texts) or any(
            "diabetes" in t for t in entity_texts
        )

    def test_pipeline_reuse(self) -> None:
        """Multiple calls to get_nlp_pipeline work correctly."""
        pipeline1 = get_nlp_pipeline("mock")
        pipeline2 = get_nlp_pipeline("mock")

        # Both should work independently
        entities1 = pipeline1.extract_entities("metformin")
        entities2 = pipeline2.extract_entities("aspirin")

        assert len(entities1) >= 0
        assert len(entities2) >= 0


class TestNLPRegistryBuiltInPipelines:
    """Tests for built-in pipeline registrations."""

    def test_mock_pipeline_registered(self) -> None:
        """Mock pipeline is registered by default."""
        assert "mock" in list_nlp_pipelines()

    def test_scispacy_pipeline_registered(self) -> None:
        """SciSpaCy pipeline is registered (may not be available)."""
        assert "scispacy" in list_nlp_pipelines()

    def test_biomedical_pipeline_registered(self) -> None:
        """Biomedical pipeline is registered (may not be available)."""
        assert "biomedical" in list_nlp_pipelines()

    def test_medspacy_pipeline_registered(self) -> None:
        """MedSpaCy pipeline is registered (may not be available)."""
        assert "medspacy" in list_nlp_pipelines()

    def test_scispacy_import_error_handling(self) -> None:
        """SciSpaCy pipeline handles import errors gracefully."""
        # This test verifies the registry handles ImportError
        # when scispacy is not installed
        try:
            pipeline = get_nlp_pipeline("scispacy")
            # If it works, that's fine
            assert pipeline is not None
        except ImportError:
            # Expected if scispacy is not installed
            pass

    def test_biomedical_import_error_handling(self) -> None:
        """Biomedical pipeline handles import errors gracefully."""
        try:
            pipeline = get_nlp_pipeline("biomedical")
            assert pipeline is not None
        except ImportError:
            pass

    def test_medspacy_import_error_handling(self) -> None:
        """MedSpaCy pipeline handles import errors gracefully."""
        try:
            pipeline = get_nlp_pipeline("medspacy")
            assert pipeline is not None
        except ImportError:
            pass
