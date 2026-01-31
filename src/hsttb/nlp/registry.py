"""
NLP pipeline registry and factory functions.

This module provides a registry of available NLP pipelines and
factory functions to instantiate them by name.

Example:
    >>> from hsttb.nlp.registry import get_nlp_pipeline, list_nlp_pipelines
    >>> print(list_nlp_pipelines())
    ['mock', 'scispacy', 'biomedical', 'medspacy']
    >>> pipeline = get_nlp_pipeline("scispacy")
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hsttb.nlp.ner_pipeline import NERPipeline

logger = logging.getLogger(__name__)

# Registry of available NLP pipelines
_NLP_PIPELINE_REGISTRY: dict[str, type[NERPipeline]] = {}

# Registry of pipeline factory functions (for lazy loading)
_NLP_PIPELINE_FACTORIES: dict[str, Callable[..., NERPipeline]] = {}


def register_nlp_pipeline(
    name: str,
) -> Callable[[type[NERPipeline]], type[NERPipeline]]:
    """
    Decorator to register an NLP pipeline class.

    Args:
        name: The name to register the pipeline under.

    Returns:
        Decorator function.

    Example:
        >>> @register_nlp_pipeline("my_ner")
        ... class MyNERPipeline(NERPipeline):
        ...     pass
    """

    def decorator(cls: type[NERPipeline]) -> type[NERPipeline]:
        _NLP_PIPELINE_REGISTRY[name] = cls
        return cls

    return decorator


def register_nlp_pipeline_factory(
    name: str,
    factory: Callable[..., NERPipeline],
) -> None:
    """
    Register a factory function for lazy pipeline instantiation.

    Useful for pipelines with optional dependencies that should
    only be loaded when requested.

    Args:
        name: The name to register the factory under.
        factory: Callable that creates the pipeline instance.

    Example:
        >>> def create_scispacy() -> NERPipeline:
        ...     from hsttb.nlp.scispacy_ner import SciSpacyNERPipeline
        ...     return SciSpacyNERPipeline()
        >>> register_nlp_pipeline_factory("scispacy", create_scispacy)
    """
    _NLP_PIPELINE_FACTORIES[name] = factory


def get_nlp_pipeline(name: str, **kwargs: Any) -> NERPipeline:
    """
    Get an NLP pipeline instance by name.

    Tries the class registry first, then factory functions.
    Supports lazy loading for pipelines with optional dependencies.

    Args:
        name: The registered pipeline name.
        **kwargs: Arguments to pass to the pipeline constructor.

    Returns:
        An instance of the requested pipeline.

    Raises:
        ValueError: If pipeline name is not registered.
        ImportError: If required dependencies are not installed.

    Example:
        >>> pipeline = get_nlp_pipeline("scispacy")
        >>> entities = pipeline.extract_entities("metformin for diabetes")
    """
    # Try class registry first
    if name in _NLP_PIPELINE_REGISTRY:
        return _NLP_PIPELINE_REGISTRY[name](**kwargs)

    # Try factory functions
    if name in _NLP_PIPELINE_FACTORIES:
        try:
            return _NLP_PIPELINE_FACTORIES[name](**kwargs)
        except ImportError as e:
            logger.warning(f"Failed to load pipeline {name!r}: {e}")
            raise

    # List available pipelines
    available = list_nlp_pipelines()
    available_str = ", ".join(available) or "none"
    raise ValueError(f"Unknown NLP pipeline: {name!r}. Available: {available_str}")


def list_nlp_pipelines() -> list[str]:
    """
    List all registered NLP pipeline names.

    Returns:
        List of registered pipeline names.
    """
    # Combine both registries
    names = set(_NLP_PIPELINE_REGISTRY.keys()) | set(_NLP_PIPELINE_FACTORIES.keys())
    return sorted(names)


def is_nlp_pipeline_registered(name: str) -> bool:
    """
    Check if an NLP pipeline is registered.

    Args:
        name: The pipeline name to check.

    Returns:
        True if registered, False otherwise.
    """
    return name in _NLP_PIPELINE_REGISTRY or name in _NLP_PIPELINE_FACTORIES


def unregister_nlp_pipeline(name: str) -> bool:
    """
    Unregister an NLP pipeline.

    Args:
        name: The pipeline name to unregister.

    Returns:
        True if pipeline was unregistered, False if not found.
    """
    removed = False
    if name in _NLP_PIPELINE_REGISTRY:
        del _NLP_PIPELINE_REGISTRY[name]
        removed = True
    if name in _NLP_PIPELINE_FACTORIES:
        del _NLP_PIPELINE_FACTORIES[name]
        removed = True
    return removed


def clear_nlp_registry() -> None:
    """Clear all registered pipelines. Mainly for testing."""
    _NLP_PIPELINE_REGISTRY.clear()
    _NLP_PIPELINE_FACTORIES.clear()


def get_pipeline_info(name: str) -> dict[str, Any]:
    """
    Get information about a registered pipeline.

    Args:
        name: The pipeline name.

    Returns:
        Dictionary with pipeline metadata.

    Raises:
        ValueError: If pipeline is not registered.
    """
    if name in _NLP_PIPELINE_REGISTRY:
        cls = _NLP_PIPELINE_REGISTRY[name]
        return {
            "name": name,
            "class": cls.__name__,
            "module": cls.__module__,
            "docstring": cls.__doc__ or "",
            "type": "class",
        }

    if name in _NLP_PIPELINE_FACTORIES:
        factory = _NLP_PIPELINE_FACTORIES[name]
        return {
            "name": name,
            "factory": factory.__name__,
            "module": factory.__module__,
            "docstring": factory.__doc__ or "",
            "type": "factory",
        }

    raise ValueError(f"Unknown NLP pipeline: {name!r}")


# ============================================================================
# Register built-in pipelines via factories (lazy loading)
# ============================================================================


def _create_mock_pipeline(**kwargs: Any) -> NERPipeline:
    """Create a mock NER pipeline with common patterns."""
    from hsttb.nlp.ner_pipeline import MockNERPipeline

    return MockNERPipeline.with_common_patterns()


def _create_scispacy_pipeline(**kwargs: Any) -> NERPipeline:
    """Create a scispaCy NER pipeline."""
    from hsttb.nlp.scispacy_ner import SciSpacyNERPipeline

    return SciSpacyNERPipeline()


def _create_biomedical_pipeline(**kwargs: Any) -> NERPipeline:
    """Create a HuggingFace biomedical NER pipeline."""
    from hsttb.nlp.biomedical_ner import BiomedicalNERPipeline

    return BiomedicalNERPipeline()


def _create_medspacy_pipeline(**kwargs: Any) -> NERPipeline:
    """Create a MedSpaCy NER pipeline with clinical NLP features."""
    from hsttb.nlp.medspacy_ner import MedSpacyNERPipeline

    return MedSpacyNERPipeline()


# Register built-in pipelines
register_nlp_pipeline_factory("mock", _create_mock_pipeline)
register_nlp_pipeline_factory("scispacy", _create_scispacy_pipeline)
register_nlp_pipeline_factory("biomedical", _create_biomedical_pipeline)
register_nlp_pipeline_factory("medspacy", _create_medspacy_pipeline)
