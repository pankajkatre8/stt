"""
Adapter registry and factory functions.

This module provides a registry of available STT adapters and
factory functions to instantiate them by name.

Example:
    >>> from hsttb.adapters import get_adapter, list_adapters
    >>> print(list_adapters())
    ['mock', 'whisper']
    >>> adapter = get_adapter("mock", responses=["test"])
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hsttb.adapters.base import STTAdapter

# Registry of available adapters
_ADAPTER_REGISTRY: dict[str, type[STTAdapter]] = {}


def register_adapter(name: str) -> Callable[[type[STTAdapter]], type[STTAdapter]]:
    """
    Decorator to register an adapter class.

    Args:
        name: The name to register the adapter under.

    Returns:
        Decorator function.

    Example:
        >>> @register_adapter("my_stt")
        ... class MySTTAdapter(STTAdapter):
        ...     pass
    """

    def decorator(cls: type[STTAdapter]) -> type[STTAdapter]:
        _ADAPTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_adapter(name: str, **kwargs: object) -> STTAdapter:
    """
    Get an adapter instance by name.

    Args:
        name: The registered adapter name.
        **kwargs: Arguments to pass to the adapter constructor.

    Returns:
        An instance of the requested adapter.

    Raises:
        ValueError: If adapter name is not registered.

    Example:
        >>> adapter = get_adapter("whisper", model_size="base")
        >>> await adapter.initialize()
    """
    if name not in _ADAPTER_REGISTRY:
        available = ", ".join(_ADAPTER_REGISTRY.keys()) or "none"
        raise ValueError(f"Unknown adapter: {name!r}. Available: {available}")
    return _ADAPTER_REGISTRY[name](**kwargs)


def list_adapters() -> list[str]:
    """
    List all registered adapter names.

    Returns:
        List of registered adapter names.
    """
    return list(_ADAPTER_REGISTRY.keys())


def is_adapter_registered(name: str) -> bool:
    """
    Check if an adapter is registered.

    Args:
        name: The adapter name to check.

    Returns:
        True if registered, False otherwise.
    """
    return name in _ADAPTER_REGISTRY


def unregister_adapter(name: str) -> bool:
    """
    Unregister an adapter.

    Args:
        name: The adapter name to unregister.

    Returns:
        True if adapter was unregistered, False if not found.
    """
    if name in _ADAPTER_REGISTRY:
        del _ADAPTER_REGISTRY[name]
        return True
    return False


def clear_registry() -> None:
    """Clear all registered adapters. Mainly for testing."""
    _ADAPTER_REGISTRY.clear()
