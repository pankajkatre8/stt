"""
STT adapters for model-agnostic evaluation.

This module provides the adapter interface and implementations
for various STT providers.

Example:
    >>> from hsttb.adapters import get_adapter, MockSTTAdapter
    >>> # Using factory
    >>> adapter = get_adapter("mock", responses=["Hello"])
    >>> # Or direct instantiation
    >>> adapter = MockSTTAdapter(responses=["Hello"])
"""
from __future__ import annotations

from hsttb.adapters.base import STTAdapter
from hsttb.adapters.mock_adapter import FailingMockAdapter, MockSTTAdapter
from hsttb.adapters.registry import (
    clear_registry,
    get_adapter,
    is_adapter_registered,
    list_adapters,
    register_adapter,
    unregister_adapter,
)

__all__ = [
    # Base class
    "STTAdapter",
    # Registry functions
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "is_adapter_registered",
    "unregister_adapter",
    "clear_registry",
    # Concrete adapters
    "MockSTTAdapter",
    "FailingMockAdapter",
]
