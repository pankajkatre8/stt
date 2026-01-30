"""
HSTTB - Healthcare Streaming STT Benchmarking Framework.

A model-agnostic evaluation framework for healthcare speech-to-text systems.

This framework provides:
- Medical term accuracy measurement (TER)
- Medical entity integrity evaluation (NER Accuracy)
- Streaming context continuity scoring (CRS)
- Model-agnostic benchmarking via adapters
- Reproducible streaming simulation

Example:
    >>> from hsttb import BenchmarkRunner, get_adapter
    >>> adapter = get_adapter("whisper", model_size="base")
    >>> runner = BenchmarkRunner(adapter, profile="ideal")
    >>> results = await runner.evaluate(audio_dir, ground_truth_dir)
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "HSTTB Team"

__all__ = [
    "__version__",
    "__author__",
]
