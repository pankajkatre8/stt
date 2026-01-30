"""
Evaluation orchestration for HSTTB.

This module provides the benchmark runner and evaluation orchestration
for running comprehensive STT evaluations.

Example:
    >>> from hsttb.evaluation import BenchmarkRunner
    >>> from hsttb.adapters import MockSTTAdapter
    >>> runner = BenchmarkRunner(MockSTTAdapter())
    >>> results = await runner.evaluate(audio_dir, ground_truth_dir)
"""
from __future__ import annotations

from hsttb.evaluation.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    EvaluationResult,
    create_benchmark_runner,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkRunner",
    "EvaluationResult",
    "create_benchmark_runner",
]
