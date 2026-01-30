"""
Reporting module for HSTTB.

This module provides report generation for benchmark results
in multiple formats.

Example:
    >>> from hsttb.reporting import ReportGenerator
    >>> generator = ReportGenerator(output_dir=Path("results"))
    >>> generator.generate_all(summary)
"""
from __future__ import annotations

from hsttb.reporting.generator import (
    ReportConfig,
    ReportGenerator,
    generate_report,
)

__all__ = [
    "ReportConfig",
    "ReportGenerator",
    "generate_report",
]
