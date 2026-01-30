"""
Web application for HSTTB evaluation.

A simple web interface for running healthcare STT benchmarks.
"""
from __future__ import annotations

from hsttb.webapp.app import app, create_app

__all__ = ["app", "create_app"]
