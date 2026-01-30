"""
Shared pytest fixtures for HSTTB tests.

This module provides common fixtures used across unit and integration tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    pass


# ==============================================================================
# Path Fixtures
# ==============================================================================


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


# ==============================================================================
# Audio Fixtures
# ==============================================================================


@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate for tests."""
    return 16000


@pytest.fixture
def audio_duration_s() -> float:
    """Standard audio duration for tests."""
    return 5.0


@pytest.fixture
def create_audio_data(sample_rate: int):
    """Factory fixture for creating test audio data."""

    def _create(duration_s: float = 5.0, noise_level: float = 0.01) -> np.ndarray:
        """
        Create test audio data.

        Args:
            duration_s: Duration in seconds.
            noise_level: Amplitude of noise (0.0 to 1.0).

        Returns:
            Audio samples as float32 numpy array.
        """
        num_samples = int(duration_s * sample_rate)
        # Generate low-level noise simulating silence
        audio = np.random.randn(num_samples).astype(np.float32) * noise_level
        return audio

    return _create


@pytest.fixture
def sample_audio_data(create_audio_data, audio_duration_s: float) -> np.ndarray:
    """Sample audio data for tests."""
    return create_audio_data(duration_s=audio_duration_s)


# ==============================================================================
# Transcript Fixtures
# ==============================================================================


@pytest.fixture
def sample_transcript() -> str:
    """Sample clinical transcript."""
    return "Patient takes metformin 500mg twice daily for diabetes."


@pytest.fixture
def sample_transcript_with_negation() -> str:
    """Sample transcript with negation."""
    return "Patient denies chest pain. No shortness of breath reported."


@pytest.fixture
def sample_transcript_pair() -> tuple[str, str]:
    """Ground truth and prediction pair (identical)."""
    text = "Patient takes metformin 500mg twice daily for diabetes."
    return (text, text)


@pytest.fixture
def sample_transcript_pair_with_error() -> tuple[str, str]:
    """Ground truth and prediction pair with drug substitution error."""
    gt = "Patient takes metformin 500mg twice daily for diabetes."
    pred = "Patient takes methotrexate 500mg twice daily for diabetes."
    return (gt, pred)


# ==============================================================================
# Medical Entity Fixtures
# ==============================================================================


@pytest.fixture
def common_drug_names() -> list[str]:
    """Common drug names for testing."""
    return [
        "metformin",
        "lisinopril",
        "atorvastatin",
        "amlodipine",
        "metoprolol",
        "omeprazole",
        "simvastatin",
        "losartan",
        "gabapentin",
        "aspirin",
    ]


@pytest.fixture
def confusable_drug_pairs() -> list[tuple[str, str]]:
    """Drug pairs that are commonly confused."""
    return [
        ("metformin", "methotrexate"),
        ("hydroxyzine", "hydralazine"),
        ("clonidine", "clonazepam"),
        ("tramadol", "trazodone"),
        ("bupropion", "buspirone"),
        ("prednisone", "prednisolone"),
    ]


# ==============================================================================
# Temporary Directory Fixtures
# ==============================================================================


@pytest.fixture
def temp_audio_dir(tmp_path: Path) -> Path:
    """Temporary directory for audio files."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    return audio_dir


@pytest.fixture
def temp_ground_truth_dir(tmp_path: Path) -> Path:
    """Temporary directory for ground truth files."""
    gt_dir = tmp_path / "ground_truth"
    gt_dir.mkdir()
    return gt_dir


# ==============================================================================
# Async Helpers
# ==============================================================================


@pytest.fixture
def event_loop_policy():
    """Event loop policy for async tests."""
    import asyncio

    return asyncio.DefaultEventLoopPolicy()
