"""
Custom exception hierarchy for HSTTB.

This module defines the exception classes used throughout the framework.
All HSTTB-specific exceptions inherit from HSSTBError for easy catching.

Exception Hierarchy:
    HSSTBError (base)
    ├── ConfigurationError
    ├── AudioError
    │   ├── AudioLoadError
    │   └── AudioFormatError
    ├── STTAdapterError
    │   ├── STTConnectionError
    │   └── STTTranscriptionError
    ├── LexiconError
    │   ├── LexiconLoadError
    │   └── LexiconLookupError
    ├── MetricComputationError
    │   ├── TERComputationError
    │   ├── NERComputationError
    │   └── CRSComputationError
    ├── EvaluationError
    │   ├── BenchmarkError
    │   └── ReportGenerationError
    └── StellicareError
        ├── StellicareConnectionError
        ├── StellicareTranscriptionError
        └── StellicareRefineError
"""

from __future__ import annotations


class HSSTBError(Exception):
    """
    Base exception for all HSTTB errors.

    All custom exceptions in the HSTTB framework inherit from this class,
    making it easy to catch all framework-specific errors.

    Example:
        >>> try:
        ...     run_benchmark()
        ... except HSSTBError as e:
        ...     logger.error(f"HSTTB error: {e}")
    """

    def __init__(self, message: str, *args: object) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message.
            *args: Additional arguments passed to Exception.
        """
        self.message = message
        super().__init__(message, *args)


# ==============================================================================
# Configuration Errors
# ==============================================================================


class ConfigurationError(HSSTBError):
    """
    Error in configuration loading or validation.

    Raised when configuration files cannot be loaded, parsed,
    or contain invalid values.

    Example:
        >>> raise ConfigurationError("Invalid chunk_size_ms: must be positive")
    """

    pass


# ==============================================================================
# Audio Errors
# ==============================================================================


class AudioError(HSSTBError):
    """
    Base class for audio-related errors.

    Raised for any errors related to audio loading, processing,
    or streaming.
    """

    pass


class AudioLoadError(AudioError):
    """
    Error loading an audio file.

    Raised when an audio file cannot be read from disk.

    Attributes:
        file_path: Path to the audio file that failed to load.
    """

    def __init__(self, message: str, file_path: str | None = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message.
            file_path: Path to the failed audio file.
        """
        self.file_path = file_path
        super().__init__(message)


class AudioFormatError(AudioError):
    """
    Error with audio format or encoding.

    Raised when audio data is in an unsupported format
    or cannot be decoded.

    Attributes:
        expected_format: Expected audio format.
        actual_format: Actual audio format encountered.
    """

    def __init__(
        self,
        message: str,
        expected_format: str | None = None,
        actual_format: str | None = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message.
            expected_format: Expected audio format.
            actual_format: Actual audio format.
        """
        self.expected_format = expected_format
        self.actual_format = actual_format
        super().__init__(message)


# ==============================================================================
# STT Adapter Errors
# ==============================================================================


class STTAdapterError(HSSTBError):
    """
    Base class for STT adapter errors.

    Raised for any errors related to STT model adapters,
    including connection and transcription errors.

    Attributes:
        adapter_name: Name of the adapter that raised the error.
    """

    def __init__(self, message: str, adapter_name: str | None = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message.
            adapter_name: Name of the failing adapter.
        """
        self.adapter_name = adapter_name
        super().__init__(message)


class STTConnectionError(STTAdapterError):
    """
    Error connecting to STT service.

    Raised when the adapter cannot establish a connection
    to the STT service (network issues, authentication, etc.).
    """

    pass


class STTTranscriptionError(STTAdapterError):
    """
    Error during transcription.

    Raised when transcription fails after connection is established.
    May be due to invalid audio, service errors, or timeouts.
    """

    pass


# ==============================================================================
# Lexicon Errors
# ==============================================================================


class LexiconError(HSSTBError):
    """
    Base class for medical lexicon errors.

    Raised for any errors related to loading or querying
    medical lexicons (RxNorm, SNOMED CT, ICD-10).

    Attributes:
        lexicon_name: Name of the lexicon that raised the error.
    """

    def __init__(self, message: str, lexicon_name: str | None = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message.
            lexicon_name: Name of the failing lexicon.
        """
        self.lexicon_name = lexicon_name
        super().__init__(message)


class LexiconLoadError(LexiconError):
    """
    Error loading a lexicon.

    Raised when a lexicon file cannot be loaded from disk
    or parsed correctly.

    Attributes:
        file_path: Path to the lexicon file that failed to load.
    """

    def __init__(
        self,
        message: str,
        lexicon_name: str | None = None,
        file_path: str | None = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message.
            lexicon_name: Name of the failing lexicon.
            file_path: Path to the failed lexicon file.
        """
        self.file_path = file_path
        super().__init__(message, lexicon_name)


class LexiconLookupError(LexiconError):
    """
    Error during lexicon lookup.

    Raised when a term lookup fails unexpectedly.
    Note: Not finding a term is not an error; this is for
    unexpected failures during the lookup process.
    """

    pass


# ==============================================================================
# Metric Computation Errors
# ==============================================================================


class MetricComputationError(HSSTBError):
    """
    Base class for metric computation errors.

    Raised when any metric computation fails unexpectedly.

    Attributes:
        metric_name: Name of the metric that failed.
    """

    def __init__(self, message: str, metric_name: str | None = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message.
            metric_name: Name of the failing metric.
        """
        self.metric_name = metric_name
        super().__init__(message)


class TERComputationError(MetricComputationError):
    """
    Error computing Term Error Rate.

    Raised when TER computation fails, typically due to
    issues with term extraction or alignment.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, metric_name="TER")


class NERComputationError(MetricComputationError):
    """
    Error computing NER accuracy.

    Raised when NER computation fails, typically due to
    issues with entity extraction or matching.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, metric_name="NER")


class CRSComputationError(MetricComputationError):
    """
    Error computing Context Retention Score.

    Raised when CRS computation fails, typically due to
    issues with embedding computation or continuity tracking.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, metric_name="CRS")


# ==============================================================================
# Evaluation Errors
# ==============================================================================


class EvaluationError(HSSTBError):
    """
    Base class for evaluation and benchmark errors.

    Raised for errors during benchmark execution or
    report generation.
    """

    pass


class BenchmarkError(EvaluationError):
    """
    Error during benchmark execution.

    Raised when the benchmark runner encounters an error
    that prevents completion.

    Attributes:
        audio_id: ID of the audio file being processed when error occurred.
    """

    def __init__(self, message: str, audio_id: str | None = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message.
            audio_id: ID of the failing audio file.
        """
        self.audio_id = audio_id
        super().__init__(message)


class ReportGenerationError(EvaluationError):
    """
    Error generating reports.

    Raised when report generation fails, typically due to
    file I/O errors or invalid data.

    Attributes:
        report_type: Type of report that failed to generate.
    """

    def __init__(self, message: str, report_type: str | None = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message.
            report_type: Type of the failing report.
        """
        self.report_type = report_type
        super().__init__(message)


# ==============================================================================
# Stellicare API Errors
# ==============================================================================


class StellicareError(HSSTBError):
    """
    Base exception for Stellicare API errors.

    Raised for any errors related to communication with
    the Stellicare STT service (WebSocket streaming, refinement API).
    """

    pass


class StellicareConnectionError(StellicareError):
    """
    Error connecting to Stellicare WebSocket.

    Raised when the WebSocket connection to the Stellicare
    transcription service cannot be established.
    """

    pass


class StellicareTranscriptionError(StellicareError):
    """
    Error during Stellicare transcription streaming.

    Raised when audio streaming or transcript reception fails
    after the WebSocket connection is established.
    """

    pass


class StellicareRefineError(StellicareError):
    """
    Error refining transcript via Stellicare API.

    Raised when the transcript refinement REST API call fails.
    """

    pass
