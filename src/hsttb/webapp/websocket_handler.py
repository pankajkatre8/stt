"""
WebSocket handler for real-time audio streaming.

Handles WebSocket connections for live audio transcription,
buffering audio chunks, streaming to STT adapters, and
returning partial and final transcripts.

Example:
    >>> # In FastAPI app
    >>> @app.websocket("/ws/transcribe")
    >>> async def websocket_transcribe(websocket: WebSocket):
    ...     handler = WebSocketHandler(websocket)
    ...     await handler.run()
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import WebSocket

    from hsttb.adapters.base import STTAdapter

from hsttb.core.types import AudioChunk, TranscriptSegment

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    AUDIO_CHUNK = "audio_chunk"
    START = "start"
    STOP = "stop"
    CONFIG = "config"

    # Server -> Client
    TRANSCRIPT = "transcript"
    PARTIAL = "partial"
    FINAL = "final"
    ERROR = "error"
    STATUS = "status"


@dataclass
class StreamConfig:
    """
    Configuration for audio streaming session.

    Attributes:
        adapter_name: STT adapter to use.
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        encoding: Audio encoding (pcm16, float32, etc.).
        language: Optional language code.
        model: Optional model variant.
    """

    adapter_name: str = "whisper"
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "pcm16"
    language: str | None = None
    model: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamConfig:
        """Create config from dictionary."""
        return cls(
            adapter_name=data.get("adapter", "whisper"),
            sample_rate=data.get("sample_rate", 16000),
            channels=data.get("channels", 1),
            encoding=data.get("encoding", "pcm16"),
            language=data.get("language"),
            model=data.get("model"),
        )


@dataclass
class StreamSession:
    """
    State for an active streaming session.

    Attributes:
        session_id: Unique session identifier.
        config: Stream configuration.
        started_at: Session start time.
        chunks_received: Number of audio chunks received.
        bytes_received: Total bytes received.
        transcripts: List of transcript segments.
    """

    session_id: str
    config: StreamConfig
    started_at: datetime = field(default_factory=datetime.now)
    chunks_received: int = 0
    bytes_received: int = 0
    transcripts: list[TranscriptSegment] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Session duration in seconds."""
        return (datetime.now() - self.started_at).total_seconds()


class WebSocketHandler:
    """
    Handler for WebSocket-based audio streaming.

    Manages the lifecycle of a streaming transcription session:
    1. Receives audio chunks from client
    2. Buffers and validates audio data
    3. Streams to STT adapter
    4. Returns partial and final transcripts

    Attributes:
        websocket: FastAPI WebSocket connection.
        adapter: STT adapter for transcription.
        session: Current streaming session.

    Example:
        >>> async with websocket_handler as handler:
        ...     await handler.run()
    """

    def __init__(
        self,
        websocket: WebSocket,
        adapter: STTAdapter | None = None,
    ) -> None:
        """
        Initialize the WebSocket handler.

        Args:
            websocket: FastAPI WebSocket connection.
            adapter: Optional pre-initialized STT adapter.
        """
        self._websocket = websocket
        self._adapter = adapter
        self._session: StreamSession | None = None
        self._audio_queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        self._running = False

    async def run(self) -> None:
        """
        Run the WebSocket handler.

        Main loop that receives messages and processes them.
        """
        await self._websocket.accept()
        logger.info("WebSocket connection accepted")

        try:
            self._running = True

            # Start transcription task
            transcription_task = asyncio.create_task(self._transcription_loop())

            # Message receiving loop
            try:
                async for message in self._receive_messages():
                    await self._handle_message(message)
            except Exception as e:
                logger.error(f"Error in message loop: {e}")
                await self._send_error(str(e))

            # Signal end of audio
            await self._audio_queue.put(None)

            # Wait for transcription to complete
            await transcription_task

        finally:
            self._running = False
            await self._cleanup()
            logger.info("WebSocket connection closed")

    async def _receive_messages(self) -> AsyncIterator[dict[str, Any]]:
        """
        Receive and parse WebSocket messages.

        Yields:
            Parsed JSON messages.
        """
        while self._running:
            try:
                data = await self._websocket.receive()

                if "text" in data:
                    yield json.loads(data["text"])
                elif "bytes" in data:
                    # Binary data is audio chunk
                    yield {
                        "type": MessageType.AUDIO_CHUNK,
                        "data": data["bytes"],
                    }

            except Exception as e:
                if "disconnect" in str(e).lower():
                    break
                raise

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """
        Handle an incoming WebSocket message.

        Args:
            message: Parsed message dictionary.
        """
        msg_type = message.get("type", "")

        if msg_type == MessageType.START:
            await self._handle_start(message)

        elif msg_type == MessageType.STOP:
            await self._handle_stop()

        elif msg_type == MessageType.CONFIG:
            await self._handle_config(message)

        elif msg_type == MessageType.AUDIO_CHUNK:
            await self._handle_audio_chunk(message)

        else:
            logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_start(self, message: dict[str, Any]) -> None:
        """Handle start message."""
        import uuid

        config_data = message.get("config", {})
        config = StreamConfig.from_dict(config_data)

        self._session = StreamSession(
            session_id=str(uuid.uuid4()),
            config=config,
        )

        # Initialize adapter if not provided
        if self._adapter is None:
            await self._initialize_adapter(config)

        await self._send_status("started", {
            "session_id": self._session.session_id,
            "adapter": config.adapter_name,
        })

        logger.info(f"Started session: {self._session.session_id}")

    async def _handle_stop(self) -> None:
        """Handle stop message."""
        if self._session:
            # Signal end of audio stream
            await self._audio_queue.put(None)

            await self._send_status("stopped", {
                "session_id": self._session.session_id,
                "duration": self._session.duration_seconds,
                "chunks": self._session.chunks_received,
            })

            logger.info(f"Stopped session: {self._session.session_id}")

        self._running = False

    async def _handle_config(self, message: dict[str, Any]) -> None:
        """Handle config update message."""
        config_data = message.get("config", {})

        if self._session:
            # Update config
            new_config = StreamConfig.from_dict(config_data)
            self._session.config = new_config

            await self._send_status("config_updated", config_data)

    async def _handle_audio_chunk(self, message: dict[str, Any]) -> None:
        """Handle incoming audio chunk."""
        if not self._session:
            await self._send_error("Session not started")
            return

        # Get audio data
        data = message.get("data")
        if data is None:
            return

        # Handle base64 encoded data
        if isinstance(data, str):
            data = base64.b64decode(data)

        # Create AudioChunk
        chunk = AudioChunk(
            data=data,
            sequence_id=self._session.chunks_received,
            timestamp_ms=int(self._session.duration_seconds * 1000),
            duration_ms=self._calculate_chunk_duration(len(data)),
            is_final=message.get("is_final", False),
        )

        # Update session stats
        self._session.chunks_received += 1
        self._session.bytes_received += len(data)

        # Queue for transcription
        await self._audio_queue.put(chunk)

    def _calculate_chunk_duration(self, byte_length: int) -> int:
        """Calculate chunk duration in milliseconds."""
        if not self._session:
            return 0

        config = self._session.config
        bytes_per_sample = 2 if config.encoding == "pcm16" else 4
        samples = byte_length // (bytes_per_sample * config.channels)
        return int((samples / config.sample_rate) * 1000)

    async def _initialize_adapter(self, config: StreamConfig) -> None:
        """Initialize the STT adapter."""
        from hsttb.adapters import get_adapter

        try:
            kwargs: dict[str, Any] = {}

            if config.model:
                kwargs["model_size"] = config.model

            if config.language:
                kwargs["language"] = config.language

            self._adapter = get_adapter(config.adapter_name, **kwargs)
            await self._adapter.initialize()

            logger.info(f"Initialized adapter: {config.adapter_name}")

        except Exception as e:
            await self._send_error(f"Failed to initialize adapter: {e}")
            raise

    async def _transcription_loop(self) -> None:
        """
        Main transcription loop.

        Reads audio chunks from queue and streams to adapter.
        """
        if not self._adapter:
            return

        async def audio_generator() -> AsyncIterator[AudioChunk]:
            """Generate audio chunks from queue."""
            while True:
                chunk = await self._audio_queue.get()
                if chunk is None:
                    break
                yield chunk

        try:
            async for segment in self._adapter.transcribe_stream(audio_generator()):
                await self._send_transcript(segment)

                if self._session:
                    self._session.transcripts.append(segment)

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            await self._send_error(f"Transcription failed: {e}")

    async def _send_transcript(self, segment: TranscriptSegment) -> None:
        """Send transcript segment to client."""
        msg_type = MessageType.FINAL if segment.is_final else MessageType.PARTIAL

        await self._websocket.send_json({
            "type": msg_type,
            "text": segment.text,
            "is_partial": segment.is_partial,
            "is_final": segment.is_final,
            "confidence": segment.confidence,
            "start_ms": segment.start_time_ms,
            "end_ms": segment.end_time_ms,
        })

    async def _send_status(self, status: str, data: dict[str, Any]) -> None:
        """Send status message to client."""
        await self._websocket.send_json({
            "type": MessageType.STATUS,
            "status": status,
            **data,
        })

    async def _send_error(self, error: str) -> None:
        """Send error message to client."""
        await self._websocket.send_json({
            "type": MessageType.ERROR,
            "error": error,
        })

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._adapter:
            await self._adapter.cleanup()
            self._adapter = None

    async def __aenter__(self) -> WebSocketHandler:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self._cleanup()


class AudioStreamBuffer:
    """
    Buffer for accumulating audio data.

    Handles chunking and buffering of audio data before
    sending to STT adapter.

    Attributes:
        sample_rate: Audio sample rate.
        buffer_size: Target buffer size in samples.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_duration_ms: int = 1000,
    ) -> None:
        """
        Initialize the audio buffer.

        Args:
            sample_rate: Audio sample rate in Hz.
            buffer_duration_ms: Target buffer duration in ms.
        """
        self._sample_rate = sample_rate
        self._buffer_duration_ms = buffer_duration_ms
        self._buffer = bytearray()
        self._sequence_id = 0
        self._timestamp_ms = 0

    @property
    def buffer_size_bytes(self) -> int:
        """Target buffer size in bytes (16-bit mono)."""
        samples = int(self._sample_rate * self._buffer_duration_ms / 1000)
        return samples * 2  # 16-bit = 2 bytes

    def add(self, data: bytes) -> list[AudioChunk]:
        """
        Add data to buffer and return complete chunks.

        Args:
            data: Raw audio bytes.

        Returns:
            List of complete AudioChunk objects.
        """
        self._buffer.extend(data)
        chunks = []

        while len(self._buffer) >= self.buffer_size_bytes:
            chunk_data = bytes(self._buffer[: self.buffer_size_bytes])
            self._buffer = self._buffer[self.buffer_size_bytes :]

            chunks.append(AudioChunk(
                data=chunk_data,
                sequence_id=self._sequence_id,
                timestamp_ms=self._timestamp_ms,
                duration_ms=self._buffer_duration_ms,
                is_final=False,
            ))

            self._sequence_id += 1
            self._timestamp_ms += self._buffer_duration_ms

        return chunks

    def flush(self) -> AudioChunk | None:
        """
        Flush remaining buffer as final chunk.

        Returns:
            Final AudioChunk or None if buffer is empty.
        """
        if not self._buffer:
            return None

        # Calculate actual duration
        samples = len(self._buffer) // 2
        duration_ms = int(samples / self._sample_rate * 1000)

        if duration_ms == 0:
            return None

        chunk = AudioChunk(
            data=bytes(self._buffer),
            sequence_id=self._sequence_id,
            timestamp_ms=self._timestamp_ms,
            duration_ms=duration_ms,
            is_final=True,
        )

        self._buffer.clear()
        return chunk

    def reset(self) -> None:
        """Reset buffer state."""
        self._buffer.clear()
        self._sequence_id = 0
        self._timestamp_ms = 0
