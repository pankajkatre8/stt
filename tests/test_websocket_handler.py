"""
Tests for WebSocket streaming handler.

Tests real-time audio streaming and transcription via WebSocket.
"""
from __future__ import annotations

import asyncio
import sys
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hsttb.adapters import MockSTTAdapter
from hsttb.core.types import AudioChunk, TranscriptSegment

# Import websocket_handler directly to avoid webapp.__init__.py -> app.py chain
# which requires python-multipart
_websocket_module: ModuleType | None = None
try:
    spec = importlib_util.spec_from_file_location(
        "websocket_handler",
        Path(__file__).parent.parent / "src" / "hsttb" / "webapp" / "websocket_handler.py"
    )
    if spec and spec.loader:
        _websocket_module = importlib_util.module_from_spec(spec)
        sys.modules["test_websocket_handler_module"] = _websocket_module
        spec.loader.exec_module(_websocket_module)
except Exception as e:
    _websocket_module = None

# Skip all tests if module not available
pytestmark = pytest.mark.skipif(
    _websocket_module is None,
    reason="websocket_handler module not available (requires python-multipart)"
)


class TestAudioStreamBuffer:
    """Tests for AudioStreamBuffer class."""

    def test_import_audio_stream_buffer(self) -> None:
        """AudioStreamBuffer can be imported."""
        assert _websocket_module is not None
        AudioStreamBuffer = _websocket_module.AudioStreamBuffer
        assert AudioStreamBuffer is not None

    def test_buffer_creation(self) -> None:
        """Create buffer with default settings."""
        assert _websocket_module is not None
        AudioStreamBuffer = _websocket_module.AudioStreamBuffer
        buffer = AudioStreamBuffer()
        assert buffer is not None

    def test_buffer_creation_with_params(self) -> None:
        """Create buffer with custom parameters."""
        assert _websocket_module is not None
        AudioStreamBuffer = _websocket_module.AudioStreamBuffer
        buffer = AudioStreamBuffer(
            chunk_size_ms=100,
            overlap_ms=25,
            sample_rate=16000,
        )
        assert buffer._chunk_size_ms == 100
        assert buffer._overlap_ms == 25

    def test_buffer_add_audio(self) -> None:
        """Add audio data to buffer."""
        assert _websocket_module is not None
        AudioStreamBuffer = _websocket_module.AudioStreamBuffer
        buffer = AudioStreamBuffer()
        buffer.add_audio(b"audio data")
        # Buffer should contain data
        assert len(buffer._buffer) > 0

    def test_buffer_clear(self) -> None:
        """Clear buffer removes all data."""
        assert _websocket_module is not None
        AudioStreamBuffer = _websocket_module.AudioStreamBuffer
        buffer = AudioStreamBuffer()
        buffer.add_audio(b"audio data")
        buffer.clear()
        assert len(buffer._buffer) == 0


class TestWebSocketHandler:
    """Tests for WebSocketHandler class."""

    def test_import_websocket_handler(self) -> None:
        """WebSocketHandler can be imported."""
        assert _websocket_module is not None
        WebSocketHandler = _websocket_module.WebSocketHandler
        assert WebSocketHandler is not None

    def test_handler_creation(self) -> None:
        """Create WebSocketHandler."""
        assert _websocket_module is not None
        WebSocketHandler = _websocket_module.WebSocketHandler
        handler = WebSocketHandler()
        assert handler is not None

    def test_handler_set_adapter(self) -> None:
        """Set STT adapter on handler."""
        assert _websocket_module is not None
        WebSocketHandler = _websocket_module.WebSocketHandler
        handler = WebSocketHandler()
        adapter = MockSTTAdapter()
        handler.set_adapter(adapter)
        assert handler._adapter is adapter

    def test_handler_get_adapter_none(self) -> None:
        """Get adapter returns None when not set."""
        assert _websocket_module is not None
        WebSocketHandler = _websocket_module.WebSocketHandler
        handler = WebSocketHandler()
        assert handler._adapter is None


class TestWebSocketHandlerTranscription:
    """Tests for transcription in WebSocketHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock adapter."""
        assert _websocket_module is not None
        WebSocketHandler = _websocket_module.WebSocketHandler
        handler = WebSocketHandler()
        handler.set_adapter(MockSTTAdapter(
            responses=["Hello world", "Test response"],
            delay_ms=0,
        ))
        return handler

    @pytest.mark.asyncio
    async def test_process_audio_chunk(self, handler) -> None:
        """Process single audio chunk."""
        # Initialize the adapter
        await handler._adapter.initialize()

        # Send a chunk
        result = await handler.process_audio_chunk(b"audio data", is_final=True)

        # Should receive transcript
        assert result is not None

    @pytest.mark.asyncio
    async def test_start_session(self, handler) -> None:
        """Start transcription session."""
        session_id = await handler.start_session()
        assert session_id is not None
        assert handler._adapter._initialized is True

    @pytest.mark.asyncio
    async def test_end_session(self, handler) -> None:
        """End transcription session."""
        await handler.start_session()
        await handler.end_session()
        # Adapter should be cleaned up
        assert handler._adapter._initialized is False


class TestWebSocketMessage:
    """Tests for WebSocket message handling."""

    def test_import_message_types(self) -> None:
        """Message types can be imported."""
        assert _websocket_module is not None
        WebSocketMessage = _websocket_module.WebSocketMessage
        MessageType = _websocket_module.MessageType
        assert WebSocketMessage is not None
        assert MessageType is not None

    def test_message_creation(self) -> None:
        """Create WebSocket message."""
        assert _websocket_module is not None
        WebSocketMessage = _websocket_module.WebSocketMessage
        MessageType = _websocket_module.MessageType
        message = WebSocketMessage(
            type=MessageType.TRANSCRIPT,
            data={"text": "Hello"},
        )
        assert message.type == MessageType.TRANSCRIPT
        assert message.data["text"] == "Hello"

    def test_message_to_dict(self) -> None:
        """Message to_dict for JSON serialization."""
        assert _websocket_module is not None
        WebSocketMessage = _websocket_module.WebSocketMessage
        MessageType = _websocket_module.MessageType
        message = WebSocketMessage(
            type=MessageType.TRANSCRIPT,
            data={"text": "Hello"},
        )
        result = message.to_dict()
        assert isinstance(result, dict)
        assert "type" in result
        assert "data" in result


class TestWebSocketHandlerErrorHandling:
    """Tests for error handling in WebSocketHandler."""

    def test_handler_no_adapter_error(self) -> None:
        """Handler raises error when no adapter set."""
        assert _websocket_module is not None
        WebSocketHandler = _websocket_module.WebSocketHandler
        handler = WebSocketHandler()

        with pytest.raises((ValueError, RuntimeError, AttributeError)):
            asyncio.run(handler.process_audio_chunk(b"data", is_final=True))

    @pytest.mark.asyncio
    async def test_handler_adapter_error_recovery(self) -> None:
        """Handler recovers from adapter errors."""
        assert _websocket_module is not None
        WebSocketHandler = _websocket_module.WebSocketHandler
        from hsttb.adapters import FailingMockAdapter

        handler = WebSocketHandler()
        handler.set_adapter(FailingMockAdapter(fail_on_calls={0}))

        # Should handle error gracefully
        await handler.start_session()
        try:
            await handler.process_audio_chunk(b"data", is_final=True)
        except Exception:
            # Error is expected
            pass


class TestGetWebSocketHandler:
    """Tests for global WebSocket handler."""

    def test_get_websocket_handler(self) -> None:
        """get_websocket_handler returns handler instance."""
        assert _websocket_module is not None
        get_websocket_handler = _websocket_module.get_websocket_handler
        handler = get_websocket_handler()
        assert handler is not None

    def test_get_websocket_handler_singleton(self) -> None:
        """get_websocket_handler returns same instance."""
        assert _websocket_module is not None
        get_websocket_handler = _websocket_module.get_websocket_handler
        handler1 = get_websocket_handler()
        handler2 = get_websocket_handler()
        assert handler1 is handler2
