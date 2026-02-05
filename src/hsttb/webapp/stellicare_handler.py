"""
WebSocket handler for Stellicare audio processing pipeline.

Manages the browser-to-backend WebSocket session and orchestrates
sequential audio streaming to the Stellicare WSS endpoint, relaying
progress and transcript results back to the browser.

Example:
    >>> # In FastAPI app
    >>> @app.websocket("/ws/stellicare")
    >>> async def websocket_stellicare(websocket: WebSocket):
    ...     handler = StellicareWebSocketHandler(websocket, get_config)
    ...     await handler.run()
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from hsttb.webapp.stellicare_client import (
    StellicareConfig,
    process_files_sequentially,
)

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = logging.getLogger(__name__)


class StellicareWebSocketHandler:
    """
    Handler for Stellicare pipeline WebSocket sessions.

    Accepts a browser WebSocket connection, receives file IDs,
    processes them sequentially through the Stellicare WSS,
    and relays progress/transcript messages back to the browser.

    Attributes:
        websocket: The FastAPI WebSocket connection.
        config_factory: Callable that returns StellicareConfig.
    """

    def __init__(
        self,
        websocket: WebSocket,
        config_factory: Callable[[], StellicareConfig],
        file_resolver: Callable[[str], Path | None] | None = None,
    ) -> None:
        """
        Initialize the handler.

        Args:
            websocket: FastAPI WebSocket connection from the browser.
            config_factory: Factory function to get StellicareConfig.
            file_resolver: Function to resolve file_id to Path.
                If None, uses AudioHandler.get_file().
        """
        self.websocket = websocket
        self.config_factory = config_factory
        self.file_resolver = file_resolver

    async def run(self) -> None:
        """
        Main handler loop.

        Accepts the WebSocket, waits for a start message with file IDs,
        processes files sequentially, and sends progress updates.
        """
        await self.websocket.accept()
        logger.info("Stellicare WebSocket client connected")

        try:
            await self._send_status("connected", "Ready to process files")

            # Wait for start message
            message = await self.websocket.receive_json()
            msg_type = message.get("type", "")

            if msg_type != "start":
                await self._send_error(
                    f"Expected 'start' message, got '{msg_type}'"
                )
                return

            file_ids = message.get("file_ids", [])
            if not file_ids:
                await self._send_error("No file IDs provided")
                return

            # Resolve file paths
            file_paths = await self._resolve_files(file_ids)
            if not file_paths:
                await self._send_error("No valid audio files found")
                return

            # Get config
            config = self.config_factory()

            # Process files sequentially
            async def progress_callback(msg: dict[str, Any]) -> None:
                try:
                    await self.websocket.send_json(msg)
                except Exception as e:
                    logger.warning(f"Failed to send progress: {e}")

            result = await process_files_sequentially(
                file_paths=file_paths,
                config=config,
                progress_callback=progress_callback,
            )

            logger.info(
                f"Stellicare processing complete: {result.files_processed}/"
                f"{len(file_paths)} files, "
                f"{len(result.raw_transcript)} chars transcript"
            )

        except Exception as e:
            logger.error(f"Stellicare WebSocket error: {e}")
            try:
                await self._send_error(str(e))
            except Exception:
                pass

    async def _resolve_files(self, file_ids: list[str]) -> list[Path]:
        """
        Resolve file IDs to filesystem paths.

        Args:
            file_ids: List of audio file IDs from the upload system.

        Returns:
            List of resolved Path objects for valid files.
        """
        paths: list[Path] = []

        for file_id in file_ids:
            path = self._get_file_path(file_id)
            if path and path.exists():
                paths.append(path)
                logger.debug(f"Resolved file {file_id} -> {path}")
            else:
                logger.warning(f"File not found for ID: {file_id}")
                try:
                    await self.websocket.send_json({
                        "type": "error",
                        "error": f"File not found: {file_id}",
                        "file_id": file_id,
                    })
                except Exception:
                    pass

        return paths

    def _get_file_path(self, file_id: str) -> Path | None:
        """
        Get the filesystem path for a file ID.

        Args:
            file_id: The upload file ID.

        Returns:
            Path to the file, or None if not found.
        """
        if self.file_resolver:
            return self.file_resolver(file_id)

        # Default: use AudioHandler
        try:
            from hsttb.webapp.audio_handler import get_audio_handler

            handler = get_audio_handler()
            return handler.get_file(file_id)
        except Exception as e:
            logger.error(f"Error resolving file {file_id}: {e}")
            return None

    async def _send_status(self, status: str, message: str) -> None:
        """Send a status message to the browser."""
        await self.websocket.send_json({
            "type": "status",
            "status": status,
            "message": message,
        })

    async def _send_error(self, error: str, **extra: Any) -> None:
        """Send an error message to the browser."""
        msg: dict[str, Any] = {
            "type": "error",
            "error": error,
        }
        msg.update(extra)
        await self.websocket.send_json(msg)
