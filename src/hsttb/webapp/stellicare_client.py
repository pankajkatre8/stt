"""
Stellicare API client for healthcare STT transcription.

Handles WebSocket streaming to the Stellicare WSS endpoint and
transcript refinement via the Stellicare REST API. This module
acts as a backend proxy, keeping API URLs and tokens server-side.

Example:
    >>> from hsttb.webapp.stellicare_client import (
    ...     StellicareConfig,
    ...     process_files_sequentially,
    ...     refine_transcript,
    ... )
    >>> config = StellicareConfig()
    >>> result = await process_files_sequentially(
    ...     file_paths=[Path("audio1.wav"), Path("audio2.wav")],
    ...     config=config,
    ...     progress_callback=lambda msg: print(msg),
    ... )
    >>> refined = await refine_transcript(result.raw_transcript, config)
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from hsttb.core.exceptions import (
    StellicareConnectionError,
    StellicareRefineError,
    StellicareTranscriptionError,
)

logger = logging.getLogger(__name__)


class StellicareConfig(BaseModel):
    """
    Configuration for Stellicare API connections.

    Attributes:
        wss_url: WebSocket URL for Stellicare audio streaming.
        refine_url: REST API URL for transcript refinement.
        chunk_size_bytes: Size of binary audio chunks to send over WSS.
        connection_timeout: Timeout for WSS connection in seconds.
        read_timeout: Timeout for waiting for remaining responses in seconds.
    """

    wss_url: str = "wss://dev-lunagen.com/transcript/ws/audio?token=123456"
    refine_url: str = "https://dev-lunagen.com/transcript/refine/test"
    chunk_size_bytes: int = 4096
    connection_timeout: float = 30.0
    read_timeout: float = 120.0

    @classmethod
    def from_env(cls) -> StellicareConfig:
        """
        Create config from environment variables with defaults.

        Environment variables:
            STELLICARE_WSS_URL: Override WSS URL.
            STELLICARE_REFINE_URL: Override refine API URL.
            STELLICARE_CHUNK_SIZE: Override chunk size in bytes.
            STELLICARE_CONNECTION_TIMEOUT: Override connection timeout.
            STELLICARE_READ_TIMEOUT: Override read timeout.

        Returns:
            StellicareConfig with environment overrides applied.
        """
        kwargs: dict[str, Any] = {}
        if url := os.environ.get("STELLICARE_WSS_URL"):
            kwargs["wss_url"] = url
        if url := os.environ.get("STELLICARE_REFINE_URL"):
            kwargs["refine_url"] = url
        if chunk_size := os.environ.get("STELLICARE_CHUNK_SIZE"):
            kwargs["chunk_size_bytes"] = int(chunk_size)
        if timeout := os.environ.get("STELLICARE_CONNECTION_TIMEOUT"):
            kwargs["connection_timeout"] = float(timeout)
        if timeout := os.environ.get("STELLICARE_READ_TIMEOUT"):
            kwargs["read_timeout"] = float(timeout)
        return cls(**kwargs)


@dataclass
class StellicarePhrase:
    """
    A single transcript phrase received from Stellicare.

    Attributes:
        text: The transcribed text.
        is_final: Whether this is a final (committed) phrase.
        file_index: Index of the audio file this phrase belongs to.
    """

    text: str
    is_final: bool
    file_index: int = 0


@dataclass
class StellicareResult:
    """
    Result of processing audio files through Stellicare.

    Attributes:
        raw_transcript: All FINAL phrases concatenated.
        phrases: All phrases received (partial and final).
        files_processed: Number of files successfully processed.
        per_file_transcripts: Transcript for each individual file.
    """

    raw_transcript: str
    phrases: list[StellicarePhrase] = field(default_factory=list)
    files_processed: int = 0
    per_file_transcripts: list[str] = field(default_factory=list)


async def _invoke_callback(
    callback: Callable[..., Any] | None, *args: Any
) -> None:
    """Invoke a callback, awaiting it if it's a coroutine function."""
    if callback is None:
        return
    result = callback(*args)
    if inspect.isawaitable(result):
        await result


async def stream_audio_to_stellicare(
    file_path: Path,
    config: StellicareConfig,
    file_index: int = 0,
    on_phrase: Callable[[str, bool], Any] | None = None,
) -> list[StellicarePhrase]:
    """
    Stream a WAV file to Stellicare WSS and collect transcript phrases.

    Opens a WebSocket connection, sends the WAV file as binary chunks,
    and receives transcript phrases. Concurrently sends audio and
    receives responses.

    Args:
        file_path: Path to the WAV audio file.
        config: Stellicare configuration.
        file_index: Index of this file in the batch (for tracking).
        on_phrase: Optional callback invoked for each phrase received.

    Returns:
        List of all phrases received (both partial and final).

    Raises:
        StellicareConnectionError: If WebSocket connection fails.
        StellicareTranscriptionError: If streaming or reception fails.
        FileNotFoundError: If the audio file does not exist.
    """
    import websockets
    from websockets.exceptions import (
        ConnectionClosed,
        InvalidStatusCode,
        WebSocketException,
    )

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    phrases: list[StellicarePhrase] = []
    # Stellicare alternates between two streams:
    #   Even messages (0, 2, 4...) = committed transcript (grows)
    #   Odd messages  (1, 3, 5...) = active word being recognized
    # Full transcript = last committed + unique tail from last active
    committed_texts: list[str] = []  # even-indexed interims
    active_texts: list[str] = []     # odd-indexed interims
    interim_count = 0

    try:
        async with websockets.connect(
            config.wss_url,
            open_timeout=config.connection_timeout,
            close_timeout=10,
        ) as ws:
            receive_done = asyncio.Event()
            all_audio_sent = asyncio.Event()
            # Idle timeout: seconds of silence after all audio sent
            # before we consider the stream complete.
            idle_timeout_secs = 5.0

            async def receive_phrases() -> None:
                """Receive and parse transcript phrases from Stellicare.

                Stellicare sends pipe-delimited messages:
                  STATUS|Voice recording session started
                  INTERIM|partial transcript text
                Messages alternate between committed text and current word.

                Uses per-message timeouts after all audio is sent to detect
                when Stellicare has finished sending transcript data.
                """
                nonlocal interim_count
                try:
                    while True:
                        # Use a short recv timeout once all audio is sent,
                        # so we detect when Stellicare stops sending.
                        if all_audio_sent.is_set():
                            try:
                                raw_message = await asyncio.wait_for(
                                    ws.recv(), timeout=idle_timeout_secs
                                )
                            except asyncio.TimeoutError:
                                logger.debug(
                                    f"No message for {idle_timeout_secs}s "
                                    "after audio sent — stream complete"
                                )
                                return
                        else:
                            raw_message = await ws.recv()

                        try:
                            if isinstance(raw_message, bytes):
                                raw_message = raw_message.decode("utf-8")
                        except UnicodeDecodeError:
                            logger.debug("Non-text message from Stellicare, skipping")
                            continue

                        # Parse pipe-delimited format: TYPE|text
                        if "|" in raw_message:
                            msg_type, text = raw_message.split("|", 1)
                            msg_type = msg_type.strip().upper()
                        else:
                            # Try JSON fallback for forward compatibility
                            try:
                                data = json.loads(raw_message)
                                text = data.get("text", data.get("transcript", ""))
                                msg_type = "FINAL" if data.get("is_final") else "INTERIM"
                            except json.JSONDecodeError:
                                logger.debug(f"Unknown message format: {raw_message!r}")
                                continue

                        if msg_type == "STATUS":
                            logger.info(f"Stellicare status: {text}")
                            continue

                        if msg_type in ("INTERIM", "FINAL"):
                            # Track alternating streams separately
                            if interim_count % 2 == 0:
                                committed_texts.append(text)
                            else:
                                active_texts.append(text)
                            interim_count += 1

                            phrase = StellicarePhrase(
                                text=text,
                                is_final=(msg_type == "FINAL"),
                                file_index=file_index,
                            )
                            phrases.append(phrase)

                            # Show interim progress to UI (not final yet)
                            await _invoke_callback(on_phrase, text, False)

                            logger.debug(
                                f"Stellicare {msg_type}: {text[:80]}"
                            )

                            # FINAL message means transcript is complete
                            if msg_type == "FINAL":
                                return
                        else:
                            logger.debug(f"Unknown Stellicare message type: {msg_type}")

                except ConnectionClosed:
                    pass
                except Exception as e:
                    logger.warning(f"Error receiving Stellicare phrases: {e}")
                finally:
                    receive_done.set()

            # Start receiving in the background
            receive_task = asyncio.create_task(receive_phrases())

            # Send audio data in chunks
            file_size = file_path.stat().st_size
            bytes_sent = 0

            try:
                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(config.chunk_size_bytes)
                        if not chunk:
                            break
                        await ws.send(chunk)
                        bytes_sent += len(chunk)
                        logger.debug(
                            f"Sent {bytes_sent}/{file_size} bytes "
                            f"({bytes_sent * 100 // file_size}%)"
                        )
            except (ConnectionClosed, WebSocketException) as e:
                raise StellicareTranscriptionError(
                    f"Connection lost while streaming audio: {e}"
                ) from e

            logger.info(
                f"All audio sent for {file_path.name} "
                f"({bytes_sent} bytes). Waiting for final transcript..."
            )

            # Signal to the receive loop that audio is done.
            # The receive loop will now use a short idle timeout
            # to detect when Stellicare stops sending.
            all_audio_sent.set()

            # Wait for receive loop to finish (FINAL, idle, or hard timeout)
            try:
                await asyncio.wait_for(
                    receive_done.wait(), timeout=config.read_timeout
                )
            except asyncio.TimeoutError:
                logger.info(
                    "Stellicare response stream ended (hard timeout). "
                    "Proceeding with phrases received."
                )

            # Clean up
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

    except (InvalidStatusCode, OSError, WebSocketException) as e:
        raise StellicareConnectionError(
            f"Failed to connect to Stellicare WSS: {e}"
        ) from e
    except StellicareTranscriptionError:
        raise
    except Exception as e:
        raise StellicareTranscriptionError(
            f"Unexpected error during Stellicare streaming: {e}"
        ) from e

    # Assemble final transcript from Stellicare's two alternating streams:
    #   Committed (even msgs): grows progressively, the stable transcript
    #   Active (odd msgs): word(s) currently being recognized
    # Full transcript = last committed + last active (if it adds new words)
    # No deduplication — output is passed through as Stellicare produced it.
    committed = committed_texts[-1] if committed_texts else ""
    active = active_texts[-1] if active_texts else ""

    if committed and active:
        # Only append active if it's not already at the end of committed
        committed_lower = committed.lower().rstrip()
        active_lower = active.lower().strip()
        if committed_lower.endswith(active_lower):
            final_text = committed
        else:
            final_text = f"{committed} {active}".strip()
    else:
        final_text = committed or active

    if final_text:
        final_phrase = StellicarePhrase(
            text=final_text,
            is_final=True,
            file_index=file_index,
        )
        phrases.append(final_phrase)
        await _invoke_callback(on_phrase, final_text, True)

    logger.info(
        f"Streamed {file_path.name}: {interim_count} interims, "
        f"final transcript: {final_text[:100]!r}"
    )

    return phrases


async def refine_transcript(
    raw_transcript: str, config: StellicareConfig
) -> str:
    """
    Refine a transcript using the Stellicare refinement API.

    Sends the raw transcript to the Stellicare PUT endpoint
    and returns the refined version.

    Args:
        raw_transcript: The unrefined transcript text.
        config: Stellicare configuration.

    Returns:
        The refined transcript text.

    Raises:
        StellicareRefineError: If the API call fails.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.put(
                config.refine_url,
                json={"transcript": raw_transcript},
            )
            response.raise_for_status()

            data = response.json()
            refined = data.get("transcript", raw_transcript)

            logger.info(
                f"Transcript refined: {len(raw_transcript)} chars -> "
                f"{len(refined)} chars"
            )
            return refined

    except httpx.HTTPStatusError as e:
        raise StellicareRefineError(
            f"Stellicare refine API returned {e.response.status_code}: "
            f"{e.response.text}"
        ) from e
    except httpx.RequestError as e:
        raise StellicareRefineError(
            f"Failed to reach Stellicare refine API: {e}"
        ) from e
    except Exception as e:
        raise StellicareRefineError(
            f"Unexpected error during transcript refinement: {e}"
        ) from e


async def process_files_sequentially(
    file_paths: list[Path],
    config: StellicareConfig,
    progress_callback: Callable[[dict[str, Any]], Any] | None = None,
) -> StellicareResult:
    """
    Process multiple audio files through Stellicare sequentially.

    Streams each file one at a time to the Stellicare WSS endpoint,
    collecting transcripts. Reports progress via the callback.

    Args:
        file_paths: List of WAV file paths to process in order.
        config: Stellicare configuration.
        progress_callback: Optional callback for progress updates.
            Can be sync or async. Receives dicts with keys:
            type, file_index, total_files, etc.

    Returns:
        StellicareResult with combined transcript from all files.

    Raises:
        StellicareConnectionError: If WSS connection fails.
        StellicareTranscriptionError: If streaming fails.
    """
    all_phrases: list[StellicarePhrase] = []
    per_file_transcripts: list[str] = []
    total_files = len(file_paths)

    async def _notify(msg: dict[str, Any]) -> None:
        await _invoke_callback(progress_callback, msg)

    for i, file_path in enumerate(file_paths):
        await _notify({
            "type": "progress",
            "file_index": i,
            "total_files": total_files,
            "status": "streaming",
            "filename": file_path.name,
        })

        async def on_phrase(text: str, is_final: bool, _idx: int = i) -> None:
            await _notify({
                "type": "phrase",
                "text": text,
                "is_final": is_final,
                "file_index": _idx,
            })

        try:
            phrases = await stream_audio_to_stellicare(
                file_path=file_path,
                config=config,
                file_index=i,
                on_phrase=on_phrase,
            )
            all_phrases.extend(phrases)

            # Build per-file transcript from FINAL phrases only
            final_phrases = [p.text for p in phrases if p.is_final]
            file_transcript = " ".join(final_phrases)
            per_file_transcripts.append(file_transcript)

            await _notify({
                "type": "file_complete",
                "file_index": i,
                "total_files": total_files,
                "transcript": file_transcript,
            })

        except (StellicareConnectionError, StellicareTranscriptionError) as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            await _notify({
                "type": "error",
                "file_index": i,
                "total_files": total_files,
                "error": str(e),
                "filename": file_path.name,
            })
            per_file_transcripts.append("")

    # Build combined transcript from all FINAL phrases
    raw_transcript = " ".join(t for t in per_file_transcripts if t)

    await _notify({
        "type": "all_complete",
        "raw_transcript": raw_transcript,
        "total_files": total_files,
        "files_processed": sum(1 for t in per_file_transcripts if t),
    })

    return StellicareResult(
        raw_transcript=raw_transcript,
        phrases=all_phrases,
        files_processed=sum(1 for t in per_file_transcripts if t),
        per_file_transcripts=per_file_transcripts,
    )
