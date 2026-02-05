"""
HSTTB Command Line Interface.

This module provides the main CLI entry point for the HSTTB framework,
including commands for benchmarking, transcription, and profile management.

Example:
    >>> # Run from command line
    >>> hsttb --help
    >>> hsttb transcribe audio.wav
    >>> hsttb profiles
    >>> hsttb adapters
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from hsttb import __version__


@click.group()
@click.version_option(version=__version__, prog_name="hsttb")
def cli() -> None:
    """Lunagen Speech-to-Text Benchmarking Tool.

    A model-agnostic evaluation framework for healthcare speech-to-text
    with streaming simulation and medical accuracy metrics.
    """


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--adapter",
    "-a",
    default="mock",
    help="STT adapter to use (default: mock)",
)
@click.option(
    "--profile",
    "-p",
    default="ideal",
    help="Streaming profile to use (default: ideal)",
)
@click.option(
    "--responses",
    "-r",
    multiple=True,
    help="Mock responses (for mock adapter)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
def transcribe(
    audio_file: Path,
    adapter: str,
    profile: str,
    responses: tuple[str, ...],
    seed: int,
) -> None:
    """Transcribe an audio file using streaming simulation.

    Simulates real-time streaming conditions and outputs transcript segments.
    """
    asyncio.run(_transcribe(audio_file, adapter, profile, responses, seed))


async def _transcribe(
    audio_file: Path,
    adapter_name: str,
    profile_name: str,
    responses: tuple[str, ...],
    seed: int,
) -> None:
    """Async implementation of transcribe command."""
    from hsttb.adapters import get_adapter
    from hsttb.audio.chunker import StreamingChunker
    from hsttb.audio.loader import AudioLoader
    from hsttb.core.config import load_profile

    # Load audio
    click.echo(f"Loading audio: {audio_file}")
    loader = AudioLoader()
    audio_data, sample_rate = loader.load(audio_file)
    click.echo(f"  Sample rate: {sample_rate} Hz")
    click.echo(f"  Duration: {len(audio_data) / sample_rate:.2f}s")

    # Load profile
    profile = load_profile(profile_name)
    click.echo(f"Using profile: {profile.profile_name}")
    click.echo(f"  Chunk size: {profile.chunking.chunk_size_ms}ms")

    # Create chunker
    chunker = StreamingChunker(profile, seed=seed)

    # Get adapter
    if adapter_name == "mock" and responses:
        stt_adapter = get_adapter(adapter_name, responses=list(responses))
    else:
        stt_adapter = get_adapter(adapter_name)

    click.echo(f"Using adapter: {stt_adapter.name}")
    click.echo()
    click.echo("--- Transcription ---")

    # Stream and transcribe
    async with stt_adapter as active_adapter:
        audio_stream = chunker.stream_audio(audio_data, sample_rate)
        async for segment in active_adapter.transcribe_stream(audio_stream):
            if segment.is_final:
                click.echo(f"[FINAL] {segment.text}")
            else:
                click.echo(f"[PARTIAL] {segment.text}")

    click.echo("--- Done ---")


@cli.command(name="profiles")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed profile info")
def list_profiles(verbose: bool) -> None:
    """List available streaming profiles."""
    from hsttb.core.config import BUILTIN_PROFILES

    click.echo("Available Streaming Profiles:")
    click.echo()

    for name, profile in BUILTIN_PROFILES.items():
        if verbose:
            click.echo(f"  {name}:")
            click.echo(f"    Description: {profile.description}")
            click.echo(f"    Chunk size: {profile.chunking.chunk_size_ms}ms")
            click.echo(f"    Chunk jitter: {profile.chunking.chunk_jitter_ms}ms")
            click.echo(f"    Network delay: {profile.network.delay_ms}ms")
            click.echo()
        else:
            click.echo(f"  {name}: {profile.description}")


@cli.command(name="adapters")
def list_adapters() -> None:
    """List available STT adapters."""
    from hsttb.adapters import list_adapters as get_adapters

    click.echo("Available STT Adapters:")
    click.echo()

    for name in get_adapters():
        click.echo(f"  {name}")


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
def info(audio_file: Path) -> None:
    """Show information about an audio file."""
    from hsttb.audio.loader import AudioLoader

    loader = AudioLoader()
    file_info = loader.get_info(audio_file)
    checksum = loader.get_checksum(audio_file)

    click.echo(f"Audio File: {audio_file}")
    click.echo()
    click.echo(f"  Format: {file_info['format']}")
    click.echo(f"  Subtype: {file_info['subtype']}")
    click.echo(f"  Sample rate: {file_info['sample_rate']} Hz")
    click.echo(f"  Channels: {file_info['channels']}")
    click.echo(f"  Duration: {file_info['duration_seconds']:.2f}s")
    click.echo(f"  Frames: {file_info['frames']}")
    click.echo(f"  Checksum: {checksum[:16]}...")


@cli.command()
@click.option(
    "--profile",
    "-p",
    default="ideal",
    help="Streaming profile to use",
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.argument("duration_ms", type=int, default=5000)
def simulate(profile: str, seed: int, duration_ms: int) -> None:
    """Simulate chunk boundaries for given duration.

    Useful for understanding how audio will be chunked.
    """
    from hsttb.audio.chunker import StreamingChunker
    from hsttb.core.config import load_profile

    streaming_profile = load_profile(profile)
    chunker = StreamingChunker(streaming_profile, seed=seed)

    click.echo(f"Profile: {profile}")
    click.echo(f"Duration: {duration_ms}ms")
    click.echo()

    boundaries = chunker.get_chunk_boundaries(duration_ms)
    expected = chunker.get_expected_chunks(duration_ms)

    click.echo(f"Expected chunks: {expected}")
    click.echo()
    click.echo("Chunk boundaries:")
    for i, (start, end) in enumerate(boundaries):
        click.echo(f"  Chunk {i}: {start}ms - {end}ms ({end - start}ms)")


def main() -> int:
    """
    Main entry point for the HSTTB CLI.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    try:
        cli()
        return 0
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
