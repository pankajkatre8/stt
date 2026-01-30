"""
HSTTB Command Line Interface.

This module provides the main CLI entry point for the HSTTB framework.
"""
from __future__ import annotations

import sys


def main() -> int:
    """
    Main entry point for the HSTTB CLI.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    print("HSTTB - Healthcare Streaming STT Benchmarking")
    print("Version: 0.1.0")
    print()
    print("Commands:")
    print("  benchmark    Run benchmark on audio files")
    print("  compare      Compare benchmark results")
    print("  profiles     List available streaming profiles")
    print("  adapters     List available STT adapters")
    print()
    print("Use --help with any command for more information.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
